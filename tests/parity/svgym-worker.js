// svgym-worker.js — Pyodide worker that runs the REAL deterministic pipeline.
//
// The pipeline's Python is unmodified. The gate's rasterizer is bridged
// SYNCHRONOUSLY to the MAIN thread's <canvas> via a SharedArrayBuffer + Atomics:
// the worker writes the SVG, signals main, and Atomics.wait()s; main renders on
// a real <canvas> (actual browser engine + the user's real fonts) and notifies.
// So rendering matches what the user will SEE, while Python stays synchronous.
//
// SAB layout:  Int32Array CTRL[0..15] in the first 64 bytes, then Uint8Array DATA.
//   CTRL[0] state: 0 idle, 1 request pending, 2 done-ok, 3 done-error
//   CTRL[1] input  SVG byte length
//   CTRL[2] render size (px, square)
//   CTRL[3] output RGB byte length (= size*size*3)

const PYODIDE = "https://cdn.jsdelivr.net/pyodide/v0.26.2/full/";
const MODULES = ["__init__.py", "config.py", "tools.py", "deterministic.py"];
const RENDER_SIZE = 512;

let CTRL, DATA, pyodide;

function renderViaMain(svgText, size) {
  const bytes = new TextEncoder().encode(svgText);
  if (bytes.length > DATA.length) return null; // SVG larger than the SAB region
  DATA.set(bytes, 0);
  Atomics.store(CTRL, 1, bytes.length);
  Atomics.store(CTRL, 2, size);
  Atomics.store(CTRL, 0, 1);              // STATE = request
  self.postMessage({ type: "render" });   // wake the main thread
  Atomics.wait(CTRL, 0, 1);               // block here until main flips STATE
  if (Atomics.load(CTRL, 0) === 3) return null;  // render failed
  const len = Atomics.load(CTRL, 3);
  return DATA.slice(0, len);              // RGB bytes (size*size*3)
}

async function init(sab, base) {
  CTRL = new Int32Array(sab, 0, 16);
  DATA = new Uint8Array(sab, 64);

  importScripts(PYODIDE + "pyodide.js");
  pyodide = await loadPyodide({ indexURL: PYODIDE });
  await pyodide.loadPackage(["numpy", "scipy"]);
  try {
    await pyodide.loadPackage("micropip");
    await pyodide.runPythonAsync('import micropip; await micropip.install("fonttools")');
  } catch (e) { /* font subsetting is optional */ }

  pyodide.FS.mkdirTree("/pkg/svgym");
  for (const m of MODULES) {
    const code = await (await fetch(`${base}/${m}`)).text();
    pyodide.FS.writeFile(`/pkg/svgym/${m}`, code);
  }

  // SVGO's own browser build, so the in-browser pipeline runs the same SVGO
  // baseline as the CLI (then optimizes beyond it). Falls back to a no-op if it
  // can't load -- the pipeline supports running without the SVGO step.
  let __svgo = null;
  try { __svgo = (await import("https://cdn.jsdelivr.net/npm/svgo@3.3.2/dist/svgo.browser.js")).optimize; }
  catch (e) { console.warn("svgo worker-load failed (will use main-thread svgo if provided):", e); }
  self.__svgo_result = null;  // SVGO output provided per-run by the main thread (preferred, reliable)
  self.__run_svgo = (svg) => {
    if (self.__svgo_result) return self.__svgo_result;
    try { return __svgo ? __svgo(svg, { multipass: true }).data : svg; }
    catch (e) { return svg; }
  };

  self.__render = (svg, size) => renderViaMain(svg, size);
  pyodide.runPython(`
import sys; sys.path.insert(0, "/pkg")
import numpy as np, js
import svgym.tools as T
import svgym.deterministic as D

def _browser_render(svg_text, size=${RENDER_SIZE}):
    buf = js.__render(svg_text, size)
    if buf is None:
        return None
    arr = np.asarray(buf.to_py(), dtype=np.uint8)
    if arr.size != size * size * 3:
        return None
    return arr.reshape((size, size, 3))

# compare() looks up render_svg in tools' globals at call time -> swaps the
# whole gate's rasterizer to the canvas bridge without touching any other code.
T.render_svg = _browser_render
D.run_svgo = lambda s: js.__run_svgo(s)   # SVGO browser build (same baseline as the CLI)

import json, hashlib
def _run(svg, level):
    r = D.optimize_svg_deterministic(svg, level=level)
    opt = r.get("optimized_svg") or svg
    traj = [[s.get("tool"), s.get("status"), s.get("args", {})]
            for s in r.get("tool_trajectory", [])]
    svgo_ssim = next((s.get("ssim") for s in r.get("tool_trajectory", [])
                      if s.get("tool") == "run_svgo" and s.get("ssim") is not None), None)
    psnr = r.get("psnr")
    return json.dumps({
        "size": len(opt.encode("utf-8")),
        "ssim": r.get("ssim"),
        "psnr": None if psnr in (float("inf"),) else psnr,
        "sha256": hashlib.sha256(opt.encode("utf-8")).hexdigest(),
        "decisions": traj,
        "optimized_svg": opt,
        "svgo_size": r.get("svgo_size"),
        "svgo_ssim": svgo_ssim,
    })
`);

  // Self-test the worker -> canvas -> worker render bridge end to end.
  let selftest = null;
  try {
    selftest = pyodide.runPython(`
import numpy as np
__t = _browser_render('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 10 10"><rect x="0" y="0" width="10" height="10" fill="black"/></svg>', 512)
("none" if __t is None else f"shape={__t.shape} mean={int(__t.mean())}")
`);
  } catch (e) { selftest = "exc: " + (e && e.message || e); }

  self.postMessage({ type: "ready", selftest });
}

self.onmessage = (e) => {
  const d = e.data;
  if (d.type === "init") {
    init(d.sab, d.base).catch(err =>
      self.postMessage({ type: "error", error: String(err && err.stack || err) }));
  } else if (d.type === "run") {
    try {
      self.__svgo_result = d.svgo || null;   // SVGO baseline from the main thread, if provided
      const run = pyodide.globals.get("_run");
      const out = run(d.svg, d.level);      // synchronous; renders block via Atomics
      run.destroy && run.destroy();
      self.postMessage({ type: "result", id: d.id, result: JSON.parse(out) });
    } catch (err) {
      self.postMessage({ type: "result", id: d.id, error: String(err && err.stack || err) });
    }
  }
};
