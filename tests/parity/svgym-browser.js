// svgym-browser.js
// Runs the REAL deterministic pipeline fully in the browser, no server.
//
// How it stays faithful to the native pipeline:
//   * Pyodide (CPython + numpy + scipy in WASM) runs the unmodified svgym
//     Python modules (config.py, tools.py, deterministic.py).
//   * The only thing swapped is the gate's rasterizer: native uses cairosvg,
//     here we use resvg-wasm, which is *synchronous*, so the Python pipeline
//     stays synchronous and unchanged (no async refactor, no SharedArrayBuffer).
//   * run_svgo is a no-op in the browser (no Node `svgo` binary); the pipeline
//     already supports running without the SVGO baseline step.
//
// Privacy: the SVG never leaves the page. Pyodide / resvg are fetched once as
// runtime code; all parsing, transforms, rendering and SSIM run locally.
//
// Exports: initSvgym(opts) -> Promise, runPipeline(svgText, level) -> result.

const DEFAULTS = {
  pyodideIndexURL: "https://cdn.jsdelivr.net/pyodide/v0.26.2/full/",
  resvgJsURL:      "https://esm.sh/@resvg/resvg-wasm@2.6.2",
  resvgWasmURL:    "https://cdn.jsdelivr.net/npm/@resvg/resvg-wasm@2.6.2/index_bg.wasm",
  // Where the svgym .py modules live, relative to the page. Default assumes the
  // package sits two dirs up (tests/parity/ -> svgym-public/svgym/).
  svgymBaseURL:    "../../svgym",
  modules: ["__init__.py", "config.py", "tools.py", "deterministic.py"],
  renderSize: 512,         // matches the native gate's default render size
  installFonttools: true,  // pure-Python; enables the font-subsetting tool
  // resvg-wasm ships with NO fonts, so it can't render <text> and the gate goes
  // blind to text-breaking transforms. Load at least one sans font so text is
  // rasterized and gross breakage (text out of place) is caught. Add more URLs
  // (CJK, etc.) for broader coverage. Non-fatal if a URL 404s.
  fontURLs: [
    "https://cdn.jsdelivr.net/npm/@expo-google-fonts/roboto/Roboto_400Regular.ttf",
  ],
};

let pyodide = null;
let Resvg = null;
let CFG = null;
let FONT_BUFFERS = [];

// Synchronous rasterizer exposed to Python: SVG text -> flat RGB Uint8Array
// (length size*size*3), white-composited. Mirrors cairosvg's square render:
// the SVG is scaled to fit size*size PRESERVING aspect ratio (xMidYMid meet)
// and letterboxed onto a white background. (cairosvg with output_width =
// output_height does NOT stretch; matching that is essential for the gate's
// SSIM to behave like the native pipeline.) Returns null on render failure so
// the Python gate treats it as a broken render (same as cairosvg -> None).
function renderRGB(svgText, size) {
  try {
    const fontOpt = { fontBuffers: FONT_BUFFERS, loadSystemFonts: false,
                      defaultFontFamily: "sans-serif" };

    // Probe the SVG's intrinsic aspect so we fit the LONGER side to `size`.
    const probe = new Resvg(svgText, { font: fontOpt });
    const iw = probe.width || 1, ih = probe.height || 1;
    probe.free && probe.free();
    const mode = iw >= ih ? "width" : "height";

    const r = new Resvg(svgText, {
      fitTo: { mode, value: size },
      background: "rgba(255,255,255,1)",
      font: fontOpt,
    });
    const img = r.render();
    const w = img.width, h = img.height;
    const rgba = img.pixels;            // Uint8Array, w*h*4, on white
    img.free && img.free();

    // Letterbox: center the w*h render onto a size*size white RGB buffer.
    const out = new Uint8Array(size * size * 3).fill(255);
    const offx = (size - w) >> 1, offy = (size - h) >> 1;
    for (let y = 0; y < h; y++) {
      const oy = y + offy;
      if (oy < 0 || oy >= size) continue;
      for (let x = 0; x < w; x++) {
        const ox = x + offx;
        if (ox < 0 || ox >= size) continue;
        const si = (y * w + x) * 4, di = (oy * size + ox) * 3;
        out[di] = rgba[si]; out[di + 1] = rgba[si + 1]; out[di + 2] = rgba[si + 2];
      }
    }
    return out;
  } catch (e) {
    console.warn("resvg render failed:", e);
    return null;
  }
}

export async function initSvgym(userOpts = {}) {
  CFG = { ...DEFAULTS, ...userOpts };

  // 1. resvg-wasm (synchronous rasterizer)
  const resvgMod = await import(/* webpackIgnore: true */ CFG.resvgJsURL);
  await resvgMod.initWasm(fetch(CFG.resvgWasmURL));
  Resvg = resvgMod.Resvg;

  // Load fonts so resvg can rasterize <text>; without this the gate is blind
  // to text-breaking transforms. Each failure is non-fatal.
  FONT_BUFFERS = [];
  for (const url of (CFG.fontURLs || [])) {
    try {
      const ab = await (await fetch(url)).arrayBuffer();
      FONT_BUFFERS.push(new Uint8Array(ab));
    } catch (e) { console.warn("font load failed:", url, e); }
  }
  if (!FONT_BUFFERS.length) console.warn("no fonts loaded; <text> verification will be unreliable");

  // 2. Pyodide + numpy + scipy
  if (!globalThis.loadPyodide) {
    await import(/* webpackIgnore: true */ CFG.pyodideIndexURL + "pyodide.mjs")
      .then(m => { globalThis.loadPyodide = m.loadPyodide; });
  }
  pyodide = await globalThis.loadPyodide({ indexURL: CFG.pyodideIndexURL });
  await pyodide.loadPackage(["numpy", "scipy"]);
  if (CFG.installFonttools) {
    try {
      await pyodide.loadPackage("micropip");
      await pyodide.runPythonAsync(
        `import micropip; await micropip.install("fonttools")`);
    } catch (e) { console.warn("fonttools install skipped:", e); }
  }

  // 3. Drop the svgym modules into Pyodide's filesystem
  pyodide.FS.mkdirTree("/pkg/svgym");
  for (const m of CFG.modules) {
    const code = await (await fetch(`${CFG.svgymBaseURL}/${m}`)).text();
    pyodide.FS.writeFile(`/pkg/svgym/${m}`, code);
  }

  // 4. Bridge JS renderer to Python and monkeypatch the pipeline
  globalThis.__svgym_render = (svgText, size) => renderRGB(svgText, size);
  pyodide.runPython(`
import sys; sys.path.insert(0, "/pkg")
import numpy as np, js
import svgym.tools as T
import svgym.deterministic as D

def _browser_render(svg_text, size=${CFG.renderSize}):
    buf = js.__svgym_render(svg_text, size)
    if buf is None:
        return None
    arr = np.asarray(buf.to_py(), dtype=np.uint8)
    return arr.reshape((size, size, 3))

# compare() looks up render_svg in tools' globals at call time, so this swaps
# the rasterizer for the whole gate without touching any other code.
T.render_svg = _browser_render
# No Node svgo in the browser: skip the SVGO baseline step (supported path).
D.run_svgo = lambda s: s

import json, hashlib
def _run(svg, level):
    r = D.optimize_svg_deterministic(svg, level=level)
    opt = r.get("optimized_svg") or svg
    traj = [[s.get("tool"), s.get("status"), s.get("args", {})]
            for s in r.get("tool_trajectory", [])]
    psnr = r.get("psnr")
    return json.dumps({
        "size": len(opt.encode("utf-8")),
        "ssim": r.get("ssim"),
        "psnr": None if psnr in (float("inf"),) else psnr,
        "sha256": hashlib.sha256(opt.encode("utf-8")).hexdigest(),
        "decisions": traj,
        "optimized_svg": opt,
    })
`);
}

export function runPipeline(svgText, level = "conservative") {
  const run = pyodide.globals.get("_run");
  const out = JSON.parse(run(svgText, level));
  run.destroy && run.destroy();
  return out;
}

export function ready() { return pyodide !== null && Resvg !== null; }
