// run_browser.mjs — drive the in-browser pipeline over the corpus with Playwright
// and emit browser_results.json (+ out/browser/*.svg) in the same schema as
// run_native.py, so compare_parity.py can diff them.
//
// Setup (on a machine with a real browser):
//     npm i -D playwright && npx playwright install chromium
// Run:
//     node tests/parity/run_browser.mjs                 # default subset
//     node tests/parity/run_browser.mjs --all
//     node tests/parity/run_browser.mjs --svgs glyph-k tux
//
// It serves svgym-public/ statically (so the page can fetch the svgym .py
// modules) and reads the SVG corpus straight from disk.

import { chromium } from "playwright";
import http from "node:http";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const PUB_ROOT = path.resolve(__dirname, "../..");            // svgym-public/
const CORPUS  = path.resolve(__dirname, "../../../demo/svgs"); // shared corpus
const OUT_DIR = path.join(__dirname, "out", "browser");
const LEVELS  = ["lossless", "conservative", "aggressive"];

const DEFAULT_SUBSET = [
  "heroicons-camera", "simple-github", "phosphor-apple-logo",
  "glyph-k", "glyph-w", "ir-lion-sun", "hr-croatia", "openmoji-dragon",
  "tux", "gophers-9", "neo4j-graph", "ggplot-timeseries", "whale",
  "stop-sign", "3-dots-move", "hover-interactive",
];

const MIME = { ".html": "text/html", ".js": "text/javascript", ".mjs": "text/javascript",
  ".py": "text/plain", ".wasm": "application/wasm", ".json": "application/json",
  ".svg": "image/svg+xml", ".css": "text/css" };

function serve(root) {
  return new Promise(resolve => {
    const srv = http.createServer((req, res) => {
      const fp = path.join(root, decodeURIComponent(req.url.split("?")[0]));
      if (!fp.startsWith(root) || !fs.existsSync(fp) || fs.statSync(fp).isDirectory()) {
        res.writeHead(404); return res.end("nf");
      }
      // Cross-origin isolation -> SharedArrayBuffer (needed by the canvas harness).
      res.writeHead(200, {
        "content-type": MIME[path.extname(fp)] || "application/octet-stream",
        "Cross-Origin-Opener-Policy": "same-origin",
        "Cross-Origin-Embedder-Policy": "credentialless",
      });
      fs.createReadStream(fp).pipe(res);
    });
    srv.listen(0, () => resolve({ srv, port: srv.address().port }));
  });
}

function allSvgs() {
  const out = [];
  (function walk(d) {
    for (const e of fs.readdirSync(d, { withFileTypes: true })) {
      const p = path.join(d, e.name);
      if (e.isDirectory()) { if (e.name !== "breakage") walk(p); }
      else if (e.name.endsWith(".svg")) out.push(p);
    }
  })(CORPUS);
  return out;
}

function pickCorpus(args) {
  const all = allSvgs();
  if (args.includes("--all")) return all;
  const i = args.indexOf("--svgs");
  const want = new Set(i >= 0 ? args.slice(i + 1) : DEFAULT_SUBSET);
  return all.filter(p => want.has(path.basename(p, ".svg")));
}

const main = async () => {
  const args = process.argv.slice(2);
  const files = pickCorpus(args);
  if (!files.length) { console.error("no corpus svgs found under", CORPUS); process.exit(1); }
  fs.mkdirSync(OUT_DIR, { recursive: true });

  const { srv, port } = await serve(PUB_ROOT);
  const browser = await chromium.launch({ args: ["--enable-features=SharedArrayBuffer"] });
  const page = await browser.newPage();
  page.on("console", m => { if (m.type() === "error" || m.type() === "warning") console.log("[page]", m.text()); });
  page.on("pageerror", err => console.log("[pageerror]", err.message));

  const HARNESS = process.env.HARNESS || "harness-canvas.html";   // canvas-gated (option A)
  await page.goto(`http://localhost:${port}/tests/parity/${HARNESS}`);
  console.log(`booting pipeline in browser (${HARNESS})…`);
  await page.waitForFunction("window.svgymReady === true || window.svgymError", null, { timeout: 240000 });
  if (await page.evaluate("window.svgymError || null"))
    throw new Error("harness init failed: " + await page.evaluate("window.svgymError"));
  console.log("ready.\n");

  const records = [];
  for (const f of files) {
    const name = path.basename(f, ".svg");
    const svg = fs.readFileSync(f, "utf8");
    for (const level of LEVELS) {
      try {
        const r = await page.evaluate(([s, lv]) => window.runPipeline(s, lv), [svg, level]);
        fs.writeFileSync(path.join(OUT_DIR, `${name}__${level}.svg`), r.optimized_svg);
        delete r.optimized_svg;
        records.push({ name, level, original_size: Buffer.byteLength(svg), ...r });
        console.log(`  ${name.padEnd(28)} ${level.padEnd(12)} -> ${r.size} B  ssim=${r.ssim}  steps=${r.decisions.length}`);
      } catch (e) {
        records.push({ name, level, error: String(e) });
        console.log(`  ${name.padEnd(28)} ${level.padEnd(12)} ERROR ${e}`);
      }
    }
  }

  fs.writeFileSync(path.join(__dirname, "browser_results.json"), JSON.stringify(records, null, 1));
  console.log(`\nWrote ${path.join(__dirname, "browser_results.json")} (${records.length} records)`);
  console.log(`Optimized SVGs in ${OUT_DIR}/`);
  await browser.close();
  srv.close();
};

main().catch(e => { console.error(e); process.exit(1); });
