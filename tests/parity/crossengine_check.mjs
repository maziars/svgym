// crossengine_check.mjs — independent, non-circular safety check.
//
// For every canvas-gated output (tests/parity/out/browser/*.svg) it renders the
// OPTIMIZED svg and its ORIGINAL in three real browser engines (Chromium,
// Firefox, WebKit) and computes SSIM(original, optimized) in each. The gate
// already guarantees high SSIM in Chromium (it gated there); Firefox + WebKit
// are the genuinely independent judges. A file is "cross-browser safe" if its
// WORST engine SSIM still clears the level's bar.
//
// Setup:  npx playwright install firefox webkit   (chromium already installed)
// Run:    node tests/parity/crossengine_check.mjs
//         node tests/parity/crossengine_check.mjs --engines chromium firefox

import { chromium, firefox, webkit } from "playwright";
import http from "node:http";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const PUB_ROOT = path.resolve(__dirname, "../..");
const CORPUS   = path.resolve(__dirname, "../../../demo/svgs");
const OUT_DIR  = path.join(__dirname, "out", "browser");
const SIZE = 512;
const BAR = { lossless: 0.999, conservative: 0.99, aggressive: 0.97 };
const ENGINES = { chromium, firefox, webkit };

const MIME = { ".html": "text/html", ".js": "text/javascript" };
function serve(root) {
  return new Promise(resolve => {
    const srv = http.createServer((req, res) => {
      const fp = path.join(root, decodeURIComponent(req.url.split("?")[0]));
      if (!fp.startsWith(root) || !fs.existsSync(fp) || fs.statSync(fp).isDirectory()) { res.writeHead(404); return res.end("nf"); }
      res.writeHead(200, { "content-type": MIME[path.extname(fp)] || "application/octet-stream" });
      fs.createReadStream(fp).pipe(res);
    });
    srv.listen(0, () => resolve({ srv, port: srv.address().port }));
  });
}

function origFor(name) {
  const hit = [];
  (function walk(d){ for (const e of fs.readdirSync(d, { withFileTypes: true })) {
    const p = path.join(d, e.name);
    if (e.isDirectory()) walk(p); else if (e.name === name + ".svg") hit.push(p);
  }})(CORPUS);
  return hit[0];
}

const main = async () => {
  const args = process.argv.slice(2);
  const ei = args.indexOf("--engines");
  const want = ei >= 0 ? args.slice(ei + 1) : ["chromium", "firefox", "webkit"];

  const outs = fs.readdirSync(OUT_DIR).filter(f => f.endsWith(".svg"));
  if (!outs.length) { console.error("no outputs in", OUT_DIR, "— run run_browser.mjs --all first"); process.exit(1); }
  const cases = outs.map(f => {
    const m = /^(.*)__(lossless|conservative|aggressive)\.svg$/.exec(f);
    return m ? { name: m[1], level: m[2], opt: path.join(OUT_DIR, f) } : null;
  }).filter(Boolean);

  const { srv, port } = await serve(PUB_ROOT);
  const results = {}; // key name__level -> {engine: ssim}

  for (const eng of want) {
    const launcher = ENGINES[eng];
    if (!launcher) { console.log("skip unknown engine", eng); continue; }
    console.log(`\n== ${eng} ==`);
    const browser = await launcher.launch();
    const page = await browser.newPage();
    await page.goto(`http://localhost:${port}/tests/parity/crossengine.html`);
    await page.waitForFunction("window.judgeReady === true", null, { timeout: 60000 });
    let i = 0;
    for (const c of cases) {
      const op = origFor(c.name);
      if (!op) continue;
      const orig = fs.readFileSync(op, "utf8"), opt = fs.readFileSync(c.opt, "utf8");
      let s;
      try { s = await page.evaluate(([o, p, z]) => window.measure(o, p, z), [orig, opt, SIZE]); }
      catch (e) { s = null; }
      (results[`${c.name}__${c.level}`] ||= { name: c.name, level: c.level })[eng] = s == null ? null : +s.toFixed(4);
      if (++i % 40 === 0) console.log(`  ${i}/${cases.length}`);
    }
    await browser.close();
  }
  srv.close();

  // ---- report: worst engine SSIM per file, flag below bar ----
  const byLevel = {};
  for (const k of Object.keys(results)) {
    const r = results[k];
    const vals = want.map(e => r[e]).filter(v => v != null);
    const worst = vals.length ? Math.min(...vals) : null;
    r.worst = worst;
    (byLevel[r.level] ||= []).push(r);
  }
  console.log("\n==== cross-engine safety (worst of " + want.join("/") + ") ====");
  for (const level of ["lossless", "conservative", "aggressive"]) {
    const rows = (byLevel[level] || []).filter(r => r.worst != null);
    if (!rows.length) continue;
    rows.sort((a, b) => a.worst - b.worst);
    const bar = BAR[level];
    const below = rows.filter(r => r.worst < bar);
    const med = rows[Math.floor(rows.length / 2)].worst;
    console.log(`\n  ${level}  (n=${rows.length}, bar ${bar})`);
    console.log(`    worst-engine SSIM: min ${rows[0].worst}  median ${med}`);
    console.log(`    below bar in some engine: ${below.length}`);
    for (const r of below.slice(0, 12))
      console.log(`      ${r.name.padEnd(26)} ${want.map(e => e[0] + ":" + (r[e] ?? "—")).join("  ")}`);
  }
  fs.writeFileSync(path.join(__dirname, "crossengine_results.json"), JSON.stringify(results, null, 1));
  console.log(`\nWrote ${path.join(__dirname, "crossengine_results.json")}`);
};

main().catch(e => { console.error(e); process.exit(1); });
