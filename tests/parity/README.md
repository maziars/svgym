# Browser ↔ native parity test

Confirms that the **in-browser** deterministic pipeline (Pyodide + resvg-wasm,
fully local) produces the same results as the **native** pipeline (cairosvg).

The pipeline *code* is identical in both — Pyodide runs the unmodified `svgym`
Python. The only thing that differs is the gate's rasterizer (cairosvg vs
resvg-wasm), so this test is really asking: **do the renderer differences change
any accept/revert decisions, and if so, is the browser output still safe?**

What to expect:

- **lossless** — renderer-independent (no pixel-gated steps), so it should be
  **100% byte-identical**.
- **conservative / aggressive** — a few gated lossy steps (rounding, curve
  simplification, viewBox rescale) can flip at the margin. Goal: a high
  byte-identical rate, and every divergent file still **safe** (clears the
  level's SSIM bar under a neutral cairosvg re-render) and within a small size
  tolerance.

## Files

| file | what it is | runs where |
|---|---|---|
| `run_native.py` | native baseline (cairosvg) → `native_results.json`, `out/native/*.svg` | any machine with cairosvg |
| `svgym-browser.js` | the browser engine: Pyodide + resvg-wasm, runs the real pipeline locally | browser |
| `harness.html` | minimal page that loads the engine and exposes `window.runPipeline` | browser |
| `run_browser.mjs` | Playwright driver → `browser_results.json`, `out/browser/*.svg` | machine with a real browser |
| `compare_parity.py` | diffs the two runs and reports parity + safety | any machine with cairosvg |

## Run it

```bash
# from svgym-public/
# 1) native baseline (subset by default; --all for the whole corpus)
PYTHONPATH=. python tests/parity/run_native.py            # or --all / --svgs glyph-k tux

# 2) browser run (needs a real browser)
npm i -D playwright && npx playwright install chromium
node tests/parity/run_browser.mjs                          # match the same subset/--all

# 3) diff them
PYTHONPATH=. python tests/parity/compare_parity.py \
    --browser tests/parity/browser_results.json \
    --browser-dir tests/parity/out/browser
```

Keep the corpus selection identical between steps 1 and 2 (both default to the
same subset; pass `--all` to both for the full run).

## Self-test

`compare_parity.py` against the native run itself must report 100% identical —
a sanity check on the diff tool:

```bash
PYTHONPATH=. python tests/parity/compare_parity.py \
    --browser tests/parity/native_results.json --browser-dir tests/parity/out/native
```

## Try one SVG by hand

Serve `svgym-public/` and open `tests/parity/harness.html`:

```bash
python -m http.server 8000        # from svgym-public/
# then visit http://localhost:8000/tests/parity/harness.html
```

## Notes

- `run_svgo` is a no-op in the browser (no Node `svgo` binary); the pipeline
  runs without the SVGO baseline step, which it already supports. If you want
  full parity including SVGO, run the native baseline with `svgo` *off* too, or
  wire the SVGO browser build into `svgym-browser.js`.
- resvg renders square (`size×size`) to mirror cairosvg's `output_width =
  output_height`; this keeps the gate self-consistent and avoids shape
  mismatches when a transform changes aspect ratio.
- Privacy: nothing is uploaded. Pyodide/resvg are fetched once as runtime code;
  the SVG is processed entirely in the tab.
