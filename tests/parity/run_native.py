#!/usr/bin/env python3
"""Native baseline for the browser-vs-native parity test.

Runs the REAL deterministic pipeline (cairosvg-rendered gate) over a corpus of
SVGs at every quality level and records, per (file, level):

  - the optimized SVG (written to out/native/<name>__<level>.svg)
  - its sha256, size, SSIM, PSNR
  - the *decision trajectory*: the ordered list of (tool, status) the gate made
    (status is "applied" / "reverted" / "no_change"), which is what actually
    determines the output.

The browser harness (Pyodide + resvg-wasm) produces a JSON with the same schema
(see run_browser.mjs); compare_parity.py then diffs the two.

Usage:
    PYTHONPATH=<svgym-public> python tests/parity/run_native.py            # default subset
    PYTHONPATH=<svgym-public> python tests/parity/run_native.py --all      # whole corpus
    PYTHONPATH=<svgym-public> python tests/parity/run_native.py --svgs glyph-k tux
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent                      # svgym-public/
CORPUS = ROOT.parent / "demo" / "svgs"         # demo/svgs/ (shared corpus)
OUT = HERE / "out" / "native"

LEVELS = ["lossless", "conservative", "aggressive"]

# A representative default subset spanning every category + size regime.
DEFAULT_SUBSET = [
    "heroicons-camera", "simple-github", "phosphor-apple-logo",   # icons
    "glyph-k", "glyph-w",                                          # glyphs
    "ir-lion-sun", "hr-croatia",                                  # flags
    "openmoji-dragon",                                            # emoji
    "tux", "gophers-9",                                           # illustrations
    "neo4j-graph", "ggplot-timeseries",                          # diagrams/charts
    "whale", "stop-sign",                                        # sketches
    "3-dots-move", "hover-interactive",                          # animated/interactive
]


def find_svgs(names: list[str] | None, all_: bool) -> list[Path]:
    every = sorted(CORPUS.rglob("*.svg"))
    if all_:
        # skip the deliberately-broken inputs used elsewhere
        return [p for p in every if "breakage" not in p.parts]
    if names:
        want = set(names)
        return [p for p in every if p.stem in want]
    want = set(DEFAULT_SUBSET)
    return [p for p in every if p.stem in want]


def decisions(trajectory: list[dict]) -> list[list]:
    """Compact ordered decision list: [tool, status, args] per step."""
    out = []
    for s in trajectory:
        out.append([s.get("tool"), s.get("status"), s.get("args", {})])
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="run the whole corpus")
    ap.add_argument("--svgs", nargs="*", help="specific SVG stems")
    ap.add_argument("--levels", nargs="*", default=LEVELS)
    args = ap.parse_args()

    from svgym.deterministic import optimize_svg_deterministic  # noqa: E402

    svgs = find_svgs(args.svgs, args.all)
    if not svgs:
        print("No SVGs found in", CORPUS, file=sys.stderr)
        return 1
    OUT.mkdir(parents=True, exist_ok=True)

    records = []
    print(f"Native baseline: {len(svgs)} SVGs x {len(args.levels)} levels "
          f"= {len(svgs) * len(args.levels)} runs\n")
    for p in svgs:
        svg = p.read_text(errors="replace")
        for level in args.levels:
            try:
                r = optimize_svg_deterministic(svg, level=level)
            except Exception as e:  # record the failure rather than aborting
                records.append({"name": p.stem, "level": level, "error": str(e)})
                print(f"  {p.stem:28s} {level:12s} ERROR {e}")
                continue
            opt = r.get("optimized_svg") or svg
            (OUT / f"{p.stem}__{level}.svg").write_text(opt)
            rec = {
                "name": p.stem,
                "level": level,
                "original_size": len(svg.encode("utf-8")),
                "size": len(opt.encode("utf-8")),
                "ssim": r.get("ssim"),
                "psnr": (None if r.get("psnr") in (float("inf"),) else r.get("psnr")),
                "sha256": hashlib.sha256(opt.encode("utf-8")).hexdigest(),
                "decisions": decisions(r.get("tool_trajectory", [])),
            }
            records.append(rec)
            print(f"  {p.stem:28s} {level:12s} {rec['original_size']:>8d} -> "
                  f"{rec['size']:>8d} B  ssim={rec['ssim']}  steps={len(rec['decisions'])}")

    (HERE / "native_results.json").write_text(json.dumps(records, indent=1))
    print(f"\nWrote {HERE / 'native_results.json'} ({len(records)} records)")
    print(f"Optimized SVGs in {OUT}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
