#!/usr/bin/env python3
"""Diff the native baseline against the browser run and report parity.

Inputs (same schema, produced by run_native.py and run_browser.mjs):
    native_results.json   browser_results.json
    out/native/*.svg      out/browser/*.svg

For each (name, level) it reports:
  - byte-identical?           (sha256 of the optimized SVG matches)
  - decisions-identical?      (same ordered [tool, status] trajectory)
  - size delta                (browser size vs native size)
  - safe?                     (browser output re-rendered with the NEUTRAL engine
                               -- cairosvg -- still clears the level's SSIM bar vs
                               the original)

Acceptance guideline:
  * lossless              -> expect 100% byte-identical
  * conservative/aggressive -> high identical rate; every divergent file must
    still be SAFE and within a small size tolerance.

Usage:
    PYTHONPATH=<svgym-public> python tests/parity/compare_parity.py \
        --browser tests/parity/browser_results.json \
        --browser-dir tests/parity/out/browser
    # self-test (native vs itself -> everything identical):
    PYTHONPATH=<svgym-public> python tests/parity/compare_parity.py \
        --browser tests/parity/native_results.json --browser-dir tests/parity/out/native
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
CORPUS = ROOT.parent / "demo" / "svgs"

THRESHOLDS = {"lossless": 0.999, "conservative": 0.99, "aggressive": 0.97}


def load(path: Path) -> dict:
    recs = json.loads(Path(path).read_text())
    return {(r["name"], r["level"]): r for r in recs if "error" not in r}


def neutral_ssim(name: str, level: str, browser_dir: Path) -> float | None:
    """Re-render the browser output and the original with cairosvg, return SSIM."""
    from svgym.tools import compare  # native cairosvg-backed gate
    orig = next(iter(CORPUS.rglob(name + ".svg")), None)
    out = browser_dir / f"{name}__{level}.svg"
    if not orig or not out.exists():
        return None
    try:
        ssim, _ = compare(orig.read_text(errors="replace"), out.read_text(errors="replace"))
        return round(ssim, 4)
    except Exception:
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--native", default=str(HERE / "native_results.json"))
    ap.add_argument("--browser", required=True)
    ap.add_argument("--browser-dir", required=True)
    ap.add_argument("--size-tol", type=float, default=2.0,
                    help="allowed browser-vs-native size delta, %% (default 2)")
    ap.add_argument("--no-safety", action="store_true",
                    help="skip the neutral-renderer safety re-check (faster)")
    args = ap.parse_args()

    nat = load(Path(args.native))
    bro = load(Path(args.browser))
    bdir = Path(args.browser_dir)
    keys = sorted(set(nat) & set(bro))
    if not keys:
        print("No overlapping (name, level) records between the two runs.", file=sys.stderr)
        return 1

    by_level: dict[str, list] = {}
    mismatches = []
    for k in keys:
        n, b = nat[k], bro[k]
        name, level = k
        identical = n["sha256"] == b["sha256"]
        same_dec = n["decisions"] == b["decisions"]
        dsize = b["size"] - n["size"]
        dpct = (dsize / n["size"] * 100) if n["size"] else 0.0
        safe = None
        if not identical and not args.no_safety:
            s = neutral_ssim(name, level, bdir)
            safe = (s is not None and s >= THRESHOLDS.get(level, 0.99))
        by_level.setdefault(level, []).append(
            {"identical": identical, "same_dec": same_dec, "dpct": dpct, "safe": safe})
        if not identical:
            mismatches.append((name, level, same_dec, dpct, safe))

    print("==== parity summary ====")
    for level in ["lossless", "conservative", "aggressive"]:
        rows = by_level.get(level)
        if not rows:
            continue
        n = len(rows)
        ident = sum(r["identical"] for r in rows)
        dec = sum(r["same_dec"] for r in rows)
        dpcts = [abs(r["dpct"]) for r in rows]
        unsafe = sum(1 for r in rows if r["safe"] is False)
        print(f"\n  {level}  (n={n})")
        print(f"    byte-identical     : {ident}/{n}  ({100*ident//n}%)")
        print(f"    same decisions     : {dec}/{n}")
        print(f"    size delta |%|     : mean {round(st.mean(dpcts),2)}  max {round(max(dpcts),2)}")
        if unsafe:
            print(f"    *** UNSAFE outputs : {unsafe}  (below SSIM bar under neutral render) ***")
        else:
            print(f"    unsafe outputs     : 0")

    if mismatches:
        print("\n==== divergent files ====")
        for name, level, same_dec, dpct, safe in mismatches:
            tag = "safe" if safe else ("UNSAFE" if safe is False else "n/a")
            print(f"  {name:28s} {level:12s} dsize={dpct:+.1f}%  "
                  f"decisions={'same' if same_dec else 'DIFFER'}  neutral={tag}")
    else:
        print("\nAll outputs byte-identical across the two renderers.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
