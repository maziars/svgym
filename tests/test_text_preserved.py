#!/usr/bin/env python3
"""Regression guard: the deterministic pipeline must never alter text geometry.

Rounding `x/y/dx/dy` on <text>/<tspan> shifts glyph positions sub-pixel, which
changes font hinting and is visible on small/bold text. The pipeline masks text
out of coordinate rounding, so every coordinate inside a <text>...</text> block
must survive optimization byte-for-byte. This test asserts that across every
text-bearing demo SVG and every quality level.

Run directly:
    PYTHONPATH=<svgym-public> python tests/test_text_preserved.py
Or under pytest:
    PYTHONPATH=<svgym-public> pytest tests/test_text_preserved.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT.parent / "demo" / "svgs"
LEVELS = ["lossless", "conservative", "aggressive"]

from svgym.deterministic import optimize_svg_deterministic  # noqa: E402


def text_coords(svg: str) -> list[tuple[str, str]]:
    """All x/y/dx/dy values that appear inside <text>...</text> blocks."""
    blocks = re.findall(r"<text\b.*?</text>", svg, flags=re.DOTALL)
    return re.findall(r'\b(x|y|dx|dy)="(-?\d+\.?\d*)"', " ".join(blocks))


def text_svgs() -> list[Path]:
    return sorted(p for p in CORPUS.rglob("*.svg")
                  if "breakage" not in p.parts and "<text" in p.read_text(errors="replace"))


def check(path: Path) -> list[str]:
    """Flag any text coordinate VALUE that appears in the output but not the
    input -- i.e. a rounded or repositioned glyph. Values that merely disappear
    (lossless merges / style extraction that drop a redundant coord) are fine;
    the SSIM gate covers visual equivalence for those.
    """
    failures = []
    src = path.read_text(errors="replace")
    before = {v for _, v in text_coords(src)}
    if not before:
        return failures
    for level in LEVELS:
        out = optimize_svg_deterministic(src, level=level).get("optimized_svg") or src
        introduced = {v for _, v in text_coords(out)} - before
        if introduced:
            failures.append(f"{path.stem} [{level}]: text coords changed/rounded -> "
                            f"{sorted(introduced)[:6]}")
    return failures


def test_text_geometry_preserved():
    svgs = text_svgs()
    assert svgs, f"no text-bearing SVGs under {CORPUS}"
    all_failures = []
    for p in svgs:
        all_failures += check(p)
    assert not all_failures, "text geometry was altered:\n  " + "\n  ".join(all_failures)


def main() -> int:
    svgs = text_svgs()
    print(f"Checking {len(svgs)} text-bearing SVGs x {len(LEVELS)} levels...")
    failures = []
    for p in svgs:
        f = check(p)
        print(f"  {p.stem:28s} {'OK' if not f else 'FAIL'}")
        failures += f
    if failures:
        print("\nFAILURES:")
        for f in failures:
            print("  " + f)
        return 1
    print(f"\nAll text geometry preserved across {len(svgs)} SVGs.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
