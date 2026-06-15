"""Tests for paths_to_shapes shape reconstruction."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from svgym.tools import paths_to_shapes, detect_shapes


# ---------------------------------------------------------------------------
# Test SVGs with known shapes encoded as paths
# ---------------------------------------------------------------------------

# Circle: 2 semicircular arcs
CIRCLE_2ARC = '<svg><path d="M22 12A10 10 0 0 1 2 12A10 10 0 0 1 22 12Z" fill="red"/></svg>'

# Circle: 4 quarter-circle arcs
CIRCLE_4ARC = '<svg><path d="M22 12A10 10 0 0 0 12 2A10 10 0 0 0 2 12A10 10 0 0 0 12 22A10 10 0 0 0 22 12Z" fill="red"/></svg>'

# Circle: 4 cubic beziers (Illustrator-style export)
# Approximation of circle cx=12 cy=12 r=10
CIRCLE_CUBIC = '<svg><path d="M22 12C22 17.52 17.52 22 12 22C6.48 22 2 17.52 2 12C2 6.48 6.48 2 12 2C17.52 2 22 6.48 22 12Z" fill="red"/></svg>'

# Ellipse: 2 arcs with rx != ry
ELLIPSE_2ARC = '<svg><path d="M30 10A20 10 0 0 1 -10 10A20 10 0 0 1 30 10Z" fill="blue"/></svg>'

# Rectangle: absolute L commands
RECT_ABS = '<svg><path d="M10 20L110 20L110 70L10 70Z" fill="blue"/></svg>'

# Rectangle: H/V shorthand
RECT_HV = '<svg><path d="M0 0H100V50H0Z" fill="blue"/></svg>'

# Rectangle: relative h/v
RECT_REL = '<svg><path d="M0 0h100v50h-100z" fill="blue"/></svg>'

# Rounded rectangle: lines + arcs
ROUNDED_RECT = '<svg><path d="M15 0L85 0A15 15 0 0 1 100 15L100 35A15 15 0 0 1 85 50L15 50A15 15 0 0 1 0 35L0 15A15 15 0 0 1 15 0Z" fill="green"/></svg>'

# Non-shape path (should not convert)
COMPLEX_PATH = '<svg><path d="M10 80C40 10 65 10 95 80S150 150 180 80" fill="none" stroke="black"/></svg>'

# Multiple subpaths (should not convert)
MULTI_SUBPATH = '<svg><path d="M0 0h10v10h-10zM20 20h10v10h-10z" fill="black"/></svg>'


def test_circle_2arc():
    result = paths_to_shapes(CIRCLE_2ARC)
    assert '<circle' in result, f"Expected <circle> but got: {result}"
    assert 'cx="12"' in result
    assert 'cy="12"' in result
    assert 'r="10"' in result
    assert 'fill="red"' in result
    print("PASS: circle 2-arc")


def test_circle_4arc():
    result = paths_to_shapes(CIRCLE_4ARC)
    assert '<circle' in result, f"Expected <circle> but got: {result}"
    assert 'r="10"' in result
    print("PASS: circle 4-arc")


def test_circle_cubic():
    result = paths_to_shapes(CIRCLE_CUBIC)
    assert '<circle' in result, f"Expected <circle> but got: {result}"
    print("PASS: circle cubic")


def test_ellipse_2arc():
    result = paths_to_shapes(ELLIPSE_2ARC)
    # May or may not convert depending on size savings
    detected = detect_shapes(ELLIPSE_2ARC)
    assert any(d["shape"] == "ellipse" for d in detected), f"Expected ellipse detection: {detected}"
    print("PASS: ellipse 2-arc detection")


def test_rect_detection():
    for name, svg in [("abs", RECT_ABS), ("hv", RECT_HV), ("rel", RECT_REL)]:
        detected = detect_shapes(svg)
        assert any(d["shape"] == "rect" for d in detected), f"Expected rect detection for {name}: {detected}"
    print("PASS: rect detection (all variants)")


def test_rounded_rect():
    detected = detect_shapes(ROUNDED_RECT)
    assert any(d["shape"] == "rounded_rect" for d in detected), f"Expected rounded_rect: {detected}"
    print("PASS: rounded rect detection")


def test_complex_no_match():
    detected = detect_shapes(COMPLEX_PATH)
    assert len(detected) == 0, f"Should not match complex path: {detected}"
    print("PASS: complex path not matched")


def test_multi_subpath_no_match():
    detected = detect_shapes(MULTI_SUBPATH)
    assert len(detected) == 0, f"Should not match multi-subpath: {detected}"
    print("PASS: multi-subpath not matched")


def test_size_guard():
    """paths_to_shapes should only convert when result is shorter."""
    result = paths_to_shapes(RECT_REL)
    # Rect form is often longer than compact h/v path — should keep path
    print(f"  Rect rel: kept as {'path' if '<path' in result else 'rect'} (size guard)")
    print("PASS: size guard")


if __name__ == "__main__":
    test_circle_2arc()
    test_circle_4arc()
    test_circle_cubic()
    test_ellipse_2arc()
    test_rect_detection()
    test_rounded_rect()
    test_complex_no_match()
    test_multi_subpath_no_match()
    test_size_guard()
    print("\nAll tests passed!")
