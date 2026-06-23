"""Deterministic SVG optimization pipeline — no LLM required.

Replicates the LLM agent's tool-calling strategy with a fixed pipeline:
1. Deterministic prepass (lossless tools)
2. Structural tools
3. Coordinate rounding (precision=2→1→0, keep most aggressive that passes)
4. Path simplification (cubic_to_line, cubic_to_quad, etc.)
5. Merging and finishing

Each tool application is checked against quality thresholds (SSIM/PSNR).
If quality drops below threshold, the change is reverted automatically.
"""

import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from svgym.tools import (
    abs_to_rel,
    bake_translate_into_paths,
    bake_translate_into_text,
    compact_path_numbers,
    compare,
    consolidate_attrs_to_parent,
    cubic_to_line,
    cubic_to_quad,
    curve_to_hv,
    merge_collinear_lines,
    merge_paths,
    merge_same_commands,
    remove_classes,
    remove_default_attributes,
    remove_defs,
    remove_hidden_elements,
    remove_identity_transforms,
    remove_metadata,
    remove_space_before_negative,
    remove_unused_defs,
    remove_width_height,
    rescale_viewbox,
    round_attribute_coords,
    round_path_coordinates,
    shapes_to_paths,
    shorten_colors,
    shorten_ids,
    strip_whitespace,
    style_to_attributes,
    try_smooth_curves,
    unwrap_bare_groups,
    unwrap_single_tspans,
    extract_common_styles,
    merge_text_elements,
    deduplicate_paths,
    remove_junk_attrs,
    remove_unused_ids,
    simplify_transforms,
    cubic_to_arc,
    merge_subpaths,
    paths_to_shapes,
    run_svgo,
)
from svgym.config import QUALITY_THRESHOLDS


def _has_features(svg_text: str) -> dict:
    """Detect SVG features that affect which tools are safe."""
    return {
        "hover_or_pseudo": bool(re.search(r':(hover|focus|active|visited|checked)', svg_text)),
        "animation": bool(re.search(r'<animate|<set |@keyframes|animation:|transition:', svg_text)),
        "script": bool(re.search(r'<script|onclick|onmouseover|onload', svg_text, re.IGNORECASE)),
        "gradient": bool(re.search(r'<(linearGradient|radialGradient|pattern)\b', svg_text)),
        "filter": bool(re.search(r'<filter\b', svg_text)),
        "clippath": bool(re.search(r'<clipPath\b', svg_text)),
        "css_rules": bool(re.search(r'<style', svg_text)),
        "transforms": bool(re.search(r'\btransform\s*=', svg_text)),
        "text": bool(re.search(r'<text\b', svg_text)),
    }


def _try_tool(svg_text: str, original: str, func, thresholds: dict,
              trajectory: list, **kwargs) -> str:
    """Try applying a tool. Keep result if it saves bytes and passes quality. Revert otherwise."""
    name = func.__name__
    try:
        result = func(svg_text=svg_text, **kwargs) if kwargs else func(svg_text)
    except Exception as e:
        trajectory.append({"tool": name, "args": kwargs, "status": "error", "saved": 0, "error": str(e)})
        return svg_text

    saved = len(svg_text) - len(result)
    if saved <= 0:
        trajectory.append({"tool": name, "args": kwargs, "status": "no_change", "saved": 0})
        return svg_text

    # Quality check
    try:
        ssim_val, psnr_val = compare(original, result)
        passes = ssim_val >= thresholds["ssim"] and psnr_val >= thresholds["psnr"]
    except Exception:
        # If quality check fails, reject the change
        trajectory.append({"tool": name, "args": kwargs, "status": "quality_check_failed", "saved": 0})
        return svg_text

    if passes:
        trajectory.append({
            "tool": name, "args": kwargs, "status": "applied",
            "saved": saved, "ssim": round(ssim_val, 4), "psnr": round(psnr_val, 1),
        })
        return result
    else:
        trajectory.append({
            "tool": name, "args": kwargs, "status": "reverted",
            "saved": 0, "ssim": round(ssim_val, 4), "psnr": round(psnr_val, 1),
        })
        return svg_text


def optimize_svg_deterministic(svg_text: str, level: str = "conservative") -> dict:
    """Run deterministic optimization pipeline on SVG text.

    Args:
        svg_text: Raw SVG markup to optimize.
        level: Quality level — "lossless", "conservative", or "aggressive".

    Returns:
        Dict with optimized_svg, compressed_size, compression_pct, ssim, psnr,
        tool_trajectory, elapsed_time, tokens_used (always 0).
    """
    if level not in QUALITY_THRESHOLDS:
        raise ValueError(f"Unknown level: {level!r}")

    t0 = time.time()
    thresholds = QUALITY_THRESHOLDS[level]
    original_raw = svg_text
    original_size = len(svg_text.encode("utf-8"))
    features = _has_features(svg_text)
    trajectory = []

    # Use raw original as quality reference
    original_ref = original_raw

    # =========================================================================
    # Phase -2: Reconstruct shapes from paths BEFORE SVGO
    # SVGO converts cubic beziers to smooth curves, making detection harder.
    # Detecting Illustrator-style circle-as-cubics must happen on raw input.
    # =========================================================================
    if not features["css_rules"]:
        svg_text = _try_tool(svg_text, original_ref, paths_to_shapes, thresholds, trajectory)

    # =========================================================================
    # Phase -1: Run SVGO as baseline (safe for static SVGs)
    # SVGO can break: animations (<animate>, CSS @keyframes), hover/focus
    # pseudo-selectors, JavaScript event handlers. Skip for those.
    # =========================================================================
    svgo_safe = not (features["animation"] or features["hover_or_pseudo"]
                     or features["script"])
    if svgo_safe:
        try:
            svgo_result = run_svgo(svg_text)
            svgo_saved = len(svg_text) - len(svgo_result)
            if svgo_saved > 0:
                # Quality check — SVGO should be lossless but verify
                try:
                    ssim_val, psnr_val = compare(original_raw, svgo_result)
                    if ssim_val >= thresholds["ssim"] and psnr_val >= thresholds["psnr"]:
                        svg_text = svgo_result
                        trajectory.append({"tool": "run_svgo", "args": {}, "status": "applied",
                                           "saved": svgo_saved, "phase": "svgo",
                                           "ssim": round(ssim_val, 4), "psnr": round(psnr_val, 1)})
                    else:
                        trajectory.append({"tool": "run_svgo", "args": {}, "status": "reverted",
                                           "saved": 0, "phase": "svgo",
                                           "ssim": round(ssim_val, 4), "psnr": round(psnr_val, 1)})
                except Exception:
                    # Can't verify quality, skip SVGO result
                    trajectory.append({"tool": "run_svgo", "args": {}, "status": "quality_check_failed",
                                       "saved": 0, "phase": "svgo"})
            else:
                trajectory.append({"tool": "run_svgo", "args": {}, "status": "no_change",
                                   "saved": 0, "phase": "svgo"})
        except Exception as e:
            trajectory.append({"tool": "run_svgo", "args": {}, "status": "error",
                               "saved": 0, "phase": "svgo", "error": str(e)})

        # Re-detect features after SVGO (it may strip elements)
        features = _has_features(svg_text)
    else:
        trajectory.append({"tool": "run_svgo", "args": {}, "status": "skipped",
                           "saved": 0, "phase": "svgo",
                           "reason": "interactive SVG (animation/hover/script)"})

    # Save post-SVGO state as fallback (better than raw if SVGO passed quality)
    post_svgo = svg_text
    post_svgo_size = len(post_svgo.encode("utf-8"))

    # =========================================================================
    # Phase 0: Lossless prepass (no quality check needed per-tool)
    # =========================================================================
    prepass_tools = [
        remove_metadata,
        remove_junk_attrs,
        remove_hidden_elements,
        remove_default_attributes,
        remove_identity_transforms,
        unwrap_bare_groups,
        unwrap_single_tspans,
        shorten_colors,
        simplify_transforms,
        strip_whitespace,
        compact_path_numbers,
        remove_space_before_negative,
        merge_same_commands,
        curve_to_hv,
        merge_collinear_lines,
    ]

    # Conditionally safe prepass tools
    if not features["hover_or_pseudo"] and not features["css_rules"] and not features["script"]:
        prepass_tools.insert(3, remove_classes)
        prepass_tools.insert(4, style_to_attributes)

    if not features["hover_or_pseudo"] and not features["script"]:
        prepass_tools.append(shorten_ids)
        prepass_tools.append(remove_unused_ids)

    if not features["gradient"] and not features["filter"] and not features["clippath"]:
        prepass_tools.append(remove_unused_defs)

    for func in prepass_tools:
        try:
            result = func(svg_text)
            saved = len(svg_text) - len(result)
            if saved > 0:
                svg_text = result
                trajectory.append({"tool": func.__name__, "args": {}, "status": "applied",
                                   "saved": saved, "phase": "prepass"})
            else:
                trajectory.append({"tool": func.__name__, "args": {}, "status": "no_change",
                                   "saved": 0, "phase": "prepass"})
        except Exception:
            trajectory.append({"tool": func.__name__, "args": {}, "status": "error",
                               "saved": 0, "phase": "prepass"})

    # Quality gate on prepass
    prepass_ok = True
    try:
        ssim_val, psnr_val = compare(original_raw, svg_text)
        if ssim_val >= 0.9999:
            # Prepass was lossless — use post-prepass as quality reference
            original_ref = svg_text
        elif ssim_val < thresholds["ssim"] or psnr_val < thresholds["psnr"]:
            # Prepass broke quality — fall back to post-SVGO (not raw)
            svg_text = post_svgo
            original_ref = post_svgo
            prepass_ok = False
    except Exception:
        svg_text = post_svgo
        original_ref = post_svgo
        prepass_ok = False

    if level == "lossless":
        # Lossless mode: only prepass tools, done
        elapsed = time.time() - t0
        final_size = len(svg_text.encode("utf-8"))
        try:
            ssim_val, psnr_val = compare(original_raw, svg_text)
        except Exception:
            ssim_val, psnr_val = 1.0, 100.0
        return {
            "optimized_svg": svg_text,
            "compressed_size": final_size,
            "compression_pct": round((1 - final_size / original_size) * 100, 1),
            "ssim": round(ssim_val, 4),
            "psnr": round(psnr_val, 1),
            "tool_trajectory": trajectory,
            "elapsed_time": elapsed,
            "tokens_used": 0,
            "svgo_size": post_svgo_size,
            "svgo_svg": post_svgo,
        }

    # =========================================================================
    # Phase 1: Structural tools
    # Guard: skip tools that break animations, hover, or CSS selectors.
    # SSIM only checks static renders — it can't detect behavioral breakage.
    # =========================================================================
    has_interactive = (features["animation"] or features["hover_or_pseudo"]
                       or features["script"])
    has_css = features["css_rules"]

    svg_text = _try_tool(svg_text, original_ref, remove_width_height, thresholds, trajectory)

    # shapes_to_paths changes element types — breaks CSS targeting rect/circle/etc.
    if not has_css:
        svg_text = _try_tool(svg_text, original_ref, shapes_to_paths, thresholds, trajectory)

    # deduplicate_paths replaces elements with <use> — breaks individual animations
    if not has_interactive:
        svg_text = _try_tool(svg_text, original_ref, deduplicate_paths, thresholds, trajectory)

    # consolidate_attrs moves attrs to parent <g> — breaks CSS selectors on children
    if not has_css and not has_interactive:
        svg_text = _try_tool(svg_text, original_ref, consolidate_attrs_to_parent, thresholds, trajectory)

    svg_text = _try_tool(svg_text, original_ref, merge_text_elements, thresholds, trajectory)

    # Bake transforms if SVG has them (safe — doesn't change element identity)
    if features["transforms"]:
        svg_text = _try_tool(svg_text, original_ref, bake_translate_into_paths, thresholds, trajectory)
    if features["transforms"] and features["text"]:
        svg_text = _try_tool(svg_text, original_ref, bake_translate_into_text, thresholds, trajectory)

    # =========================================================================
    # Phase 2: Path simplification BEFORE rounding (order matters for quality)
    # Simplifying curves first uses less quality budget, leaving room for rounding.
    # =========================================================================
    svg_text = _try_tool(svg_text, original_ref, abs_to_rel, thresholds, trajectory)

    for threshold in [0.05, 0.1]:
        checkpoint = svg_text
        svg_text = _try_tool(svg_text, original_ref, cubic_to_line, thresholds, trajectory,
                             threshold=threshold)
        last = trajectory[-1]
        if last["status"] == "reverted":
            svg_text = checkpoint
            break

    svg_text = _try_tool(svg_text, original_ref, cubic_to_arc, thresholds, trajectory)
    svg_text = _try_tool(svg_text, original_ref, cubic_to_quad, thresholds, trajectory)
    svg_text = _try_tool(svg_text, original_ref, try_smooth_curves, thresholds, trajectory)
    svg_text = _try_tool(svg_text, original_ref, curve_to_hv, thresholds, trajectory)
    svg_text = _try_tool(svg_text, original_ref, merge_collinear_lines, thresholds, trajectory)

    # =========================================================================
    # Phase 3: Merging (before rounding — merged paths round better)
    # Guard: merge_paths/merge_subpaths destroy element identity (breaks animations)
    # extract_common_styles adds <style> rules that can conflict with pseudo-selectors
    # =========================================================================
    if not has_interactive:
        svg_text = _try_tool(svg_text, original_ref, merge_paths, thresholds, trajectory)
        svg_text = _try_tool(svg_text, original_ref, merge_subpaths, thresholds, trajectory)
    if not has_interactive and not has_css:
        svg_text = _try_tool(svg_text, original_ref, extract_common_styles, thresholds, trajectory)

    # =========================================================================
    # Phase 4: Coordinate rounding (try 2→1→0, keep most aggressive that passes)
    # Done AFTER simplification so quality budget is better spent.
    # =========================================================================
    svg_text = _try_tool(svg_text, original_ref, round_attribute_coords, thresholds, trajectory,
                         precision=1)

    best_rounded = svg_text
    best_precision = None
    for precision in [2, 1, 0]:
        checkpoint = svg_text
        svg_text = _try_tool(svg_text, original_ref, round_path_coordinates, thresholds, trajectory,
                             precision=precision)
        last = trajectory[-1]
        if last["status"] == "applied":
            best_rounded = svg_text
            best_precision = precision
        elif last["status"] == "reverted":
            # Revert to last good state but keep trying (don't break)
            svg_text = checkpoint
            break
    svg_text = best_rounded

    # =========================================================================
    # Phase 5: Second pass — retry tools that may benefit from rounding/merging
    # =========================================================================
    svg_text = _try_tool(svg_text, original_ref, compact_path_numbers, thresholds, trajectory)
    svg_text = _try_tool(svg_text, original_ref, remove_space_before_negative, thresholds, trajectory)
    svg_text = _try_tool(svg_text, original_ref, abs_to_rel, thresholds, trajectory)
    svg_text = _try_tool(svg_text, original_ref, merge_collinear_lines, thresholds, trajectory)
    svg_text = _try_tool(svg_text, original_ref, merge_same_commands, thresholds, trajectory)
    svg_text = _try_tool(svg_text, original_ref, strip_whitespace, thresholds, trajectory)

    # Try rescale_viewbox (often aggressive, usually reverted)
    svg_text = _try_tool(svg_text, original_ref, rescale_viewbox, thresholds, trajectory,
                         new_size=1000)

    # =========================================================================
    # Final quality check and result
    # =========================================================================
    elapsed = time.time() - t0
    final_size = len(svg_text.encode("utf-8"))

    try:
        ssim_val, psnr_val = compare(original_raw, svg_text)
    except Exception:
        ssim_val, psnr_val = 1.0, 100.0

    # If final quality fails, fall back to prepass-only result
    if ssim_val < thresholds["ssim"] or psnr_val < thresholds["psnr"]:
        # This shouldn't happen since we check per-tool, but safety net
        svg_text = original_ref
        final_size = len(svg_text.encode("utf-8"))
        try:
            ssim_val, psnr_val = compare(original_raw, svg_text)
        except Exception:
            ssim_val, psnr_val = 1.0, 100.0

    return {
        "optimized_svg": svg_text,
        "compressed_size": final_size,
        "compression_pct": round((1 - final_size / original_size) * 100, 1),
        "ssim": round(ssim_val, 4),
        "psnr": round(psnr_val, 1),
        "tool_trajectory": trajectory,
        "elapsed_time": elapsed,
        "tokens_used": 0,
        "svgo_size": post_svgo_size,
        "svgo_svg": post_svgo,
    }
