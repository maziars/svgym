"""SVG optimization engine using Haiku with tool_use API.

Gives the model full agency over compression strategy:
- Model calls compression tools (server-side state, no svg_text passing)
- Model calls compare_quality and get_size to check its own work
- Model calls revert to undo bad changes
- Model calls read_state if it needs to see the full SVG text
- No auto quality gate — model decides what to keep/revert
"""

import re
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from svgym.llm_client import create_client

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
)
from svgym.config import (
    LOSSLESS_TOOLS,
    MAX_TURNS,
    PROVIDER,
    QUALITY_THRESHOLDS,
)


# ---------------------------------------------------------------------------
# Tool definitions for the Anthropic API
# ---------------------------------------------------------------------------

TOOLS = [
    {
        "name": "read_svg",
        "description": "Load the SVG for compression. Returns a structural summary with recommended tools and skip list.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "read_state",
        "description": "Read the full current SVG text. Use this when you need to inspect the actual markup.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "write_svg",
        "description": "Finalise and save the compressed SVG.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "get_size",
        "description": "Get the current byte size and compression percentage vs original.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "compare_quality",
        "description": "Compare current state against the original. Returns SSIM and PSNR values. Use this after applying tools to check quality.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "revert",
        "description": "Undo the last tool application, restoring the previous SVG state.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "save_checkpoint",
        "description": "Save the current SVG state under a name. Use this to bookmark a good state before trying alternatives.",
        "input_schema": {
            "type": "object",
            "properties": {"name": {"type": "string", "description": "Label for this checkpoint (e.g. 'after_rounding_p2')"}},
            "required": ["name"],
        },
    },
    {
        "name": "restore_checkpoint",
        "description": "Restore SVG state from a named checkpoint. Use this to go back to a saved state.",
        "input_schema": {
            "type": "object",
            "properties": {"name": {"type": "string", "description": "Name of the checkpoint to restore"}},
            "required": ["name"],
        },
    },
    {
        "name": "round_path_coordinates",
        "description": "Round all path coordinates to given decimal precision. Try 2 first, then 1.",
        "input_schema": {
            "type": "object",
            "properties": {"precision": {"type": "integer", "default": 2}},
            "required": [],
        },
    },
    {
        "name": "abs_to_rel",
        "description": "Convert absolute path commands to relative where shorter. Also converts L to h/v.",
        "input_schema": {
            "type": "object",
            "properties": {"precision": {"type": "integer", "default": 2}},
            "required": [],
        },
    },
    {
        "name": "cubic_to_line",
        "description": "Convert nearly-straight cubic beziers to line commands. threshold controls aggressiveness.",
        "input_schema": {
            "type": "object",
            "properties": {"threshold": {"type": "number", "default": 0.05}},
            "required": [],
        },
    },
    {
        "name": "cubic_to_quad",
        "description": "Convert cubic beziers to quadratic where error is small.",
        "input_schema": {
            "type": "object",
            "properties": {"threshold": {"type": "number", "default": 0.03}},
            "required": [],
        },
    },
    {
        "name": "try_smooth_curves",
        "description": "Convert cubic beziers to smooth curves (c->s) where control points are reflections.",
        "input_schema": {
            "type": "object",
            "properties": {"tolerance": {"type": "number", "default": 0.06}},
            "required": [],
        },
    },
    {
        "name": "curve_to_hv",
        "description": "Convert line commands to h/v shorthand when one delta is zero.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "merge_collinear_lines",
        "description": "Merge consecutive same-direction line segments into one.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "merge_same_commands",
        "description": "Merge consecutive same-type path commands (removes repeated command letters).",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "merge_paths",
        "description": "Merge multiple <path> elements with same attributes into a single <path>.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "consolidate_attrs_to_parent",
        "description": "Move repeated attributes from child elements to parent group/svg.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "remove_default_attributes",
        "description": "Remove attributes matching SVG defaults (fill='black', opacity='1', etc).",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "remove_hidden_elements",
        "description": "Remove elements with display:none or visibility:hidden.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "remove_classes",
        "description": "Remove all class attributes. Safe when no <style>/<script> references them.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "unwrap_single_tspans",
        "description": "Unwrap <tspan> when it's the sole child of <text>, merging attrs.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "remove_identity_transforms",
        "description": "Remove no-op transforms like rotate(360deg), translate(0,0).",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "remove_width_height",
        "description": "Remove width/height attributes when viewBox is present.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "style_to_attributes",
        "description": "Convert inline style='...' to individual presentation attributes.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "strip_whitespace",
        "description": "Remove unnecessary whitespace from SVG markup.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "compact_path_numbers",
        "description": "Remove leading zeros (0.5 -> .5) in path data.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "remove_space_before_negative",
        "description": "Remove spaces before negative numbers in path data.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "bake_translate_into_paths",
        "description": "Apply translate() transforms directly into path coordinates.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "bake_translate_into_text",
        "description": "Apply translate() transforms directly into text x/y attributes.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "remove_unused_defs",
        "description": "Remove <defs> entries whose id is not referenced anywhere.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "rescale_viewbox",
        "description": "Rescale viewBox and all coordinates to integer-friendly dimensions.",
        "input_schema": {
            "type": "object",
            "properties": {"new_size": {"type": "integer", "default": 1000}},
            "required": [],
        },
    },
    {
        "name": "unwrap_bare_groups",
        "description": "Remove <g> wrappers that have no attributes.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "round_attribute_coords",
        "description": "Round x, y, width, height, translate values to given precision.",
        "input_schema": {
            "type": "object",
            "properties": {"precision": {"type": "integer", "default": 1}},
            "required": [],
        },
    },
    {
        "name": "shorten_colors",
        "description": "Shorten color values: #rrggbb -> #rgb, hex -> named color, rgb() -> hex. Always lossless.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "shorten_ids",
        "description": "Shorten referenced IDs to minimal strings (A, B, C...). Removes unreferenced IDs. Lossless.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "remove_metadata",
        "description": "Remove comments, metadata, title/desc, editor namespaces (inkscape/sodipodi), XML declarations. Lossless.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "shapes_to_paths",
        "description": "Convert rect, line, polygon, polyline to <path> for further path optimization. Skips rounded rects.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "extract_common_styles",
        "description": "Extract repeated inline style/presentation attributes into CSS classes in a <style> block. Huge savings when many elements share fill, stroke, font properties. Lossless.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "merge_text_elements",
        "description": "Merge adjacent <text> elements with identical style/font attributes into one <text> with <tspan> children. Saves repeated attribute declarations. Lossless.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "deduplicate_paths",
        "description": "Replace duplicate <path> elements (identical d= and attributes) with <defs>/<use> references. Huge savings on SVGs with repeated decorative elements (heraldry, patterns). Lossless.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "remove_junk_attrs",
        "description": "Remove useless SVG attributes: version, xmlns:xlink, x='0px', y='0px', enable-background, xml:space, empty defs, null attributes, svg id. Lossless.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "simplify_transforms",
        "description": "Decompose matrix() transforms into simpler translate/scale/rotate. Rounds transform values to 3 decimals. Lossless.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "cubic_to_arc",
        "description": "Convert cubic bezier curves that approximate circular/elliptical arcs to arc commands. Huge savings on rounded corners, buttons, and icons.",
        "input_schema": {
            "type": "object",
            "properties": {"tolerance": {"type": "number", "description": "Max deviation threshold (default 0.02)"}},
            "required": [],
        },
    },
    {
        "name": "merge_subpaths",
        "description": "Merge consecutive <path> elements with identical fill/stroke/style into one compound path with multiple subpaths. Saves repeated tags and attributes. Lossless.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
]

# Map tool names to their svgym.tools functions
TOOL_FUNCS = {
    "round_path_coordinates": round_path_coordinates,
    "abs_to_rel": abs_to_rel,
    "cubic_to_line": cubic_to_line,
    "cubic_to_quad": cubic_to_quad,
    "try_smooth_curves": try_smooth_curves,
    "curve_to_hv": curve_to_hv,
    "merge_collinear_lines": merge_collinear_lines,
    "merge_same_commands": merge_same_commands,
    "merge_paths": merge_paths,
    "consolidate_attrs_to_parent": consolidate_attrs_to_parent,
    "remove_default_attributes": remove_default_attributes,
    "remove_hidden_elements": remove_hidden_elements,
    "remove_classes": remove_classes,
    "unwrap_single_tspans": unwrap_single_tspans,
    "remove_identity_transforms": remove_identity_transforms,
    "remove_width_height": remove_width_height,
    "style_to_attributes": style_to_attributes,
    "strip_whitespace": strip_whitespace,
    "compact_path_numbers": compact_path_numbers,
    "remove_space_before_negative": remove_space_before_negative,
    "bake_translate_into_paths": bake_translate_into_paths,
    "bake_translate_into_text": bake_translate_into_text,
    "remove_unused_defs": remove_unused_defs,
    "rescale_viewbox": rescale_viewbox,
    "unwrap_bare_groups": unwrap_bare_groups,
    "round_attribute_coords": round_attribute_coords,
    "shorten_colors": shorten_colors,
    "shorten_ids": shorten_ids,
    "remove_metadata": remove_metadata,
    "shapes_to_paths": shapes_to_paths,
    "extract_common_styles": extract_common_styles,
    "merge_text_elements": merge_text_elements,
    "deduplicate_paths": deduplicate_paths,
    "remove_junk_attrs": remove_junk_attrs,
    "remove_unused_ids": remove_unused_ids,
    "simplify_transforms": simplify_transforms,
    "cubic_to_arc": cubic_to_arc,
    "merge_subpaths": merge_subpaths,
}

# Control tools that don't transform the SVG
CONTROL_TOOLS = {"read_svg", "read_state", "write_svg", "get_size", "compare_quality", "revert", "save_checkpoint", "restore_checkpoint"}


def _build_system_prompt(level: str) -> str:
    """Build a system prompt that gives the model full agency."""
    thresholds = QUALITY_THRESHOLDS[level]
    ssim_str = f"{thresholds['ssim']:.2f}" if thresholds["ssim"] < 1.0 else "1.0"
    psnr_str = f"{thresholds['psnr']:.0f}" if thresholds["psnr"] != float("inf") else "inf"

    level_note = ""
    if level == "lossless":
        level_note = (
            "\n\nIMPORTANT: Lossless mode. Only apply structural tools that cannot change "
            "visual output. Do NOT use: round_path_coordinates, round_attribute_coords, "
            "cubic_to_line, cubic_to_quad, try_smooth_curves, abs_to_rel, rescale_viewbox, "
            "bake_translate_into_paths, bake_translate_into_text."
        )
    elif level == "aggressive":
        level_note = (
            "\n\nAggressive mode. Push compression hard. Try round_path_coordinates with "
            "precision=1 or even 0. Use higher thresholds for cubic_to_line (0.1-0.2). "
            "Try rescale_viewbox. Target >40% compression."
        )

    return f"""You are a world-class SVG compression expert. Your mission: achieve MAXIMUM compression while staying within quality bounds. Every byte matters. The read_svg summary tells you exactly what tools will help and how much — use it.

QUALITY BOUNDS: SSIM > {ssim_str}, PSNR > {psnr_str}dB
As long as you stay above these thresholds, compress as hard as possible.

WORKFLOW:
1. read_svg — study the structural analysis carefully. It tells you which tools will save the most.
2. Apply ALL recommended tools. Do not skip tools that the summary recommends.
3. Every tool auto-reports size savings + SSIM/PSNR + PASS/FAIL.
4. If FAIL → revert, try more conservative params. If PASS → keep going.
5. After all phases, check get_size. If compression < 30%%, go back and try:
   - More aggressive rounding (precision=1 or even 0)
   - rescale_viewbox to eliminate decimals
   - Re-run abs_to_rel (it often saves more after rounding)
6. write_svg when you've exhausted all options.

PHASES (apply ALL applicable tools in each phase):

Phase 1 — Lossless structural cleanup (always apply all of these):
  remove_metadata, remove_hidden_elements, remove_default_attributes,
  remove_width_height, unwrap_bare_groups, style_to_attributes,
  remove_classes, unwrap_single_tspans, remove_unused_defs,
  remove_identity_transforms, shorten_colors, shorten_ids, shapes_to_paths,
  deduplicate_paths (after shapes_to_paths — finds duplicate paths for <use> refs)

Phase 2 — Attributes, transforms & text:
  consolidate_attrs_to_parent, round_attribute_coords(precision=1),
  bake_translate_into_text, bake_translate_into_paths,
  merge_text_elements (group similar <text> into parent+<tspan>)

Phase 3 — Path optimization (biggest wins — typically 30-60%% of total savings):
  round_path_coordinates(precision=2) — then try precision=1 if PASS — then try precision=0 if PASS
  IMPORTANT: Always try precision=0 with save_checkpoint — it often passes quality and gives huge savings,
  especially on SVGs where coordinates are already low-precision. The quality gate protects you.
  abs_to_rel — often 5-20%% savings on absolute coordinates
  cubic_to_line(threshold=0.05) — try 0.1 if quality allows
  cubic_to_quad, try_smooth_curves, curve_to_hv
  merge_collinear_lines, merge_same_commands, merge_paths

Phase 4 — CSS extraction & final cleanup:
  extract_common_styles (run AFTER all structural changes — finds repeated attrs across elements and creates shared CSS classes)
  compact_path_numbers, remove_space_before_negative, strip_whitespace

Phase 5 — Advanced (try if compression < 30%% or if you haven't tried precision=0 yet):
  round_path_coordinates(precision=0) — try this if you haven't already, use save_checkpoint first
  rescale_viewbox(new_size=1000) — eliminates decimals on small viewBoxes
  Then re-run: abs_to_rel, compact_path_numbers, merge_same_commands

CONTROL TOOLS:
  get_size — current size + compression %%
  compare_quality — SSIM/PSNR (also auto-reported after changes)
  revert — undo last change
  save_checkpoint(name) / restore_checkpoint(name) — branch and compare strategies
  read_state — see full SVG text (use if you need to understand the markup)

IMPORTANT BEHAVIORS:
- Every change auto-reports quality (SSIM/PSNR).
- Your FINAL output MUST pass quality thresholds. If it doesn't, the system will fall back
  to an earlier checkpoint that did pass — wasting all your later work. So keep quality passing
  for your final answer, otherwise you automatically fail.
- If a tool saves 0 bytes, move on — it's a no-op, no revert needed.
- Do NOT stop early. Apply every applicable tool before calling write_svg.
- ALWAYS try precision=2, then 1, then 0 for round_path_coordinates (use save_checkpoint).
  The quality gate protects you — if precision=0 passes, keep it. It often saves 20-50%% alone.
- abs_to_rel often saves MORE after rounding — run it again after round_path_coordinates.
- Target >40%% compression. Most SVGs can achieve 50-80%%.{level_note}"""


def _summarize_svg(svg_text: str) -> str:
    """Generate a structural summary with actionable tool recommendations."""
    size = len(svg_text.encode("utf-8"))
    lines = [f"Read SVG: {size} bytes"]
    recommendations: list[str] = []

    vb = re.search(r'viewBox="([^"]*)"', svg_text)
    if vb:
        lines.append(f"viewBox: {vb.group(1)}")
    w = re.search(r'\bwidth="([^"]*)"', svg_text)
    h = re.search(r'\bheight="([^"]*)"', svg_text)
    if w and h:
        lines.append(f"width={w.group(1)} height={h.group(1)}")
        if vb:
            recommendations.append(
                f"remove_width_height: ~{len(w.group(0)) + len(h.group(0)) + 2}b savings"
            )

    tags = re.findall(r"<(\w+)[\s>/]", svg_text)
    counts = Counter(tags)
    elements = []
    for tag in [
        "path", "rect", "circle", "line", "ellipse", "polyline",
        "polygon", "text", "tspan", "g", "use", "defs", "style", "script",
    ]:
        if tag in counts:
            elements.append(f"{counts[tag]} <{tag}>")
    if elements:
        lines.append("Elements: " + ", ".join(elements))

    paths = re.findall(r'd="([^"]*)"', svg_text)
    if paths:
        total_d = sum(len(p) for p in paths)
        all_path_text = "".join(paths)
        cmds_used = set(re.findall(r"[A-Za-z]", all_path_text))
        lines.append(
            f"Path data: {total_d} bytes ({total_d * 100 // size}% of file) in {len(paths)} paths"
        )

        # Coordinate precision analysis
        decimals = re.findall(r'\.(\d+)', all_path_text)
        if decimals:
            avg_decimals = sum(len(d) for d in decimals) / len(decimals)
            max_decimals = max(len(d) for d in decimals)
            lines.append(
                f"Coordinate precision: avg {avg_decimals:.1f} decimals, max {max_decimals} "
                f"({len(decimals)} decimal numbers)"
            )
            if avg_decimals > 2:
                recommendations.append(
                    f"round_path_coordinates(precision=2): avg {avg_decimals:.1f} decimals → 2 will save ~{int(total_d * (1 - 2/avg_decimals) * 0.3)}b"
                )
            if avg_decimals > 1:
                recommendations.append(
                    f"round_path_coordinates(precision=1): aggressive, try after precision=2"
                )

        # Command type breakdown
        cmd_counts = Counter(re.findall(r"[A-Za-z]", all_path_text))
        abs_cmds = sum(cmd_counts.get(c, 0) for c in "MLCHVCSQTAZ")
        rel_cmds = sum(cmd_counts.get(c, 0) for c in "mlchvcsqtaz")
        if abs_cmds > 0:
            cmd_summary = ", ".join(f"{c}:{n}" for c, n in sorted(cmd_counts.items()) if n > 0)
            lines.append(f"Path commands: {cmd_summary}")
            if abs_cmds > rel_cmds:
                recommendations.append(
                    f"abs_to_rel: {abs_cmds} absolute vs {rel_cmds} relative commands — big savings likely"
                )

        has_curves = bool(cmds_used & {"c", "C", "s", "S", "q", "Q"})
        cubic_count = cmd_counts.get("c", 0) + cmd_counts.get("C", 0)
        if has_curves:
            recommendations.append(
                f"cubic_to_line, cubic_to_quad, try_smooth_curves: {cubic_count} cubic curves to simplify"
            )

        # Path mergeability: paths with identical non-d attributes
        path_tags = re.findall(r'<path\b([^>]*)/>', svg_text)
        if len(path_tags) > 1:
            attr_groups: dict[str, int] = Counter()
            for pt in path_tags:
                non_d = re.sub(r'\bd="[^"]*"', '', pt).strip()
                non_d = " ".join(sorted(non_d.split()))
                attr_groups[non_d] += 1
            mergeable = sum(v for v in attr_groups.values() if v > 1)
            if mergeable > 1:
                tag_overhead = mergeable * 10  # ~10 bytes per eliminated <path .../> tag
                recommendations.append(
                    f"merge_paths: {mergeable}/{len(path_tags)} paths share attributes — merge to save ~{tag_overhead}b+ in tag overhead"
                )

    class_attrs = re.findall(r' class="[^"]*"', svg_text)
    has_style_block = "<style" in svg_text
    has_script = "<script" in svg_text
    if class_attrs:
        class_bytes = sum(len(c) for c in class_attrs)
        if not has_style_block and not has_script:
            recommendations.append(
                f"remove_classes: SAFE - {len(class_attrs)} class attrs ({class_bytes}b) "
                f"with NO <style> or <script> block referencing them"
            )
        elif has_style_block:
            lines.append(
                f"Classes: {len(class_attrs)} attrs ({class_bytes}b) - referenced by <style>, do NOT remove"
            )

    transforms = re.findall(r'transform="([^"]*)"', svg_text)
    if transforms:
        path_translates = len(
            re.findall(r'<path\b[^>]*transform="translate\(', svg_text)
        )
        text_translates = len(
            re.findall(r'<text\b[^>]*transform="translate\(', svg_text)
        )
        if text_translates:
            recommendations.append(
                f"bake_translate_into_text: {text_translates} text elements with translate (~{text_translates * 30}b savings)"
            )
        if path_translates:
            recommendations.append(
                f"bake_translate_into_paths: {path_translates} paths with translate (~{path_translates * 30}b savings)"
            )

    single_tspans = re.findall(
        r"<text[^>]*><tspan[^>]*>[^<]*</tspan></text>", svg_text
    )
    if single_tspans:
        recommendations.append(
            f"unwrap_single_tspans: {len(single_tspans)} text elements with single tspan (~{len(single_tspans) * 15}b savings)"
        )

    defs_match = re.search(r"<defs[^>]*>(.*?)</defs>", svg_text, re.DOTALL)
    if defs_match:
        defs_ids = re.findall(r'\bid="([^"]*)"', defs_match.group(1))
        rest = svg_text[: defs_match.start()] + svg_text[defs_match.end() :]
        unused = [did for did in defs_ids if f"#{did}" not in rest]
        if unused:
            recommendations.append(
                f"remove_unused_defs: {len(unused)}/{len(defs_ids)} defs entries are unreferenced"
            )
        elif defs_ids:
            lines.append(
                f"Defs: {len(defs_ids)} entries, all referenced - do NOT remove"
            )

    style_count = svg_text.count('style="')
    if style_count:
        recommendations.append(
            f"style_to_attributes: {style_count} inline styles to convert"
        )

    if vb:
        parts = vb.group(1).split()
        if len(parts) == 4:
            try:
                vw, vh = float(parts[2]), float(parts[3])
                if vw <= 48 and vh <= 48 and paths:
                    decimal_count = sum(
                        1 for p in paths for _ in re.findall(r"\.\d", p)
                    )
                    if decimal_count > 20:
                        recommendations.append(
                            f"rescale_viewbox: small viewBox ({vw}x{vh}) with {decimal_count} decimal coords"
                        )
            except ValueError:
                pass

    # Check for new tool opportunities
    comments = len(re.findall(r'<!--', svg_text))
    metadata = 1 if re.search(r'<metadata|<title|<desc|xmlns:inkscape|xmlns:sodipodi', svg_text) else 0
    if comments or metadata:
        recommendations.insert(0,
            f"remove_metadata: {comments} comments, {'has' if metadata else 'no'} metadata/editor cruft"
        )

    color_matches = re.findall(r'#[0-9a-fA-F]{6}\b', svg_text)
    rgb_matches = re.findall(r'rgb\(', svg_text)
    if color_matches or rgb_matches:
        recommendations.append(
            f"shorten_colors: {len(color_matches)} hex6 colors + {len(rgb_matches)} rgb() to shorten"
        )

    id_decls = re.findall(r'\bid="([^"]*)"', svg_text)
    long_ids = [i for i in id_decls if len(i) > 2]
    if long_ids:
        recommendations.append(
            f"shorten_ids: {len(long_ids)} IDs to shorten (~{sum(len(i)-1 for i in long_ids)}b savings)"
        )

    shape_counts = sum(counts.get(t, 0) for t in ["rect", "line", "polygon", "polyline"])
    if shape_counts:
        recommendations.append(
            f"shapes_to_paths: {shape_counts} shapes to convert for path optimization"
        )

    # Check for extract_common_styles opportunity
    # Count repeated presentation attribute sets
    pres_attrs_raw = re.findall(
        r'(?:fill|stroke|stroke-width|font-family|font-size|font-style|font-weight|text-anchor)="[^"]*"',
        svg_text,
    )
    from collections import Counter as _Counter
    attr_groups = _Counter()
    for m in re.finditer(r'<(\w+)\b([^>]*?)/?>', svg_text):
        attrs = re.findall(r'((?:fill|stroke|stroke-width|font-family|font-size)="[^"]*")', m.group(2))
        if len(attrs) >= 2:
            key = " ".join(sorted(attrs))
            attr_groups[key] += 1
    repeated = sum(c for c in attr_groups.values() if c >= 2)
    if repeated >= 4 or style_count >= 6:
        recommendations.insert(0,
            f"extract_common_styles: {repeated} elements share attribute sets, {style_count} inline styles -> CSS classes"
        )

    # Check for merge_text_elements opportunity
    text_count = counts.get("text", 0)
    if text_count >= 3:
        recommendations.append(
            f"merge_text_elements: {text_count} text elements — merge adjacent with shared attrs"
        )

    # Check for deduplicate_paths opportunity
    if paths and len(paths) >= 4:
        from collections import Counter as _Counter2
        path_dupes = _Counter2(paths)
        n_dupes = sum(c - 1 for c in path_dupes.values() if c > 1)
        dupe_savings = sum((c - 1) * len(d) for d, c in path_dupes.items() if c > 1)
        if n_dupes >= 2 and dupe_savings > 100:
            recommendations.insert(0,
                f"deduplicate_paths: {n_dupes} duplicate paths -> <use> refs (~{dupe_savings}b savings)"
            )

    if recommendations:
        lines.append("\nRECOMMENDED TOOLS (in order of expected savings):")
        for r in recommendations:
            lines.append(f"  -> {r}")

    skip = []
    if not transforms:
        skip.extend([
            "bake_translate_into_paths",
            "bake_translate_into_text",
            "remove_identity_transforms",
        ])
    if not class_attrs:
        skip.append("remove_classes")
    if not defs_match:
        skip.extend(["remove_unused_defs", "remove_defs"])
    if not style_count:
        skip.append("style_to_attributes")
    if not single_tspans:
        skip.append("unwrap_single_tspans")
    if counts.get("text", 0) == 0 and counts.get("tspan", 0) == 0:
        skip.append("unwrap_single_tspans")
    if skip:
        lines.append(f"\nSKIP (not applicable): {', '.join(skip)}")

    return "\n".join(lines)


def _filter_tools_for_level(level: str) -> list[dict]:
    """Return only the tool definitions allowed for the given level."""
    if level != "lossless":
        return TOOLS
    allowed = LOSSLESS_TOOLS | CONTROL_TOOLS
    return [t for t in TOOLS if t["name"] in allowed]


def _has_interactive_features(svg_text: str) -> dict:
    """Detect SVG features that make certain tools unsafe."""
    return {
        "hover_or_pseudo": bool(re.search(r':(hover|focus|active|visited|checked)', svg_text)),
        "animation": bool(re.search(r'<animate|<set |@keyframes|animation:|transition:', svg_text)),
        "script": bool(re.search(r'<script|onclick|onmouseover|onload', svg_text, re.IGNORECASE)),
        "gradient": bool(re.search(r'<(linearGradient|radialGradient|pattern)\b', svg_text)),
        "filter": bool(re.search(r'<filter\b', svg_text)),
        "clippath": bool(re.search(r'<clipPath\b', svg_text)),
        "css_rules": bool(re.search(r'<style', svg_text)),
    }


def _deterministic_prepass(svg_text: str) -> tuple[str, bool, list[str]]:
    """Apply safe lossless tools deterministically before the model starts.

    Returns (compressed_svg, is_lossless, applied_tool_names) tuple.
    Falls back to (original, False, []) if quality check fails.
    """
    original = svg_text
    features = _has_interactive_features(svg_text)

    # Phase 1: Always safe — no risk to interactivity, gradients, or animations
    safe_tools = [
        remove_metadata,
        remove_hidden_elements,
        remove_default_attributes,
        unwrap_bare_groups,
        unwrap_single_tspans,
        shorten_colors,
        shapes_to_paths,
        strip_whitespace,
        compact_path_numbers,
        remove_space_before_negative,
        merge_same_commands,
        curve_to_hv,
        merge_collinear_lines,
    ]

    # Phase 2: Conditionally safe — only if no interactive/CSS features
    if not features["hover_or_pseudo"] and not features["css_rules"] and not features["script"]:
        safe_tools.insert(3, remove_classes)       # after remove_default_attributes
        safe_tools.insert(4, style_to_attributes)  # after remove_classes

    if not features["hover_or_pseudo"] and not features["script"]:
        safe_tools.append(shorten_ids)

    if not features["gradient"] and not features["filter"] and not features["clippath"]:
        safe_tools.append(remove_unused_defs)

    # Apply all tools in order, track which ones saved bytes
    applied = []
    for tool_func in safe_tools:
        try:
            result = tool_func(svg_text)
            if len(result) < len(svg_text):
                svg_text = result
                applied.append(tool_func.__name__)
        except Exception:
            continue  # skip tool on error, keep current state

    # Quality gate: single check at the end
    saved = len(original) - len(svg_text)
    if saved <= 0:
        return original, False, []

    try:
        ssim_val, psnr_val = compare(original, svg_text)
        if ssim_val >= 0.99 and psnr_val >= 30.0:
            is_lossless = (ssim_val >= 0.9999)
            return svg_text, is_lossless, applied
    except Exception:
        pass

    # Quality check failed — fall back to original
    return original, False, []


def _format_det_trajectory_for_llm(det_result: dict) -> str:
    """Format deterministic pipeline results as context for the LLM.

    Tells the model exactly what was tried, what worked, and what failed
    so it doesn't waste turns repeating the same operations.
    """
    traj = det_result["tool_trajectory"]
    lines = []
    lines.append(
        f"DETERMINISTIC PIPELINE ALREADY APPLIED: "
        f"compressed from {det_result.get('original_size', '?')} to "
        f"{det_result['compressed_size']} bytes "
        f"({det_result['compression_pct']}% compression), "
        f"SSIM={det_result['ssim']}, PSNR={det_result['psnr']}."
    )
    lines.append("")

    applied = []
    failed = []
    no_effect = []

    for step in traj:
        tool = step["tool"]
        args = step.get("args", {})
        args_str = ", ".join(f"{k}={v}" for k, v in args.items()) if args else ""
        label = f"{tool}({args_str})" if args_str else tool

        if step["status"] == "applied":
            saved = step.get("saved", 0)
            ssim = step.get("ssim", "")
            applied.append(f"  {label}: saved {saved}b" + (f", SSIM={ssim}" if ssim else ""))
        elif step["status"] == "reverted":
            ssim = step.get("ssim", "")
            failed.append(f"  {label}: FAILED quality gate (SSIM={ssim})")
        elif step["status"] == "no_change":
            no_effect.append(tool)

    if applied:
        lines.append("Tools that WORKED (do not re-run, already applied):")
        lines.extend(applied)
    if failed:
        lines.append("\nTools that FAILED quality gate (don't retry same params):")
        lines.extend(failed)
    if no_effect:
        lines.append(f"\nTools with NO EFFECT (skip): {', '.join(no_effect)}")

    lines.append(
        "\n\nYour job: find ADDITIONAL compression beyond what the deterministic pipeline achieved. "
        "Try different parameter combinations, tool orderings, or creative approaches. "
        "Do NOT re-run tools listed above with the same parameters — they will save 0 bytes or fail again. "
        "Focus on: (1) trying tools that failed with DIFFERENT preceding steps, "
        "(2) rescale_viewbox if viewBox is small, "
        "(3) re-running rounding after different simplification orders."
    )

    return "\n".join(lines)


def optimize_svg(svg_text: str, level: str = "conservative",
                 thinking_budget: int | None = None,
                 det_result: dict | None = None) -> dict:
    """Run the LLM agent loop on SVG text and return optimization results.

    The model has full agency: it calls tools, checks quality with
    compare_quality, and decides whether to revert or keep changes.

    Args:
        svg_text: Raw SVG markup to optimize.
        level: Quality level — "lossless", "conservative", or "aggressive".
        thinking_budget: Gemini only — token budget for thinking.
            None = auto, 0 = disable thinking (fastest/cheapest).
        det_result: Optional result from optimize_svg_deterministic(). If provided,
            the LLM starts from the deterministic result and gets context about
            what was already tried.
    """
    if level not in QUALITY_THRESHOLDS:
        raise ValueError(f"Unknown level: {level!r}. Use lossless/conservative/aggressive.")

    original_raw = svg_text

    if det_result is not None:
        # Hybrid mode: start from deterministic result
        svg_text = det_result["optimized_svg"]
        prepass_lossless = det_result.get("ssim", 1.0) >= 0.9999
        prepass_tools = [s["tool"] for s in det_result["tool_trajectory"]
                         if s["status"] == "applied"]
        prepass_saved = len(original_raw.encode()) - len(svg_text.encode())
        det_context = _format_det_trajectory_for_llm(det_result)
    else:
        # Standard mode: apply safe lossless tools before the model starts
        svg_text, prepass_lossless, prepass_tools = _deterministic_prepass(svg_text)
        prepass_saved = len(original_raw.encode()) - len(svg_text.encode())
        det_context = None

    tools_for_level = _filter_tools_for_level(level)

    client = create_client(thinking_budget=thinking_budget)
    system_prompt = _build_system_prompt(level)

    # Reset Gemini chat session if applicable
    if hasattr(client, 'reset'):
        client.reset()

    # Server-side SVG state
    svg_state = {
        "current": "",
        "previous": "",  # for revert
        "original": "",
        "done": False,
        "checkpoints": {},  # name -> svg_text
        "best_passing": "",  # smallest SVG that passed quality thresholds
        "best_passing_ssim": None,
        "best_passing_psnr": None,
    }
    trajectory: list[dict] = []

    original_size = len(original_raw.encode("utf-8"))

    if det_context:
        user_msg = (
            "A deterministic pipeline has already compressed this SVG. "
            "Start with read_svg to see what was already done. "
            "Your goal: squeeze out ADDITIONAL compression beyond what the pipeline achieved. "
            "Try different tool orderings, parameter combinations, and creative approaches. "
            "Do not call write_svg until you have exhausted all options."
        )
    else:
        user_msg = (
            "Compress this SVG to the absolute minimum size possible. "
            "Start with read_svg and follow its recommendations closely. "
            "Apply ALL applicable tools — do not stop early. "
            "After Phase 3, if compression is below 30%, try more aggressive params "
            "and rescale_viewbox. Most SVGs can reach 50-80% compression. "
            "Do not call write_svg until you have tried every tool."
        )

    messages = [{"role": "user", "content": user_msg}]

    total_input_tokens = 0
    total_output_tokens = 0
    t0 = time.time()

    for turn in range(MAX_TURNS):
        try:
            response = client.create(
                system=system_prompt,
                tools=tools_for_level,
                messages=messages,
                max_tokens=4096,
            )
        except Exception as e:
            # Rate limit or other API error — retry once after wait
            time.sleep(30)
            try:
                response = client.create(
                    system=system_prompt,
                    tools=tools_for_level,
                    messages=messages,
                    max_tokens=4096,
                )
            except Exception:
                break

        total_input_tokens += response.input_tokens
        total_output_tokens += response.output_tokens

        # Add assistant message to history (Anthropic needs this; Gemini manages internally)
        assistant_msg = client.build_assistant_message(response)
        if assistant_msg is not None:
            messages.append(assistant_msg)

        if response.stop_reason == "end_turn":
            break

        # Handle tool calls
        tool_results = []
        for tc in response.tool_calls:
            name = tc.name
            args = tc.args
            entry = {"tool": name, "args": {k: v for k, v in args.items()}}

            try:
                if name in TOOL_FUNCS:
                    # Compression tool — operate on server-side state
                    func = TOOL_FUNCS[name]
                    tool_args = {k: v for k, v in args.items() if k != "svg_text"}
                    result = func(svg_text=svg_state["current"], **tool_args)
                    saved = len(svg_state["current"]) - len(result)

                    if saved <= 0:
                        result_msg = f"{name}: no size reduction ({saved} bytes). No change made."
                        entry["status"] = "no_change"
                        entry["saved"] = 0
                    else:
                        # Save previous state for revert, apply change
                        svg_state["previous"] = svg_state["current"]
                        svg_state["current"] = result
                        total_saved = len(svg_state["original"]) - len(result)
                        total_pct = total_saved / len(svg_state["original"]) * 100

                        # Quality check — report to model, track passing checkpoints
                        try:
                            ssim_val, psnr_val = compare(svg_state["original"], result)
                            thresholds = QUALITY_THRESHOLDS[level]
                            passes = ssim_val >= thresholds["ssim"] and psnr_val >= thresholds["psnr"]
                            quality_str = (
                                f"SSIM={ssim_val:.4f} PSNR={psnr_val:.1f}dB "
                                f"{'PASS' if passes else 'FAIL — you must fix this before write_svg'}"
                            )
                            entry["ssim"] = round(ssim_val, 4)
                            entry["psnr"] = round(psnr_val, 1)

                            # Track best passing checkpoint for fallback
                            if passes and len(result) < len(svg_state.get("best_passing", result + "x")):
                                svg_state["best_passing"] = result
                                svg_state["best_passing_ssim"] = ssim_val
                                svg_state["best_passing_psnr"] = psnr_val
                        except Exception:
                            quality_str = "(quality check failed)"

                        result_msg = (
                            f"{name}: saved {saved} bytes. "
                            f"Now {len(result)} bytes ({total_pct:.1f}% total compression). "
                            f"{quality_str}"
                        )
                        entry["status"] = "applied"
                        entry["saved"] = saved

                elif name == "read_svg":
                    if svg_state["original"]:
                        result_msg = (
                            f"Already loaded. Current size: {len(svg_state['current'])} bytes "
                            f"({(1 - len(svg_state['current']) / len(svg_state['original'])) * 100:.1f}% compressed)."
                        )
                    else:
                        svg_state["current"] = svg_text
                        # If pre-pass was lossless (SSIM>0.9999), use post-prepass
                        # as the original — model gets full quality budget.
                        # Otherwise use raw original so pre-pass loss counts.
                        svg_state["original"] = svg_text if prepass_lossless else original_raw
                        svg_state["previous"] = svg_text
                        # Seed best_passing with initial SVG — guarantees a fallback
                        svg_state["best_passing"] = svg_text
                        svg_state["best_passing_ssim"] = 1.0
                        svg_state["best_passing_psnr"] = 100.0
                        summary = _summarize_svg(svg_text)
                        if det_context:
                            # Hybrid mode: give full deterministic trajectory
                            summary += "\n\n" + det_context
                        elif prepass_saved > 0 and prepass_tools:
                            prepass_pct = prepass_saved / original_size * 100
                            tools_list = ", ".join(prepass_tools)
                            summary += (
                                f"\n\nPRE-PASS ALREADY APPLIED: saved {prepass_saved} bytes "
                                f"({prepass_pct:.1f}%). Tools already run: {tools_list}. "
                                "Do NOT call these again — they will save 0 bytes. "
                                "Start from Phase 2."
                            )
                        result_msg = summary
                    entry["status"] = "ok"
                    entry["saved"] = 0

                elif name == "read_state":
                    result_msg = svg_state["current"] if svg_state["current"] else "No SVG loaded. Call read_svg first."
                    entry["status"] = "ok"
                    entry["saved"] = 0

                elif name == "write_svg":
                    svg_state["done"] = True
                    comp_pct = (1 - len(svg_state["current"]) / len(svg_state["original"])) * 100 if svg_state["original"] else 0
                    result_msg = f"Saved. Final size: {len(svg_state['current'])} bytes ({comp_pct:.1f}% compression)."
                    entry["status"] = "ok"
                    entry["saved"] = 0

                elif name == "get_size":
                    if not svg_state["original"]:
                        result_msg = "No SVG loaded. Call read_svg first."
                    else:
                        total_pct = (1 - len(svg_state["current"]) / len(svg_state["original"])) * 100
                        result_msg = (
                            f"Current: {len(svg_state['current'])} bytes. "
                            f"Original: {len(svg_state['original'])} bytes. "
                            f"Compression: {total_pct:.1f}%"
                        )
                    entry["status"] = "ok"
                    entry["saved"] = 0

                elif name == "compare_quality":
                    if not svg_state["original"] or not svg_state["current"]:
                        result_msg = "No SVG loaded. Call read_svg first."
                    else:
                        ssim_val, psnr_val = compare(svg_state["original"], svg_state["current"])
                        thresholds = QUALITY_THRESHOLDS[level]
                        passes = ssim_val >= thresholds["ssim"] and psnr_val >= thresholds["psnr"]
                        result_msg = (
                            f"SSIM={ssim_val:.4f} PSNR={psnr_val:.1f}dB — "
                            f"{'PASS' if passes else 'FAIL'} "
                            f"(thresholds: SSIM>{thresholds['ssim']:.2f}, PSNR>{thresholds['psnr']:.0f}dB)"
                        )
                        entry["ssim"] = round(ssim_val, 4)
                        entry["psnr"] = round(psnr_val, 1)
                    entry["status"] = "ok"
                    entry["saved"] = 0

                elif name == "revert":
                    if svg_state["previous"] and svg_state["previous"] != svg_state["current"]:
                        reverted_size = len(svg_state["current"])
                        svg_state["current"] = svg_state["previous"]
                        restored_size = len(svg_state["current"])
                        result_msg = (
                            f"Reverted. Restored from {reverted_size} to {restored_size} bytes."
                        )
                        entry["status"] = "reverted"
                        entry["saved"] = 0
                    else:
                        result_msg = "Nothing to revert."
                        entry["status"] = "no_change"
                        entry["saved"] = 0

                elif name == "save_checkpoint":
                    cp_name = args.get("name", "")
                    if not cp_name:
                        result_msg = "Error: checkpoint name required."
                        entry["status"] = "error"
                    else:
                        svg_state["checkpoints"][cp_name] = svg_state["current"]
                        cur_size = len(svg_state["current"])
                        total_pct = (1 - cur_size / len(svg_state["original"])) * 100 if svg_state["original"] else 0
                        result_msg = (
                            f"Checkpoint '{cp_name}' saved. "
                            f"State: {cur_size} bytes ({total_pct:.1f}% compressed). "
                            f"Active checkpoints: {', '.join(svg_state['checkpoints'].keys())}"
                        )
                        entry["status"] = "ok"
                    entry["saved"] = 0

                elif name == "restore_checkpoint":
                    cp_name = args.get("name", "")
                    if cp_name not in svg_state["checkpoints"]:
                        available = ", ".join(svg_state["checkpoints"].keys()) or "none"
                        result_msg = f"Checkpoint '{cp_name}' not found. Available: {available}"
                        entry["status"] = "error"
                    else:
                        svg_state["previous"] = svg_state["current"]
                        svg_state["current"] = svg_state["checkpoints"][cp_name]
                        cur_size = len(svg_state["current"])
                        total_pct = (1 - cur_size / len(svg_state["original"])) * 100 if svg_state["original"] else 0
                        result_msg = (
                            f"Restored checkpoint '{cp_name}'. "
                            f"Now {cur_size} bytes ({total_pct:.1f}% compressed)."
                        )
                        entry["status"] = "ok"
                    entry["saved"] = 0

                else:
                    result_msg = f"Unknown tool: {name}"
                    entry["status"] = "error"
                    entry["saved"] = 0

                trajectory.append(entry)
                tool_results.append(
                    client.make_tool_result(tc.id, result_msg, tool_name=name)
                )

            except Exception as e:
                entry["status"] = "error"
                entry["saved"] = 0
                entry["error"] = str(e)
                trajectory.append(entry)
                tool_results.append(
                    client.make_tool_result(tc.id, f"Error: {e}", is_error=True, tool_name=name)
                )

        messages.append(client.build_tool_results(tool_results))

    elapsed = time.time() - t0
    tokens_used = total_input_tokens + total_output_tokens

    # Compute final metrics — check quality, fall back to best passing checkpoint
    optimized = svg_state["current"] or svg_text
    final_ssim = None
    final_psnr = None
    used_fallback = False

    if optimized:
        try:
            final_ssim, final_psnr = compare(original_raw, optimized)
            thresholds = QUALITY_THRESHOLDS[level]
            passes = final_ssim >= thresholds["ssim"] and final_psnr >= thresholds["psnr"]

            if not passes and svg_state["best_passing"]:
                # Model's final result fails quality — use best passing checkpoint
                optimized = svg_state["best_passing"]
                final_ssim, final_psnr = compare(original_raw, optimized)
                used_fallback = True
        except Exception:
            # If quality check fails entirely, try best passing checkpoint
            if svg_state["best_passing"]:
                optimized = svg_state["best_passing"]
                try:
                    final_ssim, final_psnr = compare(original_raw, optimized)
                    used_fallback = True
                except Exception:
                    pass

    if final_ssim is not None:
        final_ssim = round(final_ssim, 4)
    if final_psnr is not None:
        final_psnr = round(final_psnr, 1)

    compressed_size = len(optimized.encode("utf-8"))
    compression_pct = (1 - compressed_size / original_size) * 100 if original_size > 0 else 0.0

    return {
        "optimized_svg": optimized,
        "original_size": original_size,
        "compressed_size": compressed_size,
        "compression_pct": round(compression_pct, 1),
        "ssim": final_ssim,
        "psnr": final_psnr,
        "used_fallback": used_fallback,
        "tool_trajectory": trajectory,
        "elapsed_time": round(elapsed, 1),
        "tokens_used": tokens_used,
    }


def verify_svg(original_svg: str, optimized_svg: str) -> dict:
    """Verify that the optimized SVG is structurally valid."""
    result = {"valid": True, "issues": []}

    if not optimized_svg or not optimized_svg.strip():
        result["valid"] = False
        result["issues"].append("Empty output")
        return result

    if "<svg" not in optimized_svg.lower():
        result["valid"] = False
        result["issues"].append("No <svg> element found")
        return result

    orig_vb = re.search(r'viewBox\s*=\s*"([^"]*)"', original_svg)
    opt_vb = re.search(r'viewBox\s*=\s*"([^"]*)"', optimized_svg)
    if orig_vb and not opt_vb:
        result["issues"].append("viewBox removed")

    return result
