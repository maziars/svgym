"""System and user prompts for SVG compression with tool use.

These prompts are designed for use with Claude (or any LLM) that has access
to the compression tools defined in svgym.compression_tools. The model is
given tools for compressing SVGs and evaluating visual fidelity, and must
meet quality constraints while maximizing compression.

Usage:
    from svgym.prompts import SYSTEM_PROMPT, make_user_prompt

    # For API calls:
    messages = [
        {"role": "user", "content": make_user_prompt(svg_text)}
    ]
    # system=SYSTEM_PROMPT

    # For RL training:
    from svgym.prompts import TOOL_DEFINITIONS
    # Pass TOOL_DEFINITIONS as the tools parameter
"""

SYSTEM_PROMPT = """\
You are an SVG compression agent. Your goal is to take an SVG file and produce \
the smallest possible compressed version while preserving visual fidelity.

## Objective

Maximize **compression_ratio** (bytes saved / original bytes) while maintaining:
- **ssim >= 0.97** (structural similarity to original)
- **pixel_match_ratio >= 0.95** (fraction of exactly matching pixels)
- The compressed SVG **must render without errors** (render_ok = true)

If any constraint is violated, your output is considered a failure regardless \
of compression achieved.

## Compression Performance Tiers

Your compression_ratio determines your score:
- **< 10%**: Poor. You barely compressed anything. Almost always means you were \
too timid and left significant headroom on the table.
- **10-30%**: Acceptable. You applied basic optimizations but likely didn't push \
hard enough. Try more aggressive rounding, path merging, or manual edits.
- **30-50%**: Good. Solid compression with meaningful size reduction.
- **> 50%**: Excellent. You found deep structural optimizations beyond the basics.

Do NOT settle for < 10% compression. The quality constraints (ssim >= 0.97, \
pixel_match >= 0.95) are your safety net — push aggressively until you hit them, \
then back off just enough to pass.

## Available Tools

You have access to the following SVG compression and evaluation tools. \
Call them via function calls.

### SVG Optimizer (Lossless)

1. **optimize_svg(svg, aggressive=False)** — Run Scour, a lossless SVG optimizer \
(similar to SVGO). Strips metadata, comments, unused IDs, XML prolog, default \
attributes, and empty groups. Use this as a FIRST PASS before other tools. \
Set aggressive=True for additional optimizations like creating groups and \
enabling viewboxing. This never changes visual appearance.

### Coordinate Compression Tools

2. **round_path_coordinates(d, decimals=2)** — Round all numbers in an SVG path \
`d` attribute to `decimals` decimal places. Reconstructs with minimal separators \
using SVG adjacency rules (negative sign and dot can act as separators). \
This is your primary compression tool.
   - decimals=3: conservative, minimal visual change
   - decimals=2: good balance (recommended starting point)
   - decimals=1: aggressive, may damage fine details
   - decimals=0: very aggressive, integers only

### Structural Compression Tools

3. **merge_paths(svg)** — Merge multiple `<path>` elements that share identical \
style attributes (fill, stroke, etc.) into a single `<path>` with concatenated \
`d` data. Only merges paths with exactly matching non-d attributes. \
Fixes lowercase `m` to `M` at merge boundaries. Can break complex SVGs — \
always evaluate after using.

4. **group_shared_attributes(svg)** — Move attributes shared by all `<path>` \
elements to a wrapping `<g>` element. Safer than merge_paths (preserves path \
boundaries) but less compression.

### Cleanup Tools

5. **remove_noop_styles(svg)** — Remove no-op style attributes like \
`rotate(360deg)` transforms (360° = identity).

6. **remove_default_attributes(svg)** — Remove attributes that match SVG \
defaults: `fill-opacity="1"`, `stroke-opacity="1"`, `stroke="none"`, \
`opacity="1"`.

7. **shorten_hex_colors(svg)** — Shorten 6-digit hex colors to 3-digit where \
exact: `#aabbcc` → `#abc`. Never shortens inexact matches.

8. **collapse_whitespace(svg)** — Remove whitespace between XML tags, collapse \
multiple spaces, strip leading/trailing whitespace.

### Pipeline Tool

9. **compress_svg(svg, decimals=2, merge=True, group_attrs=True, \
remove_noop=True, remove_defaults=True, shorten_colors=True)** — Full pipeline \
that chains cleanup tools + coordinate rounding + structural optimization. \
Use this for a quick one-shot attempt, or use individual tools for finer control.

### Evaluation Tool

10. **evaluate_compression(original_svg, compressed_svg, render_size=256)** — \
Render both SVGs and compute visual similarity metrics. Returns a dictionary with:
   - `compression_ratio`: fraction of bytes saved (higher = better)
   - `ssim`: structural similarity (must be >= 0.97)
   - `pixel_match_ratio`: exact pixel match fraction (must be >= 0.95)
   - `render_ok`: whether the compressed SVG rendered successfully
   - `error`: error message if rendering failed
   - Also: `mse`, `mae`, `psnr`, `l2_distance`, `max_pixel_error`

**Always call evaluate_compression after compressing** to verify your output \
meets the constraints. If constraints are violated, adjust and try again.

## Strategy Guide

**Start aggressive, back off only when constraints are violated.**

1. **Optimize first**: Run `optimize_svg(svg)` for free lossless compression.
2. **Go aggressive**: Try `compress_svg(svg, decimals=1, merge=True)` first.
3. **Evaluate**: Call `evaluate_compression` to check metrics.
4. **If constraints pass** (ssim >= 0.97, pixel_match >= 0.95): try to push \
even further — apply manual edits, try decimals=0 on coarse paths, remove more.
5. **If constraints fail**: back off incrementally — try decimals=2 instead of 1, \
disable merge but keep group, etc. Find the most aggressive setting that still passes.
6. **Never stop at the first thing that works** if compression_ratio < 30%. \
There is almost always more to squeeze out.

### Advanced techniques (apply directly to the SVG text for extra compression):
- Remove unreferenced `id` and `class` attributes
- Simplify `transform` attributes (e.g., bake translate into coordinates)
- Convert simple `<path>` rectangles to `<rect>` elements
- Apply different rounding precision to different paths (coarse shapes tolerate \
decimals=1 or 0; fine details may need decimals=2 or 3)
- Merge CSS classes that produce identical styles
- Remove empty `<g>` wrappers and unnecessary nesting
- Inline single-use `<defs>` references
- Remove `xmlns:xlink` if no xlink references exist
- Remove XML comments and processing instructions

You are encouraged to combine the provided tools with your own SVG knowledge \
and to write your own compression logic when the tools are not enough. \
The tools handle the mechanical work (tokenizing, rounding, rendering) while \
you provide the strategic decisions about what to compress and how aggressively. \
Do not just parameter-sweep compress_svg — analyze the SVG structure and apply \
targeted optimizations.

## Output Format

After compression and evaluation, output the final compressed SVG inside \
an <svg_output> tag:

<svg_output>
[your compressed SVG here]
</svg_output>
"""

USER_PROMPT_TEMPLATE = """\
Compress the following SVG. Maximize compression while keeping ssim >= 0.97 \
and pixel_match_ratio >= 0.95.

<svg_input>
{svg}
</svg_input>
"""


def make_user_prompt(svg: str) -> str:
    """Create the user prompt with the SVG embedded.

    Args:
        svg: The SVG string to compress.

    Returns:
        Formatted user prompt string.
    """
    return USER_PROMPT_TEMPLATE.format(svg=svg)


# Tool definitions in the format expected by the Anthropic API tool_use spec.
# These map to functions in svgym.compression_tools.
TOOL_DEFINITIONS = [
    {
        "name": "optimize_svg",
        "description": (
            "Run Scour lossless SVG optimizer (similar to SVGO). Strips metadata, "
            "comments, unused IDs, XML prolog, default attributes, and empty groups. "
            "Use as a FIRST PASS before other tools. Never changes visual appearance. "
            "Set aggressive=True for additional optimizations."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "svg": {
                    "type": "string",
                    "description": "SVG string to optimize."
                },
                "aggressive": {
                    "type": "boolean",
                    "description": "Enable additional optimizations. Default false.",
                    "default": False
                }
            },
            "required": ["svg"]
        }
    },
    {
        "name": "round_path_coordinates",
        "description": (
            "Round all coordinates in an SVG path d attribute to the specified "
            "decimal places and reconstruct with minimal separators using SVG "
            "path adjacency rules. This is the primary compression technique. "
            "Lower decimals = more compression but more visual change. "
            "decimals=2 is a good default."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "d": {
                    "type": "string",
                    "description": "SVG path d attribute string to compress."
                },
                "decimals": {
                    "type": "integer",
                    "description": "Max decimal places (0-4). Default 2.",
                    "default": 2
                }
            },
            "required": ["d"]
        }
    },
    {
        "name": "merge_paths",
        "description": (
            "Merge multiple <path> elements with identical style attributes "
            "(fill, stroke, etc.) into single <path> elements with concatenated "
            "d data. Only merges paths with exactly matching non-d attributes. "
            "Correctly converts leading lowercase m to uppercase M at merge "
            "boundaries. Can break complex SVGs — always evaluate after using."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "svg": {
                    "type": "string",
                    "description": "Full SVG string."
                }
            },
            "required": ["svg"]
        }
    },
    {
        "name": "group_shared_attributes",
        "description": (
            "Move attributes shared by all <path> elements to a wrapping <g> "
            "element. Safer than merge_paths because it preserves individual "
            "path boundaries. Reduces attribute repetition."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "svg": {
                    "type": "string",
                    "description": "Full SVG string."
                }
            },
            "required": ["svg"]
        }
    },
    {
        "name": "remove_noop_styles",
        "description": (
            "Remove no-op style attributes from SVG elements, such as "
            "rotate(360deg) transforms which are identity operations."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "svg": {
                    "type": "string",
                    "description": "Full SVG string."
                }
            },
            "required": ["svg"]
        }
    },
    {
        "name": "remove_default_attributes",
        "description": (
            "Remove SVG attributes that match their default values: "
            "fill-opacity=\"1\", stroke-opacity=\"1\", stroke=\"none\", "
            "opacity=\"1\"."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "svg": {
                    "type": "string",
                    "description": "Full SVG string."
                }
            },
            "required": ["svg"]
        }
    },
    {
        "name": "shorten_hex_colors",
        "description": (
            "Shorten 6-digit hex colors to 3-digit where the color is exactly "
            "representable. E.g. #aabbcc -> #abc. Never shortens inexact "
            "matches like #3465a4."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "svg": {
                    "type": "string",
                    "description": "Full SVG string."
                }
            },
            "required": ["svg"]
        }
    },
    {
        "name": "collapse_whitespace",
        "description": (
            "Remove unnecessary whitespace: whitespace between XML tags, "
            "multiple consecutive spaces, and leading/trailing whitespace."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "svg": {
                    "type": "string",
                    "description": "Full SVG string."
                }
            },
            "required": ["svg"]
        }
    },
    {
        "name": "compress_svg",
        "description": (
            "Full compression pipeline that chains all tools: remove no-op "
            "styles, remove default attributes, shorten hex colors, round path "
            "coordinates, merge or group paths, and collapse whitespace. "
            "Use this for a quick one-shot attempt, or use individual tools for "
            "finer control."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "svg": {
                    "type": "string",
                    "description": "Full SVG string to compress."
                },
                "decimals": {
                    "type": "integer",
                    "description": "Decimal places for coordinate rounding (0-4). Default 2.",
                    "default": 2
                },
                "merge": {
                    "type": "boolean",
                    "description": "Merge same-attribute paths into one. Default true.",
                    "default": True
                },
                "group_attrs": {
                    "type": "boolean",
                    "description": "Group shared attributes in <g> (fallback if merge=false). Default true.",
                    "default": True
                },
                "remove_noop": {
                    "type": "boolean",
                    "description": "Remove no-op styles. Default true.",
                    "default": True
                },
                "remove_defaults": {
                    "type": "boolean",
                    "description": "Remove default-valued attributes. Default true.",
                    "default": True
                },
                "shorten_colors": {
                    "type": "boolean",
                    "description": "Shorten hex colors. Default true.",
                    "default": True
                }
            },
            "required": ["svg"]
        }
    },
    {
        "name": "evaluate_compression",
        "description": (
            "Render both the original and compressed SVGs and compute visual "
            "similarity metrics. ALWAYS call this after compressing to verify "
            "quality constraints are met. Returns compression_ratio, ssim, "
            "pixel_match_ratio, render_ok, and other metrics. "
            "Constraints: ssim >= 0.97, pixel_match_ratio >= 0.95, "
            "render_ok = true."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "original_svg": {
                    "type": "string",
                    "description": "The original reference SVG string."
                },
                "compressed_svg": {
                    "type": "string",
                    "description": "The compressed SVG string to evaluate."
                },
                "render_size": {
                    "type": "integer",
                    "description": "Render resolution in pixels. Default 256.",
                    "default": 256
                }
            },
            "required": ["original_svg", "compressed_svg"]
        }
    },
]


# Mapping from tool names to their implementations
def get_tool_executor():
    """Return a dict mapping tool names to callables.

    Each callable accepts **kwargs matching the tool's input_schema
    and returns the tool result (string or dict).

    Usage:
        executor = get_tool_executor()
        result = executor["round_path_coordinates"](d="M10 20L30.5-40.2", decimals=2)
    """
    from svgym.compression_tools import (
        optimize_svg,
        round_path_coordinates,
        merge_paths,
        group_shared_attributes,
        remove_noop_styles,
        remove_default_attributes,
        shorten_hex_colors,
        collapse_whitespace,
        compress_svg,
        evaluate_compression,
    )

    return {
        "optimize_svg": optimize_svg,
        "round_path_coordinates": round_path_coordinates,
        "merge_paths": merge_paths,
        "group_shared_attributes": group_shared_attributes,
        "remove_noop_styles": remove_noop_styles,
        "remove_default_attributes": remove_default_attributes,
        "shorten_hex_colors": shorten_hex_colors,
        "collapse_whitespace": collapse_whitespace,
        "compress_svg": compress_svg,
        "evaluate_compression": evaluate_compression,
    }
