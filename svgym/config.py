"""Core configuration for SVGym (library + CLI).

The deterministic pipeline needs none of this except QUALITY_THRESHOLDS /
LOSSLESS_TOOLS. The optional AI mode reads provider/model/API keys; set them in a
.env file at the repo root or as environment variables.
"""
import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except Exception:
    pass

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

# Provider for the optional AI mode: "anthropic" or "gemini"
PROVIDER = os.environ.get("SVGYM_PROVIDER", "anthropic")
ANTHROPIC_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
MODEL = GEMINI_MODEL if PROVIDER == "gemini" else ANTHROPIC_MODEL

MAX_TURNS = 50

QUALITY_THRESHOLDS = {
    "lossless": {"ssim": 1.0, "psnr": float("inf")},
    "conservative": {"ssim": 0.99, "psnr": 30.0},
    "aggressive": {"ssim": 0.97, "psnr": 25.0},
}

LOSSLESS_TOOLS = {
    "remove_identity_transforms", "remove_hidden_elements", "remove_default_attributes",
    "unwrap_bare_groups", "style_to_attributes", "remove_classes", "unwrap_single_tspans",
    "remove_unused_defs", "consolidate_attrs_to_parent", "strip_whitespace", "merge_paths",
    "merge_same_commands", "curve_to_hv", "merge_collinear_lines", "compact_path_numbers",
    "remove_space_before_negative", "shorten_colors", "shorten_ids", "remove_metadata",
    "shapes_to_paths", "extract_common_styles", "merge_text_elements", "deduplicate_paths",
    "remove_junk_attrs", "remove_unused_ids", "simplify_transforms", "merge_subpaths",
}
