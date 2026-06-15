"""Hybrid SVG optimizer: deterministic pipeline first, LLM only when needed.

Strategy:
1. Run deterministic pipeline (free, ~3-8s)
2. If compression < threshold, call LLM with full context of what was already tried
3. LLM focuses on finding additional savings beyond the deterministic result
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from svgym.deterministic import optimize_svg_deterministic
from svgym.optimizer import optimize_svg as optimize_svg_llm


# If deterministic compression beyond SVGO is below this %, invoke the LLM
LLM_THRESHOLD_PCT = 30.0
# Files under this size skip LLM regardless of compression %
LLM_SIZE_GATE_BYTES = 5 * 1024  # 5KB


def optimize_svg_hybrid(svg_text: str, level: str = "conservative",
                        thinking_budget: int | None = None,
                        llm_threshold: float = LLM_THRESHOLD_PCT,
                        size_gate: int = LLM_SIZE_GATE_BYTES,
                        svgo_size: int | None = None,
                        original_size: int | None = None) -> dict:
    """Run hybrid optimization: deterministic first, LLM if needed.

    Args:
        svg_text: Raw SVG markup (may be SVGO output or original).
        level: Quality level.
        thinking_budget: Gemini thinking budget (None=auto, 0=disable).
        llm_threshold: Beyond-SVGO compression % below which to invoke LLM.
        svgo_size: Size of SVGO output in bytes (for beyond-SVGO calculation).
        original_size: Size of raw original SVG in bytes.

    Returns:
        Dict with all standard fields plus:
        - mode: "deterministic" or "hybrid"
        - det_compression_pct: compression from deterministic pass
        - det_elapsed: time for deterministic pass
        - llm_elapsed: time for LLM pass (0 if not invoked)
    """
    # Step 1: Deterministic pipeline
    det_result = optimize_svg_deterministic(svg_text, level=level)
    det_pct = det_result["compression_pct"]
    det_elapsed = det_result["elapsed_time"]

    # Compute beyond-SVGO compression if SVGO data available
    final_size = det_result["compressed_size"]
    if svgo_size and svgo_size > 0:
        beyond_svgo_pct = round((1 - final_size / svgo_size) * 100, 1)
    else:
        beyond_svgo_pct = det_pct  # fallback: use raw compression

    raw_size = original_size or len(svg_text.encode("utf-8"))
    if beyond_svgo_pct >= llm_threshold or raw_size < size_gate:
        # Good enough or too small to justify LLM cost
        det_result["mode"] = "deterministic"
        det_result["det_compression_pct"] = det_pct
        det_result["det_elapsed"] = det_elapsed
        det_result["llm_elapsed"] = 0
        return det_result

    # Step 2: LLM pass, starting from deterministic result
    llm_result = optimize_svg_llm(
        svg_text,
        level=level,
        thinking_budget=thinking_budget,
        det_result=det_result,
    )

    # Use whichever result is smaller
    if llm_result["compressed_size"] < det_result["compressed_size"]:
        final = llm_result
    else:
        final = det_result

    final["mode"] = "hybrid"
    final["det_compression_pct"] = det_pct
    final["det_elapsed"] = det_elapsed
    final["llm_elapsed"] = llm_result["elapsed_time"]
    # Merge token counts
    final["tokens_used"] = llm_result.get("tokens_used", 0)

    return final
