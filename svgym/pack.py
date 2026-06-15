"""Batch icon pack analyzer and optimizer.

Scans a directory of SVGs, computes cross-file statistics,
optimizes each file, and optionally generates a sprite sheet.
"""

from __future__ import annotations

import asyncio
import hashlib
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from svgym.deterministic import optimize_svg_deterministic
from svgym.render_bench import RenderBenchmark


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def analyze_pack(svg_files: list[tuple[str, str]]) -> dict:
    """Analyze a pack of SVGs for cross-file patterns.

    Args:
        svg_files: List of (filename, svg_text) tuples.

    Returns:
        Pack profile dict with attribute frequencies, duplicates,
        precision stats, viewbox groups, and outliers.
    """
    viewboxes = Counter()
    root_attrs = Counter()       # "attr=value" -> count
    element_attrs = Counter()    # "attr=value" on any element -> count
    path_hashes = defaultdict(list)  # hash -> list of filenames
    precisions = []
    file_sizes = []
    per_file = {}

    for filename, svg in svg_files:
        size = len(svg.encode("utf-8"))
        file_sizes.append(size)
        per_file[filename] = {"size": size}

        # viewBox
        vb = re.search(r'viewBox="([^"]*)"', svg)
        if vb:
            viewboxes[vb.group(1)] += 1
            per_file[filename]["viewBox"] = vb.group(1)

        # Root <svg> attributes
        svg_tag = re.search(r'<svg\s+([^>]*)>', svg)
        if svg_tag:
            for name, val in re.findall(r'([\w:-]+)="([^"]*)"', svg_tag.group(1)):
                root_attrs[f'{name}="{val}"'] += 1

        # All element attributes (for shared style extraction)
        for name, val in re.findall(r'([\w:-]+)="([^"]*)"', svg):
            if name not in ("d", "viewBox", "xmlns", "id", "class"):
                element_attrs[f'{name}="{val}"'] += 1

        # Path fingerprints
        for d in re.findall(r'\bd="([^"]*)"', svg):
            h = hashlib.md5(d.encode()).hexdigest()[:16]
            path_hashes[h].append(filename)

            # Precision analysis
            for dec in re.findall(r'\.(\d+)', d):
                precisions.append(len(dec))

    n = len(svg_files)

    # Shared attributes (>80% of files)
    shared_attrs = {
        attr: count for attr, count in root_attrs.items()
        if count > n * 0.8
    }

    # Common element attrs (>50% of files)
    common_element_attrs = {
        attr: count for attr, count in element_attrs.items()
        if count > n * 0.5
    }

    # Duplicate paths
    duplicates = {
        h: files for h, files in path_hashes.items()
        if len(files) > 1
    }

    # Precision distribution
    prec_dist = dict(Counter(precisions).most_common()) if precisions else {}

    # Dominant viewBox
    dominant_vb = viewboxes.most_common(1)[0] if viewboxes else (None, 0)

    # ViewBox groups
    vb_groups = {}
    for vb, count in viewboxes.items():
        vb_groups[vb] = count

    # Outliers: files that differ from dominant patterns
    outliers = []
    if dominant_vb[0]:
        for filename, info in per_file.items():
            if info.get("viewBox") and info["viewBox"] != dominant_vb[0]:
                # Only flag if there's a clear dominant (>60%)
                if dominant_vb[1] > n * 0.6:
                    outliers.append({
                        "file": filename,
                        "issue": "viewBox",
                        "expected": dominant_vb[0],
                        "actual": info["viewBox"],
                    })

    file_sizes.sort()
    size_median = file_sizes[len(file_sizes) // 2] if file_sizes else 0

    # Estimate extractable savings from shared root attrs
    extractable_bytes = 0
    for attr, count in shared_attrs.items():
        # Each file that has this attr saves len(attr)+3 bytes (attr="value" + space)
        extractable_bytes += (len(attr) + 3) * count

    return {
        "file_count": n,
        "total_size": sum(file_sizes),
        "size_min": min(file_sizes) if file_sizes else 0,
        "size_max": max(file_sizes) if file_sizes else 0,
        "size_median": size_median,
        "viewbox_groups": vb_groups,
        "dominant_viewbox": dominant_vb[0],
        "shared_root_attrs": shared_attrs,
        "common_element_attrs": common_element_attrs,
        "duplicate_paths": {h: len(files) for h, files in duplicates.items()},
        "duplicate_path_count": len(duplicates),
        "precision_distribution": prec_dist,
        "outliers": outliers,
        "extractable_bytes_estimate": extractable_bytes,
    }


# ---------------------------------------------------------------------------
# Optimize
# ---------------------------------------------------------------------------

def _benchmark_pack_renders(
    svg_files: list[tuple[str, str]],
    results: list[dict],
) -> None:
    """Benchmark render speed for each file in the pack.

    Measures comparative ratios (not absolute times):
    - speedup_vs_original: original_ms / optimized_ms
    - speedup_vs_svgo: svgo_ms / optimized_ms
    - render_vs_pack_median: median_optimized_ms / this_optimized_ms
      (>1 = faster than median, <1 = slower than median)

    Mutates results in-place to add render fields.
    """
    originals = {name: svg for name, svg in svg_files}

    async def _run():
        async with RenderBenchmark() as bench:
            opt_times = []
            for r in results:
                fname = r["filename"]
                original_svg = originals.get(fname, "")
                optimized_svg = r["optimized_svg"]
                svgo_svg = r.get("svgo_svg")

                orig_ms = await bench.measure(original_svg, iterations=30)
                opt_ms = await bench.measure(optimized_svg, iterations=30)

                r["speedup_vs_original"] = round(orig_ms / opt_ms, 2) if opt_ms > 0 else 1.0

                if svgo_svg:
                    svgo_ms = await bench.measure(svgo_svg, iterations=30)
                    r["speedup_vs_svgo"] = round(svgo_ms / opt_ms, 2) if opt_ms > 0 else 1.0

                opt_times.append((fname, opt_ms))

            # Compute pack median render time
            times_sorted = sorted(t for _, t in opt_times)
            n = len(times_sorted)
            if n > 0:
                median_ms = times_sorted[n // 2] if n % 2 else (times_sorted[n // 2 - 1] + times_sorted[n // 2]) / 2
                for r in results:
                    fname = r["filename"]
                    file_ms = next((t for f, t in opt_times if f == fname), None)
                    if file_ms and file_ms > 0:
                        r["render_vs_pack_median"] = round(median_ms / file_ms, 2)

    asyncio.run(_run())


def optimize_pack(
    svg_files: list[tuple[str, str]],
    level: str = "conservative",
    pack_name: str = "unnamed",
    on_progress: callable = None,
) -> dict:
    """Optimize every SVG in a pack and return aggregated results.

    Args:
        svg_files: List of (filename, svg_text) tuples.
        level: Quality level for optimization.
        pack_name: Name for logging.
        on_progress: Optional callback(filename, index, total) for progress.

    Returns:
        Dict with per-file results, aggregated stats, and analysis.
    """
    t0 = time.time()

    # Step 1: Analyze
    analysis = analyze_pack(svg_files)

    # Step 2: Optimize each file
    results = []
    total_input = 0
    total_output = 0
    ssim_sum = 0
    ssim_count = 0

    for i, (filename, svg_text) in enumerate(svg_files):
        if on_progress:
            on_progress(filename, i, len(svg_files))

        input_size = len(svg_text.encode("utf-8"))
        total_input += input_size

        try:
            result = optimize_svg_deterministic(svg_text, level=level)
            output_size = result["compressed_size"]
            total_output += output_size

            if result.get("ssim") is not None:
                ssim_sum += result["ssim"]
                ssim_count += 1

            results.append({
                "filename": filename,
                "input_size": input_size,
                "output_size": output_size,
                "svgo_size": result.get("svgo_size"),
                "svgo_svg": result.get("svgo_svg"),
                "compression_pct": result["compression_pct"],
                "ssim": result.get("ssim"),
                "psnr": result.get("psnr"),
                "optimized_svg": result["optimized_svg"],
                "tool_trajectory": result["tool_trajectory"],
            })
        except Exception as e:
            # Don't fail the whole pack for one file
            total_output += input_size
            results.append({
                "filename": filename,
                "input_size": input_size,
                "output_size": input_size,
                "compression_pct": 0,
                "ssim": None,
                "psnr": None,
                "optimized_svg": svg_text,
                "tool_trajectory": [],
                "error": str(e),
            })

    # Step 3: Render benchmarking (comparative ratios)
    try:
        _benchmark_pack_renders(svg_files, results)
    except Exception:
        pass  # Playwright not installed or failed — skip render metrics

    elapsed = time.time() - t0
    total_pct = round((1 - total_output / total_input) * 100, 1) if total_input > 0 else 0
    avg_ssim = round(ssim_sum / ssim_count, 4) if ssim_count > 0 else None

    return {
        "pack_name": pack_name,
        "analysis": analysis,
        "results": results,
        "total_input_size": total_input,
        "total_output_size": total_output,
        "total_compression_pct": total_pct,
        "avg_ssim": avg_ssim,
        "file_count": len(svg_files),
        "elapsed_time": round(elapsed, 1),
        "level": level,
    }


# ---------------------------------------------------------------------------
# Sprite generation
# ---------------------------------------------------------------------------

def generate_sprite(
    optimized_files: list[tuple[str, str]],
    analysis: dict,
) -> str:
    """Generate an SVG sprite sheet from optimized icons.

    Groups icons by viewBox, extracts shared attributes to a <style> block,
    and wraps each icon in a <symbol>.

    Args:
        optimized_files: List of (filename, optimized_svg_text) tuples.
        analysis: Pack analysis from analyze_pack().

    Returns:
        Sprite SVG string.
    """
    # Group files by viewBox
    groups = defaultdict(list)
    for filename, svg in optimized_files:
        vb = re.search(r'viewBox="([^"]*)"', svg)
        vb_str = vb.group(1) if vb else "0 0 24 24"
        groups[vb_str].append((filename, svg))

    # Build shared style from common element attributes
    shared_attrs = analysis.get("common_element_attrs", {})
    style_rules = []
    attr_to_remove = set()
    for attr_str, count in shared_attrs.items():
        # Parse "name=value"
        m = re.match(r'([\w-]+)="([^"]*)"', attr_str)
        if not m:
            continue
        name, value = m.group(1), m.group(2)
        # Only extract presentation attributes (not structural)
        if name in ("fill", "stroke", "stroke-width", "stroke-linecap",
                     "stroke-linejoin", "opacity", "fill-rule", "clip-rule"):
            css_name = name  # SVG presentation attrs are valid CSS
            style_rules.append(f"  {css_name}: {value};")
            attr_to_remove.add(f'{name}="{value}"')

    # Build sprite
    parts = ['<svg xmlns="http://www.w3.org/2000/svg">']

    if style_rules:
        parts.append("<defs><style>")
        parts.append("symbol {")
        parts.extend(style_rules)
        parts.append("}")
        parts.append("</style></defs>")

    for vb_str, files in sorted(groups.items()):
        for filename, svg in files:
            # Extract icon ID from filename
            icon_id = Path(filename).stem
            # Make ID safe (no spaces, special chars)
            icon_id = re.sub(r'[^a-zA-Z0-9_-]', '-', icon_id)

            # Extract inner content (everything between <svg> and </svg>)
            inner = re.sub(r'<svg[^>]*>', '', svg)
            inner = re.sub(r'</svg>\s*$', '', inner)
            inner = re.sub(r'/>\s*$', '/>', inner)  # handle self-closing <svg .../>

            # Remove shared attributes from inner elements
            for attr_str in attr_to_remove:
                inner = inner.replace(f' {attr_str}', '')

            inner = inner.strip()
            if not inner:
                continue

            parts.append(f'<symbol id="{icon_id}" viewBox="{vb_str}">')
            parts.append(f"  {inner}")
            parts.append("</symbol>")

    parts.append("</svg>")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def _percentile(values: list[float], p: float) -> float:
    """Return the p-th percentile (0-100) of a sorted list."""
    if not values:
        return 0
    k = (len(values) - 1) * p / 100
    f = int(k)
    c = f + 1 if f + 1 < len(values) else f
    d = k - f
    return values[f] + d * (values[c] - values[f])


def _histogram(values: list[float], bins: list[tuple[float, float]]) -> list[dict]:
    """Bucket values into bins. Each bin is (low, high]."""
    result = []
    for low, high in bins:
        count = sum(1 for v in values if low < v <= high)
        result.append({"range": f"{low}-{high}", "count": count})
    return result


def generate_report(
    pack_result: dict,
    sprite_svg: str | None = None,
) -> dict:
    """Generate a comprehensive pack optimization report.

    Args:
        pack_result: Output from optimize_pack().
        sprite_svg: Optional sprite SVG string (from generate_sprite).

    Returns:
        Report dict with summary, distributions, per-file table, and outliers.
    """
    results = pack_result["results"]
    analysis = pack_result["analysis"]
    n = len(results)

    # --- Per-file metrics ---
    compressions = []
    sizes_before = []
    sizes_after = []
    beyond_svgo = []
    file_table = []

    for r in results:
        comp = r["compression_pct"]
        compressions.append(comp)
        sizes_before.append(r["input_size"])
        sizes_after.append(r["output_size"])

        row = {
            "filename": r["filename"],
            "input_size": r["input_size"],
            "output_size": r["output_size"],
            "compression_pct": comp,
            "ssim": r.get("ssim"),
        }

        # Beyond-SVGO comparison (from deterministic pipeline's internal SVGO run)
        svgo_sz = r.get("svgo_size")
        if svgo_sz and svgo_sz > 0:
            row["svgo_size"] = svgo_sz
            pct = round((1 - r["output_size"] / svgo_sz) * 100, 1)
            row["beyond_svgo_pct"] = pct
            beyond_svgo.append(pct)

        # Render speedup ratios
        if r.get("speedup_vs_original") is not None:
            row["speedup_vs_original"] = r["speedup_vs_original"]
        if r.get("speedup_vs_svgo") is not None:
            row["speedup_vs_svgo"] = r["speedup_vs_svgo"]
        if r.get("render_vs_pack_median") is not None:
            row["render_vs_pack_median"] = r["render_vs_pack_median"]

        file_table.append(row)

    compressions_sorted = sorted(compressions)
    sizes_before_sorted = sorted(sizes_before)
    sizes_after_sorted = sorted(sizes_after)

    # --- Medians ---
    median_before = _percentile(sizes_before_sorted, 50)
    median_after = _percentile(sizes_after_sorted, 50)
    median_compression = _percentile(compressions_sorted, 50)

    # --- Size outliers (>2x median) ---
    size_outliers = []
    for r in results:
        if median_before > 0:
            ratio = r["input_size"] / median_before
            if ratio > 2.0:
                size_outliers.append({
                    "filename": r["filename"],
                    "size": r["input_size"],
                    "ratio_to_median": round(ratio, 1),
                })

    # --- Compression histogram ---
    comp_bins = [(i, i + 10) for i in range(0, 100, 10)]
    comp_histogram = _histogram(compressions, comp_bins)

    # --- Summary ---
    summary = {
        "pack_name": pack_result["pack_name"],
        "file_count": n,
        "level": pack_result["level"],
        "elapsed_time": pack_result["elapsed_time"],
        "total_input_size": pack_result["total_input_size"],
        "total_output_size": pack_result["total_output_size"],
        "total_compression_pct": pack_result["total_compression_pct"],
        "avg_ssim": pack_result["avg_ssim"],
        "median_compression_pct": round(median_compression, 1),
        "min_compression_pct": round(min(compressions), 1) if compressions else 0,
        "max_compression_pct": round(max(compressions), 1) if compressions else 0,
        "p10_compression_pct": round(_percentile(compressions_sorted, 10), 1),
        "p90_compression_pct": round(_percentile(compressions_sorted, 90), 1),
        "size_before_median": round(median_before),
        "size_after_median": round(median_after),
    }

    # Beyond SVGO stats
    if beyond_svgo:
        beyond_sorted = sorted(beyond_svgo)
        summary["avg_beyond_svgo_pct"] = round(sum(beyond_svgo) / len(beyond_svgo), 1)
        summary["median_beyond_svgo_pct"] = round(_percentile(beyond_sorted, 50), 1)

    # Render speedup stats
    speedups_orig = [f.get("speedup_vs_original") for f in file_table if f.get("speedup_vs_original") is not None]
    speedups_svgo = [f.get("speedup_vs_svgo") for f in file_table if f.get("speedup_vs_svgo") is not None]
    if speedups_orig:
        so = sorted(speedups_orig)
        summary["avg_speedup_vs_original"] = round(sum(so) / len(so), 2)
        summary["median_speedup_vs_original"] = round(_percentile(so, 50), 2)
    if speedups_svgo:
        ss = sorted(speedups_svgo)
        summary["avg_speedup_vs_svgo"] = round(sum(ss) / len(ss), 2)
        summary["median_speedup_vs_svgo"] = round(_percentile(ss, 50), 2)

    # Sprite stats
    if sprite_svg:
        sprite_size = len(sprite_svg.encode("utf-8"))
        summary["sprite_size"] = sprite_size
        summary["sprite_savings_pct"] = round(
            (1 - sprite_size / pack_result["total_output_size"]) * 100, 1
        ) if pack_result["total_output_size"] > 0 else 0
        summary["sprite_vs_original_pct"] = round(
            (1 - sprite_size / pack_result["total_input_size"]) * 100, 1
        ) if pack_result["total_input_size"] > 0 else 0

    # Sort file table by compression (worst first — most actionable)
    file_table.sort(key=lambda r: r["compression_pct"])

    report = {
        "summary": summary,
        "compression_histogram": comp_histogram,
        "size_outliers": size_outliers,
        "pack_outliers": analysis.get("outliers", []),
        "duplicate_paths": analysis.get("duplicate_path_count", 0),
        "shared_attrs_count": len(analysis.get("shared_root_attrs", {})),
        "viewbox_groups": analysis.get("viewbox_groups", {}),
        "files": file_table,
    }

    return report


def format_report_markdown(report: dict) -> str:
    """Format a report dict as a readable markdown string."""
    s = report["summary"]
    lines = []

    lines.append(f"# Pack Report: {s['pack_name']}")
    lines.append("")
    lines.append(f"## Summary")
    lines.append(f"- **Files**: {s['file_count']}")
    lines.append(f"- **Level**: {s['level']}")
    lines.append(f"- **Time**: {s['elapsed_time']}s")
    lines.append(f"- **Total size**: {s['total_input_size']:,}B -> {s['total_output_size']:,}B ({s['total_compression_pct']}%)")
    lines.append(f"- **Median compression**: {s['median_compression_pct']}% (range: {s['min_compression_pct']}% - {s['max_compression_pct']}%)")
    lines.append(f"- **P10/P90 compression**: {s['p10_compression_pct']}% / {s['p90_compression_pct']}%")
    if s.get("avg_ssim"):
        lines.append(f"- **Avg SSIM**: {s['avg_ssim']}")
    if s.get("avg_beyond_svgo_pct") is not None:
        lines.append(f"- **Beyond SVGO**: avg {s['avg_beyond_svgo_pct']}%, median {s['median_beyond_svgo_pct']}%")
    if s.get("avg_speedup_vs_original") is not None:
        lines.append(f"- **Render speedup vs original**: avg {s['avg_speedup_vs_original']}x, median {s['median_speedup_vs_original']}x")
    if s.get("avg_speedup_vs_svgo") is not None:
        lines.append(f"- **Render speedup vs SVGO**: avg {s['avg_speedup_vs_svgo']}x, median {s['median_speedup_vs_svgo']}x")
    if s.get("sprite_size"):
        lines.append(f"- **Sprite size**: {s['sprite_size']:,}B (saves {s['sprite_savings_pct']}% vs individual files, {s['sprite_vs_original_pct']}% vs originals)")
    lines.append(f"- **Size median**: {s['size_before_median']}B -> {s['size_after_median']}B")

    lines.append("")
    lines.append("## Compression Distribution")
    for bucket in report["compression_histogram"]:
        if bucket["count"] > 0:
            bar = "#" * bucket["count"]
            lines.append(f"  {bucket['range']:>6}%: {bar} ({bucket['count']})")

    if report["size_outliers"]:
        lines.append("")
        lines.append("## Size Outliers (>2x median)")
        for o in report["size_outliers"]:
            lines.append(f"  - {o['filename']}: {o['size']}B ({o['ratio_to_median']}x median)")

    if report["pack_outliers"]:
        lines.append("")
        lines.append("## Pack Outliers")
        for o in report["pack_outliers"]:
            lines.append(f"  - {o['file']}: {o['issue']} (expected {o['expected']}, got {o['actual']})")

    lines.append("")
    lines.append(f"## Details ({report['duplicate_paths']} duplicate paths, {report['shared_attrs_count']} shared attrs)")
    lines.append(f"ViewBox groups: {report['viewbox_groups']}")

    lines.append("")
    lines.append("## Per-File Results (worst compression first)")
    lines.append("")
    has_render = any(f.get("speedup_vs_original") is not None for f in report["files"])
    if has_render:
        lines.append("| File | Before | After | Comp% | SSIM | Beyond SVGO | vs Orig | vs SVGO | vs Median |")
        lines.append("|------|--------|-------|-------|------|-------------|---------|---------|-----------|")
    else:
        lines.append("| File | Before | After | Comp% | SSIM | Beyond SVGO |")
        lines.append("|------|--------|-------|-------|------|-------------|")
    for f in report["files"]:
        ssim = f"{f['ssim']:.2f}" if f.get("ssim") is not None else "—"
        beyond = f"{f['beyond_svgo_pct']}%" if f.get("beyond_svgo_pct") is not None else "—"
        if has_render:
            vs_orig = f"{f['speedup_vs_original']}x" if f.get("speedup_vs_original") is not None else "—"
            vs_svgo = f"{f['speedup_vs_svgo']}x" if f.get("speedup_vs_svgo") is not None else "—"
            vs_med = f"{f['render_vs_pack_median']}x" if f.get("render_vs_pack_median") is not None else "—"
            lines.append(f"| {f['filename']} | {f['input_size']}B | {f['output_size']}B | {f['compression_pct']}% | {ssim} | {beyond} | {vs_orig} | {vs_svgo} | {vs_med} |")
        else:
            lines.append(f"| {f['filename']} | {f['input_size']}B | {f['output_size']}B | {f['compression_pct']}% | {ssim} | {beyond} |")

    return "\n".join(lines)
