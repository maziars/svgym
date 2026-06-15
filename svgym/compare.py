"""Side-by-side SVG comparison with visual metrics and compression stats."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

from svgym.render import (
    render_svg,
    compute_metrics,
    compression_stats,
    ImageMetrics,
    CompressionStats,
)


def compare_svgs(
    svg_a: str,
    svg_b: str,
    label_a: str = "Original",
    label_b: str = "Compressed",
    render_size: int = 256,
    output_path: str | Path | None = None,
    show: bool = True,
) -> tuple[ImageMetrics | None, CompressionStats]:
    """Render two SVGs side-by-side with diff map, metrics, and compression stats.

    Args:
        svg_a: First SVG string (typically the original / SVGO output).
        svg_b: Second SVG string (typically the compressed version).
        label_a: Label for the first image.
        label_b: Label for the second image.
        render_size: Pixel size for rendering (square).
        output_path: If set, save the figure to this path.
        show: If True, display the figure interactively.

    Returns:
        (ImageMetrics or None, CompressionStats)
    """
    img_a = render_svg(svg_a, size=render_size)
    img_b = render_svg(svg_b, size=render_size)

    comp = compression_stats(svg_a, svg_b)

    if img_a is None or img_b is None:
        print(f"Render failed: {'A' if img_a is None else ''} {'B' if img_b is None else ''}")
        return None, comp

    metrics = compute_metrics(img_a, img_b)

    # Build figure: [img_a | img_b | diff | metrics text]
    fig = plt.figure(figsize=(16, 5))
    gs = gridspec.GridSpec(1, 4, width_ratios=[1, 1, 1, 1.2], wspace=0.3)

    # Image A
    ax0 = fig.add_subplot(gs[0])
    ax0.imshow(img_a)
    ax0.set_title(label_a, fontsize=12, fontweight="bold")
    ax0.axis("off")

    # Image B
    ax1 = fig.add_subplot(gs[1])
    ax1.imshow(img_b)
    ax1.set_title(label_b, fontsize=12, fontweight="bold")
    ax1.axis("off")

    # Diff map (amplified for visibility)
    ax2 = fig.add_subplot(gs[2])
    diff = np.abs(img_a.astype(float) - img_b.astype(float))
    # Amplify diff for visibility (scale to 0-255 range)
    diff_max = diff.max() if diff.max() > 0 else 1.0
    diff_vis = (diff / diff_max * 255).astype(np.uint8)
    ax2.imshow(diff_vis)
    ax2.set_title(f"Difference (x{255/diff_max:.0f})", fontsize=12, fontweight="bold")
    ax2.axis("off")

    # Metrics text panel
    ax3 = fig.add_subplot(gs[3])
    ax3.axis("off")

    text = (
        "COMPRESSION\n"
        "───────────────────────\n"
        f"{comp.summary()}\n\n"
        "VISUAL SIMILARITY\n"
        "───────────────────────\n"
        f"{metrics.summary()}"
    )

    ax3.text(
        0.05, 0.95, text,
        transform=ax3.transAxes,
        fontsize=9,
        fontfamily="monospace",
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#f0f0f0", edgecolor="#cccccc"),
    )

    if output_path:
        fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
        print(f"Saved to {output_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return metrics, comp


def compare_files(
    path_a: str | Path,
    path_b: str | Path,
    label_a: str | None = None,
    label_b: str | None = None,
    **kwargs,
) -> tuple[ImageMetrics | None, CompressionStats]:
    """Compare two SVG files by path."""
    path_a, path_b = Path(path_a), Path(path_b)
    svg_a = path_a.read_text(errors="replace")
    svg_b = path_b.read_text(errors="replace")

    if label_a is None:
        label_a = path_a.name
    if label_b is None:
        label_b = path_b.name

    return compare_svgs(svg_a, svg_b, label_a=label_a, label_b=label_b, **kwargs)


def compare_batch(
    pairs: list[tuple[Path, Path]],
    output_dir: str | Path | None = None,
    render_size: int = 256,
) -> list[dict]:
    """Compare a batch of SVG pairs. Returns metrics for each.

    Args:
        pairs: List of (original_path, compressed_path) tuples.
        output_dir: If set, save comparison images here.
        render_size: Pixel size for rendering.

    Returns:
        List of dicts with filename, metrics, and compression stats.
    """
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for path_a, path_b in pairs:
        out = None
        if output_dir:
            out = output_dir / f"{path_a.stem}_compare.png"

        metrics, comp = compare_files(
            path_a, path_b,
            render_size=render_size,
            output_path=out,
            show=False,
        )

        result = {
            "filename": path_a.name,
            "compression": {
                "original_bytes": comp.original_bytes,
                "compressed_bytes": comp.compressed_bytes,
                "ratio": comp.ratio,
                "savings": comp.absolute_savings,
            },
        }

        if metrics:
            result["metrics"] = {
                "ssim": metrics.ssim,
                "psnr": metrics.psnr,
                "mse": metrics.mse,
                "mae": metrics.mae,
                "l2_distance": metrics.l2_distance,
                "pixel_match_ratio": metrics.pixel_match_ratio,
                "max_pixel_error": metrics.max_pixel_error,
            }

        results.append(result)

    return results
