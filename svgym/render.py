"""SVG rendering and visual comparison utilities."""

import io
from dataclasses import dataclass

import cairosvg
import numpy as np
from PIL import Image


def render_svg(svg_text: str, size: int = 256) -> np.ndarray | None:
    """Render SVG string to RGB numpy array on white background.

    SVGs with transparency are composited onto white.
    Returns None if rendering fails.
    """
    try:
        png_data = cairosvg.svg2png(
            bytestring=svg_text.encode("utf-8"),
            output_width=size,
            output_height=size,
        )
        img = Image.open(io.BytesIO(png_data)).convert("RGBA")
        # Composite onto white background (SVGs often have transparency)
        bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
        composite = Image.alpha_composite(bg, img).convert("RGB")
        return np.array(composite)
    except Exception:
        return None


def render_svg_rgba(svg_text: str, size: int = 256) -> np.ndarray | None:
    """Render SVG string to RGBA numpy array (preserves transparency)."""
    try:
        png_data = cairosvg.svg2png(
            bytestring=svg_text.encode("utf-8"),
            output_width=size,
            output_height=size,
        )
        img = Image.open(io.BytesIO(png_data)).convert("RGBA")
        return np.array(img)
    except Exception:
        return None


@dataclass
class ImageMetrics:
    """All distance/similarity metrics between two images."""

    l2_distance: float          # Euclidean distance (lower = more similar)
    mse: float                  # Mean Squared Error (lower = better)
    psnr: float                 # Peak Signal-to-Noise Ratio in dB (higher = better)
    ssim: float                 # Structural Similarity Index (higher = better, max 1.0)
    mae: float                  # Mean Absolute Error (lower = better)
    pixel_match_ratio: float    # Fraction of exactly matching pixels
    max_pixel_error: float      # Worst-case single pixel difference (0-255 scale)

    def summary(self) -> str:
        lines = [
            f"  SSIM:              {self.ssim:.4f}  (1.0 = identical)",
            f"  PSNR:              {self.psnr:.1f} dB  (>40 = excellent, >30 = good)",
            f"  MSE:               {self.mse:.2f}  (0 = identical)",
            f"  MAE:               {self.mae:.2f}  (0 = identical)",
            f"  L2 distance:       {self.l2_distance:.1f}",
            f"  Pixel match:       {self.pixel_match_ratio:.2%}",
            f"  Max pixel error:   {self.max_pixel_error:.0f} / 255",
        ]
        return "\n".join(lines)


def compute_metrics(img_a: np.ndarray, img_b: np.ndarray) -> ImageMetrics:
    """Compute all visual similarity metrics between two RGB images.

    Both images must have the same shape (H, W, C).
    """
    from skimage.metrics import structural_similarity as ssim_fn

    assert img_a.shape == img_b.shape, (
        f"Shape mismatch: {img_a.shape} vs {img_b.shape}"
    )

    a = img_a.astype(np.float64)
    b = img_b.astype(np.float64)
    diff = a - b

    # L2 (Euclidean distance across all pixels)
    l2 = np.sqrt(np.sum(diff ** 2))

    # MSE
    mse = np.mean(diff ** 2)

    # PSNR
    if mse == 0:
        psnr = float("inf")
    else:
        psnr = 10 * np.log10(255.0 ** 2 / mse)

    # SSIM
    ssim_val = ssim_fn(img_a, img_b, channel_axis=2, data_range=255)

    # MAE
    mae = np.mean(np.abs(diff))

    # Pixel-level exact match ratio
    pixel_match = np.all(img_a == img_b, axis=2).mean()

    # Max single-pixel error
    max_err = np.max(np.abs(diff))

    return ImageMetrics(
        l2_distance=float(l2),
        mse=float(mse),
        psnr=float(psnr),
        ssim=float(ssim_val),
        mae=float(mae),
        pixel_match_ratio=float(pixel_match),
        max_pixel_error=float(max_err),
    )


@dataclass
class CompressionStats:
    """Compression statistics for an SVG pair."""

    original_bytes: int
    compressed_bytes: int
    ratio: float               # 1 - compressed/original (higher = more compression)
    absolute_savings: int      # bytes saved

    def summary(self) -> str:
        return (
            f"  Original:   {self.original_bytes:>8,} bytes\n"
            f"  Compressed: {self.compressed_bytes:>8,} bytes\n"
            f"  Savings:    {self.absolute_savings:>8,} bytes ({self.ratio:.1%} reduction)"
        )


def compression_stats(original: str, compressed: str) -> CompressionStats:
    orig_bytes = len(original.encode("utf-8"))
    comp_bytes = len(compressed.encode("utf-8"))
    return CompressionStats(
        original_bytes=orig_bytes,
        compressed_bytes=comp_bytes,
        ratio=1 - comp_bytes / orig_bytes if orig_bytes > 0 else 0.0,
        absolute_savings=orig_bytes - comp_bytes,
    )
