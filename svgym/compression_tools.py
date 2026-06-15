"""SVG compression tools extracted from Sonnet agent behavior.

These are the compression techniques that Claude Sonnet 4.6 discovered and
implemented when asked to compress SVGs beyond SVGO. They were extracted from
130 Python scripts that Sonnet wrote and executed via Bash during zero-shot
SVG compression experiments across 29 SVG files in 3 categories (charts,
icons, line drawings).

These tools are designed to be given to smaller models as callable functions
during training and inference, so they can achieve Sonnet-level compression
without needing to independently discover these techniques.

Usage:
    from svgym.compression_tools import compress_svg
    compressed = compress_svg(svg_text, decimals=2)

Individual tools can also be used standalone:
    from svgym.compression_tools import (
        tokenize_path,
        round_path_coordinates,
        merge_paths,
        group_shared_attributes,
        remove_noop_styles,
        remove_default_attributes,
        collapse_whitespace,
        evaluate_compression,
        format_evaluation,
    )
"""
import re
from typing import List, Optional, Tuple


# ---------------------------------------------------------------------------
# Tool 1: SVG Path Tokenizer
# ---------------------------------------------------------------------------

def tokenize_path(d: str) -> List[Tuple[str, str]]:
    """Tokenize an SVG path `d` attribute into (type, value) tuples.

    Returns a list of tuples where type is one of:
    - 'cmd': SVG path command letter (M, m, L, l, C, c, S, s, Q, q, T, t, A, a, Z, z, H, h, V, v)
    - 'num': A number string (integer, decimal, or scientific notation)

    This correctly handles SVG path number format edge cases:
    - Negative signs as implicit separators: "1-2" = two numbers [1, -2]
    - Dot as implicit separator: "1.5.3" = two numbers [1.5, .3]
    - Scientific notation: "1e-4"
    - Leading dots: ".5" = 0.5
    """
    tokens = []
    i = 0
    n = len(d)
    while i < n:
        c = d[i]
        if c in 'MmZzLlHhVvCcSsQqTtAa':
            tokens.append(('cmd', c))
            i += 1
        elif c in ' ,\t\n\r':
            i += 1  # skip whitespace and comma separators
        elif c == '-' or c == '+' or c.isdigit() or c == '.':
            j = i
            if j < n and d[j] in '+-':
                j += 1
            while j < n and d[j].isdigit():
                j += 1
            if j < n and d[j] == '.':
                j += 1
                while j < n and d[j].isdigit():
                    j += 1
            # Handle scientific notation (e.g., 1e-4, 2.5E+3)
            if j < n and d[j] in 'eE':
                j += 1
                if j < n and d[j] in '+-':
                    j += 1
                while j < n and d[j].isdigit():
                    j += 1
            tokens.append(('num', d[i:j]))
            i = j
        else:
            i += 1  # skip unknown characters
    return tokens


# ---------------------------------------------------------------------------
# Tool 2: Number Formatter (compact SVG number representation)
# ---------------------------------------------------------------------------

def format_number(val: float, decimals: int = 2) -> str:
    """Format a number for minimal SVG path representation.

    - Rounds to `decimals` decimal places
    - Strips trailing zeros: 1.50 -> 1.5
    - Removes leading zero: 0.5 -> .5, -0.5 -> -.5
    - Converts to integer when possible: 1.0 -> 1

    Args:
        val: The number to format.
        decimals: Maximum decimal places to keep.

    Returns:
        Minimal string representation.
    """
    rounded = round(val, decimals)
    if rounded == int(rounded):
        return str(int(rounded))
    s = f'{rounded:.{decimals}f}'
    s = s.rstrip('0').rstrip('.')
    # Remove leading zero for sub-1 values
    if s.startswith('0.'):
        s = s[1:]
    elif s.startswith('-0.'):
        s = '-' + s[2:]
    return s if s else '0'


# ---------------------------------------------------------------------------
# Tool 3: Separator Logic (SVG path number adjacency rules)
# ---------------------------------------------------------------------------

def needs_separator(prev: str, curr: str) -> bool:
    """Determine if a separator is needed between two adjacent numbers in SVG path data.

    SVG path data has implicit separator rules that allow omitting spaces:
    1. A '-' sign acts as a separator: "1-2" parses as [1, -2]
    2. A '.' after a decimal number starts a new number: "1.5.3" parses as [1.5, .3]
    3. But '.' after an integer is ambiguous: "1.3" parses as [1.3], NOT [1, .3]

    Args:
        prev: Previous number's formatted string.
        curr: Current number's formatted string.

    Returns:
        True if a space separator is needed between prev and curr.
    """
    if not prev or not curr:
        return False
    # Negative sign acts as separator
    if curr[0] == '-':
        return False
    # Dot can act as separator only if prev already contains a dot
    if curr[0] == '.':
        if '.' in prev:
            # e.g., prev="1.5", curr=".3" -> "1.5.3" is valid (two numbers)
            return False
        else:
            # e.g., prev="1", curr=".3" -> "1.3" would be ONE number -> need space
            return True
    # Any other case: digit follows digit -> need separator
    return True


# ---------------------------------------------------------------------------
# Tool 4: Path Coordinate Rounding
# ---------------------------------------------------------------------------

def round_path_coordinates(d: str, decimals: int = 2) -> str:
    """Round all coordinates in an SVG path `d` attribute and reconstruct with minimal separators.

    This is the core compression technique. It:
    1. Tokenizes the path into commands and numbers
    2. Rounds each number to `decimals` decimal places
    3. Formats numbers compactly (no leading zeros, no trailing zeros)
    4. Reconstructs with minimal separators using SVG adjacency rules

    Args:
        d: SVG path d attribute string.
        decimals: Decimal places to round to (default 2). Lower = more compression, more visual change.
            - 3: Very conservative (minimal visual change)
            - 2: Good balance (Sonnet's default)
            - 1: Aggressive (Haiku's default, can damage fine details)
            - 0: Very aggressive (integers only, significant visual change)

    Returns:
        Compressed path d attribute string.
    """
    tokens = tokenize_path(d)
    result = []
    prev_type = None
    prev_val = None

    for ttype, tval in tokens:
        if ttype == 'cmd':
            result.append(tval)
            prev_type = 'cmd'
            prev_val = tval
        elif ttype == 'num':
            try:
                rounded = format_number(float(tval), decimals)
            except ValueError:
                rounded = tval  # keep as-is if can't parse
            if prev_type == 'num' and needs_separator(prev_val, rounded):
                result.append(' ')
            result.append(rounded)
            prev_type = 'num'
            prev_val = rounded

    return ''.join(result)


# ---------------------------------------------------------------------------
# Tool 5: Multi-Path Merging
# ---------------------------------------------------------------------------

def merge_paths(svg: str) -> str:
    """Merge multiple <path> elements with identical non-d attributes into single paths.

    Only merges paths that share the exact same style attributes (fill, stroke, etc.).
    Paths with different styles are kept separate. This prevents visual corruption
    when merging paths from complex SVGs with varied styling.

    When merging, lowercase 'm' (relative moveto) at the start of non-first
    paths must be converted to uppercase 'M' (absolute moveto), because the initial 'm'
    in a standalone path is treated as absolute per SVG spec, but after concatenation it
    would be relative to the previous path's endpoint.

    Args:
        svg: SVG string containing multiple <path> elements.

    Returns:
        SVG string with same-attribute paths merged.
    """
    # Find all <path .../> elements with their full attributes
    path_elements = re.findall(r'<path\s+([^>]+?)/>', svg)
    if len(path_elements) <= 1:
        return svg

    def parse_path_attrs(attr_str):
        """Parse attributes from a path element string."""
        attrs = {}
        for m in re.finditer(r'([\w-]+)="([^"]*)"', attr_str):
            attrs[m.group(1)] = m.group(2)
        return attrs

    # Group paths by their non-d attributes
    from collections import OrderedDict
    groups = OrderedDict()  # key: frozenset of (attr, val) pairs -> list of d values
    for attr_str in path_elements:
        attrs = parse_path_attrs(attr_str)
        d_val = attrs.pop('d', '')
        # Sort to make key order-independent
        key = frozenset(attrs.items())
        if key not in groups:
            groups[key] = {'attrs': attrs, 'd_values': []}
        groups[key]['d_values'].append(d_val)

    # Only merge groups with >1 path
    if all(len(g['d_values']) == 1 for g in groups.values()):
        return svg  # nothing to merge

    # Build merged SVG
    svg_open = re.match(r'(<svg[^>]*>)', svg)
    if not svg_open:
        return svg

    # Collect non-path content (text, rect, circle, g, style, defs, etc.)
    non_path_content = []
    for m in re.finditer(r'<(?!path\b)(?!/svg\b)(?!svg\b)([^>]+)(?:/>|>[^<]*</[^>]+>)', svg):
        non_path_content.append(m.group(0))

    parts = [svg_open.group(1)]

    # Add non-path elements back
    for elem in non_path_content:
        parts.append(elem)

    for key, group in groups.items():
        d_values = group['d_values']
        attrs = group['attrs']
        attrs_str = ' '.join(f'{k}="{v}"' for k, v in attrs.items())

        if len(d_values) == 1:
            merged_d = d_values[0]
        else:
            # Fix leading 'm' -> 'M' for non-first paths
            merged_parts = []
            for i, d in enumerate(d_values):
                d = d.strip()
                if i > 0 and d and d[0] == 'm':
                    d = 'M' + d[1:]
                merged_parts.append(d)
            merged_d = ' '.join(merged_parts)

        parts.append(f'<path {attrs_str} d="{merged_d}"/>')

    parts.append('</svg>')
    return ''.join(parts)


# ---------------------------------------------------------------------------
# Tool 6: Attribute Grouping with <g>
# ---------------------------------------------------------------------------

def group_shared_attributes(svg: str) -> str:
    """Move shared attributes from multiple <path> elements to a wrapping <g> element.

    When paths share attributes (fill, stroke, stroke-width, etc.), those can be
    moved to a parent <g> element, reducing repetition without merging the paths.
    This is safer than merging because it preserves individual path boundaries.

    Args:
        svg: SVG string containing multiple <path> elements.

    Returns:
        SVG string with shared attributes grouped in <g>.
    """
    # Find all path elements
    paths = re.findall(r'<path\s+([^>]+?)/>', svg)
    if len(paths) <= 1:
        return svg

    # Parse attributes from each path
    def parse_attrs(attr_str):
        attrs = {}
        for m in re.finditer(r'(\w[\w-]*)="([^"]*)"', attr_str):
            attrs[m.group(1)] = m.group(2)
        return attrs

    all_attrs = [parse_attrs(p) for p in paths]

    # Find attributes shared by ALL paths (excluding 'd')
    shared = {}
    if all_attrs:
        first = all_attrs[0]
        for key, val in first.items():
            if key == 'd':
                continue
            if all(a.get(key) == val for a in all_attrs):
                shared[key] = val

    if not shared:
        return svg

    # Build shared attribute string
    shared_str = ' '.join(f'{k}="{v}"' for k, v in shared.items())

    # Remove shared attributes from each path
    result = svg
    for key, val in shared.items():
        # Remove the attribute from path elements
        result = re.sub(rf'\s*{re.escape(key)}="{re.escape(val)}"', '', result)

    # Wrap paths in <g> with shared attributes
    # Find where paths start
    result = re.sub(r'(<svg[^>]*>)', rf'\1<g {shared_str}>', result)
    result = re.sub(r'</svg>', '</g></svg>', result)

    return result


# ---------------------------------------------------------------------------
# Tool 7: Remove No-Op Styles
# ---------------------------------------------------------------------------

def remove_noop_styles(svg: str) -> str:
    """Remove no-op style attributes from SVG elements.

    Common no-ops:
    - rotate(360deg) transforms (360° = identity)
    - Vendor-prefixed transforms that are no-ops
    - fill-opacity:1 (default)
    - stroke:none (default)

    Args:
        svg: SVG string.

    Returns:
        SVG string with no-op styles removed.
    """
    # Remove style attributes containing only rotate(360deg)
    # These are common in font icon SVGs
    svg = re.sub(r'\s+style="[^"]*rotate\(360deg\)[^"]*"', '', svg)

    return svg


# ---------------------------------------------------------------------------
# Tool 8: Remove Default Attributes
# ---------------------------------------------------------------------------

def remove_default_attributes(svg: str) -> str:
    """Remove attributes that match SVG defaults.

    Args:
        svg: SVG string.

    Returns:
        SVG string with default-valued attributes removed.
    """
    # fill-opacity="1" is the default
    svg = re.sub(r'\s+fill-opacity="1"', '', svg)
    # stroke-opacity="1" is the default
    svg = re.sub(r'\s+stroke-opacity="1"', '', svg)
    # stroke="none" is the default
    svg = re.sub(r'\s+stroke="none"', '', svg)
    # opacity="1" is the default
    svg = re.sub(r'\s+opacity="1"', '', svg)

    return svg


# ---------------------------------------------------------------------------
# Tool 9: Collapse Whitespace
# ---------------------------------------------------------------------------

def collapse_whitespace(svg: str) -> str:
    """Remove unnecessary whitespace from SVG markup.

    - Strips whitespace between XML tags
    - Collapses multiple spaces to single space
    - Removes trailing/leading whitespace

    Args:
        svg: SVG string.

    Returns:
        Compacted SVG string.
    """
    # Remove whitespace between tags
    svg = re.sub(r'>\s+<', '><', svg)
    # Collapse multiple spaces
    svg = re.sub(r'  +', ' ', svg)
    return svg.strip()


# ---------------------------------------------------------------------------
# Tool 10: Shorten Hex Colors
# ---------------------------------------------------------------------------

def shorten_hex_colors(svg: str) -> str:
    """Shorten 6-digit hex colors to 3-digit where possible.

    Only shortens when the color can be exactly represented in 3 digits.
    e.g., #aabbcc -> #abc, but #abcdef stays as-is.

    Note: Haiku was observed incorrectly shortening colors (e.g., #3465a4 -> #36a
    which is actually #336688, a different color). This function avoids that mistake.

    Args:
        svg: SVG string.

    Returns:
        SVG string with shortened hex colors where safe.
    """
    def shorten(m):
        h = m.group(1)
        if len(h) == 6 and h[0] == h[1] and h[2] == h[3] and h[4] == h[5]:
            return f'#{h[0]}{h[2]}{h[4]}'
        return m.group(0)

    return re.sub(r'#([0-9a-fA-F]{6})\b', shorten, svg)


# ---------------------------------------------------------------------------
# Tool 11: Scour (Python-native SVG optimizer, similar to SVGO)
# ---------------------------------------------------------------------------

def optimize_svg(svg: str, aggressive: bool = False) -> str:
    """Run Scour SVG optimizer — a Python-native tool similar to SVGO.

    Scour applies safe, lossless SVG optimizations:
    - Strip XML prolog and metadata
    - Remove comments and descriptive elements (<desc>, <title>)
    - Strip unused IDs and shorten remaining ones
    - Remove empty/default attributes
    - Collapse groups
    - Convert styles to attributes where possible
    - Remove unused namespace declarations

    This is best used as a FIRST PASS before applying coordinate rounding
    and path merging. On raw SVGs it can achieve 30-50% compression losslessly.
    On already-optimized SVGs (e.g., SVGO output) it may have minimal effect.

    Args:
        svg: SVG string to optimize.
        aggressive: If True, enable additional optimizations that very rarely
            cause visual changes (e.g., convert colors to shorter forms,
            simplify transforms).

    Returns:
        Optimized SVG string.
    """
    try:
        from scour.scour import scourString, parse_args
    except ImportError:
        return svg  # scour not installed, return unchanged

    args = [
        '--enable-id-stripping',
        '--enable-comment-stripping',
        '--shorten-ids',
        '--remove-metadata',
        '--strip-xml-prolog',
        '--remove-descriptive-elements',
        '--strip-xml-space',
        '--no-line-breaks',
        '--indent=none',
    ]
    if aggressive:
        args.extend([
            '--enable-viewboxing',
            '--create-groups',
        ])

    opts = parse_args(args)
    try:
        return scourString(svg, opts)
    except Exception:
        return svg  # on any error, return unchanged


# ---------------------------------------------------------------------------
# Main Compression Pipeline
# ---------------------------------------------------------------------------

def compress_svg(
    svg: str,
    decimals: int = 2,
    merge: bool = True,
    group_attrs: bool = True,
    remove_noop: bool = True,
    remove_defaults: bool = True,
    shorten_colors: bool = True,
) -> str:
    """Apply all compression techniques to an SVG string.

    This chains together all the individual tools into a complete pipeline.
    The order matters — some techniques are mutually exclusive (merge vs group).

    Args:
        svg: Input SVG string.
        decimals: Decimal places for coordinate rounding (default 2).
        merge: Whether to merge multiple paths into one (default True).
        group_attrs: Whether to group shared attributes in <g> (default True).
            Only used if merge=False or merge fails.
        remove_noop: Whether to remove no-op styles (default True).
        remove_defaults: Whether to remove default-valued attributes (default True).
        shorten_colors: Whether to shorten hex colors (default True).

    Returns:
        Compressed SVG string.
    """
    if remove_noop:
        svg = remove_noop_styles(svg)

    if remove_defaults:
        svg = remove_default_attributes(svg)

    if shorten_colors:
        svg = shorten_hex_colors(svg)

    # Round path coordinates
    def round_paths(svg_str):
        def replacer(m):
            d = m.group(1)
            return f'd="{round_path_coordinates(d, decimals)}"'
        return re.sub(r'd="([^"]+)"', replacer, svg_str)

    svg = round_paths(svg)

    # Structural optimizations: merge or group
    path_count = len(re.findall(r'<path\b', svg))
    if path_count > 1:
        if merge:
            svg = merge_paths(svg)
        elif group_attrs:
            svg = group_shared_attributes(svg)

    svg = collapse_whitespace(svg)

    return svg


# ---------------------------------------------------------------------------
# Tool 11: Visual Similarity Evaluation
# ---------------------------------------------------------------------------

def evaluate_compression(
    original_svg: str,
    compressed_svg: str,
    render_size: int = 256,
) -> dict:
    """Render both SVGs and compute visual similarity + compression metrics.

    This is the feedback tool — it lets the model see how well its compression
    preserved visual appearance. Returns a flat dictionary of metrics suitable
    for printing or further processing.

    Both SVGs are rendered to `render_size x render_size` PNG on white background,
    then compared pixel-by-pixel.

    Args:
        original_svg: The original (reference) SVG string.
        compressed_svg: The compressed SVG string to evaluate.
        render_size: Resolution for rendering comparison (default 256).

    Returns:
        Dictionary with keys:
        - 'original_bytes': int — size of original SVG in bytes
        - 'compressed_bytes': int — size of compressed SVG in bytes
        - 'compression_ratio': float — fraction of bytes saved (0-1, higher = smaller)
        - 'bytes_saved': int — absolute bytes saved
        - 'ssim': float — Structural Similarity Index (0-1, higher = more similar)
        - 'psnr': float — Peak Signal-to-Noise Ratio in dB (higher = better, inf = identical)
        - 'mse': float — Mean Squared Error (0 = identical)
        - 'mae': float — Mean Absolute Error (0 = identical)
        - 'l2_distance': float — Euclidean distance between images
        - 'pixel_match_ratio': float — fraction of exactly matching pixels (0-1)
        - 'max_pixel_error': float — worst single pixel difference (0-255)
        - 'render_ok': bool — whether both SVGs rendered successfully
        - 'error': str or None — error message if rendering failed

    Example:
        >>> result = evaluate_compression(original_svg, compressed_svg)
        >>> print(f"SSIM: {result['ssim']:.4f}, saved {result['compression_ratio']:.1%}")
        SSIM: 0.9847, saved 23.5%
    """
    import io

    # Compression stats (always available, no rendering needed)
    orig_bytes = len(original_svg.encode('utf-8'))
    comp_bytes = len(compressed_svg.encode('utf-8'))
    result = {
        'original_bytes': orig_bytes,
        'compressed_bytes': comp_bytes,
        'compression_ratio': 1 - comp_bytes / orig_bytes if orig_bytes > 0 else 0.0,
        'bytes_saved': orig_bytes - comp_bytes,
        'ssim': None,
        'psnr': None,
        'mse': None,
        'mae': None,
        'l2_distance': None,
        'pixel_match_ratio': None,
        'max_pixel_error': None,
        'render_ok': False,
        'error': None,
    }

    # Render both SVGs
    try:
        import cairosvg
        import numpy as np
        from PIL import Image

        def _render(svg_text):
            png_data = cairosvg.svg2png(
                bytestring=svg_text.encode('utf-8'),
                output_width=render_size,
                output_height=render_size,
            )
            img = Image.open(io.BytesIO(png_data)).convert('RGBA')
            bg = Image.new('RGBA', img.size, (255, 255, 255, 255))
            return np.array(Image.alpha_composite(bg, img).convert('RGB'))

        img_a = _render(original_svg)
        img_b = _render(compressed_svg)
    except Exception as e:
        result['error'] = f'Render failed: {e}'
        return result

    # Compute metrics
    try:
        from skimage.metrics import structural_similarity as ssim_fn

        a = img_a.astype(np.float64)
        b = img_b.astype(np.float64)
        diff = a - b

        mse = float(np.mean(diff ** 2))
        result['mse'] = mse
        result['psnr'] = float('inf') if mse == 0 else float(10 * np.log10(255.0 ** 2 / mse))
        result['ssim'] = float(ssim_fn(img_a, img_b, channel_axis=2, data_range=255))
        result['mae'] = float(np.mean(np.abs(diff)))
        result['l2_distance'] = float(np.sqrt(np.sum(diff ** 2)))
        result['pixel_match_ratio'] = float(np.all(img_a == img_b, axis=2).mean())
        result['max_pixel_error'] = float(np.max(np.abs(diff)))
        result['render_ok'] = True
    except Exception as e:
        result['error'] = f'Metrics failed: {e}'

    return result


def format_evaluation(metrics: dict) -> str:
    """Format evaluation metrics as a human-readable string.

    Args:
        metrics: Dictionary returned by evaluate_compression().

    Returns:
        Multi-line string summarizing the evaluation.
    """
    lines = []
    lines.append(f"Compression: {metrics['original_bytes']} -> {metrics['compressed_bytes']} bytes "
                 f"({metrics['compression_ratio']:.1%} reduction, {metrics['bytes_saved']} bytes saved)")

    if metrics['render_ok']:
        lines.append(f"SSIM:            {metrics['ssim']:.4f}  (1.0 = identical)")
        lines.append(f"PSNR:            {metrics['psnr']:.1f} dB  (>40 excellent, >30 good)")
        lines.append(f"MSE:             {metrics['mse']:.2f}  (0 = identical)")
        lines.append(f"MAE:             {metrics['mae']:.2f}  (0 = identical)")
        lines.append(f"L2 distance:     {metrics['l2_distance']:.1f}")
        lines.append(f"Pixel match:     {metrics['pixel_match_ratio']:.2%}")
        lines.append(f"Max pixel error: {metrics['max_pixel_error']:.0f} / 255")
    elif metrics['error']:
        lines.append(f"ERROR: {metrics['error']}")
    else:
        lines.append("Rendering not available")

    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# CLI interface
# ---------------------------------------------------------------------------

def main():
    """CLI: compress_svg < input.svg > output.svg"""
    import sys
    import argparse

    parser = argparse.ArgumentParser(description='Compress SVG using extracted Sonnet techniques')
    parser.add_argument('input', nargs='?', help='Input SVG file (default: stdin)')
    parser.add_argument('-o', '--output', help='Output SVG file (default: stdout)')
    parser.add_argument('-d', '--decimals', type=int, default=2, help='Decimal places for rounding (default: 2)')
    parser.add_argument('--no-merge', action='store_true', help='Disable path merging')
    parser.add_argument('--no-group', action='store_true', help='Disable attribute grouping')
    parser.add_argument('--stats', action='store_true', help='Print compression stats to stderr')

    args = parser.parse_args()

    if args.input:
        with open(args.input) as f:
            svg = f.read()
    else:
        svg = sys.stdin.read()

    compressed = compress_svg(
        svg,
        decimals=args.decimals,
        merge=not args.no_merge,
        group_attrs=not args.no_group,
    )

    if args.output:
        with open(args.output, 'w') as f:
            f.write(compressed)
    else:
        sys.stdout.write(compressed)

    if args.stats:
        orig = len(svg.encode('utf-8'))
        comp = len(compressed.encode('utf-8'))
        pct = (1 - comp / orig) * 100 if orig > 0 else 0
        print(f'Original: {orig} bytes', file=sys.stderr)
        print(f'Compressed: {comp} bytes', file=sys.stderr)
        print(f'Reduction: {pct:.1f}%', file=sys.stderr)


if __name__ == '__main__':
    main()
