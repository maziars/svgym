"""SVG compression tools extracted from Opus agent transcripts.

This module provides reusable Python functions for compressing SVG files
beyond what SVGO achieves. Functions are organized by category and extracted
from three Opus agent compression sessions:
  - fonts/000125: font icon compression (path-heavy single-path SVGs)
  - simple/001155: line drawing compression (multi-path SVGs with arcs)
  - full/001521: chart compression (complex structure with text, groups, styles)

Each function takes SVG text (string) as input and returns compressed SVG text.
Functions are independently usable with no cross-dependencies.

Usage:
    from svgym.tools import compress_svg
    compressed = compress_svg(svg_text)

    # Or use individual tools:
    from svgym.tools import (
        parse_path, format_number, format_path,
        round_path_coordinates, abs_to_rel,
        cubic_to_line, cubic_to_quad, try_smooth_curves,
        merge_collinear_lines, merge_same_commands,
        merge_paths, consolidate_attrs_to_parent,
        remove_default_attributes, remove_hidden_elements,
        strip_whitespace, unwrap_single_tspans,
        render_svg, compare,
    )

Dependencies:
    Standard library: re, math, io
    External (for render/compare only): cairosvg, numpy, PIL, scipy
"""

import re
import math
import xml.etree.ElementTree as ET

# ---------------------------------------------------------------------------
# Path data tools: parsing, number formatting, coordinate rounding
# ---------------------------------------------------------------------------

# Argument counts for SVG path commands
_ARG_COUNTS = {
    'M': 2, 'm': 2, 'L': 2, 'l': 2, 'H': 1, 'h': 1, 'V': 1, 'v': 1,
    'C': 6, 'c': 6, 'S': 4, 's': 4, 'Q': 4, 'q': 4, 'T': 2, 't': 2,
    'A': 7, 'a': 7, 'Z': 0, 'z': 0,
}


def parse_path(d):
    """Parse an SVG path data string into a list of (command, args) tuples.

    Handles implicit command repetition (e.g. M followed by coordinate pairs
    becomes M then implicit L commands), exponent notation, arc flag parsing
    (flags can be concatenated without separators), and all standard SVG path
    commands.

    Args:
        d: SVG path data string (the value of a 'd' attribute).

    Returns:
        List of (cmd, args) where cmd is a single letter and args is a list
        of floats.

    Typical savings: N/A (utility function used by other tools).
    """
    tokens = re.findall(
        r'[A-Za-z]|[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?', d
    )
    # Make mutable list so we can split tokens for arc flag handling
    tokens = list(tokens)
    commands = []
    i = 0
    current_cmd = None
    while i < len(tokens):
        if re.match(r'^[A-Za-z]$', tokens[i]):
            current_cmd = tokens[i]
            i += 1
        if current_cmd is None:
            break
        n_args = _ARG_COUNTS.get(current_cmd, 0)
        if n_args == 0:
            commands.append((current_cmd, []))
            current_cmd = None
            continue

        is_arc = current_cmd in ('a', 'A')

        if is_arc:
            # Arc has 7 args: rx ry rotation large-arc-flag sweep-flag x y
            # Flags (args 3,4) are single digits (0/1) that may be concatenated
            # with each other or with the following number without separators.
            args = []
            # Read 3 normal args (rx, ry, x-rotation)
            for _ in range(3):
                if i < len(tokens) and not re.match(r'^[A-Za-z]$', tokens[i]):
                    args.append(float(tokens[i]))
                    i += 1
            # Read 2 flags - each is a single digit, may be embedded in token
            for _ in range(2):
                if i >= len(tokens) or re.match(r'^[A-Za-z]$', tokens[i]):
                    break
                tok = tokens[i]
                if tok and tok[0] in '01':
                    args.append(float(tok[0]))
                    rest = tok[1:]
                    if rest:
                        # Put remaining chars back as a new token
                        # Handle: "0150" → flag=0, insert "150"
                        # Handle: "01" → flag=0, insert "1"
                        # Handle: "0.5" → flag=0, insert ".5"
                        tokens[i] = rest
                    else:
                        i += 1
                else:
                    args.append(float(tok))
                    i += 1
            # Read 2 normal args (x, y)
            for _ in range(2):
                if i < len(tokens) and not re.match(r'^[A-Za-z]$', tokens[i]):
                    args.append(float(tokens[i]))
                    i += 1
            commands.append((current_cmd, args))
        else:
            args = []
            for _ in range(n_args):
                if i < len(tokens) and not re.match(r'^[A-Za-z]$', tokens[i]):
                    args.append(float(tokens[i]))
                    i += 1
            commands.append((current_cmd, args))

        # Implicit command repetition
        if current_cmd == 'M':
            current_cmd = 'L'
        elif current_cmd == 'm':
            current_cmd = 'l'
        # If next token is a letter, reset
        if i < len(tokens) and re.match(r'^[A-Za-z]$', tokens[i]):
            current_cmd = None
    return commands


def format_number(n, precision=2):
    """Format a number compactly for SVG path data.

    Removes trailing zeros, unnecessary decimal points, and leading zeros
    on fractional values (e.g. 0.5 -> .5, -0.3 -> -.3).

    Args:
        n: Number to format.
        precision: Decimal places to round to (default 2).

    Returns:
        Compact string representation.

    Typical savings: 1-3 bytes per number (adds up across hundreds of coords).
    """
    r = round(n, precision)
    if abs(r - round(r)) < 1e-9 and abs(r) < 100000:
        return str(int(round(r)))
    s = f"{r:.{precision}f}".rstrip('0').rstrip('.')
    if s.startswith('0.'):
        s = s[1:]
    elif s.startswith('-0.'):
        s = '-' + s[2:]
    return s


def _needs_separator(prev_str, next_str):
    """Determine if a separator is needed between two formatted numbers.

    In SVG path data, separators can be omitted when:
    - The next number starts with '-' (acts as separator)
    - The next number starts with '.' and the previous number already
      contains a '.' (second dot starts a new number)
    - The previous character is a letter (command)
    """
    if not prev_str:
        return False
    last_char = prev_str[-1]
    if last_char.isalpha():
        return False
    if next_str.startswith('-'):
        return False
    if next_str.startswith('.'):
        # Safe if prev already has a dot (second dot starts new number)
        if '.' in prev_str and not last_char.isalpha():
            return False
        return True
    return True


def format_path(commands, precision=2):
    """Format parsed path commands back into a compact path data string.

    Applies command merging (omitting repeated command letters), minimal
    separators, and compact number formatting. Handles arc flag parameters
    specially (they are always 0 or 1 and can be concatenated without
    separators).

    Args:
        commands: List of (cmd, args) tuples from parse_path().
        precision: Decimal precision for coordinates.

    Returns:
        Compact path data string.

    Typical savings: 10-30% of path data size from formatting alone.
    """
    out = []
    for idx, (cmd, args) in enumerate(commands):
        # Determine if we can merge (omit command letter)
        merge = False
        if idx > 0:
            prev_cmd = commands[idx - 1][0]
            if cmd == prev_cmd and cmd not in ('M', 'm', 'Z', 'z'):
                merge = True
            elif (prev_cmd == 'M' and cmd == 'L') or (prev_cmd == 'm' and cmd == 'l'):
                merge = True

        if not merge:
            out.append(cmd)

        is_arc = cmd in ('a', 'A')

        for j, a in enumerate(args):
            # Arc flags (positions 3, 4 in 7-arg arc) are always 0 or 1
            if is_arc and j in (3, 4):
                flag = str(int(a))
                if j == 3:
                    # After rotation number - may need separator
                    prev_s = out[-1] if out else ''
                    lc = prev_s[-1] if prev_s else ''
                    if lc.isdigit():
                        out.append(' ' + flag)
                    else:
                        out.append(flag)
                else:
                    # Sweep flag after large-arc flag - always unambiguous
                    out.append(flag)
                continue

            f = format_number(a, precision)

            if j == 0 and not merge:
                # First arg right after command letter
                out.append(f)
            elif is_arc and j == 5:
                # dx/dy after sweep flag
                prev_s = out[-1] if out else ''
                if f.startswith('-') or f.startswith('.'):
                    out.append(f)
                else:
                    out.append(' ' + f)
            else:
                prev_s = out[-1] if out else ''
                if _needs_separator(prev_s, f):
                    out.append(' ' + f)
                else:
                    out.append(f)

    return ''.join(out)


def round_path_coordinates(svg_text, precision=2):
    """Round all coordinates in path data to the given decimal precision.

    Parses each path 'd' attribute, rounds all coordinates, and reformats
    with compact number formatting.

    Args:
        svg_text: SVG string.
        precision: Decimal places (0 for integers, 1, 2, etc.).

    Returns:
        SVG string with rounded path coordinates.

    Typical savings: 5-25% depending on original precision and target.
    """
    def process(m):
        d = m.group(1)
        cmds = parse_path(d)
        rounded = []
        for cmd, args in cmds:
            if cmd in ('a', 'A') and len(args) == 7:
                new_args = [
                    round(args[0], precision), round(args[1], precision),
                    round(args[2], precision),
                    round(args[3]), round(args[4]),  # flags stay as-is
                    round(args[5], precision), round(args[6], precision),
                ]
                rounded.append((cmd, new_args))
            else:
                rounded.append((cmd, [round(a, precision) for a in args]))
        return 'd="' + format_path(rounded, precision) + '"'
    return re.sub(r'(?<![a-zA-Z-])d="([^"]*)"', process, svg_text)


def to_absolute(commands):
    """Convert parsed path commands to absolute coordinates.

    Resolves all relative commands (lowercase) to absolute (uppercase) while
    tracking the current point. This is a prerequisite for many transformations
    that need to reason about absolute positions.

    Args:
        commands: List of (cmd, args) from parse_path().

    Returns:
        List of (cmd, args) with all commands in absolute form.
    """
    cx, cy = 0.0, 0.0
    sx, sy = 0.0, 0.0  # subpath start
    result = []
    for cmd, args in commands:
        if cmd == 'M':
            result.append(('M', [args[0], args[1]]))
            cx, cy = args[0], args[1]
            sx, sy = cx, cy
            for j in range(2, len(args), 2):
                result.append(('L', [args[j], args[j + 1]]))
                cx, cy = args[j], args[j + 1]
        elif cmd == 'm':
            ax, ay = cx + args[0], cy + args[1]
            result.append(('M', [ax, ay]))
            cx, cy = ax, ay
            sx, sy = cx, cy
            for j in range(2, len(args), 2):
                cx += args[j]
                cy += args[j + 1]
                result.append(('L', [cx, cy]))
        elif cmd == 'l':
            for j in range(0, len(args), 2):
                cx += args[j]
                cy += args[j + 1]
                result.append(('L', [cx, cy]))
        elif cmd == 'L':
            for j in range(0, len(args), 2):
                cx, cy = args[j], args[j + 1]
                result.append(('L', [cx, cy]))
        elif cmd == 'h':
            for v in args:
                cx += v
                result.append(('L', [cx, cy]))
        elif cmd == 'H':
            for v in args:
                cx = v
                result.append(('L', [cx, cy]))
        elif cmd == 'v':
            for v in args:
                cy += v
                result.append(('L', [cx, cy]))
        elif cmd == 'V':
            for v in args:
                cy = v
                result.append(('L', [cx, cy]))
        elif cmd == 'c':
            for j in range(0, len(args), 6):
                a = [cx + args[j], cy + args[j + 1],
                     cx + args[j + 2], cy + args[j + 3],
                     cx + args[j + 4], cy + args[j + 5]]
                result.append(('C', a))
                cx, cy = a[4], a[5]
        elif cmd == 'C':
            for j in range(0, len(args), 6):
                result.append(('C', list(args[j:j + 6])))
                cx, cy = args[j + 4], args[j + 5]
        elif cmd == 's':
            for j in range(0, len(args), 4):
                a = [cx + args[j], cy + args[j + 1],
                     cx + args[j + 2], cy + args[j + 3]]
                result.append(('S', a))
                cx, cy = a[2], a[3]
        elif cmd == 'S':
            for j in range(0, len(args), 4):
                result.append(('S', list(args[j:j + 4])))
                cx, cy = args[j + 2], args[j + 3]
        elif cmd == 'q':
            for j in range(0, len(args), 4):
                a = [cx + args[j], cy + args[j + 1],
                     cx + args[j + 2], cy + args[j + 3]]
                result.append(('Q', a))
                cx, cy = a[2], a[3]
        elif cmd == 'Q':
            for j in range(0, len(args), 4):
                result.append(('Q', list(args[j:j + 4])))
                cx, cy = args[j + 2], args[j + 3]
        elif cmd == 'a':
            for j in range(0, max(len(args) - 6, 1), 7):
                if j + 6 < len(args):
                    ax, ay = cx + args[j + 5], cy + args[j + 6]
                    result.append(('A', [args[j], args[j + 1], args[j + 2],
                                         args[j + 3], args[j + 4], ax, ay]))
                    cx, cy = ax, ay
        elif cmd == 'A':
            for j in range(0, max(len(args) - 6, 1), 7):
                if j + 6 < len(args):
                    result.append(('A', list(args[j:j + 7])))
                    cx, cy = args[j + 5], args[j + 6]
        elif cmd in ('Z', 'z'):
            result.append(('Z', []))
            cx, cy = sx, sy
    return result


def abs_to_rel(svg_text, precision=2):
    """Convert absolute path commands to relative where shorter.

    For each path, converts to absolute first, then decides per-command
    whether absolute or relative form is shorter. Also converts L to h/v
    when one coordinate delta is zero.

    Args:
        svg_text: SVG string.
        precision: Decimal precision for formatting.

    Returns:
        SVG string with optimized abs/rel path commands.

    Typical savings: 5-20% on paths with large absolute coordinates.
    """
    def process(m):
        d = m.group(1)
        cmds = parse_path(d)
        abs_cmds = to_absolute(cmds)
        abs_cmds = [(cmd, [round(a, precision) for a in args]
                     if cmd not in ('A', 'Z') else
                     ([round(args[0], precision), round(args[1], precision),
                       round(args[2], precision), round(args[3]),
                       round(args[4]), round(args[5], precision),
                       round(args[6], precision)] if cmd == 'A' else args))
                    for cmd, args in abs_cmds]

        cx, cy = 0.0, 0.0
        sx, sy = 0.0, 0.0
        result = []
        fmt = lambda n: format_number(n, precision)

        for cmd, args in abs_cmds:
            if cmd == 'M':
                dx, dy = args[0] - cx, args[1] - cy
                abs_s = 'M' + fmt(args[0]) + ',' + fmt(args[1])
                rel_s = 'm' + fmt(dx) + ',' + fmt(dy)
                if not result or len(abs_s) <= len(rel_s):
                    result.append(('M', [args[0], args[1]]))
                else:
                    result.append(('m', [dx, dy]))
                cx, cy = args[0], args[1]
                sx, sy = cx, cy
            elif cmd == 'L':
                dx, dy = args[0] - cx, args[1] - cy
                if abs(dy) < 1e-10 and abs(dx) > 1e-10:
                    result.append(('h', [dx]))
                elif abs(dx) < 1e-10 and abs(dy) > 1e-10:
                    result.append(('v', [dy]))
                else:
                    rel_s = fmt(dx) + ',' + fmt(dy)
                    abs_s = fmt(args[0]) + ',' + fmt(args[1])
                    if len(rel_s) <= len(abs_s):
                        result.append(('l', [dx, dy]))
                    else:
                        result.append(('L', [args[0], args[1]]))
                cx, cy = args[0], args[1]
            elif cmd == 'C':
                rel = [args[0] - cx, args[1] - cy, args[2] - cx, args[3] - cy,
                       args[4] - cx, args[5] - cy]
                abs_len = sum(len(fmt(a)) for a in args) + 5
                rel_len = sum(len(fmt(a)) for a in rel) + 5
                if rel_len <= abs_len:
                    result.append(('c', rel))
                else:
                    result.append(('C', list(args)))
                cx, cy = args[4], args[5]
            elif cmd == 'S':
                rel = [args[0] - cx, args[1] - cy, args[2] - cx, args[3] - cy]
                abs_len = sum(len(fmt(a)) for a in args) + 3
                rel_len = sum(len(fmt(a)) for a in rel) + 3
                if rel_len <= abs_len:
                    result.append(('s', rel))
                else:
                    result.append(('S', list(args)))
                cx, cy = args[2], args[3]
            elif cmd == 'Q':
                rel = [args[0] - cx, args[1] - cy, args[2] - cx, args[3] - cy]
                abs_len = sum(len(fmt(a)) for a in args) + 3
                rel_len = sum(len(fmt(a)) for a in rel) + 3
                if rel_len <= abs_len:
                    result.append(('q', rel))
                else:
                    result.append(('Q', list(args)))
                cx, cy = args[2], args[3]
            elif cmd == 'A':
                rel_xy = [args[5] - cx, args[6] - cy]
                result.append(('a', [args[0], args[1], args[2],
                                     args[3], args[4]] + rel_xy))
                cx, cy = args[5], args[6]
            elif cmd == 'Z':
                result.append(('z', []))
                cx, cy = sx, sy
            else:
                result.append((cmd, list(args)))

        # Merge consecutive same commands
        merged = [list(result[0])] if result else []
        for cmd, args in result[1:]:
            if cmd == merged[-1][0] and cmd.lower() in ('l', 'c', 's', 'h', 'v', 'q', 't', 'a'):
                merged[-1][1].extend(args)
            else:
                merged.append([cmd, list(args)])

        return 'd="' + format_path([(c, a) for c, a in merged], precision) + '"'

    return re.sub(r'(?<![a-zA-Z-])d="([^"]*)"', process, svg_text)


# ---------------------------------------------------------------------------
# Curve simplification: cubic-to-line, cubic-to-quadratic, cubic-to-smooth
# ---------------------------------------------------------------------------

def _cubic_line_error(args):
    """Compute max deviation of a relative cubic bezier from a straight line.

    Samples the bezier at 9 points and returns the maximum distance from
    the corresponding point on the straight line from origin to endpoint.

    Args:
        args: [dx1, dy1, dx2, dy2, dx, dy] - relative cubic bezier args.

    Returns:
        Maximum deviation in SVG units.
    """
    dx1, dy1, dx2, dy2, dx, dy = args
    mx = 0
    for t in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        bx = 3 * (1 - t) ** 2 * t * dx1 + 3 * (1 - t) * t ** 2 * dx2 + t ** 3 * dx
        by = 3 * (1 - t) ** 2 * t * dy1 + 3 * (1 - t) * t ** 2 * dy2 + t ** 3 * dy
        mx = max(mx, ((bx - t * dx) ** 2 + (by - t * dy) ** 2) ** 0.5)
    return mx


def _cubic_to_quad_error(args):
    """Compute best quadratic approximation of a relative cubic and its error.

    Finds the optimal single control point for a quadratic bezier that
    approximates the cubic, and returns the max error plus the control point.

    Args:
        args: [dx1, dy1, dx2, dy2, dx, dy] - relative cubic bezier args.

    Returns:
        (max_error, qx1, qy1) where qx1/qy1 are the quadratic control point.
    """
    dx1, dy1, dx2, dy2, dx, dy = args
    qx1 = (1.5 * dx1 + 1.5 * dx2 - 0.5 * dx) / 2
    qy1 = (1.5 * dy1 + 1.5 * dy2 - 0.5 * dy) / 2
    mx = 0
    for t in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        cx = 3 * (1 - t) ** 2 * t * dx1 + 3 * (1 - t) * t ** 2 * dx2 + t ** 3 * dx
        cy = 3 * (1 - t) ** 2 * t * dy1 + 3 * (1 - t) * t ** 2 * dy2 + t ** 3 * dy
        qx = 2 * (1 - t) * t * qx1 + t ** 2 * dx
        qy = 2 * (1 - t) * t * qy1 + t ** 2 * dy
        mx = max(mx, ((cx - qx) ** 2 + (cy - qy) ** 2) ** 0.5)
    return mx, qx1, qy1


def cubic_to_line(svg_text, threshold=0.05, precision=2):
    """Convert nearly-straight cubic beziers to line commands.

    For each relative cubic 'c' command, measures deviation from a straight
    line. If below threshold, replaces with 'l' (or 'h'/'v' when one delta
    is zero).

    Args:
        svg_text: SVG string.
        threshold: Maximum allowed deviation in SVG units (default 0.05).
        precision: Decimal precision for formatting.

    Returns:
        SVG string with simplified curves.

    Typical savings: 2-15% on icon/font SVGs with many near-linear curves.
    """
    def process(m):
        d = m.group(1)
        cmds = parse_path(d)
        result = []
        for cmd, args in cmds:
            if cmd == 'c' and len(args) == 6:
                err = _cubic_line_error(args)
                if err < threshold:
                    dx, dy = args[4], args[5]
                    if abs(dy) < 0.001:
                        result.append(('h', [dx]))
                    elif abs(dx) < 0.001:
                        result.append(('v', [dy]))
                    else:
                        result.append(('l', [dx, dy]))
                    continue
            result.append((cmd, args))
        return 'd="' + format_path(result, precision) + '"'
    return re.sub(r'(?<![a-zA-Z-])d="([^"]*)"', process, svg_text)


def cubic_to_quad(svg_text, threshold=0.03, precision=2):
    """Convert cubic beziers to quadratic where error is small.

    For each relative cubic 'c' command, computes the best quadratic
    approximation. If the error is below threshold, replaces 'c' with 'q'.

    Args:
        svg_text: SVG string.
        threshold: Maximum allowed approximation error (default 0.03).
        precision: Decimal precision for formatting.

    Returns:
        SVG string with quadratic curves replacing qualifying cubics.

    Typical savings: 2-8% (saves 2 args per converted curve).
    """
    def process(m):
        d = m.group(1)
        cmds = parse_path(d)
        result = []
        for cmd, args in cmds:
            if cmd == 'c' and len(args) == 6:
                err, qx1, qy1 = _cubic_to_quad_error(args)
                if err < threshold:
                    result.append(('q', [qx1, qy1, args[4], args[5]]))
                    continue
            result.append((cmd, args))
        return 'd="' + format_path(result, precision) + '"'
    return re.sub(r'(?<![a-zA-Z-])d="([^"]*)"', process, svg_text)


def try_smooth_curves(svg_text, tolerance=0.06, precision=2):
    """Convert cubic beziers to smooth curves (c->s) where control points match.

    When a cubic bezier's first control point is the reflection of the
    previous curve's second control point, the 'c' can be replaced with 's'
    (smooth cubic), saving 2 coordinate values.

    Args:
        svg_text: SVG string.
        tolerance: Max distance between cp1 and reflected cp2 (default 0.06).
        precision: Decimal precision for formatting.

    Returns:
        SVG string with smooth curve substitutions.

    Typical savings: 3-10% on paths with many connected bezier curves.
    """
    def process(m):
        d = m.group(1)
        cmds = parse_path(d)
        result = []
        prev_abs_cp2 = None
        cx, cy = 0.0, 0.0
        prev_cmd = None

        for cmd, args in cmds:
            converted = False

            if (cmd == 'c' and len(args) == 6 and
                    prev_cmd in ('c', 's', 'C', 'S') and
                    prev_abs_cp2 is not None):
                cp1_abs = (cx + args[0], cy + args[1])
                refl = (2 * cx - prev_abs_cp2[0], 2 * cy - prev_abs_cp2[1])
                if (abs(cp1_abs[0] - refl[0]) < tolerance and
                        abs(cp1_abs[1] - refl[1]) < tolerance):
                    result.append(('s', [args[2], args[3], args[4], args[5]]))
                    prev_abs_cp2 = (cx + args[2], cy + args[3])
                    cx += args[4]
                    cy += args[5]
                    prev_cmd = 's'
                    converted = True

            if not converted:
                result.append((cmd, args))
                if cmd == 'c' and len(args) == 6:
                    prev_abs_cp2 = (cx + args[2], cy + args[3])
                    cx += args[4]
                    cy += args[5]
                elif cmd == 'C' and len(args) == 6:
                    prev_abs_cp2 = (args[2], args[3])
                    cx, cy = args[4], args[5]
                elif cmd == 's' and len(args) == 4:
                    prev_abs_cp2 = (cx + args[0], cy + args[1])
                    cx += args[2]
                    cy += args[3]
                elif cmd == 'S' and len(args) == 4:
                    prev_abs_cp2 = (args[0], args[1])
                    cx, cy = args[2], args[3]
                elif cmd == 'l' and len(args) == 2:
                    cx += args[0]
                    cy += args[1]
                    prev_abs_cp2 = None
                elif cmd == 'L' and len(args) == 2:
                    cx, cy = args[0], args[1]
                    prev_abs_cp2 = None
                elif cmd == 'h':
                    for v in args:
                        cx += v
                    prev_abs_cp2 = None
                elif cmd == 'H':
                    cx = args[-1]
                    prev_abs_cp2 = None
                elif cmd == 'v':
                    for v in args:
                        cy += v
                    prev_abs_cp2 = None
                elif cmd == 'V':
                    cy = args[-1]
                    prev_abs_cp2 = None
                elif cmd == 'M' and len(args) >= 2:
                    cx, cy = args[0], args[1]
                    prev_abs_cp2 = None
                elif cmd == 'm' and len(args) >= 2:
                    cx += args[0]
                    cy += args[1]
                    prev_abs_cp2 = None
                elif cmd in ('z', 'Z'):
                    prev_abs_cp2 = None
                elif cmd == 'a' and len(args) >= 7:
                    cx += args[5]
                    cy += args[6]
                    prev_abs_cp2 = None
                elif cmd == 'A' and len(args) >= 7:
                    cx, cy = args[5], args[6]
                    prev_abs_cp2 = None
                else:
                    prev_abs_cp2 = None
                prev_cmd = cmd

        return 'd="' + format_path(result, precision) + '"'
    return re.sub(r'(?<![a-zA-Z-])d="([^"]*)"', process, svg_text)


def curve_to_hv(svg_text, precision=2):
    """Convert line commands to h/v shorthand when one delta is zero.

    Scans for 'l dx 0' -> 'h dx' and 'l 0 dy' -> 'v dy' conversions.
    Also handles absolute L commands.

    Args:
        svg_text: SVG string.
        precision: Decimal precision.

    Returns:
        SVG string with h/v shorthand.

    Typical savings: 1-5% on SVGs with axis-aligned line segments.
    """
    def process(m):
        d = m.group(1)
        cmds = parse_path(d)
        result = []
        for cmd, args in cmds:
            if cmd == 'l' and len(args) == 2:
                if abs(args[1]) < 0.001:
                    result.append(('h', [args[0]]))
                elif abs(args[0]) < 0.001:
                    result.append(('v', [args[1]]))
                else:
                    result.append((cmd, args))
            else:
                result.append((cmd, args))
        return 'd="' + format_path(result, precision) + '"'
    return re.sub(r'(?<![a-zA-Z-])d="([^"]*)"', process, svg_text)


# ---------------------------------------------------------------------------
# Path merging: collinear lines, consecutive commands, multi-path merge
# ---------------------------------------------------------------------------

def merge_collinear_lines(svg_text, threshold=0.01, precision=2):
    """Merge consecutive collinear relative line segments into one.

    When two consecutive 'l' commands point in the same direction (cross
    product near zero, dot product positive), they can be merged.

    Args:
        svg_text: SVG string.
        threshold: Cross product threshold for collinearity (default 0.01).
        precision: Decimal precision.

    Returns:
        SVG string with merged line segments.

    Typical savings: 1-5% on paths with many small collinear segments.
    """
    def process(m):
        d = m.group(1)
        cmds = parse_path(d)
        result = []
        i = 0
        while i < len(cmds):
            if (cmds[i][0] == 'l' and len(cmds[i][1]) == 2 and
                    i + 1 < len(cmds) and
                    cmds[i + 1][0] == 'l' and len(cmds[i + 1][1]) == 2):
                dx1, dy1 = cmds[i][1]
                dx2, dy2 = cmds[i + 1][1]
                cross = dx1 * dy2 - dy1 * dx2
                dot = dx1 * dx2 + dy1 * dy2
                if abs(cross) < threshold and dot > 0:
                    result.append(('l', [dx1 + dx2, dy1 + dy2]))
                    i += 2
                    continue
            result.append(cmds[i])
            i += 1
        return 'd="' + format_path(result, precision) + '"'
    return re.sub(r'(?<![a-zA-Z-])d="([^"]*)"', process, svg_text)


def merge_same_commands(svg_text, precision=2):
    """Merge consecutive same-type path commands into one.

    When multiple consecutive commands have the same letter (e.g. l l l),
    their arguments can be concatenated into a single command.

    Args:
        svg_text: SVG string.
        precision: Decimal precision.

    Returns:
        SVG string with merged commands.

    Typical savings: 1-3% (saves one command letter per merge).
    """
    def process(m):
        d = m.group(1)
        cmds = parse_path(d)
        if not cmds:
            return m.group(0)
        merged = [list(cmds[0])]
        for cmd, args in cmds[1:]:
            if (cmd == merged[-1][0] and
                    cmd.lower() in ('l', 'c', 's', 'h', 'v', 'q', 't', 'a')):
                merged[-1][1].extend(args)
            else:
                merged.append([cmd, list(args)])
        return 'd="' + format_path([(c, a) for c, a in merged], precision) + '"'
    return re.sub(r'(?<![a-zA-Z-])d="([^"]*)"', process, svg_text)


def merge_paths(svg_text):
    """Merge multiple <path> elements with the same attributes into one.

    Finds all <path> elements that share the same non-d attributes and
    combines their path data into a single path using multiple M commands.

    Args:
        svg_text: SVG string.

    Returns:
        SVG string with merged paths.

    Typical savings: 10-30% on SVGs with many simple paths sharing attributes.
    """
    # Find all self-closing path elements
    path_pattern = r'<path\s+([^>]*)d="([^"]*)"([^>]*)/>'
    matches = list(re.finditer(path_pattern, svg_text))
    if len(matches) < 2:
        return svg_text

    # Group by non-d attributes
    groups = {}
    for m in matches:
        before_d = m.group(1).strip()
        after_d = m.group(3).strip()
        attrs_key = (before_d, after_d)
        if attrs_key not in groups:
            groups[attrs_key] = []
        groups[attrs_key].append(m)

    # Build all replacements: map each match to its replacement string
    # (first in group -> merged path, rest -> empty string)
    replacements = {}  # match object -> replacement string
    for attrs_key, group_matches in groups.items():
        if len(group_matches) < 2:
            continue
        before_d, after_d = attrs_key
        # Strip trailing / from after_d (the regex captures it from />)
        after_d_clean = after_d.rstrip('/')
        # Combine all d values
        combined_d = ''.join(gm.group(2) for gm in group_matches)
        # Build replacement element
        parts = ['<path']
        if before_d:
            parts.append(' ' + before_d)
        parts.append(f' d="{combined_d}"')
        if after_d_clean:
            parts.append(' ' + after_d_clean)
        parts.append('/>')
        replacements[group_matches[0]] = ''.join(parts)
        for gm in group_matches[1:]:
            replacements[gm] = ''

    if not replacements:
        return svg_text

    # Apply all replacements in reverse document order to preserve positions
    sorted_matches = sorted(replacements.keys(), key=lambda m: m.start(), reverse=True)
    result = svg_text
    for m in sorted_matches:
        result = result[:m.start()] + replacements[m] + result[m.end():]

    return result


# ---------------------------------------------------------------------------
# Attribute tools: remove defaults, consolidate to parent, style conversion
# ---------------------------------------------------------------------------

def remove_default_attributes(svg_text):
    """Remove attributes that match their SVG default values.

    Removes known defaults like fill="black" (default for most elements),
    dy="0em", stroke-width="1", opacity="1", etc.

    Args:
        svg_text: SVG string.

    Returns:
        SVG string with default attributes removed.

    Typical savings: 1-5% depending on how many defaults are present.
    """
    defaults = [
        (' dy="0em"', ''),
        (' dy="0"', ''),
        (' opacity="1"', ''),
        (' fill-opacity="1"', ''),
        (' stroke-opacity="1"', ''),
        (' fill-rule="nonzero"', ''),
    ]
    for old, new in defaults:
        svg_text = svg_text.replace(old, new)
    return svg_text


def consolidate_attrs_to_parent(svg_text):
    """Move repeated attributes from child elements to their parent group.

    When ALL direct children of a <g> share the same attribute value, that
    attribute is moved to the <g> and removed from children. Only consolidates
    within groups, never to the <svg> root (which would affect elements that
    don't have the attribute set).

    Args:
        svg_text: SVG string.

    Returns:
        SVG string with consolidated attributes.

    Typical savings: 5-15% on SVGs with many elements sharing styles.
    """
    import xml.etree.ElementTree as ET

    # SVG namespace handling
    ns_match = re.match(r'<svg[^>]*\bxmlns="([^"]*)"', svg_text)
    ns = ns_match.group(1) if ns_match else 'http://www.w3.org/2000/svg'
    ET.register_namespace('', ns)
    # Preserve other namespace declarations
    for ns_prefix, ns_uri in re.findall(r'xmlns:(\w+)="([^"]*)"', svg_text):
        ET.register_namespace(ns_prefix, ns_uri)

    try:
        root = ET.fromstring(svg_text)
    except ET.ParseError:
        return svg_text

    consolidate_attrs = ['fill', 'stroke', 'stroke-width', 'font-family', 'font-size']
    changed = False

    def process_group(elem):
        nonlocal changed
        children = list(elem)
        if len(children) < 2:
            for child in children:
                process_group(child)
            return

        for attr in consolidate_attrs:
            # Check if ALL children have this attribute with the same value
            values = []
            all_have = True
            for child in children:
                val = child.get(attr)
                if val is None:
                    all_have = False
                    break
                values.append(val)
            if not all_have or not values:
                continue
            if len(set(values)) != 1:
                continue
            # All children have the same value — move to parent
            common_val = values[0]
            if elem.get(attr) is not None:
                continue  # Parent already has this attr
            elem.set(attr, common_val)
            for child in children:
                del child.attrib[attr]
            changed = True

        for child in children:
            process_group(child)

    process_group(root)

    if not changed:
        return svg_text

    result = ET.tostring(root, encoding='unicode')
    # ET may reorder attributes or change formatting; preserve original
    # XML declaration if present
    if svg_text.startswith('<?xml'):
        decl_end = svg_text.index('?>') + 2
        decl = svg_text[:decl_end]
        if not result.startswith('<?xml'):
            result = decl + result
    return result


def style_to_attributes(svg_text):
    """Convert inline style attributes to presentation attributes.

    Replaces style="prop:value;..." with individual XML attributes where
    the SVG presentation attribute exists. This can enable further
    optimizations and is sometimes shorter.

    Args:
        svg_text: SVG string.

    Returns:
        SVG string with style properties converted to attributes.

    Typical savings: 0-5% (mainly enables further optimizations).
    """
    def convert_style(m):
        full = m.group(0)
        style_m = re.search(r'style="([^"]*)"', full)
        if not style_m:
            return full
        style = style_m.group(1)
        # Known SVG presentation properties
        svg_props = {
            'fill', 'stroke', 'stroke-width', 'stroke-linecap',
            'stroke-linejoin', 'stroke-dasharray', 'stroke-dashoffset',
            'opacity', 'fill-opacity', 'stroke-opacity', 'font-family',
            'font-size', 'font-weight', 'font-style', 'text-anchor',
            'text-decoration', 'letter-spacing', 'word-spacing',
            'dominant-baseline', 'fill-rule', 'clip-rule',
        }
        converted = []
        remaining = []
        for prop_val in style.split(';'):
            prop_val = prop_val.strip()
            if not prop_val:
                continue
            parts = prop_val.split(':', 1)
            if len(parts) == 2:
                prop, val = parts[0].strip(), parts[1].strip()
                if prop in svg_props:
                    converted.append(f'{prop}="{val}"')
                else:
                    remaining.append(prop_val)
            else:
                remaining.append(prop_val)

        if not converted:
            return full

        # Remove old style, add new attributes
        result = full.replace(style_m.group(0), '')
        # Clean up double spaces
        result = re.sub(r'\s+', ' ', result)
        # Insert attributes before closing
        attr_str = ' '.join(converted)
        if remaining:
            attr_str += f' style="{";".join(remaining)}"'
        # Insert before /> or >
        result = re.sub(r'\s*(/?>)', f' {attr_str}\\1', result)
        return result

    return re.sub(r'<[^>]+style="[^"]*"[^>]*/?>', convert_style, svg_text)


def remove_hidden_elements(svg_text):
    """Remove elements with display:none, visibility:hidden, or hidden classes.

    Scans for common patterns indicating hidden elements and removes them
    entirely, including nested content.

    Args:
        svg_text: SVG string.

    Returns:
        SVG string with hidden elements removed.

    Typical savings: 0-30% depending on hidden content (e.g. tooltips in charts).
    """
    # Remove elements with display:none or visibility:hidden
    svg_text = re.sub(
        r'<[^>]+(?:display\s*:\s*none|visibility\s*:\s*hidden)[^>]*>.*?</\w+>',
        '', svg_text, flags=re.DOTALL
    )
    # Remove elements with common "hide" class patterns
    svg_text = re.sub(
        r'<g[^>]*class="[^"]*\bhide\b[^"]*"[^>]*>.*?</g>',
        '', svg_text, flags=re.DOTALL
    )
    # Remove empty groups (self-closing or with only whitespace, even if they have attributes)
    svg_text = re.sub(r'<g[^>]*/>', '', svg_text)
    svg_text = re.sub(r'<g[^>]*>\s*</g>', '', svg_text)
    return svg_text


def remove_classes(svg_text):
    """Remove all class attributes from SVG elements.

    This is safe when classes are not referenced by any CSS rules within
    the SVG. Always verify with a quality check after applying.

    Args:
        svg_text: SVG string.

    Returns:
        SVG string with class attributes removed.

    Typical savings: 2-10% on SVGs with many class attributes.
    """
    return re.sub(r' class="[^"]*"', '', svg_text)


def remove_defs(svg_text):
    """Remove the entire <defs> block from an SVG.

    Safe when defs only contain unused filters, gradients, or patterns
    (e.g. after removing the elements that reference them).

    Args:
        svg_text: SVG string.

    Returns:
        SVG string with defs removed.

    Typical savings: Variable (depends on defs content).
    """
    return re.sub(r'<defs>.*?</defs>', '', svg_text, flags=re.DOTALL)


# ---------------------------------------------------------------------------
# Structure tools: viewBox, transform baking, group unwrapping, whitespace
# ---------------------------------------------------------------------------

def rescale_viewbox(svg_text, new_size=1000):
    """Rescale viewBox and all coordinates to use integer-friendly dimensions.

    Changes the viewBox to 0 0 <new_size> <new_size> and scales all path
    coordinates accordingly. This can make coordinates round to integers,
    saving decimal formatting bytes.

    Note: This is most effective for square SVGs (e.g. icons). For
    non-square SVGs, both dimensions are scaled to fit new_size.

    Args:
        svg_text: SVG string.
        new_size: New viewBox dimension (default 1000).

    Returns:
        SVG string with rescaled viewBox and coordinates.

    Typical savings: 5-20% when coordinates become integers.
    """
    # Extract current viewBox
    vb_match = re.search(r'viewBox="([^"]*)"', svg_text)
    if not vb_match:
        return svg_text
    parts = vb_match.group(1).split()
    if len(parts) != 4:
        return svg_text
    try:
        vx, vy, vw, vh = float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])
    except ValueError:
        return svg_text

    scale_x = new_size / vw
    scale_y = new_size / vh

    def process_path(m):
        d = m.group(1)
        cmds = parse_path(d)
        scaled = []
        for cmd, args in cmds:
            if cmd in ('a', 'A') and len(args) == 7:
                new_args = [
                    args[0] * scale_x, args[1] * scale_y,
                    args[2], args[3], args[4],
                    args[5] * scale_x, args[6] * scale_y,
                ]
                scaled.append((cmd, new_args))
            elif cmd in ('h', 'H'):
                scaled.append((cmd, [a * scale_x for a in args]))
            elif cmd in ('v', 'V'):
                scaled.append((cmd, [a * scale_y for a in args]))
            elif cmd in ('Z', 'z'):
                scaled.append((cmd, []))
            else:
                # For M, L, C, S, Q, T and their relative variants:
                # alternating x, y coordinates
                new_args = []
                for i, a in enumerate(args):
                    if i % 2 == 0:
                        new_args.append(a * scale_x)
                    else:
                        new_args.append(a * scale_y)
                scaled.append((cmd, new_args))
        return 'd="' + format_path(scaled, 0) + '"'

    result = re.sub(r'(?<![a-zA-Z-])d="([^"]*)"', process_path, svg_text)
    # Update viewBox
    new_vb = f'viewBox="{int(vx * scale_x)} {int(vy * scale_y)} {new_size} {int(vh * scale_y)}"'
    if vw == vh:
        new_vb = f'viewBox="0 0 {new_size} {new_size}"'
    result = result.replace(vb_match.group(0), new_vb)
    return result


def bake_translate_into_paths(svg_text):
    """Bake translate() transforms into path coordinates.

    When a <path> has transform="translate(tx ty)", the translation is
    added directly to absolute M/L/H/V/A coordinates and the transform
    attribute is removed. Uses proper path parsing to handle all commands.

    Args:
        svg_text: SVG string.

    Returns:
        SVG string with translate transforms baked into paths.

    Typical savings: 20-40 bytes per path with a transform attribute.
    """
    def _fmt(v):
        if v == int(v) and abs(v) < 100000:
            return str(int(v))
        return f'{v:.2f}'.rstrip('0').rstrip('.')

    def bake_one(m):
        full = m.group(0)
        tr = re.search(r'transform="translate\(([0-9.e+-]+)[\s,]+([0-9.e+-]+)\)"', full)
        if not tr:
            # Single-value translate (x only)
            tr = re.search(r'transform="translate\(([0-9.e+-]+)\)"', full)
            if not tr:
                return full
            tx, ty = float(tr.group(1)), 0.0
        else:
            tx, ty = float(tr.group(1)), float(tr.group(2))

        d_match = re.search(r'd="([^"]*)"', full)
        if not d_match:
            return full

        cmds = parse_path(d_match.group(1))
        result = []
        for cmd, args in cmds:
            if cmd == 'M' and len(args) >= 2:
                result.append(('M', [args[0] + tx, args[1] + ty]))
            elif cmd == 'L' and len(args) >= 2:
                result.append(('L', [args[0] + tx, args[1] + ty]))
            elif cmd == 'H' and len(args) >= 1:
                result.append(('H', [args[0] + tx]))
            elif cmd == 'V' and len(args) >= 1:
                result.append(('V', [args[0] + ty]))
            elif cmd == 'A' and len(args) >= 7:
                result.append(('A', [args[0], args[1], args[2], args[3], args[4],
                                     args[5] + tx, args[6] + ty]))
            else:
                # Relative commands and Z don't need translation
                result.append((cmd, args))

        new_d = format_path(result, 2)
        new_full = full.replace(f'd="{d_match.group(1)}"', f'd="{new_d}"')
        new_full = re.sub(r'\s*transform="translate\([^)]+\)"', '', new_full)
        return new_full

    svg_text = re.sub(
        r'<path\b[^>]*/?>',
        lambda m: bake_one(m) if 'transform="translate(' in m.group(0) and 'd="' in m.group(0) else m.group(0),
        svg_text
    )
    return svg_text


def bake_translate_into_text(svg_text):
    """Bake translate() transforms into text element x/y attributes.

    When a <text> has transform="translate(tx ty)", the translation is
    added to the x/y attributes and the transform is removed.

    Args:
        svg_text: SVG string.

    Returns:
        SVG string with text translates baked into x/y.

    Typical savings: 20-40 bytes per text element with a translate.
    """
    def _fmt(v):
        if v == int(v) and abs(v) < 100000:
            return str(int(v))
        return f'{v:.2f}'.rstrip('0').rstrip('.')

    def bake_text(m):
        full = m.group(0)
        tr = re.search(r'transform="translate\(([0-9.e+-]+)[\s,]+([0-9.e+-]+)\)"', full)
        if not tr:
            tr = re.search(r'transform="translate\(([0-9.e+-]+)\)"', full)
            if not tr:
                return full
            tx, ty = float(tr.group(1)), 0.0
        else:
            tx, ty = float(tr.group(1)), float(tr.group(2))

        # Update x/y on the <text> element
        x_match = re.search(r'\bx="([^"]*)"', full)
        y_match = re.search(r'\by="([^"]*)"', full)
        if not x_match or not y_match:
            return full

        new_x = float(x_match.group(1)) + tx
        new_y = float(y_match.group(1)) + ty

        result = full
        result = result.replace(f'x="{x_match.group(1)}"', f'x="{_fmt(new_x)}"')
        result = result.replace(f'y="{y_match.group(1)}"', f'y="{_fmt(new_y)}"')
        result = re.sub(r'\s*transform="translate\([^)]+\)"', '', result)

        # Also update x/y on child tspan elements
        def offset_tspan(tm):
            tspan = tm.group(0)
            tx_m = re.search(r'\bx="([^"]*)"', tspan)
            ty_m = re.search(r'\by="([^"]*)"', tspan)
            if tx_m:
                nx = float(tx_m.group(1)) + tx
                tspan = tspan.replace(f'x="{tx_m.group(1)}"', f'x="{_fmt(nx)}"')
            if ty_m:
                ny = float(ty_m.group(1)) + ty
                tspan = tspan.replace(f'y="{ty_m.group(1)}"', f'y="{_fmt(ny)}"')
            return tspan

        result = re.sub(r'<tspan[^>]*>', offset_tspan, result)
        return result

    svg_text = re.sub(
        r'<text\b[^>]*transform="translate\([^)]+\)"[^>]*>.*?</text>',
        bake_text, svg_text, flags=re.DOTALL
    )
    return svg_text


def remove_unused_defs(svg_text):
    """Remove individual <defs> entries that aren't referenced in the SVG.

    Checks each child of <defs> for its id, and removes it if that id
    isn't referenced anywhere else in the document (via url(#id),
    href="#id", or filter="url(#id)" etc.).

    Args:
        svg_text: SVG string.

    Returns:
        SVG string with unreferenced defs removed. If all defs entries are
        unused, the entire <defs> block is removed.

    Typical savings: Variable — can be 0 or hundreds of bytes.
    """
    import xml.etree.ElementTree as ET

    defs_match = re.search(r'<defs[^>]*>(.*?)</defs>', svg_text, re.DOTALL)
    if not defs_match:
        return svg_text

    defs_content = defs_match.group(1)
    # Find all ids defined in defs
    ids_in_defs = re.findall(r'\bid="([^"]*)"', defs_content)
    if not ids_in_defs:
        return svg_text

    # Check which ids are referenced outside defs
    rest = svg_text[:defs_match.start()] + svg_text[defs_match.end():]
    unused_ids = []
    for def_id in ids_in_defs:
        # Check for url(#id), href="#id", xlink:href="#id", filter="url(#id)"
        if (f'url(#{def_id})' not in rest and
            f'href="#{def_id}"' not in rest and
            f'#{def_id}' not in rest):
            unused_ids.append(def_id)

    if not unused_ids:
        return svg_text

    # Remove unused entries from defs
    new_defs = defs_content
    for uid in unused_ids:
        # Remove the element with this id (self-closing or with children)
        new_defs = re.sub(
            rf'<(\w+)\b[^>]*\bid="{re.escape(uid)}"[^>]*/>', '', new_defs
        )
        new_defs = re.sub(
            rf'<(\w+)\b([^>]*\bid="{re.escape(uid)}"[^>]*)>.*?</\1>',
            '', new_defs, flags=re.DOTALL
        )

    new_defs = new_defs.strip()
    if not new_defs:
        # All defs removed — remove the whole block
        return svg_text[:defs_match.start()] + svg_text[defs_match.end():]

    return svg_text[:defs_match.start()] + f'<defs>{new_defs}</defs>' + svg_text[defs_match.end():]


def unwrap_single_tspans(svg_text):
    """Unwrap <tspan> elements that are the sole child of <text>.

    When a <text> has exactly one <tspan> child, the tspan attributes can
    be merged into the text element and the tspan removed. Handles
    duplicate attribute conflicts by preferring tspan values.

    Args:
        svg_text: SVG string.

    Returns:
        SVG string with unwrapped tspans.

    Typical savings: 15-20 bytes per unwrapped tspan.
    """
    pattern = r'<text([^>]*)><tspan([^>]*)>([^<]*)</tspan></text>'

    def unwrap(m):
        text_attrs = m.group(1)
        tspan_attrs = m.group(2)
        content = m.group(3)
        # Remove duplicate attrs from text (tspan values take precedence)
        text_names = set(re.findall(r'(\w+)=', text_attrs))
        tspan_names = set(re.findall(r'(\w+)=', tspan_attrs))
        for name in text_names & tspan_names:
            text_attrs = re.sub(rf'\s*{name}="[^"]*"', '', text_attrs)
        return f'<text{text_attrs}{tspan_attrs}>{content}</text>'

    return re.sub(pattern, unwrap, svg_text)


def unwrap_bare_groups(svg_text):
    """Remove <g> wrapper elements that have no attributes.

    Bare <g>...</g> wrappers with no attributes serve no purpose and
    can be removed, keeping their content.

    Args:
        svg_text: SVG string.

    Returns:
        SVG string with bare groups unwrapped.

    Typical savings: 7 bytes per removed group (<g></g>).
    """
    # Remove <g> with single child that could be promoted
    svg_text = re.sub(
        r'<g>(<(?:text|path|rect|circle|line|ellipse|polyline|polygon)[^>]*(?:>[^<]*</\w+>|/>))</g>',
        r'\1', svg_text
    )
    # Remove empty groups
    svg_text = re.sub(r'<g>\s*</g>', '', svg_text)
    return svg_text


def remove_width_height(svg_text):
    """Remove explicit width/height when viewBox is present.

    When a viewBox attribute exists, width and height are optional as the
    SVG will scale to fit its container.

    Args:
        svg_text: SVG string.

    Returns:
        SVG string with width/height removed if viewBox exists.

    Typical savings: 20-40 bytes.
    """
    if 'viewBox=' in svg_text:
        svg_text = re.sub(r'\s+(?<![-])width="[^"]*"', '', svg_text)
        svg_text = re.sub(r'\s+(?<![-])height="[^"]*"', '', svg_text)
    return svg_text


def remove_identity_transforms(svg_text):
    """Remove transform attributes that are identity operations.

    Removes rotate(360deg) and translate(0 0) transforms that have
    no visual effect.

    Args:
        svg_text: SVG string.

    Returns:
        SVG string with identity transforms removed.

    Typical savings: 20-100 bytes per identity transform.
    """
    # rotate(360deg) = identity
    svg_text = re.sub(
        r'\s*style="[^"]*(?:-ms-transform:rotate\(360deg\);)?'
        r'(?:-webkit-transform:rotate\(360deg\);)?'
        r'transform:rotate\(360deg\)[^"]*"',
        '', svg_text
    )
    # translate(0 0) and translate(0,0)
    svg_text = re.sub(r'\s*transform="translate\(0[\s,]+0\)"', '', svg_text)
    return svg_text


def strip_whitespace(svg_text):
    """Remove unnecessary whitespace from SVG markup.

    Removes spaces before />, between >< tags, and collapses multiple
    spaces. Does not affect whitespace inside text content.

    Args:
        svg_text: SVG string.

    Returns:
        SVG string with whitespace stripped.

    Typical savings: 2-10% on SVGs with generous formatting.
    """
    svg_text = re.sub(r'\s+/>', '/>', svg_text)
    svg_text = re.sub(r'>\s+<', '><', svg_text)
    svg_text = svg_text.strip()
    return svg_text


def compact_path_numbers(svg_text):
    """Remove leading zeros from fractional numbers in path data.

    Converts 0.5 to .5, -0.3 to -.3, etc. inside path d attributes.
    Relies on the SVG spec allowing implicit leading zeros.

    Args:
        svg_text: SVG string.

    Returns:
        SVG string with compacted path numbers.

    Typical savings: 1-5% on paths with many sub-1 values.
    """
    def opt(m):
        d = m.group(1)
        d = re.sub(r'(?<![0-9])0\.', '.', d)
        return f'd="{d}"'
    return re.sub(r'(?<![a-zA-Z-])d="([^"]+)"', opt, svg_text)


def remove_space_before_negative(svg_text):
    """Remove spaces before negative numbers in path data.

    The minus sign acts as an implicit separator in SVG path data, so
    spaces before negative numbers are redundant.

    Args:
        svg_text: SVG string.

    Returns:
        SVG string with spaces removed before negatives.

    Typical savings: 1-3%.
    """
    def opt(m):
        d = m.group(1)
        d = re.sub(r'([MmLlHhVvCcSsQqTtAaZz])\s+', r'\1', d)
        d = re.sub(r' (-)', r'\1', d)
        return f'd="{d}"'
    return re.sub(r'(?<![a-zA-Z-])d="([^"]+)"', opt, svg_text)


def round_attribute_coords(svg_text, precision=1):
    """Round coordinate attributes (x, y, width, height, etc.) to given precision.

    Rounds numeric values in common SVG geometric attributes. Also rounds
    translate() transform values.

    Args:
        svg_text: SVG string.
        precision: Decimal places (default 1).

    Returns:
        SVG string with rounded attribute values.

    Typical savings: 2-8% on SVGs with many decimal attribute values.
    """
    def round_attr(m):
        val = float(m.group(1))
        rounded = round(val, precision)
        if rounded == int(rounded):
            return str(int(rounded))
        return str(rounded)

    for attr in ['x', 'y', 'x1', 'y1', 'x2', 'y2', 'width', 'height',
                 'dx', 'dy', 'cx', 'cy', 'r', 'rx', 'ry']:
        svg_text = re.sub(
            r'(?<=' + attr + r'=")(-?\d+\.?\d*?)(?=")', round_attr, svg_text
        )

    # Round translate values
    def round_translate(m):
        parts = re.findall(r'[-+]?\d+\.?\d*', m.group(1))
        rounded = []
        for p in parts:
            v = float(p)
            r = round(v, precision)
            if r == int(r):
                rounded.append(str(int(r)))
            else:
                rounded.append(str(r))
        return 'translate(' + ' '.join(rounded) + ')'

    svg_text = re.sub(r'translate\(([^)]+)\)', round_translate, svg_text)
    return svg_text


# ---------------------------------------------------------------------------
# Font subsetting
# ---------------------------------------------------------------------------

def subset_fonts(svg_text):
    """Remove unused glyphs from embedded fonts in SVG.

    Finds @font-face rules with base64-encoded font data, determines which
    characters are actually used in <text>/<tspan> elements, and rebuilds
    each font containing only the needed glyphs.

    Args:
        svg_text: SVG string.

    Returns:
        SVG string with subsetted fonts.

    Typical savings: 50-97% on SVGs with embedded fonts (e.g., 246KB to 6KB).
    Requires: fonttools, brotli (for WOFF2).
    """
    import base64
    import io
    from fontTools.ttLib import TTFont
    from fontTools.subset import Subsetter, Options

    # Find all @font-face rules with embedded font data
    font_face_pattern = re.compile(
        r'(@font-face\s*\{[^}]*?src:\s*url\(["\']?data:(?:application|font)/[\w+-]+;base64,)'
        r'([A-Za-z0-9+/=\s]+)'
        r'(["\']?\)[^}]*?\})',
        re.DOTALL
    )
    matches = list(font_face_pattern.finditer(svg_text))
    if not matches:
        return svg_text

    # Collect all characters used in text content
    used_chars = set()
    for text_match in re.finditer(r'<(?:text|tspan)[^>]*>([^<]+)</', svg_text):
        used_chars.update(text_match.group(1))
    # Also check text content in nested elements
    for text_match in re.finditer(r'>([^<]+)</', svg_text):
        content = text_match.group(1).strip()
        if content and not content.startswith('{') and not content.startswith('@'):
            used_chars.update(content)

    if not used_chars:
        return svg_text

    # Build the unicode set string for subsetter
    unicodes = {ord(c) for c in used_chars}

    result = svg_text
    for match in reversed(matches):  # reverse to preserve offsets
        prefix = match.group(1)
        b64_data = match.group(2).replace('\n', '').replace(' ', '')
        suffix = match.group(3)

        try:
            font_bytes = base64.b64decode(b64_data)
            font = TTFont(io.BytesIO(font_bytes))

            # Determine original format for re-encoding
            is_woff2 = font_bytes[:4] == b'wOF2'
            is_woff = font_bytes[:4] == b'wOFF'

            # Subset the font
            options = Options()
            options.layout_features = ['*']  # keep all layout features
            options.name_IDs = ['*']
            options.notdef_outline = True
            subsetter = Subsetter(options=options)
            subsetter.populate(unicodes=unicodes)
            subsetter.subset(font)

            # Re-encode
            out = io.BytesIO()
            if is_woff2:
                font.flavor = 'woff2'
            elif is_woff:
                font.flavor = 'woff'
            font.save(out)
            new_bytes = out.getvalue()

            # Only replace if we actually saved space
            if len(new_bytes) >= len(font_bytes):
                continue

            new_b64 = base64.b64encode(new_bytes).decode('ascii')
            replacement = prefix + new_b64 + suffix
            result = result[:match.start()] + replacement + result[match.end():]

        except Exception:
            # Skip this font face on any error — don't break the SVG
            continue

    return result


# ---------------------------------------------------------------------------
# Dereference <use> elements
# ---------------------------------------------------------------------------

def dereference_use_elements(svg_text):
    """Replace <use> elements with inlined copies of their referenced content.

    Finds <use href="#id"> (or xlink:href="#id") elements, locates the
    referenced element in <defs> or elsewhere, deep-clones it, applies
    the <use> element's x/y/transform attributes, and substitutes inline.
    Removes dereferenced defs entries if no longer referenced.

    Args:
        svg_text: SVG string.

    Returns:
        SVG string with <use> elements replaced by inlined content.

    Typical savings: Varies. Can increase size if element is reused many times,
    but enables further optimization of the inlined content.
    """
    # Parse namespace-aware
    SVG_NS = 'http://www.w3.org/2000/svg'
    XLINK_NS = 'http://www.w3.org/1999/xlink'

    # Register namespaces to preserve them in output
    namespaces = dict(re.findall(r'xmlns(?::(\w+))?="([^"]*)"', svg_text))
    for prefix, uri in namespaces.items():
        if prefix:
            ET.register_namespace(prefix, uri)
        else:
            ET.register_namespace('', uri)

    try:
        root = ET.fromstring(svg_text)
    except ET.ParseError:
        return svg_text

    # Build id -> element map
    id_map = {}
    for elem in root.iter():
        eid = elem.get('id')
        if eid:
            id_map[eid] = elem

    # Find all <use> elements
    use_elements = list(root.iter(f'{{{SVG_NS}}}use')) + list(root.iter('use'))
    if not use_elements:
        return svg_text

    import copy
    replaced = 0
    for use_elem in use_elements:
        # Get referenced id
        href = (use_elem.get('href') or
                use_elem.get(f'{{{XLINK_NS}}}href') or
                use_elem.get('xlink:href'))
        if not href or not href.startswith('#'):
            continue

        ref_id = href[1:]
        ref_elem = id_map.get(ref_id)
        if ref_elem is None:
            continue

        # Deep clone the referenced element
        clone = copy.deepcopy(ref_elem)
        # Remove the id from clone to avoid duplicates
        if 'id' in clone.attrib:
            del clone.attrib['id']

        # Apply x/y as translate transform
        x = use_elem.get('x', '0')
        y = use_elem.get('y', '0')
        use_transform = use_elem.get('transform', '')

        try:
            x_val = float(x)
            y_val = float(y)
        except (ValueError, TypeError):
            x_val = 0
            y_val = 0

        # Build combined transform
        transforms = []
        if use_transform:
            transforms.append(use_transform)
        if x_val != 0 or y_val != 0:
            if y_val == 0:
                transforms.append(f'translate({x})')
            else:
                transforms.append(f'translate({x},{y})')

        if transforms:
            existing = clone.get('transform', '')
            if existing:
                transforms.append(existing)
            clone.set('transform', ' '.join(transforms))

        # Copy over other attributes from <use> (width, height, style, etc.)
        skip_attrs = {'href', f'{{{XLINK_NS}}}href', 'xlink:href',
                      'x', 'y', 'transform', 'id'}
        for attr, val in use_elem.attrib.items():
            if attr not in skip_attrs:
                clone.set(attr, val)

        # Replace <use> with clone in parent
        parent = None
        for p in root.iter():
            if use_elem in list(p):
                parent = p
                break
        if parent is not None:
            idx = list(parent).index(use_elem)
            parent.remove(use_elem)
            parent.insert(idx, clone)
            replaced += 1

    if replaced == 0:
        return svg_text

    # Serialize back
    result = ET.tostring(root, encoding='unicode')
    # Restore XML declaration if present
    if svg_text.startswith('<?xml'):
        decl_end = svg_text.index('?>') + 2
        decl = svg_text[:decl_end]
        result = decl + result

    return result


# ---------------------------------------------------------------------------
# Flatten clipPaths
# ---------------------------------------------------------------------------

def flatten_clip_paths(svg_text):
    """Remove simple clipPath definitions and their clip-path references.

    Handles two common cases:
    1. ClipPaths that clip to a rectangle matching or exceeding the viewBox
       (i.e., no-op clips) — removes the clipPath and the clip-path attribute.
    2. ClipPaths containing a single simple shape — removes the clipPath def
       and the clip-path attribute, since the quality gate will catch any
       visual regression.

    This is a lossy operation that relies on the SSIM/PSNR quality gate
    to verify the result.

    Args:
        svg_text: SVG string.

    Returns:
        SVG string with simple clipPaths removed.

    Typical savings: 50-200 bytes per removed clipPath, plus enables
    further optimization of previously-clipped content.
    """
    # Find all clipPath definitions
    clip_path_pattern = re.compile(
        r'<clipPath\b[^>]*\bid="([^"]*)"[^>]*>(.*?)</clipPath>',
        re.DOTALL
    )
    clip_paths = list(clip_path_pattern.finditer(svg_text))
    if not clip_paths:
        return svg_text

    # Get viewBox dimensions for no-op detection
    vb_match = re.search(r'viewBox="([^"]*)"', svg_text)
    vb_x, vb_y, vb_w, vb_h = 0, 0, float('inf'), float('inf')
    if vb_match:
        parts = vb_match.group(1).split()
        if len(parts) == 4:
            try:
                vb_x, vb_y, vb_w, vb_h = (float(p) for p in parts)
            except ValueError:
                pass

    ids_to_remove = []

    for match in clip_paths:
        clip_id = match.group(1)
        content = match.group(2).strip()

        # Check if this clipPath is actually referenced
        ref_count = svg_text.count(f'url(#{clip_id})')
        ref_count += svg_text.count(f'clip-path="#{clip_id}"')
        if ref_count == 0:
            # Unreferenced clipPath — safe to remove the def
            ids_to_remove.append(clip_id)
            continue

        # Case 1: clipPath contains a rect that covers the full viewBox (no-op)
        rect_match = re.match(
            r'<rect\s+([^>]*)/?>', content
        )
        if rect_match:
            attrs = rect_match.group(1)
            try:
                rx = float(re.search(r'\bx="([^"]*)"', attrs).group(1)) if re.search(r'\bx="([^"]*)"', attrs) else 0
                ry = float(re.search(r'\by="([^"]*)"', attrs).group(1)) if re.search(r'\by="([^"]*)"', attrs) else 0
                rw = float(re.search(r'\bwidth="([^"]*)"', attrs).group(1)) if re.search(r'\bwidth="([^"]*)"', attrs) else 0
                rh = float(re.search(r'\bheight="([^"]*)"', attrs).group(1)) if re.search(r'\bheight="([^"]*)"', attrs) else 0

                # If rect covers the viewBox, it's a no-op clip
                if rx <= vb_x and ry <= vb_y and rw >= vb_w and rh >= vb_h:
                    ids_to_remove.append(clip_id)
                    continue
            except (ValueError, AttributeError):
                pass

        # Case 2: Simple single-shape clipPath — remove and let quality gate decide
        single_shape = re.match(
            r'<(?:rect|circle|ellipse|polygon|path)\b[^>]*/?>',
            content
        )
        if single_shape and content.count('<') == 1:
            ids_to_remove.append(clip_id)
            continue

    if not ids_to_remove:
        return svg_text

    result = svg_text
    for clip_id in ids_to_remove:
        # Remove the clipPath definition
        result = re.sub(
            rf'<clipPath\b[^>]*\bid="{re.escape(clip_id)}"[^>]*>.*?</clipPath>\s*',
            '', result, flags=re.DOTALL
        )
        # Remove clip-path references to this id
        result = re.sub(
            rf'\s*clip-path="url\(#{re.escape(clip_id)}\)"', '', result
        )
        # Also remove from style attributes
        result = re.sub(
            rf'clip-path:\s*url\(#{re.escape(clip_id)}\);?\s*', '', result
        )

    return result


# ---------------------------------------------------------------------------
# Rendering and validation (requires cairosvg, numpy, PIL, scipy)
# ---------------------------------------------------------------------------

def render_svg(svg_text, size=512):
    """Render SVG text to a numpy RGB array.

    Renders the SVG at the given size using CairoSVG, composites onto
    a white background, and returns an RGB numpy array.

    Args:
        svg_text: SVG string.
        size: Output image width and height in pixels (default 512).

    Returns:
        numpy array of shape (size, size, 3) with dtype uint8.

    Requires: cairosvg, numpy, PIL
    """
    import io
    import cairosvg
    import numpy as np
    from PIL import Image

    png_data = cairosvg.svg2png(
        bytestring=svg_text.encode('utf-8'),
        output_width=size, output_height=size,
    )
    img = Image.open(io.BytesIO(png_data)).convert('RGBA')
    bg = Image.new('RGBA', img.size, (255, 255, 255, 255))
    return np.array(Image.alpha_composite(bg, img).convert('RGB'))


def compare(svg_a, svg_b, size=512):
    """Compare two SVGs visually using SSIM and PSNR metrics.

    Renders both SVGs and computes:
    - SSIM (Structural Similarity Index): 1.0 = identical
    - PSNR (Peak Signal-to-Noise Ratio): higher = more similar, inf = identical
    - MSE (Mean Squared Error): 0 = identical

    Args:
        svg_a: Reference SVG string.
        svg_b: Comparison SVG string.
        size: Render size for comparison (default 512).

    Returns:
        (ssim, psnr) tuple.

    Requires: cairosvg, numpy, PIL, scipy
    """
    import numpy as np
    from scipy.ndimage import uniform_filter

    a, b = render_svg(svg_a, size), render_svg(svg_b, size)
    diff = a.astype(np.float64) - b.astype(np.float64)
    mse = np.mean(diff ** 2)
    psnr = 10 * np.log10(255 ** 2 / mse) if mse > 0 else float('inf')

    def _ssim_channel(x, y):
        C1 = (0.01 * 255) ** 2
        C2 = (0.03 * 255) ** 2
        ux = uniform_filter(x, 11)
        uy = uniform_filter(y, 11)
        uxx = uniform_filter(x * x, 11)
        uyy = uniform_filter(y * y, 11)
        uxy = uniform_filter(x * y, 11)
        vx = uxx - ux * ux
        vy = uyy - uy * uy
        vxy = uxy - ux * uy
        return np.mean(
            (2 * ux * uy + C1) * (2 * vxy + C2) /
            ((ux ** 2 + uy ** 2 + C1) * (vx + vy + C2))
        )

    af = a.astype(np.float64)
    bf = b.astype(np.float64)
    ssim = float(np.mean([_ssim_channel(af[:, :, c], bf[:, :, c]) for c in range(3)]))
    return ssim, psnr


# ---------------------------------------------------------------------------
# Pipeline: compress_svg
# ---------------------------------------------------------------------------

def compress_svg(svg_text, quality_check=True, precision=2,
                 line_threshold=0.05, quad_threshold=0.03,
                 smooth_tolerance=0.06, ssim_min=0.97, psnr_min=30):
    """Apply all safe compression techniques in optimal order.

    Pipeline:
    1. Remove identity transforms
    2. Remove hidden elements
    3. Remove default attributes
    4. Remove width/height (if viewBox exists)
    5. Strip whitespace
    6. Round attribute coordinates
    7. Round path coordinates
    8. Convert abs to rel (choosing shorter per command)
    9. Cubic to line simplification
    10. Cubic to quadratic simplification
    11. Smooth curve conversion (c->s)
    12. H/V shorthand
    13. Merge collinear lines
    14. Merge same commands
    15. Compact path numbers
    16. Remove space before negatives
    17. Unwrap single tspans
    18. Unwrap bare groups

    If quality_check is True, each lossy step (rounding, curve simplification)
    is verified against the original using SSIM/PSNR, and reverted if quality
    drops below thresholds.

    Args:
        svg_text: Input SVG string.
        quality_check: Whether to verify quality after lossy steps (default True).
            Requires cairosvg, numpy, PIL, scipy.
        precision: Decimal precision for coordinate rounding (default 2).
        line_threshold: Error threshold for cubic-to-line (default 0.05).
        quad_threshold: Error threshold for cubic-to-quad (default 0.03).
        smooth_tolerance: Tolerance for smooth curve detection (default 0.06).
        ssim_min: Minimum SSIM for quality gate (default 0.97).
        psnr_min: Minimum PSNR for quality gate (default 30).

    Returns:
        Compressed SVG string.
    """
    original = svg_text

    def _check(candidate, label=""):
        """Return candidate if it passes quality, else return None."""
        if not quality_check:
            return candidate
        try:
            ssim, psnr = compare(original, candidate)
            if ssim >= ssim_min and psnr >= psnr_min:
                return candidate
            return None
        except Exception:
            return None

    # --- Lossless steps (always safe) ---
    svg_text = remove_identity_transforms(svg_text)
    svg_text = remove_hidden_elements(svg_text)
    svg_text = remove_default_attributes(svg_text)
    svg_text = remove_width_height(svg_text)
    svg_text = strip_whitespace(svg_text)
    svg_text = unwrap_single_tspans(svg_text)
    svg_text = unwrap_bare_groups(svg_text)
    svg_text = compact_path_numbers(svg_text)
    svg_text = remove_space_before_negative(svg_text)

    # --- Lossy steps (with quality gate) ---
    # Round attribute coordinates
    candidate = round_attribute_coords(svg_text, precision)
    result = _check(candidate, "round_attrs")
    if result is not None:
        svg_text = result

    # Round path coordinates
    candidate = round_path_coordinates(svg_text, precision)
    result = _check(candidate, "round_paths")
    if result is not None:
        svg_text = result

    # Abs to rel conversion
    candidate = abs_to_rel(svg_text, precision)
    result = _check(candidate, "abs_to_rel")
    if result is not None:
        svg_text = result

    # Curve simplification: cubic to line
    candidate = cubic_to_line(svg_text, line_threshold, precision)
    result = _check(candidate, "cubic_to_line")
    if result is not None:
        svg_text = result

    # Curve simplification: cubic to quadratic
    candidate = cubic_to_quad(svg_text, quad_threshold, precision)
    result = _check(candidate, "cubic_to_quad")
    if result is not None:
        svg_text = result

    # Smooth curve conversion
    candidate = try_smooth_curves(svg_text, smooth_tolerance, precision)
    result = _check(candidate, "smooth_curves")
    if result is not None:
        svg_text = result

    # H/V shorthand
    candidate = curve_to_hv(svg_text, precision)
    result = _check(candidate, "curve_to_hv")
    if result is not None:
        svg_text = result

    # Merge collinear lines
    candidate = merge_collinear_lines(svg_text, 0.01, precision)
    result = _check(candidate, "merge_collinear")
    if result is not None:
        svg_text = result

    # Merge same commands
    candidate = merge_same_commands(svg_text, precision)
    result = _check(candidate, "merge_commands")
    if result is not None:
        svg_text = result

    # Final whitespace cleanup
    svg_text = strip_whitespace(svg_text)

    return svg_text


# ---------------------------------------------------------------------------
# Additional tools inspired by Vecta Nano
# ---------------------------------------------------------------------------

# Named SVG colors sorted by length (shortest first) for optimal substitution
_NAMED_COLORS = {
    "#f00": "red", "#0f0": "lime", "#00f": "blue", "#ff0": "yellow",
    "#0ff": "cyan", "#f0f": "magenta", "#800000": "maroon", "#808000": "olive",
    "#008000": "green", "#800080": "purple", "#008080": "teal", "#000080": "navy",
    "#ffa500": "orange", "#ffc0cb": "pink", "#ee82ee": "violet", "#4b0082": "indigo",
    "#f5f5dc": "beige", "#fffff0": "ivory", "#f0e68c": "khaki", "#e6e6fa": "lavender",
    "#ffd700": "gold", "#d2b48c": "tan", "#fa8072": "salmon", "#ff7f50": "coral",
    "#dda0dd": "plum", "#f5deb3": "wheat", "#ffe4c4": "bisque", "#ff6347": "tomato",
    "#fffafa": "snow", "#ffe4e1": "mistyrose", "#fff0f5": "lavenderblush",
    "#fffaf0": "floralwhite", "#f0fff0": "honeydew", "#f0ffff": "azure",
    "#f5f5f5": "whitesmoke", "#fff5ee": "seashell", "#fdf5e6": "oldlace",
    "#faf0e6": "linen", "#faebd7": "antiquewhite", "#fff8dc": "cornsilk",
    "#ffffe0": "lightyellow", "#ffe4b5": "moccasin", "#ffdead": "navajowhite",
    "#ffdab9": "peachpuff", "#d2691e": "chocolate", "#a0522d": "sienna",
    "#cd853f": "peru", "#b22222": "firebrick", "#dc143c": "crimson",
    "#ff4500": "orangered", "#ff8c00": "darkorange", "#ff1493": "deeppink",
    "#ff69b4": "hotpink", "#da70d6": "orchid", "#ba55d3": "mediumorchid",
    "#9370db": "mediumpurple", "#6a5acd": "slateblue", "#7b68ee": "mediumslateblue",
    "#4169e1": "royalblue", "#1e90ff": "dodgerblue", "#87ceeb": "skyblue",
    "#00ced1": "darkturquoise", "#20b2aa": "lightseagreen", "#2e8b57": "seagreen",
    "#3cb371": "mediumseagreen", "#228b22": "forestgreen", "#32cd32": "limegreen",
    "#7cfc00": "lawngreen", "#adff2f": "greenyellow", "#9acd32": "yellowgreen",
    "#6b8e23": "olivedrab", "#bdb76b": "darkkhaki", "#f0e68c": "khaki",
    "#fff": "white", "#000": "black", "#808080": "gray", "#c0c0c0": "silver",
}

# Reverse map: name -> hex (for converting names to shorter hex when possible)
_COLOR_NAME_TO_HEX = {v: k for k, v in _NAMED_COLORS.items()}


def shorten_colors(svg_text: str) -> str:
    """Shorten color values: #rrggbb -> #rgb, hex -> named color (whichever shorter).

    Examples: #000000 -> #000, #ff0000 -> red, rgb(255,0,0) -> red
    """
    def _shorten_hex(m):
        h = m.group(0).lower()
        # Try to shorten #rrggbb to #rgb
        if len(h) == 7 and h[1] == h[2] and h[3] == h[4] and h[5] == h[6]:
            short = f"#{h[1]}{h[3]}{h[5]}"
        else:
            short = h
        # Check if a named color is shorter
        named = _NAMED_COLORS.get(short)
        if named and len(named) < len(short):
            return named
        named = _NAMED_COLORS.get(h)
        if named and len(named) < len(short):
            return named
        return short

    # Shorten #rrggbb and #rgb hex colors
    result = re.sub(r'#[0-9a-fA-F]{6}\b', _shorten_hex, svg_text)
    result = re.sub(r'#[0-9a-fA-F]{3}\b', _shorten_hex, result)

    # Convert rgb(r,g,b) to hex
    def _rgb_to_hex(m):
        r, g, b = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if 0 <= r <= 255 and 0 <= g <= 255 and 0 <= b <= 255:
            h = f"#{r:02x}{g:02x}{b:02x}"
            return _shorten_hex(re.match(r'#[0-9a-fA-F]{6}', h))
        return m.group(0)

    result = re.sub(r'rgb\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)', _rgb_to_hex, result)

    return result


def shorten_ids(svg_text: str) -> str:
    """Shorten referenced IDs to minimal strings (A, B, C, ..., AA, AB, ...).

    Only shortens IDs that are actually referenced elsewhere (url(#id), href="#id", etc).
    Removes unreferenced IDs entirely.
    """
    # Find all id="..." declarations
    id_decls = re.findall(r'\bid="([^"]*)"', svg_text)
    if not id_decls:
        return svg_text

    # Check which IDs are referenced
    referenced = set()
    unreferenced = set()
    for oid in id_decls:
        # Check for references: url(#id), href="#id", xlink:href="#id", clip-path="url(#id)"
        ref_pattern = re.compile(
            r'(?:url\(\s*#' + re.escape(oid) + r'\s*\))|'
            r'(?:href\s*=\s*"#' + re.escape(oid) + r'")|'
            r'(?:xlink:href\s*=\s*"#' + re.escape(oid) + r'")'
        )
        if ref_pattern.search(svg_text):
            referenced.add(oid)
        else:
            unreferenced.add(oid)

    # Remove unreferenced IDs
    result = svg_text
    for oid in unreferenced:
        result = re.sub(r'\s*id="' + re.escape(oid) + r'"', '', result)

    # Generate short names
    def _short_name(n):
        chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
        if n < len(chars):
            return chars[n]
        return chars[n // len(chars) - 1] + chars[n % len(chars)]

    # Sort referenced IDs by length (longest first to avoid partial replacements)
    sorted_refs = sorted(referenced, key=len, reverse=True)

    # Map old -> new, only if new is shorter
    renames = {}
    idx = 0
    for oid in sorted_refs:
        short = _short_name(idx)
        if len(short) < len(oid):
            renames[oid] = short
        idx += 1

    # Apply renames
    for old_id, new_id in renames.items():
        # Replace id="old" with id="new"
        result = result.replace(f'id="{old_id}"', f'id="{new_id}"')
        # Replace all references
        result = result.replace(f'url(#{old_id})', f'url(#{new_id})')
        result = result.replace(f'href="#{old_id}"', f'href="#{new_id}"')
        result = result.replace(f'xlink:href="#{old_id}"', f'xlink:href="#{new_id}"')

    return result


def remove_metadata(svg_text: str) -> str:
    """Remove comments, metadata, editor namespaces, and other non-rendering content.

    Removes:
    - XML comments (<!-- ... -->)
    - <metadata> blocks
    - <title> and <desc> elements
    - Editor namespace declarations (inkscape, sodipodi, sketch, illustrator)
    - Editor-specific attributes (inkscape:*, sodipodi:*, sketch:*, data-name, etc)
    - XML processing instructions (<?xml ...?>)
    - DOCTYPE declarations
    """
    # Remove XML comments
    result = re.sub(r'<!--.*?-->', '', svg_text, flags=re.DOTALL)

    # Remove <?xml ... ?>
    result = re.sub(r'<\?xml[^?]*\?>\s*', '', result)

    # Remove DOCTYPE
    result = re.sub(r'<!DOCTYPE[^>]*>\s*', '', result)

    # Remove <metadata>...</metadata>
    result = re.sub(r'<metadata[^>]*>.*?</metadata>\s*', '', result, flags=re.DOTALL)

    # Remove <title>...</title> and <desc>...</desc>
    result = re.sub(r'<title[^>]*>.*?</title>\s*', '', result, flags=re.DOTALL)
    result = re.sub(r'<desc[^>]*>.*?</desc>\s*', '', result, flags=re.DOTALL)

    # Remove editor namespace declarations
    result = re.sub(
        r'\s+xmlns:(?:inkscape|sodipodi|sketch|dc|cc|rdf|ns\d+|i|x|graph|illustrator)\s*=\s*"[^"]*"',
        '', result
    )

    # Remove editor-specific attributes
    result = re.sub(
        r'\s+(?:inkscape|sodipodi|sketch|illustrator|data-name):[a-zA-Z_-]+\s*=\s*"[^"]*"',
        '', result
    )

    # Remove sodipodi: and inkscape: elements
    result = re.sub(r'<(?:sodipodi|inkscape):[^>]*/?>\s*', '', result)
    result = re.sub(r'<(?:sodipodi|inkscape):[^>]*>.*?</(?:sodipodi|inkscape):[^>]*>\s*', '', result, flags=re.DOTALL)

    # Remove RDF metadata blocks
    result = re.sub(r'<rdf:RDF[^>]*>.*?</rdf:RDF>\s*', '', result, flags=re.DOTALL)
    result = re.sub(r'<cc:[^>]*>.*?</cc:[^>]*>\s*', '', result, flags=re.DOTALL)

    return result


def shapes_to_paths(svg_text: str) -> str:
    """Convert basic shapes (rect, line, polygon, polyline, ellipse, circle) to <path>.

    Paths are generally more compact and enable further path optimizations.
    Only converts shapes without rounded corners (rx/ry on rect).
    """
    def _rect_to_path(m):
        attrs_str = m.group(1)
        # Don't convert rects with rx/ry (rounded corners)
        if re.search(r'\br[xy]\s*=', attrs_str):
            return m.group(0)

        def _get(name, default="0"):
            match = re.search(r'\b' + name + r'\s*=\s*"([^"]*)"', attrs_str)
            if not match:
                return float(default)
            val = match.group(1)
            # Skip percentage/non-numeric values
            if '%' in val or not re.match(r'^[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?$', val.strip()):
                return None
            return float(val)

        x, y = _get("x"), _get("y")
        w, h = _get("width"), _get("height")
        if x is None or y is None or w is None or h is None:
            return m.group(0)  # can't convert non-numeric
        if w == 0 or h == 0:
            return ''  # invisible rect

        d = f"M{_fmt(x)} {_fmt(y)}h{_fmt(w)}v{_fmt(h)}h{_fmt(-w)}z"

        # Preserve non-geometry attributes
        # Use (?<![-]) to avoid matching stroke-width, line-height etc.
        other = re.sub(r'(?<!-)(?:x|y|width|height)\s*=\s*"[^"]*"', '', attrs_str).strip()
        return f'<path d="{d}" {other}/>'

    def _line_to_path(m):
        attrs_str = m.group(1)

        def _get(name):
            match = re.search(r'\b' + name + r'\s*=\s*"([^"]*)"', attrs_str)
            return float(match.group(1)) if match else 0.0

        x1, y1, x2, y2 = _get("x1"), _get("y1"), _get("x2"), _get("y2")
        d = f"M{_fmt(x1)} {_fmt(y1)}L{_fmt(x2)} {_fmt(y2)}"

        other = re.sub(r'\b(?:x1|y1|x2|y2)\s*=\s*"[^"]*"', '', attrs_str).strip()
        return f'<path d="{d}" {other}/>'

    def _polygon_to_path(m):
        tag = m.group(1)  # polygon or polyline
        attrs_str = m.group(2)
        points_match = re.search(r'\bpoints\s*=\s*"([^"]*)"', attrs_str)
        if not points_match:
            return m.group(0)

        nums = re.findall(r'[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?', points_match.group(1))
        if len(nums) < 4:
            return m.group(0)

        parts = [f"M{nums[0]} {nums[1]}"]
        for i in range(2, len(nums) - 1, 2):
            parts.append(f"L{nums[i]} {nums[i+1]}")
        if tag == "polygon":
            parts.append("z")

        d = "".join(parts)
        other = re.sub(r'\bpoints\s*=\s*"[^"]*"', '', attrs_str).strip()
        return f'<path d="{d}" {other}/>'

    def _fmt(v):
        """Format number minimally."""
        if v == int(v):
            return str(int(v))
        return f"{v:.3g}"

    result = svg_text

    # rect -> path (skip rounded rects)
    result = re.sub(r'<rect\b([^>]*?)/?>', _rect_to_path, result)

    # line -> path
    result = re.sub(r'<line\b([^>]*?)/?>', _line_to_path, result)

    # polygon/polyline -> path
    result = re.sub(r'<(polygon|polyline)\b([^>]*?)/?>', _polygon_to_path, result)

    return result


def extract_common_styles(svg_text: str, min_occurrences: int = 2) -> str:
    """Extract repeated presentation attributes into CSS classes.

    Works at the individual property level (like Nano): finds the most
    commonly repeated (property, value) pairs across elements, creates
    atomic CSS classes for the highest-savings combinations, and assigns
    multiple classes per element.  Strips default/redundant values first.

    Args:
        svg_text: SVG string.
        min_occurrences: Minimum times a property must appear to extract.

    Returns:
        SVG string with common styles in a <style> block and class references.
    """
    PRESENTATION_ATTRS = {
        "fill", "fill-opacity", "fill-rule", "stroke", "stroke-width",
        "stroke-linecap", "stroke-linejoin", "stroke-miterlimit",
        "stroke-dasharray", "stroke-dashoffset", "stroke-opacity",
        "opacity", "font-family", "font-size", "font-style", "font-weight",
        "text-anchor", "text-decoration", "dominant-baseline",
        "alignment-baseline", "letter-spacing", "word-spacing",
        "color", "display", "visibility", "clip-rule",
    }

    # Default values that can be stripped (they have no visual effect)
    DEFAULTS = {
        "fill-opacity": {"1", "1.0"},
        "stroke-opacity": {"1", "1.0"},
        "opacity": {"1", "1.0"},
        "display": {"inline"},
        "visibility": {"visible"},
        "stroke-dasharray": {"none"},
        "stroke-linecap": {"butt"},
        "stroke-linejoin": {"miter"},
        "stroke-miterlimit": {"4"},
        "fill-rule": {"nonzero"},
        "clip-rule": {"nonzero"},
        "font-style": {"normal"},
        "font-weight": {"normal", "400"},
        "text-decoration": {"none"},
        "letter-spacing": {"normal", "0"},
        "word-spacing": {"normal", "0"},
    }

    from collections import Counter
    import xml.etree.ElementTree as ET

    try:
        root = ET.fromstring(svg_text)
    except ET.ParseError:
        return svg_text

    ns = ""
    ns_match = re.match(r'\{([^}]+)\}', root.tag)
    if ns_match:
        ns = ns_match.group(1)

    def parse_style(style_str):
        props = {}
        for part in style_str.split(";"):
            part = part.strip()
            if ":" in part:
                k, v = part.split(":", 1)
                props[k.strip()] = v.strip()
        return props

    def get_props(elem):
        """Get meaningful presentation props, stripping defaults."""
        props = {}
        # From direct attributes
        for attr in PRESENTATION_ATTRS:
            val = elem.get(attr)
            if val is not None:
                props[attr] = val
        # From style= (overrides direct attrs)
        style = elem.get("style")
        if style:
            for k, v in parse_style(style).items():
                if k in PRESENTATION_ATTRS:
                    props[k] = v
        # Strip defaults
        cleaned = {}
        for k, v in props.items():
            defaults = DEFAULTS.get(k)
            if defaults and v in defaults:
                continue
            cleaned[k] = v
        return cleaned

    # Collect all (property, value) pairs and count occurrences
    elem_list = list(root.iter())
    pair_counts = Counter()  # (prop, val) -> count
    elem_props = []  # parallel list of props per element

    for elem in elem_list:
        props = get_props(elem)
        elem_props.append(props)
        for k, v in props.items():
            pair_counts[(k, v)] += 1

    # Find pairs worth extracting: savings = (count - 1) * len("prop:val") - overhead
    # Overhead per class: ".X{prop:val}" in <style> = ~len + 5
    extractable = {}
    for (k, v), count in pair_counts.items():
        if count < min_occurrences:
            continue
        prop_str = f"{k}:{v}"
        # Savings: remove from N elements, add 1 CSS rule + class refs
        savings = count * (len(prop_str) + 2) - (len(prop_str) + 5) - count * 2
        if savings > 0:
            extractable[(k, v)] = (count, savings)

    if not extractable:
        return svg_text

    # Group extractable pairs that always co-occur into compound classes
    # First, build element signatures (which extractable pairs each element has)
    pair_to_elems = {}
    for i, props in enumerate(elem_props):
        for k, v in props.items():
            if (k, v) in extractable:
                pair_to_elems.setdefault((k, v), set()).add(i)

    # Find groups of pairs that share the exact same element set
    sig_to_pairs = {}
    for pair, elems in pair_to_elems.items():
        sig = frozenset(elems)
        sig_to_pairs.setdefault(sig, []).append(pair)

    # Create classes for each group
    def gen_class_names():
        chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
        for c in chars:
            yield c
        for c1 in chars:
            for c2 in chars:
                yield c1 + c2

    name_gen = gen_class_names()
    css_rules = []
    # elem_index -> list of class names
    elem_classes = {}

    # Sort by total savings (most impactful first)
    sorted_sigs = sorted(
        sig_to_pairs.items(),
        key=lambda x: sum(extractable[p][1] for p in x[1]),
        reverse=True,
    )

    extracted_pairs = set()
    for sig, pairs in sorted_sigs:
        # Only include pairs not yet extracted
        new_pairs = [p for p in pairs if p not in extracted_pairs]
        if not new_pairs:
            continue

        # Check total savings for this class
        count = len(sig)
        props_str = ";".join(f"{k}:{v}" for k, v in sorted(new_pairs))
        overhead = len(props_str) + 5 + count * 2  # .X{...} + class="X" refs
        inline_cost = count * sum(len(f"{k}:{v}") + 2 for k, v in new_pairs)
        if inline_cost <= overhead:
            continue

        cls_name = next(name_gen)
        css_rules.append(f".{cls_name}{{{props_str}}}")

        for elem_idx in sig:
            elem_classes.setdefault(elem_idx, []).append(cls_name)

        extracted_pairs.update(new_pairs)

    if not css_rules:
        return svg_text

    # Apply: remove extracted props from elements, add class attrs
    modified = False
    for i, elem in enumerate(elem_list):
        if i not in elem_classes:
            continue

        props = elem_props[i]
        # Remove extracted props from direct attributes
        for k, v in list(props.items()):
            if (k, v) in extracted_pairs:
                if k in elem.attrib:
                    del elem.attrib[k]

        # Remove extracted props from style= attribute
        style = elem.get("style")
        if style:
            remaining = []
            for part in style.split(";"):
                part = part.strip()
                if ":" in part:
                    k = part.split(":", 1)[0].strip()
                    v = part.split(":", 1)[1].strip()
                    if (k, v) not in extracted_pairs:
                        remaining.append(part)
                elif part:
                    remaining.append(part)
            if remaining:
                elem.set("style", ";".join(remaining))
            else:
                if "style" in elem.attrib:
                    del elem.attrib["style"]

        # Also strip default values from remaining style
        style2 = elem.get("style")
        if style2:
            remaining2 = []
            for part in style2.split(";"):
                part = part.strip()
                if ":" in part:
                    k = part.split(":", 1)[0].strip()
                    v = part.split(":", 1)[1].strip()
                    defaults = DEFAULTS.get(k)
                    if defaults and v in defaults:
                        continue
                    remaining2.append(part)
                elif part:
                    remaining2.append(part)
            if remaining2:
                elem.set("style", ";".join(remaining2))
            else:
                if "style" in elem.attrib:
                    del elem.attrib["style"]

        # Add class names
        existing_class = elem.get("class", "")
        new_classes = " ".join(elem_classes[i])
        if existing_class:
            elem.set("class", f"{existing_class} {new_classes}")
        else:
            elem.set("class", new_classes)
        modified = True

    if not modified:
        return svg_text

    ET.register_namespace("", ns) if ns else None
    result = ET.tostring(root, encoding="unicode")

    # Inject <style> block right after <svg ...>
    style_block = "<style><![CDATA[" + "".join(css_rules) + "]]></style>"
    svg_end = result.find(">")
    if svg_end != -1:
        result = result[:svg_end + 1] + style_block + result[svg_end + 1:]

    return result


def merge_text_elements(svg_text: str) -> str:
    """Merge consecutive <text> elements into fewer elements with <tspan> children.

    Finds runs of consecutive <text> siblings, computes their shared
    (intersection) attributes, and merges them into a single <text> with
    shared attrs on the parent and per-element attrs on <tspan>s.
    A run of N texts must share at least 2 attributes to be worth merging.

    Returns:
        SVG string with merged text elements.
    """
    import xml.etree.ElementTree as ET

    try:
        root = ET.fromstring(svg_text)
    except ET.ParseError:
        return svg_text

    ns = ""
    ns_match = re.match(r'\{([^}]+)\}', root.tag)
    if ns_match:
        ns = ns_match.group(1)
    ns_prefix = f"{{{ns}}}" if ns else ""

    # Attrs that can be hoisted to parent <text>
    HOISTABLE = {
        "font-family", "font-size", "font-style", "font-weight",
        "text-anchor", "text-decoration", "dominant-baseline",
        "alignment-baseline", "letter-spacing", "fill", "stroke",
        "stroke-width", "class", "clip-path", "opacity",
    }

    # Attrs that stay on tspan (per-element positioning)
    POSITION_ATTRS = {
        "x", "y", "dx", "dy", "textLength", "lengthAdjust",
        "rotate", "transform",
    }

    def get_hoistable(elem):
        """Get hoistable attrs as a dict."""
        d = {}
        for attr in elem.attrib:
            clean = attr.split("}")[-1] if "}" in attr else attr
            if clean in HOISTABLE:
                d[clean] = elem.get(attr)
        return d

    def find_common(group):
        """Find attrs with identical values across all elements in group."""
        if not group:
            return {}
        common = get_hoistable(group[0]).copy()
        for elem in group[1:]:
            h = get_hoistable(elem)
            common = {k: v for k, v in common.items() if h.get(k) == v}
        return common

    modified = False

    def process_parent(parent):
        nonlocal modified
        children = list(parent)

        # Recurse into non-text children first
        for child in children:
            tag = child.tag.split("}")[-1] if "}" in child.tag else child.tag
            if tag != "text":
                process_parent(child)

        # Collect runs of consecutive <text> elements
        runs = []
        current_run = []
        for child in children:
            tag = child.tag.split("}")[-1] if "}" in child.tag else child.tag
            if tag == "text":
                current_run.append(child)
            else:
                if len(current_run) >= 2:
                    runs.append(current_run)
                current_run = []
        if len(current_run) >= 2:
            runs.append(current_run)

        for run in runs:
            common = find_common(run)
            # Need at least 2 shared attrs for savings to be worth it
            if len(common) < 2:
                continue

            # Estimate savings: N elements * len(common attrs) - 1 parent copy
            attr_bytes = sum(len(k) + len(v) + 4 for k, v in common.items())
            savings = attr_bytes * (len(run) - 1)
            # Overhead: <tspan> tags per element
            overhead = len(run) * 15  # rough <tspan ...></tspan>
            if savings <= overhead:
                continue

            # Find insert position (before the first text in the run)
            insert_idx = list(parent).index(run[0])

            # Create merged <text> with common attrs
            text_tag = f"{ns_prefix}text" if ns_prefix else "text"
            tspan_tag = f"{ns_prefix}tspan" if ns_prefix else "tspan"
            merged = ET.Element(text_tag)
            for attr, val in sorted(common.items()):
                merged.set(attr, val)

            for text_elem in run:
                # If text_elem already has tspan children, copy them directly
                sub_elems = list(text_elem)
                has_tspans = any(
                    (s.tag.split("}")[-1] if "}" in s.tag else s.tag) == "tspan"
                    for s in sub_elems
                )

                if has_tspans:
                    # Copy existing tspans directly
                    for sub in sub_elems:
                        merged.append(sub)
                else:
                    # Create a tspan for this text element
                    tspan = ET.SubElement(merged, tspan_tag)

                    # Copy non-common and position attrs to tspan
                    for attr, val in text_elem.attrib.items():
                        clean = attr.split("}")[-1] if "}" in attr else attr
                        if clean in common:
                            continue  # hoisted to parent
                        if clean in HOISTABLE or clean in POSITION_ATTRS:
                            tspan.set(attr, val)

                    # Copy text content
                    if text_elem.text:
                        tspan.text = text_elem.text

                # Remove original
                parent.remove(text_elem)

            # Insert merged element at the original position
            parent.insert(insert_idx, merged)
            modified = True

    process_parent(root)

    if not modified:
        return svg_text

    ET.register_namespace("", ns) if ns else None
    return ET.tostring(root, encoding="unicode")


def deduplicate_paths(svg_text: str) -> str:
    """Replace duplicate path elements with <defs>/<use> references.

    Groups paths by identical d= data (ignoring style differences).
    Puts one unstyled copy in <defs> and replaces all instances with
    <use> elements that carry per-instance style attributes.
    This handles the common SVG pattern of drawing a shape filled then
    drawing its outline with the same path data.

    Returns:
        SVG string with deduplicated paths using <defs>/<use>.
    """
    import xml.etree.ElementTree as ET
    from collections import defaultdict

    try:
        root = ET.fromstring(svg_text)
    except ET.ParseError:
        return svg_text

    ns = ""
    ns_match = re.match(r'\{([^}]+)\}', root.tag)
    if ns_match:
        ns = ns_match.group(1)
    ns_prefix = f"{{{ns}}}" if ns else ""

    path_tag = f"{ns_prefix}path" if ns_prefix else "path"
    defs_tag = f"{ns_prefix}defs" if ns_prefix else "defs"
    use_tag = f"{ns_prefix}use" if ns_prefix else "use"

    # Collect all path elements with parent refs
    path_info = []  # (elem, parent, d_value)

    def collect_paths(parent):
        for elem in list(parent):
            if elem.tag == path_tag:
                d = elem.get("d")
                if d and len(d) >= 20:
                    path_info.append((elem, parent, d))
            collect_paths(elem)

    collect_paths(root)

    if len(path_info) < 2:
        return svg_text

    # Group paths by d= value only (style may differ)
    d_groups = defaultdict(list)
    for elem, parent, d in path_info:
        d_groups[d].append((elem, parent))

    # Filter to groups with duplicates
    dup_groups = {d: elems for d, elems in d_groups.items() if len(elems) >= 2}

    if not dup_groups:
        return svg_text

    # Estimate savings
    total_savings = 0
    for d, elems in dup_groups.items():
        n_dups = len(elems) - 1
        # Each dup saves: len(d) + 4 (d="...") minus use_cost (~20 bytes)
        per_dup_savings = len(d) + 4 - 20
        if per_dup_savings > 0:
            total_savings += n_dups * per_dup_savings

    if total_savings < 50:
        return svg_text

    # Find or create <defs>
    defs = root.find(defs_tag)
    if defs is None:
        defs = ET.Element(defs_tag)
        root.insert(0, defs)

    # Generate short IDs
    def gen_ids():
        chars = "abcdefghijklmnopqrstuvwxyz"
        for c in chars:
            yield f"u{c}"
        for c1 in chars:
            for c2 in chars:
                yield f"u{c1}{c2}"

    id_gen = gen_ids()
    modified = False

    for d, elems in sorted(dup_groups.items(), key=lambda x: -len(x[1]) * len(x[0])):
        # Skip if savings not worth it for this group
        per_dup = len(d) + 4 - 20
        if per_dup <= 0:
            continue

        ref_id = next(id_gen)

        # Create unstyled path in <defs> with just d=
        def_path = ET.SubElement(defs, path_tag)
        def_path.set("id", ref_id)
        def_path.set("d", d)

        # Replace each instance with <use> keeping its own style attrs
        for elem, parent in elems:
            idx = list(parent).index(elem)
            parent.remove(elem)

            use = ET.Element(use_tag)
            use.set("href", f"#{ref_id}")

            # Copy all attributes except d and id to the <use>
            for k, v in elem.attrib.items():
                clean = k.split("}")[-1] if "}" in k else k
                if clean not in ("d", "id"):
                    use.set(k, v)

            parent.insert(idx, use)
            modified = True

    if not modified:
        return svg_text

    ET.register_namespace("", ns) if ns else None
    return ET.tostring(root, encoding="unicode")


# ---------------------------------------------------------------------------
# New tools: closing gaps vs SVGO
# ---------------------------------------------------------------------------


def remove_junk_attrs(svg_text: str) -> str:
    """Remove useless SVG attributes and elements that have no visual effect.

    Strips attributes like version="1.1", xmlns:xlink, x="0px", y="0px",
    enable-background, xml:space="preserve", empty <defs></defs>, and
    attributes with value "null" (invalid, no-op).
    """
    # Remove version="1.1" or version="1.0" from <svg> tag
    result = re.sub(r'(<svg[^>]*?)\s+version="1\.[01]"', r'\1', svg_text)

    # Remove xmlns:xlink only if xlink: isn't actually used
    if 'xlink:' not in re.sub(r'xmlns:xlink\s*=\s*"[^"]*"', '', result):
        result = re.sub(r'\s+xmlns:xlink\s*=\s*"[^"]*"', '', result)

    # Remove x="0px" y="0px" or x="0" y="0" on <svg>
    result = re.sub(r'(<svg[^>]*?)\s+x="0(?:px)?"', r'\1', result)
    result = re.sub(r'(<svg[^>]*?)\s+y="0(?:px)?"', r'\1', result)

    # Remove enable-background (deprecated)
    result = re.sub(r'\s+enable-background="[^"]*"', '', result)

    # Remove xml:space="preserve" if no <text> elements
    if not re.search(r'<text\b', result):
        result = re.sub(r'\s+xml:space="preserve"', '', result)

    # Remove empty <defs></defs> or <defs/>
    result = re.sub(r'\s*<defs\s*/>\s*', '', result)
    result = re.sub(r'\s*<defs\s*>\s*</defs>\s*', '', result)

    # Remove <description>...</description> (Sketch artifact)
    result = re.sub(r'\s*<description[^>]*>.*?</description>\s*', '', result, flags=re.DOTALL)

    # Remove attributes with value "null"
    result = re.sub(r'\s+\w+="null"', '', result)

    # Remove id on root <svg> element
    result = re.sub(r'(<svg[^>]*?)\s+id="[^"]*"', r'\1', result)

    # Remove empty style attributes
    result = re.sub(r'\s+style=""', '', result)

    # Remove editor namespace declarations (inkscape, sodipodi, sketch, etc.)
    for ns in ('inkscape', 'sodipodi', 'sketch', 'dc', 'cc', 'rdf', 'ns1', 'ns2'):
        # Only remove xmlns:ns if ns: isn't used as attribute prefix or element prefix
        cleaned = re.sub(r'\s+xmlns:' + ns + r'\s*=\s*"[^"]*"', '', result)
        # Check if ns: is still referenced anywhere (as attr or element)
        if not re.search(ns + r':', cleaned):
            result = cleaned

    # Remove editor-specific attributes (inkscape:*, sodipodi:*, sketch:*)
    result = re.sub(r'\s+(?:inkscape|sodipodi|sketch):\w+="[^"]*"', '', result)

    # Remove editor-specific elements (sodipodi:namedview, etc.)
    result = re.sub(r'\s*<sodipodi:\w+[^>]*/>\s*', '', result)
    result = re.sub(r'\s*<sodipodi:\w+[^>]*>.*?</sodipodi:\w+>\s*', '', result, flags=re.DOTALL)

    # Remove RDF metadata blocks
    result = re.sub(r'\s*<metadata[^>]*>.*?</metadata>\s*', '', result, flags=re.DOTALL)

    # Remove empty <defs> with only id attribute
    result = re.sub(r'\s*<defs\s+id="[^"]*"\s*/>\s*', '', result)
    result = re.sub(r'\s*<defs\s+id="[^"]*"\s*>\s*</defs>\s*', '', result)

    return result


def remove_unused_ids(svg_text: str) -> str:
    """Remove id attributes that are not referenced anywhere in the SVG.

    Keeps ids that are referenced by url(#id), href="#id", xlink:href="#id",
    or begin/end animation references.
    """
    # Find all id values
    all_ids = re.findall(r'\bid="([^"]*)"', svg_text)
    if not all_ids:
        return svg_text

    result = svg_text
    for id_val in all_ids:
        # Check if this id is referenced anywhere (url(#id), href="#id", etc.)
        escaped = re.escape(id_val)
        refs = [
            r'url\(\s*#' + escaped + r'\s*\)',       # url(#id)
            r'href\s*=\s*"#' + escaped + r'"',        # href="#id"
            r'begin\s*=\s*"' + escaped,                # animation begin
            r'end\s*=\s*"' + escaped,                  # animation end
        ]
        referenced = any(re.search(pat, result) for pat in refs)
        if not referenced:
            # Remove this id attribute
            result = re.sub(r'\s+id="' + escaped + r'"', '', result)

    return result


def simplify_transforms(svg_text: str) -> str:
    """Decompose matrix() transforms into simpler translate/scale/rotate forms.

    A matrix(a,b,c,d,e,f) that is pure translation becomes translate(e,f).
    Uniform scaling+translation becomes translate(...)scale(s).
    Values are rounded to 3 decimal places.
    """
    def _simplify_matrix(m):
        try:
            parts = [float(x) for x in re.split(r'[\s,]+', m.strip())]
        except ValueError:
            return None
        if len(parts) != 6:
            return None
        a, b, c, d, e, f = parts

        def fmt(v):
            s = f"{v:.3f}".rstrip('0').rstrip('.')
            return '0' if s == '-0' else s

        # Pure translation
        if abs(a - 1) < 1e-9 and abs(b) < 1e-9 and abs(c) < 1e-9 and abs(d - 1) < 1e-9:
            if abs(e) < 1e-9 and abs(f) < 1e-9:
                return ""
            return f"translate({fmt(e)} {fmt(f)})"

        # Uniform scale + translation
        if abs(b) < 1e-9 and abs(c) < 1e-9 and abs(a - d) < 1e-9:
            r = []
            if abs(e) > 1e-9 or abs(f) > 1e-9:
                r.append(f"translate({fmt(e)} {fmt(f)})")
            if abs(a - 1) > 1e-9:
                r.append(f"scale({fmt(a)})")
            return "".join(r) if r else ""

        # Non-uniform scale + translation
        if abs(b) < 1e-9 and abs(c) < 1e-9:
            r = []
            if abs(e) > 1e-9 or abs(f) > 1e-9:
                r.append(f"translate({fmt(e)} {fmt(f)})")
            r.append(f"scale({fmt(a)} {fmt(d)})")
            return "".join(r) if r else ""

        # Rotation
        if abs(a - d) < 1e-9 and abs(b + c) < 1e-9:
            angle = math.degrees(math.atan2(b, a))
            r = []
            if abs(e) > 1e-9 or abs(f) > 1e-9:
                r.append(f"translate({fmt(e)} {fmt(f)})")
            r.append(f"rotate({fmt(angle)})")
            return "".join(r) if r else ""

        # Can't simplify — round values
        rounded = " ".join(fmt(x) for x in [a, b, c, d, e, f])
        return f"matrix({rounded})"

    def _replace_matrix(match):
        simplified = _simplify_matrix(match.group(1))
        if simplified is None:
            return match.group(0)
        if simplified == "":
            return ""
        return f'transform="{simplified}"'

    result = re.sub(
        r'transform\s*=\s*"matrix\(([^)]+)\)"',
        _replace_matrix, svg_text
    )

    # Round values in existing translate/scale/rotate
    def _round_transform(match):
        func, args = match.group(1), match.group(2)
        try:
            rounded = []
            for n in re.split(r'[\s,]+', args.strip()):
                v = float(n)
                s = f"{v:.3f}".rstrip('0').rstrip('.')
                rounded.append('0' if s == '-0' else s)
            return f'{func}({" ".join(rounded)})'
        except ValueError:
            return match.group(0)

    return re.sub(r'(translate|scale|rotate)\(([^)]+)\)', _round_transform, result)


def _cubic_is_arc(dx1, dy1, dx2, dy2, dx, dy, tolerance=0.02):
    """Check if a relative cubic bezier approximates a circular arc."""
    KAPPA = 0.5522847498
    end_len = math.sqrt(dx * dx + dy * dy)
    if end_len < 0.1:
        return (False,)

    # Pattern A: horizontal start tangent
    err_a = (abs(dx1 - KAPPA * dx) + abs(dy1) +
             abs(dx2 - dx) + abs(dy2 - (1 - KAPPA) * dy))
    if err_a / end_len < tolerance:
        rx, ry = abs(dx), abs(dy)
        if rx < 0.1 or ry < 0.1:
            return (False,)
        sweep = 1 if (dx > 0) == (dy > 0) else 0
        return (True, rx, ry, 0, sweep, dx, dy)

    # Pattern B: vertical start tangent
    err_b = (abs(dx1) + abs(dy1 - KAPPA * dy) +
             abs(dx2 - (1 - KAPPA) * dx) + abs(dy2 - dy))
    if err_b / end_len < tolerance:
        rx, ry = abs(dx), abs(dy)
        if rx < 0.1 or ry < 0.1:
            return (False,)
        sweep = 1 if (dx > 0) != (dy > 0) else 0
        return (True, rx, ry, 0, sweep, dx, dy)

    return (False,)


def cubic_to_arc(svg_text: str, tolerance: float = 0.02, precision: int = 2) -> str:
    """Convert cubic bezier curves that approximate arcs to arc commands.

    A quarter-circle arc `a r r 0 0 1 dx dy` is much shorter than the
    cubic bezier equivalent. This is one of SVGO's biggest wins on
    rounded-corner SVGs (buttons, icons with rounded rects).
    """
    d_pattern = re.compile(r'(\sd=")([^"]+)(")')

    def _process_path(match):
        prefix, d_str, suffix = match.group(1), match.group(2), match.group(3)
        try:
            commands = parse_path(d_str)
            abs_cmds = to_absolute(commands)
        except Exception:
            return match.group(0)

        new_commands = []
        changed = False
        cx, cy = 0.0, 0.0

        for cmd, args in abs_cmds:
            if cmd == 'C' and len(args) == 6:
                dx1 = args[0] - cx
                dy1 = args[1] - cy
                dx2 = args[2] - cx
                dy2 = args[3] - cy
                ddx = args[4] - cx
                ddy = args[5] - cy

                arc = _cubic_is_arc(dx1, dy1, dx2, dy2, ddx, ddy, tolerance)
                if arc[0]:
                    _, rx, ry, large, sweep, _, _ = arc
                    new_commands.append(('A', [rx, ry, 0, large, sweep, args[4], args[5]]))
                    cx, cy = args[4], args[5]
                    changed = True
                    continue

            new_commands.append((cmd, list(args)))
            if cmd.upper() in ('M', 'L', 'C', 'S', 'Q', 'T', 'A') and len(args) >= 2:
                cx, cy = args[-2], args[-1]
            elif cmd.upper() == 'H' and args:
                cx = args[0]
            elif cmd.upper() == 'V' and args:
                cy = args[0]
            elif cmd.upper() == 'Z':
                for pc, pa in new_commands:
                    if pc.upper() == 'M':
                        cx, cy = pa[-2], pa[-1]

        if not changed:
            return match.group(0)
        return f'{prefix}{format_path(new_commands, precision=precision)}{suffix}'

    return d_pattern.sub(_process_path, svg_text)


# ---------------------------------------------------------------------------
# Path → Shape reconstruction
# ---------------------------------------------------------------------------

def _near(a, b, tol=0.5):
    """Check if two values are within tolerance."""
    return abs(a - b) <= tol


def _detect_circle(abs_cmds, tol=0.5):
    """Detect if absolute commands form a circle.

    Returns (cx, cy, r) or None.
    """
    # Strip Z at end
    cmds = [(c, a) for c, a in abs_cmds if c != 'Z']
    if len(cmds) < 2 or cmds[0][0] != 'M':
        return None
    sx, sy = cmds[0][1][0], cmds[0][1][1]

    # Case A: M + 2 arcs (two semicircles)
    if len(cmds) == 3 and cmds[1][0] == 'A' and cmds[2][0] == 'A':
        a1, a2 = cmds[1][1], cmds[2][1]
        rx1, ry1 = a1[0], a1[1]
        rx2, ry2 = a2[0], a2[1]
        ex1, ey1 = a1[5], a1[6]  # end of first arc
        ex2, ey2 = a2[5], a2[6]  # end of second arc
        # Both arcs must have rx == ry (circular) and same radius
        if not (_near(rx1, ry1, tol) and _near(rx2, ry2, tol) and _near(rx1, rx2, tol)):
            return None
        # Path must close
        if not (_near(ex2, sx, tol) and _near(ey2, sy, tol)):
            return None
        r = (rx1 + ry1 + rx2 + ry2) / 4
        cx = (sx + ex1) / 2
        cy = (sy + ey1) / 2
        # Verify: distance from center to start should be ~r
        dist = math.sqrt((sx - cx) ** 2 + (sy - cy) ** 2)
        if not _near(dist, r, tol):
            return None
        return (cx, cy, r)

    # Case B: M + 4 arcs (four quarter-circles)
    if len(cmds) == 5 and all(cmds[i][0] == 'A' for i in range(1, 5)):
        arcs = [cmds[i][1] for i in range(1, 5)]
        # All must be circular with same radius
        radii = []
        for a in arcs:
            if not _near(a[0], a[1], tol):
                return None
            radii.append((a[0] + a[1]) / 2)
        r = sum(radii) / 4
        if any(not _near(ri, r, tol) for ri in radii):
            return None
        # Path must close
        ex, ey = arcs[-1][5], arcs[-1][6]
        if not (_near(ex, sx, tol) and _near(ey, sy, tol)):
            return None
        # Collect all points on the circle
        pts = [(sx, sy)]
        for a in arcs:
            pts.append((a[5], a[6]))
        # Center = average of extreme points
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        cx = (min(xs) + max(xs)) / 2
        cy = (min(ys) + max(ys)) / 2
        # Verify all points are ~r from center
        for px, py in pts[:-1]:  # skip last (same as first)
            dist = math.sqrt((px - cx) ** 2 + (py - cy) ** 2)
            if not _near(dist, r, tol):
                return None
        return (cx, cy, r)

    # Case C: M + 4 cubic beziers (approximate quarter-circle arcs)
    if len(cmds) == 5 and all(cmds[i][0] == 'C' for i in range(1, 5)):
        cx_cur, cy_cur = sx, sy
        arc_radii = []
        pts = [(sx, sy)]
        for i in range(1, 5):
            args = cmds[i][1]
            dx1 = args[0] - cx_cur
            dy1 = args[1] - cy_cur
            dx2 = args[2] - cx_cur
            dy2 = args[3] - cy_cur
            ddx = args[4] - cx_cur
            ddy = args[5] - cy_cur
            arc = _cubic_is_arc(dx1, dy1, dx2, dy2, ddx, ddy, tolerance=0.05)
            if not arc[0]:
                return None
            _, rx, ry, _, _, _, _ = arc
            if not _near(rx, ry, tol):
                return None
            arc_radii.append((rx + ry) / 2)
            cx_cur, cy_cur = args[4], args[5]
            pts.append((cx_cur, cy_cur))
        r = sum(arc_radii) / 4
        if any(not _near(ri, r, tol) for ri in arc_radii):
            return None
        # Path must close
        if not (_near(pts[-1][0], sx, tol) and _near(pts[-1][1], sy, tol)):
            return None
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        center_x = (min(xs) + max(xs)) / 2
        center_y = (min(ys) + max(ys)) / 2
        for px, py in pts[:-1]:
            dist = math.sqrt((px - center_x) ** 2 + (py - center_y) ** 2)
            if not _near(dist, r, tol):
                return None
        return (center_x, center_y, r)

    return None


def _detect_ellipse(abs_cmds, tol=0.5):
    """Detect if absolute commands form an ellipse (rx != ry).

    Returns (cx, cy, rx, ry) or None.
    """
    cmds = [(c, a) for c, a in abs_cmds if c != 'Z']
    if len(cmds) < 2 or cmds[0][0] != 'M':
        return None
    sx, sy = cmds[0][1][0], cmds[0][1][1]

    # Case A: M + 2 arcs
    if len(cmds) == 3 and cmds[1][0] == 'A' and cmds[2][0] == 'A':
        a1, a2 = cmds[1][1], cmds[2][1]
        rx1, ry1 = a1[0], a1[1]
        rx2, ry2 = a2[0], a2[1]
        ex1, ey1 = a1[5], a1[6]
        ex2, ey2 = a2[5], a2[6]
        if not (_near(rx1, rx2, tol) and _near(ry1, ry2, tol)):
            return None
        if not (_near(ex2, sx, tol) and _near(ey2, sy, tol)):
            return None
        rx = (rx1 + rx2) / 2
        ry = (ry1 + ry2) / 2
        # Skip if it's actually a circle
        if _near(rx, ry, tol):
            return None
        cx = (sx + ex1) / 2
        cy = (sy + ey1) / 2
        return (cx, cy, rx, ry)

    # Case B: M + 4 arcs
    if len(cmds) == 5 and all(cmds[i][0] == 'A' for i in range(1, 5)):
        arcs = [cmds[i][1] for i in range(1, 5)]
        rx_vals = [a[0] for a in arcs]
        ry_vals = [a[1] for a in arcs]
        rx = sum(rx_vals) / 4
        ry = sum(ry_vals) / 4
        if any(not _near(v, rx, tol) for v in rx_vals):
            return None
        if any(not _near(v, ry, tol) for v in ry_vals):
            return None
        if _near(rx, ry, tol):
            return None  # It's a circle
        ex, ey = arcs[-1][5], arcs[-1][6]
        if not (_near(ex, sx, tol) and _near(ey, sy, tol)):
            return None
        pts = [(sx, sy)] + [(a[5], a[6]) for a in arcs]
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        cx = (min(xs) + max(xs)) / 2
        cy = (min(ys) + max(ys)) / 2
        return (cx, cy, rx, ry)

    return None


def _detect_rect(abs_cmds, tol=0.5):
    """Detect if absolute commands form an axis-aligned rectangle.

    Returns (x, y, width, height) or None.
    """
    cmds = [(c, a) for c, a in abs_cmds if c != 'Z']
    if len(cmds) < 2 or cmds[0][0] != 'M':
        return None
    sx, sy = cmds[0][1][0], cmds[0][1][1]

    # Collect all corner points from L/H/V commands
    pts = [(sx, sy)]
    cx_cur, cy_cur = sx, sy
    line_cmds = cmds[1:]

    # Must have exactly 3 or 4 line-type commands
    if len(line_cmds) not in (3, 4):
        return None

    for cmd, args in line_cmds:
        if cmd == 'L' and len(args) == 2:
            cx_cur, cy_cur = args[0], args[1]
        elif cmd == 'H' and len(args) == 1:
            cx_cur = args[0]
        elif cmd == 'V' and len(args) == 1:
            cy_cur = args[0]
        else:
            return None  # Non-line command
        pts.append((cx_cur, cy_cur))

    # If 4 line commands, last point must close to start
    if len(line_cmds) == 4:
        if not (_near(pts[-1][0], sx, tol) and _near(pts[-1][1], sy, tol)):
            return None
        pts = pts[:-1]  # Remove closing duplicate

    # Need exactly 4 corners
    if len(pts) != 4:
        return None

    # Check axis-aligned: each edge must be horizontal or vertical
    for i in range(4):
        p1 = pts[i]
        p2 = pts[(i + 1) % 4]
        if not (_near(p1[0], p2[0], tol) or _near(p1[1], p2[1], tol)):
            return None  # Diagonal edge

    # Extract bounding box
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    x_unique = sorted(set(round(v, 1) for v in xs))
    y_unique = sorted(set(round(v, 1) for v in ys))
    if len(x_unique) != 2 or len(y_unique) != 2:
        return None  # Not a proper rectangle

    x = min(xs)
    y = min(ys)
    w = max(xs) - x
    h = max(ys) - y
    if w < 0.01 or h < 0.01:
        return None
    return (x, y, w, h)


def _detect_rounded_rect(abs_cmds, tol=0.5):
    """Detect if absolute commands form a rounded rectangle.

    Expects pattern: M + alternating lines and arcs (4 of each) + Z.
    Returns (x, y, width, height, rx, ry) or None.
    """
    cmds = [(c, a) for c, a in abs_cmds if c != 'Z']
    if len(cmds) < 2 or cmds[0][0] != 'M':
        return None

    # A rounded rect has M + 8 commands (4 lines + 4 arcs, interleaved)
    body = cmds[1:]
    if len(body) != 8:
        return None

    # Separate lines and arcs, check alternating pattern
    # Could start with line or arc depending on start point
    lines = []
    arcs = []
    # Try pattern: L A L A L A L A
    is_la = all(body[i][0] in ('L', 'H', 'V') for i in range(0, 8, 2)) and \
            all(body[i][0] == 'A' for i in range(1, 8, 2))
    # Try pattern: A L A L A L A L
    is_al = all(body[i][0] == 'A' for i in range(0, 8, 2)) and \
            all(body[i][0] in ('L', 'H', 'V') for i in range(1, 8, 2))

    if not (is_la or is_al):
        return None

    if is_la:
        arcs = [body[i][1] for i in range(1, 8, 2)]
    else:
        arcs = [body[i][1] for i in range(0, 8, 2)]

    # All arcs must have same rx, ry
    rx_vals = [a[0] for a in arcs]
    ry_vals = [a[1] for a in arcs]
    rx = sum(rx_vals) / 4
    ry = sum(ry_vals) / 4
    if any(not _near(v, rx, tol) for v in rx_vals):
        return None
    if any(not _near(v, ry, tol) for v in ry_vals):
        return None
    if rx < 0.01 or ry < 0.01:
        return None

    # Trace all points to find bounding box
    sx, sy = cmds[0][1][0], cmds[0][1][1]
    pts = [(sx, sy)]
    cx_cur, cy_cur = sx, sy
    for cmd, args in body:
        if cmd == 'L':
            cx_cur, cy_cur = args[0], args[1]
        elif cmd == 'H':
            cx_cur = args[0]
        elif cmd == 'V':
            cy_cur = args[0]
        elif cmd == 'A':
            cx_cur, cy_cur = args[5], args[6]
        pts.append((cx_cur, cy_cur))

    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    x = min(xs)
    y = min(ys)
    w = max(xs) - x
    h = max(ys) - y
    if w < 0.01 or h < 0.01:
        return None

    # Verify rx/ry don't exceed half the dimensions
    if rx > w / 2 + tol or ry > h / 2 + tol:
        return None

    return (x, y, w, h, rx, ry)


def _fmt_attr(n):
    """Format a number for shape attributes (no trailing zeros)."""
    if n == int(n):
        return str(int(n))
    s = f"{n:.2f}".rstrip('0').rstrip('.')
    return '0' if s == '-0' else s


def paths_to_shapes(svg_text: str, tolerance: float = 0.5) -> str:
    """Convert path elements back to semantic shape elements where possible.

    Detects paths that represent circles, ellipses, rectangles, and
    rounded rectangles, converting them to their shorter semantic
    equivalents (<circle>, <ellipse>, <rect>).

    Only converts when the shape form is shorter than the path form.
    """
    path_pattern = re.compile(r'<path\s+([^>]*)d="([^"]*)"([^>]*)/?>')

    def _try_convert(match):
        before_d = match.group(1)
        d_str = match.group(2)
        after_d = match.group(3)

        try:
            cmds = parse_path(d_str)
            abs_cmds = to_absolute(cmds)
        except Exception:
            return match.group(0)

        # Skip multi-subpath paths
        m_count = sum(1 for cmd, _ in abs_cmds if cmd == 'M')
        if m_count != 1:
            return match.group(0)

        # Build non-d attribute string, strip trailing / from self-closing tag
        other = (before_d + ' ' + after_d).strip()
        other = other.rstrip('/')
        other = re.sub(r'\s+', ' ', other).strip()
        if other:
            other = ' ' + other

        original = match.group(0)

        # Try circle
        result = _detect_circle(abs_cmds, tolerance)
        if result:
            cx, cy, r = result
            candidate = f'<circle cx="{_fmt_attr(cx)}" cy="{_fmt_attr(cy)}" r="{_fmt_attr(r)}"{other}/>'
            if len(candidate) < len(original):
                return candidate

        # Try ellipse
        result = _detect_ellipse(abs_cmds, tolerance)
        if result:
            cx, cy, rx, ry = result
            candidate = f'<ellipse cx="{_fmt_attr(cx)}" cy="{_fmt_attr(cy)}" rx="{_fmt_attr(rx)}" ry="{_fmt_attr(ry)}"{other}/>'
            if len(candidate) < len(original):
                return candidate

        # Try rounded rect (before plain rect — more specific)
        result = _detect_rounded_rect(abs_cmds, tolerance)
        if result:
            x, y, w, h, rx, ry = result
            if _near(rx, ry, tolerance):
                candidate = f'<rect x="{_fmt_attr(x)}" y="{_fmt_attr(y)}" width="{_fmt_attr(w)}" height="{_fmt_attr(h)}" rx="{_fmt_attr(rx)}"{other}/>'
            else:
                candidate = f'<rect x="{_fmt_attr(x)}" y="{_fmt_attr(y)}" width="{_fmt_attr(w)}" height="{_fmt_attr(h)}" rx="{_fmt_attr(rx)}" ry="{_fmt_attr(ry)}"{other}/>'
            if len(candidate) < len(original):
                return candidate

        # Try plain rect
        result = _detect_rect(abs_cmds, tolerance)
        if result:
            x, y, w, h = result
            candidate = f'<rect x="{_fmt_attr(x)}" y="{_fmt_attr(y)}" width="{_fmt_attr(w)}" height="{_fmt_attr(h)}"{other}/>'
            if len(candidate) < len(original):
                return candidate

        return original

    return path_pattern.sub(_try_convert, svg_text)


def detect_shapes(svg_text: str, tolerance: float = 0.5) -> list[dict]:
    """Detect paths that could be converted to semantic shapes.

    Returns a list of dicts, one per detected shape, with:
      - shape: "circle", "ellipse", "rect", "rounded_rect"
      - params: dict of shape parameters (cx, cy, r, etc.)
      - path_bytes: size of original path element
      - shape_bytes: size of equivalent shape element
      - savings: bytes saved (negative means shape is larger)
      - d: the path d attribute that matched

    Use this to inspect what shapes exist before deciding to convert.
    """
    path_pattern = re.compile(r'<path\s+([^>]*)d="([^"]*)"([^>]*)/?>')
    results = []

    for match in path_pattern.finditer(svg_text):
        before_d = match.group(1)
        d_str = match.group(2)
        after_d = match.group(3)

        try:
            cmds = parse_path(d_str)
            abs_cmds = to_absolute(cmds)
        except Exception:
            continue

        m_count = sum(1 for cmd, _ in abs_cmds if cmd == 'M')
        if m_count != 1:
            continue

        other = (before_d + ' ' + after_d).strip().rstrip('/')
        other = re.sub(r'\s+', ' ', other).strip()
        if other:
            other = ' ' + other

        path_bytes = len(match.group(0))

        # Try each detector
        det = _detect_circle(abs_cmds, tolerance)
        if det:
            cx, cy, r = det
            shape_str = f'<circle cx="{_fmt_attr(cx)}" cy="{_fmt_attr(cy)}" r="{_fmt_attr(r)}"{other}/>'
            results.append({
                "shape": "circle", "params": {"cx": cx, "cy": cy, "r": r},
                "path_bytes": path_bytes, "shape_bytes": len(shape_str),
                "savings": path_bytes - len(shape_str), "d": d_str,
            })
            continue

        det = _detect_ellipse(abs_cmds, tolerance)
        if det:
            cx, cy, rx, ry = det
            shape_str = f'<ellipse cx="{_fmt_attr(cx)}" cy="{_fmt_attr(cy)}" rx="{_fmt_attr(rx)}" ry="{_fmt_attr(ry)}"{other}/>'
            results.append({
                "shape": "ellipse", "params": {"cx": cx, "cy": cy, "rx": rx, "ry": ry},
                "path_bytes": path_bytes, "shape_bytes": len(shape_str),
                "savings": path_bytes - len(shape_str), "d": d_str,
            })
            continue

        det = _detect_rounded_rect(abs_cmds, tolerance)
        if det:
            x, y, w, h, rx, ry = det
            if _near(rx, ry, tolerance):
                shape_str = f'<rect x="{_fmt_attr(x)}" y="{_fmt_attr(y)}" width="{_fmt_attr(w)}" height="{_fmt_attr(h)}" rx="{_fmt_attr(rx)}"{other}/>'
            else:
                shape_str = f'<rect x="{_fmt_attr(x)}" y="{_fmt_attr(y)}" width="{_fmt_attr(w)}" height="{_fmt_attr(h)}" rx="{_fmt_attr(rx)}" ry="{_fmt_attr(ry)}"{other}/>'
            results.append({
                "shape": "rounded_rect", "params": {"x": x, "y": y, "w": w, "h": h, "rx": rx, "ry": ry},
                "path_bytes": path_bytes, "shape_bytes": len(shape_str),
                "savings": path_bytes - len(shape_str), "d": d_str,
            })
            continue

        det = _detect_rect(abs_cmds, tolerance)
        if det:
            x, y, w, h = det
            shape_str = f'<rect x="{_fmt_attr(x)}" y="{_fmt_attr(y)}" width="{_fmt_attr(w)}" height="{_fmt_attr(h)}"{other}/>'
            results.append({
                "shape": "rect", "params": {"x": x, "y": y, "w": w, "h": h},
                "path_bytes": path_bytes, "shape_bytes": len(shape_str),
                "savings": path_bytes - len(shape_str), "d": d_str,
            })
            continue

    return results


def merge_subpaths(svg_text: str) -> str:
    """Merge consecutive <path> elements with identical attributes into one compound path.

    Paths sharing the same fill, stroke, etc. are combined into a single <path>
    with multiple subpaths. Saves repeated tags and duplicate attributes.
    """
    try:
        root = ET.fromstring(svg_text)
    except ET.ParseError:
        return svg_text

    ns_match = re.match(r'\{([^}]+)\}', root.tag)
    ns = ns_match.group(1) if ns_match else ""
    ns_prefix = f"{{{ns}}}" if ns else ""
    path_tag = f"{ns_prefix}path"

    def _style_key(elem):
        attrs = dict(elem.attrib)
        attrs.pop("d", None)
        attrs.pop("id", None)
        return tuple(sorted(attrs.items()))

    def _process(parent):
        changed = False
        children = list(parent)
        i = 0
        while i < len(children):
            child = children[i]
            if child.tag != path_tag:
                if _process(child):
                    changed = True
                i += 1
                continue
            d = child.get("d")
            if not d:
                i += 1
                continue
            key = _style_key(child)
            group = [(child, d)]
            j = i + 1
            while j < len(children):
                nc = children[j]
                if nc.tag != path_tag:
                    break
                nd = nc.get("d")
                if not nd or _style_key(nc) != key:
                    break
                group.append((nc, nd))
                j += 1
            if len(group) >= 2:
                combined = "".join(d for _, d in group)
                group[0][0].set("d", combined)
                for elem, _ in group[1:]:
                    parent.remove(elem)
                changed = True
                children = list(parent)
            else:
                i += 1
        return changed

    if _process(root):
        ET.register_namespace("", ns) if ns else None
        return ET.tostring(root, encoding="unicode")
    return svg_text


# ---------------------------------------------------------------------------
# SVGO integration: run SVGO as a subprocess tool
# ---------------------------------------------------------------------------

def run_svgo(svg_text: str) -> str:
    """Run SVGO on SVG text and return the optimized result.

    Uses the svgo binary with --multipass for maximum compression.
    Falls back to returning the original if svgo isn't available or fails.
    """
    import subprocess
    import shutil
    import tempfile
    import os

    svgo_bin = shutil.which("svgo")
    svgo_dir = None
    if not svgo_bin:
        # Try conda env path
        if os.path.exists("/opt/anaconda3/envs/svgym/bin/svgo"):
            svgo_bin = "/opt/anaconda3/envs/svgym/bin/svgo"
            svgo_dir = "/opt/anaconda3/envs/svgym/bin"
        else:
            return svg_text
    else:
        svgo_dir = os.path.dirname(svgo_bin)

    # Ensure node is in PATH (svgo needs it)
    env = os.environ.copy()
    if svgo_dir:
        env["PATH"] = svgo_dir + os.pathsep + env.get("PATH", "")

    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.svg', delete=False) as f:
            f.write(svg_text)
            tmp_in = f.name

        tmp_out = tmp_in + '.out.svg'
        result = subprocess.run(
            [svgo_bin, tmp_in, "-o", tmp_out, "--multipass"],
            capture_output=True, text=True, timeout=30, env=env,
        )

        if result.returncode == 0 and os.path.exists(tmp_out):
            optimized = open(tmp_out).read()
            os.unlink(tmp_in)
            os.unlink(tmp_out)
            return optimized
        else:
            os.unlink(tmp_in)
            if os.path.exists(tmp_out):
                os.unlink(tmp_out)
            return svg_text
    except Exception:
        return svg_text
