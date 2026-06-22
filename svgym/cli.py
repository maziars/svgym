"""SVGym CLI — optimize SVG packs from the command line.

Usage:
    svgym pack ./icons/ -o ./dist/
    svgym pack ./icons/ --level aggressive --no-sprite
    svgym pack ./icons/ --json report.json

Folder structure handling:
    Flat:    icons/*.svg        → treated as one pack
    Nested:  icons/outline/*.svg
             icons/solid/*.svg  → each subfolder is a separate pack
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import os

sys.path.insert(0, str(Path(__file__).parent.parent))

# NOTE: heavy modules (svgym.pack / svgym.config / svgym.optimizer) are imported
# lazily inside the command functions, so that --provider/--model can set the
# relevant environment variables BEFORE svgym.config reads them at import time.


def discover_packs(input_dir: Path) -> dict[str, list[tuple[str, str]]]:
    """Discover SVG packs from a directory.

    Returns a dict of pack_name -> [(filename, svg_text), ...].

    Rules:
    - If input_dir contains SVGs directly, it's one flat pack.
    - If input_dir has subdirectories with SVGs, each subdir is a pack.
    - If both exist, root SVGs are one pack + each subdir is a pack.
    """
    packs = {}

    # Root-level SVGs
    root_svgs = sorted(input_dir.glob("*.svg"))
    if root_svgs:
        packs[input_dir.name] = [(f.name, f.read_text()) for f in root_svgs]

    # Subdirectory SVGs (one level deep)
    for subdir in sorted(input_dir.iterdir()):
        if subdir.is_dir():
            sub_svgs = sorted(subdir.glob("*.svg"))
            if sub_svgs:
                packs[subdir.name] = [(f.name, f.read_text()) for f in sub_svgs]

    return packs


def run_pack(args: argparse.Namespace) -> int:
    """Run the pack optimization command."""
    from svgym.pack import (
        optimize_pack,
        generate_sprite,
        generate_report,
        format_report_markdown,
    )
    input_dir = Path(args.input).resolve()
    if not input_dir.is_dir():
        print(f"Error: {input_dir} is not a directory", file=sys.stderr)
        return 1

    packs = discover_packs(input_dir)
    if not packs:
        print(f"Error: no SVG files found in {input_dir}", file=sys.stderr)
        return 1

    total_files = sum(len(files) for files in packs.values())
    print(f"Found {len(packs)} pack(s), {total_files} SVGs total")
    for name, files in packs.items():
        print(f"  {name}/: {len(files)} files")
    print()

    output_dir = Path(args.output).resolve() if args.output else None
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    all_reports = {}
    t0 = time.time()

    for pack_name, svg_files in packs.items():
        print(f"Optimizing {pack_name}/ ({len(svg_files)} files)...")

        def on_progress(filename, i, total):
            if not args.quiet:
                pct = int((i / total) * 100)
                print(f"  [{pct:3d}%] {filename}", end="\r", flush=True)

        result = optimize_pack(
            svg_files,
            level=args.level,
            pack_name=pack_name,
            on_progress=None if args.quiet else on_progress,
        )

        if not args.quiet:
            print(f"  [100%] done" + " " * 40)

        # Generate sprite
        sprite_svg = None
        if not args.no_sprite:
            optimized = [(r["filename"], r["optimized_svg"]) for r in result["results"]]
            sprite_svg = generate_sprite(optimized, result["analysis"])

        # Generate report
        report = generate_report(result, sprite_svg=sprite_svg)
        all_reports[pack_name] = report

        # Write output files
        if output_dir:
            pack_out = output_dir / pack_name if len(packs) > 1 else output_dir
            pack_out.mkdir(parents=True, exist_ok=True)

            # Write optimized SVGs
            for r in result["results"]:
                (pack_out / r["filename"]).write_text(r["optimized_svg"])

            # Write sprite
            if sprite_svg:
                sprite_path = pack_out / f"{pack_name}-sprite.svg"
                sprite_path.write_text(sprite_svg)
                if not args.quiet:
                    print(f"  Sprite: {sprite_path}")

        # Print report
        if not args.json_output:
            md = format_report_markdown(report)
            print(md)
            print()

    elapsed = time.time() - t0

    # JSON output
    if args.json_output:
        json_path = Path(args.json_output)
        json_path.write_text(json.dumps(all_reports, indent=2))
        print(f"Report written to {json_path}")

    # Summary
    if len(packs) > 1 and not args.quiet:
        total_in = sum(r["summary"]["total_input_size"] for r in all_reports.values())
        total_out = sum(r["summary"]["total_output_size"] for r in all_reports.values())
        pct = round((1 - total_out / total_in) * 100, 1) if total_in > 0 else 0
        print(f"Total: {total_in:,}B -> {total_out:,}B ({pct}%) in {elapsed:.1f}s")

    return 0


def run_optimize(args: argparse.Namespace) -> int:
    """Optimize a single SVG file (deterministic by default; --ai for hybrid)."""
    in_path = Path(args.input)
    if not in_path.exists():
        print(f"Error: {in_path} not found", file=sys.stderr)
        return 1
    svg = in_path.read_text(errors="replace")
    orig = len(svg.encode("utf-8"))

    if args.ai:
        # Provider/model must be set BEFORE importing the optimizer, because
        # svgym.config reads these env vars at import time.
        if args.provider:
            os.environ["SVGYM_PROVIDER"] = args.provider
        if args.model:
            prov = args.provider or os.environ.get("SVGYM_PROVIDER", "anthropic")
            os.environ["GEMINI_MODEL" if prov == "gemini" else "ANTHROPIC_MODEL"] = args.model
        # --ai-always opens the escalation gate: run the model on every file,
        # regardless of how much the deterministic pass already saved or how small
        # the file is. (The smaller of {deterministic, model} is still returned.)
        threshold = 100.0 if args.ai_always else args.ai_threshold
        size_gate = 0 if args.ai_always else args.ai_size_gate
        from svgym.hybrid import optimize_svg_hybrid
        result = optimize_svg_hybrid(
            svg, level=args.level,
            llm_threshold=threshold,
            size_gate=size_gate,
            original_size=orig,
        )
        mode = "AI (always-on)" if args.ai_always else "AI hybrid"
    else:
        from svgym.deterministic import optimize_svg_deterministic
        result = optimize_svg_deterministic(svg, level=args.level)
        mode = "deterministic"

    out_svg = result.get("optimized_svg") or svg
    out_path = Path(args.output) if args.output else in_path.with_suffix(".min.svg")
    out_path.write_text(out_svg)

    new = len(out_svg.encode("utf-8"))
    pct = (1 - new / orig) * 100 if orig else 0.0
    ssim = result.get("ssim")
    if not args.quiet:
        line = f"{in_path.name}: {orig} -> {new} bytes ({pct:.1f}% smaller, {mode})"
        if ssim is not None:
            line += f", SSIM {ssim}"
        print(line)
        print(f"Wrote {out_path}")
    return 0


def main():
    parser = argparse.ArgumentParser(
        prog="svgym",
        description="SVGym — SVG optimization beyond SVGO",
    )
    subparsers = parser.add_subparsers(dest="command")

    # pack command
    pack_parser = subparsers.add_parser(
        "pack",
        help="Optimize a pack of SVG icons",
        description="Optimize a directory of SVGs with cross-file analysis, "
                    "sprite generation, and render benchmarking.",
    )
    pack_parser.add_argument(
        "input",
        help="Input directory containing SVGs (flat or with variant subdirs)",
    )
    pack_parser.add_argument(
        "-o", "--output",
        help="Output directory for optimized SVGs and sprite",
    )
    pack_parser.add_argument(
        "--level",
        choices=["lossless", "conservative", "aggressive"],
        default="conservative",
        help="Quality level (default: conservative)",
    )
    pack_parser.add_argument(
        "--no-sprite",
        action="store_true",
        help="Skip sprite sheet generation",
    )
    pack_parser.add_argument(
        "--json",
        dest="json_output",
        metavar="FILE",
        help="Write report as JSON to FILE",
    )
    pack_parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress progress output",
    )

    # optimize command (single file)
    opt_parser = subparsers.add_parser(
        "optimize",
        help="Optimize a single SVG file",
        description="Optimize one SVG with the deterministic pipeline (default), "
                    "or the AI hybrid with --ai.",
    )
    opt_parser.add_argument("input", help="Input .svg file")
    opt_parser.add_argument("-o", "--output", help="Output path (default: <name>.min.svg)")
    opt_parser.add_argument(
        "--level", choices=["lossless", "conservative", "aggressive"],
        default="conservative", help="Quality level (default: conservative)",
    )
    opt_parser.add_argument(
        "--ai", action="store_true",
        help="Enable the AI fallback (needs an API key in .env or environment)",
    )
    opt_parser.add_argument(
        "--provider", choices=["anthropic", "gemini"],
        help="AI provider for --ai (default: SVGYM_PROVIDER env / .env, else anthropic)",
    )
    opt_parser.add_argument(
        "--model", metavar="NAME",
        help="AI model name for --ai (overrides the provider's default model)",
    )
    opt_parser.add_argument(
        "--ai-threshold", type=float, default=30.0, metavar="PCT",
        help="With --ai: call the model only when the deterministic result is still "
             "below this %% reduction beyond SVGO (default: 30)",
    )
    opt_parser.add_argument(
        "--ai-size-gate", type=int, default=5120, metavar="BYTES",
        help="With --ai: skip the model for files smaller than this (default: 5120)",
    )
    opt_parser.add_argument(
        "--ai-always", action="store_true",
        help="With --ai: run the model on every file (shorthand for "
             "--ai-threshold 100 --ai-size-gate 0); overrides both",
    )
    opt_parser.add_argument("-q", "--quiet", action="store_true", help="Suppress output")

    args = parser.parse_args()

    if args.command == "pack":
        sys.exit(run_pack(args))
    elif args.command == "optimize":
        sys.exit(run_optimize(args))
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
