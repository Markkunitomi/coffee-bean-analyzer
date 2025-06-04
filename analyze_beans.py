#!/usr/bin/env python3
"""analyze_beans.py - Coffee Bean Analysis CLI

Usage examples:
    python analyze_beans.py image.tif
    python analyze_beans.py image.tif --ground-truth tests/data/beans_ground_truth.csv
    python analyze_beans.py image.tif --output results/ --preset aggressive
    python analyze_beans.py *.tif --batch --optimize
    python analyze_beans.py image.tif --no-optimize --preset quick
"""

import argparse
import glob
import sys
from pathlib import Path

# Import your comprehensive analyzer
try:
    from comprehensive_analyzer import ComprehensiveCoffeeBeanAnalyzer
except ImportError:
    print("❌ Error: Could not import ComprehensiveCoffeeBeanAnalyzer")
    print("💡 Make sure comprehensive_analyzer.py is in the same directory")
    sys.exit(1)


def parse_args():
    """Parse command line arguments with comprehensive options."""
    parser = argparse.ArgumentParser(
        description="Coffee Bean Analysis System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze single image
  python analyze_beans.py coffee_beans.tif
  
  # With ground truth for optimization  
  python analyze_beans.py image.tif --ground-truth tests/data/beans_ground_truth.csv
  
  # Custom output directory and preset
  python analyze_beans.py image.tif --output my_results/ --preset aggressive
  
  # Batch process multiple images
  python analyze_beans.py *.tif --batch
  
  # Quick analysis without optimization
  python analyze_beans.py image.tif --preset quick --no-optimize
  
Available presets: default, aggressive, conservative, quick
        """,
    )

    # Required arguments
    parser.add_argument(
        "image",
        help="Input image file path.",
    )

    # Optional arguments
    parser.add_argument(
        "--ground-truth",
        "-gt",
        type=str,
        default=None,
        help="Ground truth CSV file (default: None)",
    )

    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        help="Output directory (default: auto-generated with timestamp)",
    )

    parser.add_argument(
        "--preset",
        "-p",
        choices=["default", "aggressive", "conservative", "quick"],
        default="default",
        help="Analysis preset configuration (default: %(default)s)",
    )

    # Analysis control
    parser.add_argument(
        "--optimize",
        action="store_true",
        help="Force parameter optimization (requires ground truth)",
    )

    parser.add_argument(
        "--no-optimize",
        action="store_true",
        help="Skip parameter optimization even if ground truth available",
    )

    parser.add_argument(
        "--batch",
        action="store_true",
        help="Batch mode: process multiple images efficiently",
    )

    # Coin detection parameters
    coin_group = parser.add_argument_group("coin detection parameters")
    coin_group.add_argument("--coin-dp", type=float, help="Coin detection dp parameter")
    coin_group.add_argument(
        "--coin-min-dist", type=int, help="Minimum distance between coin centers"
    )
    coin_group.add_argument(
        "--coin-param1", type=int, help="Coin detection param1 (edge threshold)"
    )
    coin_group.add_argument(
        "--coin-param2", type=int, help="Coin detection param2 (accumulator threshold)"
    )
    coin_group.add_argument("--coin-min-radius", type=int, help="Minimum coin radius")
    coin_group.add_argument("--coin-max-radius", type=int, help="Maximum coin radius")

    # Preprocessing parameters
    preprocess_group = parser.add_argument_group("preprocessing parameters")
    preprocess_group.add_argument(
        "--gaussian-kernel", type=int, help="Gaussian blur kernel size"
    )
    preprocess_group.add_argument("--clahe-clip", type=float, help="CLAHE clip limit")

    # Segmentation parameters
    segment_group = parser.add_argument_group("segmentation parameters")
    segment_group.add_argument(
        "--morph-kernel-size", type=int, help="Morphological kernel size"
    )
    segment_group.add_argument(
        "--close-iterations", type=int, help="Morphological closing iterations"
    )
    segment_group.add_argument(
        "--open-iterations", type=int, help="Morphological opening iterations"
    )
    segment_group.add_argument(
        "--min-distance", type=int, help="Minimum distance between peaks"
    )
    segment_group.add_argument(
        "--threshold-factor", type=float, help="Watershed threshold factor"
    )

    # Measurement parameters
    measure_group = parser.add_argument_group("measurement parameters")
    measure_group.add_argument("--min-area", type=int, help="Minimum bean area")
    measure_group.add_argument(
        "--row-threshold", type=int, help="Row grouping threshold"
    )
    measure_group.add_argument(
        "--coin-overlap-threshold", type=float, help="Coin overlap threshold"
    )

    # Output control
    output_group = parser.add_argument_group("output control")
    output_group.add_argument(
        "--verbose", "-v", action="store_true", help="Verbose output"
    )
    output_group.add_argument(
        "--quiet", "-q", action="store_true", help="Quiet mode (minimal output)"
    )
    output_group.add_argument("--debug", action="store_true", help="Enable debug mode")
    output_group.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without executing",
    )
    output_group.add_argument(
        "--list-files", action="store_true", help="List found files and exit"
    )

    return parser.parse_args()


def expand_file_patterns(patterns):
    """Expand file patterns and wildcards into actual file paths."""
    expanded_files = []

    for pattern in patterns:
        if "*" in pattern or "?" in pattern:
            # Handle wildcards
            matches = glob.glob(pattern)
            if matches:
                expanded_files.extend(matches)
            else:
                print(f"⚠️ Warning: No files matched pattern '{pattern}'")
        else:
            # Regular file path
            if Path(pattern).exists():
                expanded_files.append(pattern)
            else:
                print(f"❌ Error: File not found: {pattern}")
                return None

    # Remove duplicates and sort
    return sorted(list(set(expanded_files)))


def validate_files(image_files, ground_truth_file=None):
    """Validate that all specified files exist and are readable."""
    valid_extensions = [".tif", ".tiff", ".jpg", ".jpeg", ".png", ".bmp"]
    valid_images = []

    for image_file in image_files:
        path = Path(image_file)

        if not path.exists():
            print(f"❌ Error: Image file not found: {image_file}")
            continue

        if path.suffix.lower() not in valid_extensions:
            print(f"⚠️ Warning: Unsupported image format: {image_file}")
            print(f"   Supported formats: {', '.join(valid_extensions)}")
            continue

        valid_images.append(image_file)

    # Check ground truth file
    if ground_truth_file:
        gt_path = Path(ground_truth_file)
        if not gt_path.exists():
            print(f"⚠️ Warning: Ground truth file not found: {ground_truth_file}")
            print("   Analysis will proceed without optimization")
            ground_truth_file = None
        elif gt_path.suffix.lower() != ".csv":
            print(
                f"❌ Error: Ground truth file must be CSV format: {ground_truth_file}"
            )
            ground_truth_file = None

    return valid_images, ground_truth_file


def create_custom_analyzer(args):
    """Create analyzer with custom parameters from command line."""
    analyzer = ComprehensiveCoffeeBeanAnalyzer(output_base_dir=args.output_dir)

    # Update coin detector configuration
    coin_config = {}
    if args.coin_dp is not None:
        coin_config["dp"] = args.coin_dp
    if args.coin_min_dist is not None:
        coin_config["min_dist"] = args.coin_min_dist
    if args.coin_param1 is not None:
        coin_config["param1"] = args.coin_param1
    if args.coin_param2 is not None:
        coin_config["param2"] = args.coin_param2
    if args.coin_min_radius is not None:
        coin_config["min_radius"] = args.coin_min_radius
    if args.coin_max_radius is not None:
        coin_config["max_radius"] = args.coin_max_radius

    if coin_config:
        from coffee_bean_analyzer.core.detector import CoinDetector

        analyzer.coin_detector = CoinDetector(coin_config)
        if not args.quiet:
            print(f"🔧 Custom coin detection parameters: {coin_config}")

    return analyzer


def print_analysis_plan(args, image_files, ground_truth_file):
    """Print the analysis plan before execution."""
    print("📋 ANALYSIS PLAN")
    print("=" * 50)
    print(f"Images to process: {len(image_files)}")
    for i, img in enumerate(image_files, 1):
        print(f"  {i}. {img}")

    print("\nConfiguration:")
    print(f"  Preset: {args.preset}")
    print(f"  Ground truth: {ground_truth_file if ground_truth_file else 'None'}")
    print(f"  Output directory: {args.output_dir if args.output_dir else 'Auto-generated'}")

    # Optimization settings
    if args.no_optimize:
        print("  Optimization: Disabled")
    elif args.optimize:
        print("  Optimization: Forced ON")
    elif ground_truth_file:
        print("  Optimization: Auto (ground truth available)")
    else:
        print("  Optimization: Auto (no ground truth)")

    # Custom parameters
    custom_count = sum(
        [
            args.gaussian_kernel is not None,
            args.clahe_clip is not None,
            args.morph_kernel_size is not None,
            args.close_iterations is not None,
            args.open_iterations is not None,
            args.min_distance is not None,
            args.threshold_factor is not None,
            args.min_area is not None,
            args.row_threshold is not None,
            args.coin_overlap_threshold is not None,
        ]
    )

    if custom_count > 0:
        print(f"  Custom parameters: {custom_count} overrides")

    print(f"  Batch mode: {'Yes' if args.batch else 'No'}")
    print(f"  Debug mode: {'Yes' if args.debug else 'No'}")


def main():
    """Main CLI function."""
    # Parse arguments
    args = parse_args()

    # Handle special modes
    if args.dry_run:
        print("🔍 DRY RUN MODE - No analysis will be performed")

    # Handle single image file
    image_files = [args.image]
    
    # Validate files
    valid_images, ground_truth_file = validate_files(image_files, args.ground_truth)
    if not valid_images:
        print("❌ No valid images to process!")
        sys.exit(1)

    # Handle list files mode
    if args.list_files:
        print(f"\n📁 Found {len(valid_images)} valid image file(s):")
        for i, img in enumerate(valid_images, 1):
            size = Path(img).stat().st_size / (1024 * 1024)  # MB
            print(f"  {i:2d}. {img} ({size:.1f} MB)")

        if ground_truth_file:
            size = Path(ground_truth_file).stat().st_size / 1024  # KB
            print(f"\n📋 Ground truth file: {ground_truth_file} ({size:.1f} KB)")

        sys.exit(0)

    # Validate optimization settings
    if args.optimize and not ground_truth_file:
        print("❌ Error: --optimize requires ground truth file")
        sys.exit(1)

    if args.optimize and args.no_optimize:
        print("❌ Error: Cannot specify both --optimize and --no-optimize")
        sys.exit(1)

    # Print analysis plan
    if not args.quiet:
        print_analysis_plan(args, valid_images, ground_truth_file)

    # Dry run mode - exit after showing plan
    if args.dry_run:
        print(f"\n✅ Dry run complete. Would process {len(valid_images)} image(s).")
        sys.exit(0)

    # Confirm before processing multiple files
    if len(valid_images) > 1 and not args.batch and not args.quiet:
        response = input(f"\n❓ Process {len(valid_images)} images? [Y/n]: ")
        if response.lower() in ["n", "no"]:
            print("❌ Analysis cancelled by user")
            sys.exit(0)

    # Create analyzer
    if not args.quiet:
        print("\n🚀 Initializing analyzer...")

    try:
        analyzer = ComprehensiveCoffeeBeanAnalyzer(output_base_dir=args.output_dir)

    except Exception as e:
        print(f"❌ Error initializing analyzer: {e}")
        if args.debug:
            import traceback

            traceback.print_exc()
        sys.exit(1)

    # Determine optimization setting
    if args.no_optimize:
        run_optimization = False
    elif args.optimize:
        run_optimization = True
    elif ground_truth_file:
        run_optimization = None  # Let analyzer decide
    else:
        run_optimization = False  # Default to False when no ground truth

    # Process images
    success_count = 0

    for i, image_path in enumerate(valid_images, 1):
        if not args.quiet:
            print(f"\n{'=' * 60}")
            print(f"🖼️ Processing {i}/{len(valid_images)}: {Path(image_path).name}")
            print(f"{'=' * 60}")

        try:
            # Run analysis
            results = analyzer.analyze_image_comprehensive(
                image_path,
                ground_truth_path=ground_truth_file,
                config_preset=args.preset,
                run_optimization=run_optimization,
            )

            success_count += 1

            if not args.quiet:
                orig_count = len(results["original"]["measurements"])
                print(f"✅ Analysis complete: {orig_count} beans detected")

                if results["optimized"]:
                    opt_count = len(results["optimized"]["measurements"])
                    improvement = opt_count - orig_count
                    print(f"✅ Optimization: {opt_count} beans ({improvement:+d})")

        except Exception as e:
            print(f"❌ Error processing {image_path}: {e}")
            if args.debug:
                import traceback

                traceback.print_exc()

    # Generate final summary
    if success_count > 0:
        if not args.quiet:
            print("\n📊 Generating final summary...")

        try:
            analyzer.generate_analysis_summary()

            print("\n🎉 ANALYSIS COMPLETE!")
            print("=" * 60)
            print(
                f"✅ Successfully processed: {success_count}/{len(valid_images)} images"
            )
            print(f"📁 Results saved to: {analyzer.output_dir}")
            print(f"📄 Summary report: {analyzer.output_dir}/ANALYSIS_SUMMARY.txt")

            # Show quick stats
            total_original = sum(
                len(r["original"]["measurements"])
                for r in analyzer.analysis_results.values()
            )
            total_optimized = sum(
                len(r["optimized"]["measurements"])
                for r in analyzer.analysis_results.values()
                if r["optimized"]
            )

            print("\n📈 Quick Stats:")
            print(f"   Total beans found (original): {total_original}")
            if total_optimized > 0:
                print(f"   Total beans found (optimized): {total_optimized}")
                print(
                    f"   Net improvement: {total_optimized - total_original:+d} beans"
                )

        except Exception as e:
            print(f"❌ Error generating summary: {e}")
            if args.debug:
                import traceback

                traceback.print_exc()

    else:
        print("\n❌ No images were successfully processed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
