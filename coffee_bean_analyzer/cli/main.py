"""Coffee Bean Analyzer - Command Line Interface

Main CLI entry point providing commands for image analysis, batch processing,
and parameter optimization.
"""

import logging
import sys
from pathlib import Path
from typing import Optional

import click

from ..utils.logging_config import setup_logging
from .commands.analyze import analyze_command
from .commands.batch import batch_command
from .commands.optimize import optimize_command

# Version information
__version__ = "0.1.0"


@click.group()
@click.version_option(version=__version__)
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
@click.option("--quiet", "-q", is_flag=True, help="Suppress all output except errors")
@click.option("--log-file", type=click.Path(), help="Write logs to file")
@click.pass_context
def cli(ctx, verbose: bool, quiet: bool, log_file: Optional[str]):
    """Coffee Bean Size Analysis Tool

    A computer vision tool for analyzing coffee bean dimensions using
    watershed segmentation and scale calibration.

    Examples:
        # Analyze single image
        coffee-bean-analyzer analyze image.jpg -o results/

        # Batch process directory
        coffee-bean-analyzer batch images/ -o results/ --parallel

        # Optimize parameters
        coffee-bean-analyzer optimize image.jpg --ground-truth truth.csv
    """
    # Ensure context object exists
    ctx.ensure_object(dict)

    # Setup logging
    log_level = logging.ERROR if quiet else (logging.DEBUG if verbose else logging.INFO)
    setup_logging(level=log_level, log_file=log_file)

    # Store global options in context
    ctx.obj["verbose"] = verbose
    ctx.obj["quiet"] = quiet
    ctx.obj["log_file"] = log_file


@cli.command()
@click.argument("input_path", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(path_type=Path),
    default=Path("output"),
    help="Output directory for results",
)
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True, path_type=Path),
    help="Configuration file path",
)
@click.option(
    "--save-annotated/--no-save-annotated",
    default=True,
    help="Save annotated images with detected beans",
)
@click.option(
    "--save-individuals/--no-save-individuals",
    default=False,
    help="Save individual bean images",
)
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["csv", "json", "xlsx"]),
    default="csv",
    help="Output data format",
)
@click.option(
    "--ground-truth",
    "-g",
    type=click.Path(exists=True, path_type=Path),
    help="Ground truth CSV for parameter optimization",
)
@click.option(
    "--optimize/--no-optimize",
    default=None,
    help="Run parameter optimization (auto-enabled if ground truth provided)",
)
@click.option(
    "--comprehensive-report/--no-comprehensive-report",
    default=True,
    help="Generate comprehensive analysis report with visualizations",
)
@click.pass_context
def analyze(
    ctx,
    input_path: Path,
    output_dir: Path,
    config: Optional[Path],
    save_annotated: bool,
    save_individuals: bool,
    output_format: str,
):
    """Analyze coffee beans in a single image.

    INPUT_PATH: Path to the input image file

    This command processes a single image to detect and measure coffee beans.
    It automatically detects US quarter coins for scale calibration and
    segments individual beans using watershed algorithm.

    Examples:
        coffee-bean-analyzer analyze beans.jpg
        coffee-bean-analyzer analyze beans.jpg -o results/ -c custom_config.yaml
        coffee-bean-analyzer analyze beans.jpg --format json --save-individuals
    """
    try:
        analyze_command(
            input_path=input_path,
            output_dir=output_dir,
            config_path=config,
            save_annotated=save_annotated,
            save_individuals=save_individuals,
            output_format=output_format,
            verbose=ctx.obj.get("verbose", False),
            # ADD THESE NEW PARAMETERS:
            ground_truth_path=ground_truth,
            run_optimization=optimize,
            save_comprehensive_report=comprehensive_report,
        )
        click.echo(f"✅ Analysis complete. Results saved to {output_dir}")
    except Exception as e:
        logging.error(f"Analysis failed: {e}")
        if ctx.obj.get("verbose", False):
            raise
        sys.exit(1)


@cli.command()
@click.argument(
    "input_dir", type=click.Path(exists=True, file_okay=False, path_type=Path)
)
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(path_type=Path),
    default=Path("output"),
    help="Output directory for results",
)
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True, path_type=Path),
    help="Configuration file path",
)
@click.option("--parallel", "-p", is_flag=True, help="Enable parallel processing")
@click.option(
    "--max-workers", type=int, default=4, help="Maximum number of parallel workers"
)
@click.option(
    "--pattern",
    default="*.jpg,*.jpeg,*.png,*.tiff",
    help="File patterns to process (comma-separated)",
)
@click.option(
    "--recursive", "-r", is_flag=True, help="Process subdirectories recursively"
)
@click.option("--resume", is_flag=True, help="Resume interrupted batch processing")
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["csv", "json", "xlsx"]),
    default="csv",
    help="Output data format",
)
@click.pass_context
def batch(
    ctx,
    input_dir: Path,
    output_dir: Path,
    config: Optional[Path],
    parallel: bool,
    max_workers: int,
    pattern: str,
    recursive: bool,
    resume: bool,
    output_format: str,
):
    """Batch process multiple images in a directory.

    INPUT_DIR: Directory containing images to process

    This command processes all images in a directory, generating
    comprehensive reports and measurements for each image.

    Examples:
        coffee-bean-analyzer batch images/
        coffee-bean-analyzer batch images/ -o results/ --parallel
        coffee-bean-analyzer batch images/ -r --pattern "*.jpg,*.png"
    """
    try:
        # Parse file patterns
        patterns = [p.strip() for p in pattern.split(",")]

        batch_command(
            input_dir=input_dir,
            output_dir=output_dir,
            config_path=config,
            parallel=parallel,
            max_workers=max_workers,
            file_patterns=patterns,
            recursive=recursive,
            resume=resume,
            output_format=output_format,
            verbose=ctx.obj.get("verbose", False),
        )
        click.echo(f"✅ Batch processing complete. Results saved to {output_dir}")
    except Exception as e:
        logging.error(f"Batch processing failed: {e}")
        if ctx.obj.get("verbose", False):
            raise
        sys.exit(1)


@cli.command()
@click.argument("input_path", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--ground-truth",
    "-g",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Ground truth measurements CSV file",
)
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(path_type=Path),
    default=Path("optimization_results"),
    help="Output directory for results",
)
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True, path_type=Path),
    help="Base configuration file path",
)
@click.option(
    "--param-ranges",
    type=click.Path(exists=True, path_type=Path),
    help="Parameter ranges configuration file",
)
@click.option(
    "--metric",
    type=click.Choice(["accuracy", "precision", "recall", "f1"]),
    default="accuracy",
    help="Optimization metric",
)
@click.option("--n-trials", type=int, default=100, help="Number of optimization trials")
@click.option("--timeout", type=int, help="Timeout in seconds for optimization")
@click.option(
    "--n-jobs", type=int, default=1, help="Number of parallel jobs for optimization"
)
@click.pass_context
def optimize(
    ctx,
    input_path: Path,
    ground_truth: Path,
    output_dir: Path,
    config: Optional[Path],
    param_ranges: Optional[Path],
    metric: str,
    n_trials: int,
    timeout: Optional[int],
    n_jobs: int,
):
    """Optimize segmentation parameters using ground truth data.

    INPUT_PATH: Path to the input image file
    GROUND_TRUTH: Path to ground truth measurements CSV file

    This command performs grid search optimization to find the best
    parameters for bean segmentation and measurement accuracy.

    Examples:
        coffee-bean-analyzer optimize image.jpg --ground-truth truth.csv
        coffee-bean-analyzer optimize image.jpg -g truth.csv --metric f1 --n-trials 200
        coffee-bean-analyzer optimize image.jpg -g truth.csv --param-ranges ranges.yaml
    """
    try:
        optimize_command(
            input_path=input_path,
            ground_truth_path=ground_truth,
            output_dir=output_dir,
            config_path=config,
            param_ranges_path=param_ranges,
            metric=metric,
            n_trials=n_trials,
            timeout=timeout,
            n_jobs=n_jobs,
            verbose=ctx.obj.get("verbose", False),
        )
        click.echo(f"✅ Parameter optimization complete. Results saved to {output_dir}")
    except Exception as e:
        logging.error(f"Parameter optimization failed: {e}")
        if ctx.obj.get("verbose", False):
            raise
        sys.exit(1)


@cli.command()
@click.option(
    "--config-template",
    type=click.Path(path_type=Path),
    default=Path("config_template.yaml"),
    help="Output path for configuration template",
)
@click.option("--sample-data", is_flag=True, help="Download sample data for testing")
def init(config_template: Path, sample_data: bool):
    """Initialize a new coffee bean analysis project.

    Creates configuration templates and optionally downloads sample data.

    Examples:
        coffee-bean-analyzer init
        coffee-bean-analyzer init --sample-data
        coffee-bean-analyzer init --config-template my_config.yaml
    """
    try:
        from ..utils.project_init import initialize_project

        initialize_project(
            config_template_path=config_template, download_sample_data=sample_data
        )

        click.echo("✅ Project initialized successfully!")
        click.echo(f"📄 Configuration template created: {config_template}")

        if sample_data:
            click.echo("📦 Sample data downloaded to data/sample/")

        click.echo("\n🚀 Get started with:")
        click.echo(
            "   coffee-bean-analyzer analyze data/sample/images/sample_beans_1.jpg"
        )

    except Exception as e:
        logging.error(f"Project initialization failed: {e}")
        sys.exit(1)


@cli.command()
def validate():
    """Validate installation and dependencies.

    Checks that all required dependencies are installed and working correctly.
    """
    try:
        from ..utils.validation import validate_installation

        click.echo("🔍 Validating installation...")

        issues = validate_installation()

        if not issues:
            click.echo("✅ Installation validation successful!")
            click.echo("All dependencies are working correctly.")
        else:
            click.echo("❌ Installation validation failed!")
            for issue in issues:
                click.echo(f"   • {issue}")
            sys.exit(1)

    except Exception as e:
        logging.error(f"Validation failed: {e}")
        sys.exit(1)


@cli.command()
@click.option(
    "--output-format",
    type=click.Choice(["text", "json"]),
    default="text",
    help="Output format for system info",
)
def info(output_format: str):
    """Display system and package information.

    Shows version information, system details, and dependency status.
    """
    try:
        from ..utils.system_info import get_system_info

        info_data = get_system_info()

        if output_format == "json":
            import json

            click.echo(json.dumps(info_data, indent=2))
        else:
            # Text format
            click.echo(f"Coffee Bean Analyzer v{__version__}")
            click.echo("=" * 40)
            click.echo(f"Python: {info_data['python_version']}")
            click.echo(f"Platform: {info_data['platform']}")
            click.echo(f"OpenCV: {info_data['opencv_version']}")
            click.echo(f"NumPy: {info_data['numpy_version']}")
            click.echo(f"SciPy: {info_data['scipy_version']}")
            click.echo(f"scikit-image: {info_data['skimage_version']}")

    except Exception as e:
        logging.error(f"Failed to get system info: {e}")
        sys.exit(1)


# Error handler for better user experience
def handle_exception(exc_type, exc_value, exc_traceback):
    """Global exception handler for CLI"""
    if issubclass(exc_type, KeyboardInterrupt):
        click.echo("\n⚠️  Operation cancelled by user")
        sys.exit(1)

    logging.error(
        "Unexpected error occurred", exc_info=(exc_type, exc_value, exc_traceback)
    )
    click.echo("❌ An unexpected error occurred. Use --verbose for details.")
    sys.exit(1)


# Set global exception handler
sys.excepthook = handle_exception


if __name__ == "__main__":
    cli()
