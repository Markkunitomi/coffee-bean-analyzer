"""Coffee Bean Analyzer - Command Line Interface.

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

# from .commands.batch import batch_command
# from .commands.optimize import optimize_command

# Version information
__version__ = "0.1.0"


@click.group()
@click.version_option(version=__version__)
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
@click.option("--quiet", "-q", is_flag=True, help="Suppress all output except errors")
@click.option("--log-file", type=click.Path(), help="Write logs to file")
@click.pass_context
def cli(ctx, verbose: bool, quiet: bool, log_file: Optional[str]):
    """Coffee Bean Size Analysis Tool.

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
    "--detailed-report/--no-detailed-report",
    default=True,
    help="Generate detailed analysis report with visualizations",
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
    ground_truth: Optional[Path] = None,
    optimize: Optional[bool] = None,
    detailed_report: bool = True,
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
            save_detailed_report=detailed_report,
        )
        click.echo(f"✅ Analysis complete. Results saved to {output_dir}")
    except Exception as e:
        logging.error(f"Analysis failed: {e}")
        if ctx.obj.get("verbose", False):
            raise
        sys.exit(1)


# Commented out batch command - module not implemented yet
# @cli.command()
# def batch(...):
#     """Batch process multiple images in a directory."""
#     pass

# Commented out optimize command - module not implemented yet
# @cli.command()
# def optimize(...):
#     """Optimize segmentation parameters using ground truth data."""
#     pass


# Commented out init command - module not implemented yet
# @cli.command()
# def init(...):
#     """Initialize a new coffee bean analysis project."""
#     pass


# Commented out validate command - module not implemented yet
# @cli.command()
# def validate():
#     """Validate installation and dependencies."""
#     pass


# Commented out info command - module not implemented yet
# @cli.command()
# def info(...):
#     """Display system and package information."""
#     pass


# Error handler for better user experience
def handle_exception(exc_type, exc_value, exc_traceback):
    """Global exception handler for CLI."""
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
