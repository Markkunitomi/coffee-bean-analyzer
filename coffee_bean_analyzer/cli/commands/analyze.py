#!/usr/bin/env python3
"""Enhanced analyze command that integrates detailed analysis features
This should update: coffee_bean_analyzer/cli/commands/analyze.py.
"""

import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pandas as pd

from ...config.config_loader import load_config
from ...core.detector import CoinDetector
from ...core.measurer import create_measurer
from ...core.optimizer import ParameterGrid, create_optimizer
from ...core.preprocessor import create_preprocessor
from ...core.segmentor import create_segmentor
from ...io.data_handler import DataHandler
from ...utils.report_generator import generate_analysis_report
from ...utils.visualization import (
    create_analysis_visualization,
    create_optimization_comparison,
)

logger = logging.getLogger(__name__)


def analyze_command(
    input_path: Path,
    output_dir: Path,
    config_path: Optional[Path] = None,
    save_annotated: bool = True,
    save_individuals: bool = False,
    output_format: str = "csv",
    verbose: bool = False,
    ground_truth_path: Optional[Path] = None,
    run_optimization: Optional[bool] = None,
    save_detailed_report: bool = True,
):
    """Enhanced analyze command with detailed analysis features.

    Args:
        input_path: Path to input image
        output_dir: Output directory for results
        config_path: Optional configuration file
        save_annotated: Save annotated images
        save_individuals: Save individual bean crops
        output_format: Output format (csv, json, xlsx)
        verbose: Verbose logging
        ground_truth_path: Optional ground truth for optimization
        run_optimization: Force optimization on/off
        save_detailed_report: Generate detailed analysis report
    """
    logger.info(f"Starting analysis of {input_path}")

    # Load configuration
    if config_path:
        config = load_config(config_path)
    else:
        config = load_config()  # Load defaults

    # Create output structure
    data_handler = DataHandler(output_dir)

    # Load image
    image = cv2.imread(str(input_path))
    if image is None:
        raise ValueError(f"Could not load image: {input_path}")

    image_name = input_path.stem

    # Load ground truth if provided
    ground_truth = None
    if ground_truth_path and ground_truth_path.exists():
        ground_truth = pd.read_csv(ground_truth_path)
        logger.info(f"Loaded ground truth with {len(ground_truth)} beans")

    # Auto-decide optimization
    if run_optimization is None:
        run_optimization = ground_truth is not None

    # Initialize components
    coin_detector = CoinDetector(config.get("coin_detector", {}))
    create_preprocessor(config.get("preprocessing", {}).get("preset", "default"))
    segmentor = create_segmentor(
        config.get("segmentation", {}).get("preset", "default")
    )
    measurer = create_measurer(config.get("measurement", {}).get("preset", "default"))

    # Step 1: Coin detection
    logger.info("Detecting calibration coin...")
    coin_detections = coin_detector.detect(image, debug=verbose)
    coin_detection = (
        coin_detector.get_best_coin(coin_detections) if coin_detections else None
    )

    if coin_detection:
        logger.info(
            f"Coin detected at ({coin_detection.center[0]}, {coin_detection.center[1]}) "
            f"with radius {coin_detection.radius}px"
        )
    else:
        logger.warning("No calibration coin detected - measurements will be in pixels")

    # Step 2: Original analysis
    logger.info("Running segmentation and measurement...")
    segmentation_result = segmentor.segment(image, coin_detection, debug=verbose)
    measurement_result = measurer.measure(
        segmentation_result.labels, image, coin_detection, debug=verbose
    )

    original_results = {
        "measurements": measurement_result.measurements,
        "segmentation_result": segmentation_result,
        "measurement_result": measurement_result,
        "parameters": segmentor.get_configuration(),
    }

    logger.info(f"Detected {len(measurement_result.measurements)} beans")

    # Save original visualization
    if save_annotated:
        viz_path = create_analysis_visualization(
            image=image,
            labels=segmentation_result.labels,
            measurements=measurement_result.measurements,
            coin_detection=coin_detection,
            output_path=data_handler.images_dir / f"{image_name}_original_analysis.png",
            title=f"{image_name} - Original Analysis",
        )
        logger.info(f"Saved visualization to {viz_path}")

    # Step 3: Optimization (if requested)
    optimized_results = None
    if run_optimization and ground_truth is not None:
        logger.info("Running parameter optimization...")

        optimizer = create_optimizer("count_focused")
        param_grid = ParameterGrid.create_default_grid()

        optimization_result = optimizer.optimize(
            image, ground_truth, param_grid, debug=verbose
        )

        logger.info(
            f"Optimization complete: best score={optimization_result.best_score:.3f}"
        )

        # Run analysis with optimized parameters
        segmentor.update_configuration(optimization_result.best_params)
        opt_segmentation = segmentor.segment(image, coin_detection, debug=verbose)
        opt_measurement = measurer.measure(
            opt_segmentation.labels, image, coin_detection, debug=verbose
        )

        optimized_results = {
            "measurements": opt_measurement.measurements,
            "segmentation_result": opt_segmentation,
            "measurement_result": opt_measurement,
            "parameters": optimization_result.best_params,
            "optimization_result": optimization_result,
        }

        logger.info(f"Optimized detection: {len(opt_measurement.measurements)} beans")

        # Save optimized visualization
        if save_annotated:
            opt_viz_path = create_analysis_visualization(
                image=image,
                labels=opt_segmentation.labels,
                measurements=opt_measurement.measurements,
                coin_detection=coin_detection,
                output_path=data_handler.images_dir
                / f"{image_name}_optimized_analysis.png",
                title=f"{image_name} - Optimized Analysis",
            )
            logger.info(f"Saved optimized visualization to {opt_viz_path}")

            # Create comparison visualization
            comparison_path = create_optimization_comparison(
                image_name=image_name,
                original_results=original_results,
                optimized_results=optimized_results,
                output_path=data_handler.images_dir
                / f"{image_name}_optimization_comparison.png",
            )
            logger.info(f"Saved comparison to {comparison_path}")

    # Save measurement data
    data_handler.save_measurements(
        measurements=measurement_result.measurements,
        image_name=image_name,
        output_format=output_format,
        metadata={
            "coin_detected": coin_detection is not None,
            "scale_factor": coin_detection.pixels_per_mm if coin_detection else None,
            "parameters": original_results["parameters"],
        },
    )

    # Save individual bean crops if requested
    if save_individuals:
        save_individual_beans(
            image=image,
            labels=segmentation_result.labels,
            measurements=measurement_result.measurements,
            output_dir=data_handler.images_dir / "individual_beans" / image_name,
        )

    # Generate detailed report if requested
    if save_detailed_report:
        report_path = generate_analysis_report(
            image_name=image_name,
            original_results=original_results,
            optimized_results=optimized_results,
            coin_detection=coin_detection,
            ground_truth=ground_truth,
            output_path=data_handler.reports_dir / f"{image_name}_analysis_report.txt",
        )
        logger.info(f"Generated report at {report_path}")

    # Return results for potential further processing
    return {
        "original": original_results,
        "optimized": optimized_results,
        "coin_detection": coin_detection,
        "output_dir": output_dir,
    }


def save_individual_beans(
    image: np.ndarray, labels: np.ndarray, measurements: list, output_dir: Path
):
    """Save individual bean crops."""
    output_dir.mkdir(parents=True, exist_ok=True)

    for i, measurement in enumerate(measurements):
        # Get bean mask
        bean_mask = labels == measurement.label

        # Get bounding box
        bbox = get_bbox_from_mask(bean_mask)

        # Crop image
        crop = image[bbox[1] : bbox[3], bbox[0] : bbox[2]]

        # Save crop
        cv2.imwrite(str(output_dir / f"bean_{i + 1:03d}.png"), crop)


def get_bbox_from_mask(mask: np.ndarray) -> tuple:
    """Get bounding box from binary mask."""
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    y1, y2 = np.where(rows)[0][[0, -1]]
    x1, x2 = np.where(cols)[0][[0, -1]]
    return (x1, y1, x2, y2)


# Also update the CLI main.py to add detailed analysis options:
"""
Update the analyze command in main.py to include:

@click.option('--ground-truth', '-g', type=click.Path(exists=True, path_type=Path),
              help='Ground truth CSV for optimization')
@click.option('--optimize/--no-optimize', default=None,
              help='Run parameter optimization')
@click.option('--detailed-report/--no-detailed-report', default=True,
              help='Generate detailed analysis report')

And pass these to analyze_command:
    ground_truth_path=ground_truth,
    run_optimization=optimize,
    save_detailed_report=detailed_report
"""
