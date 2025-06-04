#!/usr/bin/env python3
"""Coffee Bean Analyzer - Analysis Module
This belongs in: coffee_bean_analyzer/analysis/analyzer.py
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ..core.detector import CoinDetector
from ..core.measurer import create_measurer
from ..core.optimizer import ParameterGrid, create_optimizer
from ..core.preprocessor import create_preprocessor
from ..core.segmentor import create_segmentor
from ..io.data_handler import DataHandler
from ..utils.report_generator import ReportGenerator
from ..utils.visualization import VisualizationGenerator


class CoffeeBeanAnalyzer:
    """Main analyzer class that orchestrates the coffee bean analysis pipeline."""

    def __init__(self, output_dir: Optional[Path] = None):
        """Initialize the analyzer with specified output directory.

        Args:
            output_dir: Output directory path. If None, creates timestamped directory.
        """
        self.data_handler = DataHandler(output_dir)
        self.visualization_generator = VisualizationGenerator(
            self.data_handler.images_dir
        )
        self.report_generator = ReportGenerator(self.data_handler.reports_dir)

        # Initialize coin detector with default config
        self.coin_detector = CoinDetector(
            {
                "dp": 1,
                "min_dist": 100,
                "param1": 50,
                "param2": 30,
                "min_radius": 50,
                "max_radius": 150,
                "gaussian_kernel": 15,
            }
        )

        # These will be initialized based on config preset
        self.preprocessor = None
        self.segmentor = None
        self.measurer = None
        self.optimizer = None

        # Results storage
        self.analysis_results = {}
        self.optimization_results = {}

    def analyze_image(
        self,
        image_path: Path,
        ground_truth_path: Optional[Path] = None,
        config_preset: str = "default",
        run_optimization: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Analyze a single image with optional optimization.

        Args:
            image_path: Path to the image file
            ground_truth_path: Optional path to ground truth CSV
            config_preset: Configuration preset name
            run_optimization: Whether to run optimization. If None, auto-decides based on ground truth

        Returns:
            Dictionary containing analysis results
        """
        image_name = image_path.stem

        # Load image
        image = self.data_handler.load_image(image_path)

        # Load ground truth if provided
        ground_truth = None
        if ground_truth_path and ground_truth_path.exists():
            ground_truth = self.data_handler.load_ground_truth(ground_truth_path)

        # Auto-decide optimization
        if run_optimization is None:
            run_optimization = ground_truth is not None

        # Initialize components with preset
        self._initialize_components(config_preset)

        # Run analysis pipeline
        results = self._run_analysis_pipeline(
            image, image_name, ground_truth, run_optimization
        )

        # Store results
        self.analysis_results[image_name] = results

        return results

    def _initialize_components(self, config_preset: str):
        """Initialize analysis components with given preset."""
        self.preprocessor = create_preprocessor(config_preset)
        self.segmentor = create_segmentor(config_preset)
        self.measurer = create_measurer(config_preset)

    def _run_analysis_pipeline(
        self,
        image: np.ndarray,
        image_name: str,
        ground_truth: Optional[pd.DataFrame],
        run_optimization: bool,
    ) -> Dict[str, Any]:
        """Run the complete analysis pipeline."""
        # Step 1: Coin detection
        coin_detections = self.coin_detector.detect(image, debug=True)
        coin_detection = (
            self.coin_detector.get_best_coin(coin_detections)
            if coin_detections
            else None
        )

        # Step 2: Original analysis
        original_results = self._run_single_analysis(image, coin_detection)

        # Save original visualization
        self.visualization_generator.create_analysis_visualization(
            image, original_results, coin_detection, image_name, is_optimized=False
        )

        # Step 3: Optimization (if requested)
        optimized_results = None
        if run_optimization and ground_truth is not None:
            optimized_results = self._run_optimization(
                image, ground_truth, coin_detection, image_name
            )

        # Save data and generate reports
        self.data_handler.save_measurement_data(
            image_name, original_results, optimized_results, ground_truth
        )

        self.report_generator.generate_analysis_report(
            image_name, original_results, optimized_results, coin_detection
        )

        return {
            "original": original_results,
            "optimized": optimized_results,
            "coin_detection": coin_detection,
            "image_path": str(image_path),
        }

    def _run_single_analysis(self, image: np.ndarray, coin_detection) -> Dict[str, Any]:
        """Run a single analysis pass."""
        segmentation_result = self.segmentor.segment(image, coin_detection, debug=True)
        measurement_result = self.measurer.measure(
            segmentation_result.labels, image, coin_detection, debug=True
        )

        return {
            "measurements": measurement_result.measurements,
            "segmentation_result": segmentation_result,
            "measurement_result": measurement_result,
            "parameters": self.segmentor.get_configuration(),
        }

    def _run_optimization(
        self,
        image: np.ndarray,
        ground_truth: pd.DataFrame,
        coin_detection,
        image_name: str,
    ) -> Dict[str, Any]:
        """Run parameter optimization."""
        self.optimizer = create_optimizer("count_focused")
        param_grid = ParameterGrid.create_default_grid()

        optimization_result = self.optimizer.optimize(
            image, ground_truth, param_grid, debug=True
        )

        # Save optimization results
        self.data_handler.save_optimization_results(optimization_result)

        # Run analysis with optimized parameters
        self.segmentor.update_configuration(optimization_result.best_params)
        optimized_results = self._run_single_analysis(image, coin_detection)
        optimized_results["optimization_result"] = optimization_result

        # Save optimized visualization
        self.visualization_generator.create_analysis_visualization(
            image, optimized_results, coin_detection, image_name, is_optimized=True
        )

        # Create comparison visualization
        self.visualization_generator.create_optimization_comparison(
            image_name, self.analysis_results[image_name]["original"], optimized_results
        )

        self.optimization_results[image_name] = optimization_result

        return optimized_results

    def generate_summary(self):
        """Generate analysis summary report."""
        self.report_generator.generate_summary_report(
            self.analysis_results,
            self.optimization_results,
            self.data_handler.output_dir,
        )

    def run_batch_analysis(
        self,
        image_paths: List[Path],
        ground_truth_path: Optional[Path] = None,
        config_preset: str = "default",
    ) -> Path:
        """Run analysis on multiple images.

        Args:
            image_paths: List of image paths to analyze
            ground_truth_path: Optional ground truth CSV path
            config_preset: Configuration preset

        Returns:
            Path to output directory
        """
        for i, image_path in enumerate(image_paths, 1):
            print(f"\nProcessing {i}/{len(image_paths)}: {image_path.name}")

            try:
                self.analyze_image(image_path, ground_truth_path, config_preset)
                print(f"✅ Analysis complete for {image_path.name}")

            except Exception as e:
                print(f"❌ Error processing {image_path}: {e}")
                import traceback

                traceback.print_exc()

        # Generate final summary
        self.generate_summary()

        return self.data_handler.output_dir
