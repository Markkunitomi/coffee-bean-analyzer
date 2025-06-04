"""Coffee Bean Analyzer - Parameter Optimization Module.

Adapted from the original coffee_bean_analyzer.py script.
Handles automated parameter optimization using grid search for bean segmentation.
"""

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from itertools import product
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Import our own modules
from .detector import CoinDetector, DetectionResult
from .measurer import BeanMeasurement, BeanMeasurer
from .segmentor import BeanSegmentor


@dataclass
class OptimizationResult:
    """Container for optimization results."""

    best_params: Dict[str, Any]
    best_score: float
    best_measurements: List[BeanMeasurement]
    all_results: List[Dict[str, Any]]
    optimization_metadata: Dict[str, Any]
    total_combinations_tested: int
    optimization_time: float


class ScoringFunction(ABC):
    """Abstract base class for optimization scoring functions."""

    @abstractmethod
    def calculate_score(
        self,
        detected_measurements: List[BeanMeasurement],
        ground_truth: Optional[pd.DataFrame],
        metadata: Dict[str, Any],
    ) -> float:
        """Calculate optimization score for given measurements."""
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Get the name of this scoring function."""
        pass


class CountAccuracyScorer(ScoringFunction):
    """Scoring function based primarily on bean count accuracy."""

    def __init__(self, count_weight: float = 0.7, measurement_weight: float = 0.3):
        """Initialize count accuracy scorer.

        Args:
            count_weight: Weight for count accuracy in final score
            measurement_weight: Weight for measurement accuracy in final score
        """
        self.count_weight = count_weight
        self.measurement_weight = measurement_weight
        self.logger = logging.getLogger(__name__)

    def calculate_score(
        self,
        detected_measurements: List[BeanMeasurement],
        ground_truth: Optional[pd.DataFrame],
        metadata: Dict[str, Any],
    ) -> float:
        """Calculate optimization score based on bean count accuracy and measurement errors.

        This implements your original calculate_optimization_score logic.
        """
        if ground_truth is None or len(ground_truth) == 0:
            return float("-inf")

        n_detected = len(detected_measurements)
        n_ground_truth = len(ground_truth)

        # Bean count accuracy (primary objective) (your original logic)
        count_accuracy = 1.0 - abs(n_detected - n_ground_truth) / n_ground_truth

        # If no beans detected, return low score (your original logic)
        if n_detected == 0:
            return count_accuracy * 0.5

        # Measurement accuracy (secondary objective) (your original logic)
        n_pairs = min(n_detected, n_ground_truth)
        if n_pairs == 0:
            return count_accuracy * 0.5

        # Calculate measurement errors for available pairs (your original logic)
        length_errors = []
        width_errors = []

        # Get pixels_per_mm from metadata for unit conversion
        pixels_per_mm = metadata.get("pixels_per_mm")

        for i in range(n_pairs):
            auto_length = detected_measurements[i].length
            auto_width = detected_measurements[i].width

            # Convert ground truth from pixels to mm if needed (your original logic)
            gt_length = ground_truth.iloc[i]["length"]
            gt_width = ground_truth.iloc[i]["width"]

            if pixels_per_mm is not None:
                gt_length = gt_length / pixels_per_mm
                gt_width = gt_width / pixels_per_mm

            # Calculate relative errors (your original logic)
            if gt_length > 0:
                length_error = abs(auto_length - gt_length) / gt_length
                length_errors.append(length_error)

            if gt_width > 0:
                width_error = abs(auto_width - gt_width) / gt_width
                width_errors.append(width_error)

        # Calculate mean measurement accuracy (your original logic)
        measurement_accuracy = 0.0
        if length_errors:
            measurement_accuracy += (1.0 - np.mean(length_errors)) * 0.5
        if width_errors:
            measurement_accuracy += (1.0 - np.mean(width_errors)) * 0.5

        # Combined score (weighted towards count accuracy) (your original logic)
        total_score = (
            count_accuracy * self.count_weight
            + measurement_accuracy * self.measurement_weight
        )

        return max(0.0, total_score)  # Ensure non-negative score

    def get_name(self) -> str:
        return "count_accuracy"


class ParameterGrid:
    """Handles parameter grid generation and management."""

    def __init__(self, param_grid: Dict[str, List[Any]]):
        """Initialize parameter grid.

        Args:
            param_grid: Dictionary mapping parameter names to lists of values
        """
        self.param_grid = param_grid
        self.logger = logging.getLogger(__name__)

        # Generate all combinations
        self.combinations = self._generate_combinations()
        self.logger.info(f"Generated {len(self.combinations)} parameter combinations")

    def _generate_combinations(self) -> List[Dict[str, Any]]:
        """Generate all parameter combinations from the grid."""
        keys = self.param_grid.keys()
        values = self.param_grid.values()
        combinations = list(product(*values))

        return [dict(zip(keys, combination)) for combination in combinations]

    def get_combinations(self) -> List[Dict[str, Any]]:
        """Get all parameter combinations."""
        return self.combinations

    def get_total_combinations(self) -> int:
        """Get total number of combinations."""
        return len(self.combinations)

    @classmethod
    def create_default_grid(cls) -> "ParameterGrid":
        """Create default parameter grid matching your original grid_search_optimization."""
        param_grid = {
            "gaussian_kernel": [3, 5, 7],
            "clahe_clip": [1.0, 2.0, 3.0],
            "morph_kernel_size": [3, 5],
            "close_iterations": [1, 2, 3],
            "open_iterations": [1, 2],
            "min_distance": [15, 20, 25, 30],
            "threshold_factor": [0.2, 0.3, 0.4],
        }

        return cls(param_grid)

    @classmethod
    def create_quick_grid(cls) -> "ParameterGrid":
        """Create a smaller grid for quick testing."""
        param_grid = {
            "gaussian_kernel": [5, 7],
            "clahe_clip": [2.0, 3.0],
            "morph_kernel_size": [3, 5],
            "close_iterations": [2, 3],
            "open_iterations": [1],
            "min_distance": [20, 25],
            "threshold_factor": [0.3, 0.4],
        }

        return cls(param_grid)


class ProgressTracker:
    """Handles progress tracking during optimization."""

    def __init__(self, total_combinations: int, update_interval: int = 50):
        """Initialize progress tracker.

        Args:
            total_combinations: Total number of combinations to test
            update_interval: How often to print progress updates
        """
        self.total_combinations = total_combinations
        self.update_interval = update_interval
        self.start_time = None
        self.logger = logging.getLogger(__name__)

    def start(self):
        """Start progress tracking."""
        self.start_time = time.time()
        self.logger.info(
            f"Starting optimization with {self.total_combinations} combinations..."
        )

    def update(self, completed: int, best_score: float):
        """Update progress."""
        if (
            completed + 1
        ) % self.update_interval == 0 or completed == self.total_combinations - 1:
            elapsed = time.time() - self.start_time
            self.logger.info(
                f"Processed {completed + 1}/{self.total_combinations} combinations "
                f"(Best score: {best_score:.3f}, Time: {elapsed:.1f}s)"
            )

    def finish(self, best_score: float, best_params: Optional[Dict[str, Any]]):
        """Finish progress tracking."""
        total_time = time.time() - self.start_time
        self.logger.info("\nOptimization completed!")
        self.logger.info(f"Best score: {best_score:.3f}")
        self.logger.info(f"Total time: {total_time:.1f}s")
        self.logger.info("Best parameters:")
        if best_params is not None:
            for key, value in best_params.items():
                self.logger.info(f"  {key}: {value}")
        else:
            self.logger.info("  No valid parameters found")


class ParameterOptimizer:
    """Main parameter optimization class using grid search.

    Adapted from your original grid_search_optimization() function with added
    modularity and extensibility.
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize parameter optimizer.

        Args:
            config: Dictionary containing optimization configuration
        """
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Initialize components
        self.scoring_function = self._create_scoring_function()
        self.coin_detector = self._create_coin_detector()

    def _create_scoring_function(self) -> ScoringFunction:
        """Create scoring function from configuration."""
        scorer_type = self.config.get("scorer", "count_accuracy")

        if scorer_type == "count_accuracy":
            return CountAccuracyScorer(
                count_weight=self.config.get("count_weight", 0.7),
                measurement_weight=self.config.get("measurement_weight", 0.3),
            )
        raise ValueError(f"Unknown scorer type: {scorer_type}")

    def _create_coin_detector(self) -> CoinDetector:
        """Create coin detector for consistent coin detection."""
        coin_config = self.config.get(
            "coin_detection",
            {
                "dp": 1,
                "min_dist": 100,
                "param1": 50,
                "param2": 30,
                "min_radius": 50,
                "max_radius": 150,
                "gaussian_kernel": 15,
            },
        )

        return CoinDetector(coin_config)

    def _process_single_combination(
        self,
        image: np.ndarray,
        params: Dict[str, Any],
        coin_detection: Optional[DetectionResult],
    ) -> Tuple[float, List[BeanMeasurement], Dict[str, Any]]:
        """Process image with a single parameter combination.

        This implements your original process_image logic within the optimization loop.
        """
        try:
            # Create segmentor with current parameters
            segmentor = BeanSegmentor(params)

            # Perform segmentation
            segmentation_result = segmentor.segment(image, coin_detection, debug=False)

            # Create measurer for extracting measurements
            measurer_config = {
                "min_area": 100,  # Use consistent measurement parameters
                "row_threshold": 30,
                "coin_overlap_threshold": 1.1,
                "quarter_diameter_mm": 24.26,
            }
            measurer = BeanMeasurer(measurer_config)

            # Perform measurement
            measurement_result = measurer.measure(
                segmentation_result.labels, image, coin_detection, debug=False
            )

            # Create metadata for scoring
            metadata = {
                "pixels_per_mm": measurement_result.pixels_per_mm,
                "num_segments": segmentation_result.num_segments,
                "segmentation_metadata": segmentation_result.segmentation_metadata,
            }

            return (
                0.0,
                measurement_result.measurements,
                metadata,
            )  # Score will be calculated externally

        except Exception as e:
            self.logger.warning(f"Error processing combination {params}: {e}")
            return float("-inf"), [], {}

    def optimize(
        self,
        image: np.ndarray,
        ground_truth: Optional[pd.DataFrame] = None,
        parameter_grid: Optional[ParameterGrid] = None,
        debug: bool = False,
    ) -> OptimizationResult:
        """Perform grid search optimization to find optimal segmentation parameters.

        This is your original grid_search_optimization() function adapted to the new structure.

        Args:
            image: Input image for optimization
            ground_truth: Ground truth measurements for scoring
            parameter_grid: Parameter grid to search (uses default if None)
            debug: Enable debug output

        Returns:
            OptimizationResult containing optimization results
        """
        if debug:
            self.logger.info("Starting parameter optimization")
            if ground_truth is not None:
                self.logger.info(f"Target bean count: {len(ground_truth)}")

        # Use default grid if none provided
        if parameter_grid is None:
            parameter_grid = ParameterGrid.create_default_grid()

        combinations = parameter_grid.get_combinations()
        total_combinations = len(combinations)

        # Detect coin once for consistency (your original logic)
        coin_detections = self.coin_detector.detect(image, debug=debug)
        coin_detection = (
            self.coin_detector.get_best_coin(coin_detections)
            if coin_detections
            else None
        )

        if debug and coin_detection:
            self.logger.info(
                f"Using coin detection: center={coin_detection.center}, radius={coin_detection.radius:.1f}"
            )

        # Initialize tracking
        progress_tracker = ProgressTracker(total_combinations)
        progress_tracker.start()

        best_score = float("-inf")
        best_params = None
        best_measurements = []
        all_results = []

        # Test each parameter combination (your original logic)
        for i, params in enumerate(combinations):
            # Process image with current parameters
            _, measurements, metadata = self._process_single_combination(
                image, params, coin_detection
            )

            # Calculate optimization score (your original logic)
            score = self.scoring_function.calculate_score(
                measurements, ground_truth, metadata
            )

            # Record result
            result_record = {
                "params": params.copy(),
                "score": score,
                "num_detected": len(measurements),
                "metadata": metadata.copy(),
            }
            all_results.append(result_record)

            # Update best result (your original logic)
            if score > best_score:
                best_score = score
                best_params = params.copy()
                best_measurements = measurements.copy()

            # Update progress
            progress_tracker.update(i, best_score)

        # Finish tracking
        progress_tracker.finish(best_score, best_params)

        # Prepare optimization metadata
        optimization_metadata = {
            "total_combinations": total_combinations,
            "scorer_used": self.scoring_function.get_name(),
            "ground_truth_count": len(ground_truth) if ground_truth is not None else 0,
            "coin_detected": coin_detection is not None,
            "best_detected_count": len(best_measurements),
            "optimization_config": self.config.copy(),
        }

        # Ensure we always have valid best_params (fallback to first combination)
        if best_params is None and combinations:
            best_params = combinations[0].copy()
            self.logger.warning(
                "No valid optimization score found, using first parameter combination"
            )

        if debug:
            self.logger.info("Optimization completed:")
            self.logger.info(f"  Best score: {best_score:.3f}")
            self.logger.info(f"  Best detected count: {len(best_measurements)}")
            if ground_truth is not None:
                self.logger.info(f"  Target count: {len(ground_truth)}")

        return OptimizationResult(
            best_params=best_params,
            best_score=best_score,
            best_measurements=best_measurements,
            all_results=all_results,
            optimization_metadata=optimization_metadata,
            total_combinations_tested=total_combinations,
            optimization_time=time.time() - progress_tracker.start_time,
        )

    def get_top_results(
        self, optimization_result: OptimizationResult, top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """Get top K parameter combinations from optimization results."""
        sorted_results = sorted(
            optimization_result.all_results, key=lambda x: x["score"], reverse=True
        )
        return sorted_results[:top_k]

    def analyze_parameter_importance(
        self, optimization_result: OptimizationResult
    ) -> Dict[str, float]:
        """Analyze which parameters have the most impact on optimization score.

        Returns dictionary mapping parameter names to importance scores.
        """
        if not optimization_result.all_results:
            return {}

        # Get all parameter names
        param_names = list(optimization_result.all_results[0]["params"].keys())
        importance_scores = {}

        for param_name in param_names:
            # Get parameter values and scores
            param_values = [
                result["params"][param_name]
                for result in optimization_result.all_results
            ]
            scores = [result["score"] for result in optimization_result.all_results]

            # Check if parameter varies and is numeric
            unique_values = list(set(param_values))
            if len(unique_values) <= 1:
                importance_scores[param_name] = 0.0
                continue

            # Try to convert to numeric values for correlation
            try:
                # If all values are numeric, use them directly
                numeric_values = [float(val) for val in param_values]
                numeric_scores = [float(score) for score in scores]

                # Calculate correlation coefficient
                correlation = np.corrcoef(numeric_values, numeric_scores)[0, 1]
                importance_scores[param_name] = (
                    abs(correlation) if not np.isnan(correlation) else 0.0
                )

            except (ValueError, TypeError):
                # For categorical parameters, use variance of scores within each category
                category_scores = {}
                for val, score in zip(param_values, scores):
                    if val not in category_scores:
                        category_scores[val] = []
                    category_scores[val].append(score)

                # Calculate between-category variance as importance measure
                category_means = [
                    np.mean(scores_list) for scores_list in category_scores.values()
                ]
                if len(category_means) > 1:
                    np.mean(scores)
                    between_variance = np.var(category_means)
                    total_variance = np.var(scores) if np.var(scores) > 0 else 1.0
                    importance_scores[param_name] = between_variance / total_variance
                else:
                    importance_scores[param_name] = 0.0

        return importance_scores


# Legacy compatibility function
def grid_search_optimization_legacy(image_path, ground_truth, debug=False):
    """Legacy wrapper to maintain compatibility with your existing code.

    This function provides the same interface as your original grid_search_optimization()
    but uses the new ParameterOptimizer class internally.

    Args:
        image_path: Path to image or image array
        ground_truth: Ground truth DataFrame
        debug: Enable debug output

    Returns:
        Tuple of (best_params, best_result, best_score) in your original format
    """
    from pathlib import Path

    import cv2

    # Load image if path provided
    if isinstance(image_path, (str, Path)):
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
    else:
        image = image_path

    # Create optimizer with default configuration
    config = {
        "scorer": "count_accuracy",
        "count_weight": 0.7,
        "measurement_weight": 0.3,
    }

    optimizer = ParameterOptimizer(config)

    # Perform optimization
    result = optimizer.optimize(image, ground_truth, debug=debug)

    # Convert to original format
    # Create a mock result object that matches your original process_image output
    best_result = {
        "image_name": "optimization_result",
        "measurements": result.best_measurements,
        "pixels_per_mm": None,  # Will be set based on coin detection
    }

    # Extract pixels_per_mm from best measurements if available
    if result.best_measurements and hasattr(result.best_measurements[0], "unit"):
        if result.best_measurements[0].unit == "mm":
            # Find pixels_per_mm from optimization metadata
            for res in result.all_results:
                if res["params"] == result.best_params:
                    best_result["pixels_per_mm"] = res["metadata"].get("pixels_per_mm")
                    break

    return result.best_params, best_result, result.best_score


# Factory function for common optimization configurations
def create_optimizer(preset: str = "default", **kwargs) -> ParameterOptimizer:
    """Factory function to create optimizers with common configurations.

    Args:
        preset: Preset configuration name
        **kwargs: Override specific parameters

    Returns:
        Configured ParameterOptimizer instance
    """
    presets = {
        "default": {
            "scorer": "count_accuracy",
            "count_weight": 0.7,
            "measurement_weight": 0.3,
        },
        "count_focused": {
            "scorer": "count_accuracy",
            "count_weight": 0.9,
            "measurement_weight": 0.1,
        },
        "measurement_focused": {
            "scorer": "count_accuracy",
            "count_weight": 0.5,
            "measurement_weight": 0.5,
        },
        "quick": {
            "scorer": "count_accuracy",
            "count_weight": 0.7,
            "measurement_weight": 0.3,
            "use_quick_grid": True,
        },
    }

    if preset not in presets:
        raise ValueError(f"Unknown preset: {preset}. Available: {list(presets.keys())}")

    config = presets[preset].copy()
    config.update(kwargs)  # Allow parameter overrides

    return ParameterOptimizer(config)
