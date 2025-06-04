"""Unit tests for the optimization module using pytest."""

import sys
from pathlib import Path
from unittest.mock import Mock, patch

import cv2
import numpy as np
import pandas as pd
import pytest

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from coffee_bean_analyzer.core.detector import DetectionResult
from coffee_bean_analyzer.core.measurer import BeanMeasurement
from coffee_bean_analyzer.core.optimizer import (
    CountAccuracyScorer,
    OptimizationResult,
    ParameterGrid,
    ParameterOptimizer,
    ProgressTracker,
    create_optimizer,
    grid_search_optimization_legacy,
)


class TestParameterGrid:
    """Test parameter grid generation and management."""

    def test_parameter_grid_creation(self):
        """Test parameter grid creation."""
        param_grid = {"param1": [1, 2, 3], "param2": ["a", "b"], "param3": [10.0, 20.0]}

        grid = ParameterGrid(param_grid)

        assert grid.get_total_combinations() == 3 * 2 * 2  # 12 combinations
        combinations = grid.get_combinations()

        assert len(combinations) == 12
        assert all(isinstance(combo, dict) for combo in combinations)
        assert all(
            "param1" in combo and "param2" in combo and "param3" in combo
            for combo in combinations
        )

    def test_default_grid(self):
        """Test default parameter grid creation."""
        grid = ParameterGrid.create_default_grid()

        combinations = grid.get_combinations()
        assert len(combinations) > 0

        # Check that all expected parameters are present
        expected_params = [
            "gaussian_kernel",
            "clahe_clip",
            "morph_kernel_size",
            "close_iterations",
            "open_iterations",
            "min_distance",
            "threshold_factor",
        ]

        for combo in combinations:
            for param in expected_params:
                assert param in combo

    def test_quick_grid(self):
        """Test quick parameter grid creation."""
        default_grid = ParameterGrid.create_default_grid()
        quick_grid = ParameterGrid.create_quick_grid()

        # Quick grid should have fewer combinations than default
        assert (
            quick_grid.get_total_combinations() < default_grid.get_total_combinations()
        )
        assert quick_grid.get_total_combinations() > 0

    def test_empty_grid(self):
        """Test parameter grid with empty parameters."""
        param_grid = {}
        grid = ParameterGrid(param_grid)

        assert grid.get_total_combinations() == 1  # Empty combination
        combinations = grid.get_combinations()
        assert len(combinations) == 1
        assert combinations[0] == {}


class TestCountAccuracyScorer:
    """Test count accuracy scoring function."""

    @pytest.fixture
    def scorer(self):
        """Create CountAccuracyScorer instance."""
        return CountAccuracyScorer(count_weight=0.7, measurement_weight=0.3)

    @pytest.fixture
    def sample_measurements(self):
        """Create sample measurements."""
        return [
            BeanMeasurement(1, 50, 50, 100, 12.0, 8.0, 0, 0.5, 0.9, 30, 1.5, "mm"),
            BeanMeasurement(2, 60, 60, 120, 14.0, 9.0, 0, 0.6, 0.9, 35, 1.56, "mm"),
            BeanMeasurement(3, 70, 70, 110, 13.0, 8.5, 0, 0.55, 0.9, 32, 1.53, "mm"),
        ]

    @pytest.fixture
    def sample_ground_truth(self):
        """Create sample ground truth data."""
        return pd.DataFrame(
            {
                "length": [48.0, 56.0, 52.0],  # In pixels (will be converted to mm)
                "width": [32.0, 36.0, 34.0],  # In pixels (will be converted to mm)
            }
        )

    def test_scorer_name(self, scorer):
        """Test scorer name."""
        assert scorer.get_name() == "count_accuracy"

    def test_score_no_ground_truth(self, scorer, sample_measurements):
        """Test scoring when no ground truth is provided."""
        score = scorer.calculate_score(sample_measurements, None, {})
        assert score == float("-inf")

        empty_gt = pd.DataFrame()
        score = scorer.calculate_score(sample_measurements, empty_gt, {})
        assert score == float("-inf")

    def test_score_no_detections(self, scorer, sample_ground_truth):
        """Test scoring when no beans are detected."""
        metadata = {"pixels_per_mm": 4.0}
        score = scorer.calculate_score([], sample_ground_truth, metadata)

        # Should return count accuracy * 0.5
        expected_count_accuracy = 1.0 - abs(0 - 3) / 3  # 0.0
        expected_score = expected_count_accuracy * 0.5
        assert abs(score - expected_score) < 0.01

    def test_score_perfect_match(
        self, scorer, sample_measurements, sample_ground_truth
    ):
        """Test scoring with perfect count match."""
        # Metadata with pixels_per_mm for unit conversion
        metadata = {"pixels_per_mm": 4.0}  # 4 pixels per mm

        score = scorer.calculate_score(
            sample_measurements, sample_ground_truth, metadata
        )

        # Should be positive score (count accuracy = 1.0, some measurement accuracy)
        assert score > 0.0
        assert score <= 1.0

    def test_score_count_mismatch(
        self, scorer, sample_measurements, sample_ground_truth
    ):
        """Test scoring with count mismatch."""
        # Use only 2 measurements instead of 3
        metadata = {"pixels_per_mm": 4.0}
        score = scorer.calculate_score(
            sample_measurements[:2], sample_ground_truth, metadata
        )

        # Count accuracy should be reduced
        1.0 - abs(2 - 3) / 3  # 0.667
        assert score > 0.0
        assert score < 1.0

    def test_score_no_pixels_per_mm(
        self, scorer, sample_measurements, sample_ground_truth
    ):
        """Test scoring without pixels_per_mm in metadata."""
        metadata = {}
        score = scorer.calculate_score(
            sample_measurements, sample_ground_truth, metadata
        )

        # Should still calculate score, treating ground truth as already in correct units
        assert score >= 0.0


class TestProgressTracker:
    """Test progress tracking functionality."""

    def test_progress_tracker_creation(self):
        """Test progress tracker creation."""
        tracker = ProgressTracker(total_combinations=100, update_interval=10)

        assert tracker.total_combinations == 100
        assert tracker.update_interval == 10
        assert tracker.start_time is None

    def test_progress_tracking_flow(self):
        """Test complete progress tracking flow."""
        tracker = ProgressTracker(total_combinations=20, update_interval=5)

        # Start tracking
        tracker.start()
        assert tracker.start_time is not None

        # Update progress (should not raise errors)
        tracker.update(4, 0.5)  # Should print (multiple of update_interval)
        tracker.update(7, 0.6)  # Should not print
        tracker.update(19, 0.8)  # Should print (last item)

        # Finish tracking
        best_params = {"param1": 5, "param2": "test"}
        tracker.finish(0.8, best_params)


class TestParameterOptimizer:
    """Test the main ParameterOptimizer class."""

    @pytest.fixture
    def default_config(self):
        """Default optimizer configuration."""
        return {
            "scorer": "count_accuracy",
            "count_weight": 0.7,
            "measurement_weight": 0.3,
        }

    @pytest.fixture
    def optimizer(self, default_config):
        """Create ParameterOptimizer instance."""
        return ParameterOptimizer(default_config)

    @pytest.fixture
    def sample_image(self):
        """Create a sample image for optimization."""
        # Create an image with bean-like objects and a coin
        img = np.zeros((200, 200, 3), dtype=np.uint8)

        # Add a coin-like circle
        cv2.circle(img, (50, 50), 25, (200, 200, 200), -1)

        # Add some bean-like ellipses
        cv2.ellipse(img, (100, 60), (15, 10), 45, 0, 360, (100, 100, 100), -1)
        cv2.ellipse(img, (150, 60), (12, 8), 30, 0, 360, (110, 110, 110), -1)
        cv2.ellipse(img, (100, 140), (14, 9), 60, 0, 360, (105, 105, 105), -1)
        cv2.ellipse(img, (150, 140), (13, 10), 20, 0, 360, (95, 95, 95), -1)

        return img

    @pytest.fixture
    def sample_ground_truth(self):
        """Create sample ground truth data."""
        return pd.DataFrame(
            {
                "length": [60.0, 48.0, 56.0, 52.0],  # In pixels
                "width": [40.0, 32.0, 36.0, 40.0],  # In pixels
            }
        )

    def test_optimizer_initialization(self, default_config):
        """Test optimizer initialization."""
        optimizer = ParameterOptimizer(default_config)

        assert optimizer.config == default_config
        assert optimizer.scoring_function is not None
        assert optimizer.coin_detector is not None

    def test_invalid_scorer(self):
        """Test optimizer with invalid scorer."""
        config = {"scorer": "invalid_scorer"}

        with pytest.raises(ValueError, match="Unknown scorer type"):
            ParameterOptimizer(config)

    @patch("coffee_bean_analyzer.core.optimizer.BeanSegmentor")
    @patch("coffee_bean_analyzer.core.optimizer.BeanMeasurer")
    def test_process_single_combination(
        self, mock_measurer_class, mock_segmentor_class, optimizer, sample_image
    ):
        """Test processing a single parameter combination."""
        # Mock the segmentor and measurer
        mock_segmentor = Mock()
        mock_segmentation_result = Mock()
        mock_segmentation_result.labels = np.zeros((200, 200), dtype=np.int32)
        mock_segmentation_result.num_segments = 2
        mock_segmentation_result.segmentation_metadata = {}
        mock_segmentor.segment.return_value = mock_segmentation_result
        mock_segmentor_class.return_value = mock_segmentor

        mock_measurer = Mock()
        mock_measurement_result = Mock()
        mock_measurement_result.measurements = []
        mock_measurement_result.pixels_per_mm = 4.0
        mock_measurer.measure.return_value = mock_measurement_result
        mock_measurer_class.return_value = mock_measurer

        # Test processing
        params = {"gaussian_kernel": 5, "clahe_clip": 2.0}
        coin_detection = DetectionResult(
            center=(50, 50), radius=25.0, confidence=0.9, bbox=(25, 25, 50, 50)
        )

        score, measurements, metadata = optimizer._process_single_combination(
            sample_image, params, coin_detection
        )

        assert score == 0.0  # Score is calculated externally
        assert isinstance(measurements, list)
        assert isinstance(metadata, dict)
        assert "pixels_per_mm" in metadata

    def test_optimize_basic(self, optimizer, sample_image, sample_ground_truth):
        """Test basic optimization functionality."""
        # Create a very small parameter grid for testing
        small_grid = ParameterGrid(
            {
                "gaussian_kernel": [5],
                "clahe_clip": [2.0],
                "morph_kernel_size": [3],
                "close_iterations": [2],
                "open_iterations": [1],
                "min_distance": [20],
                "threshold_factor": [0.3],
            }
        )

        result = optimizer.optimize(
            sample_image,
            ground_truth=sample_ground_truth,
            parameter_grid=small_grid,
            debug=False,
        )

        assert isinstance(result, OptimizationResult)
        assert result.best_params is not None
        assert isinstance(result.best_score, float)
        assert isinstance(result.best_measurements, list)
        assert len(result.all_results) == 1  # Only one combination
        assert result.total_combinations_tested == 1
        assert result.optimization_time > 0

    def test_optimize_no_ground_truth(self, optimizer, sample_image):
        """Test optimization without ground truth."""
        small_grid = ParameterGrid(
            {
                "gaussian_kernel": [5],
                "clahe_clip": [2.0],
                "morph_kernel_size": [3],
                "close_iterations": [2],
                "open_iterations": [1],
                "min_distance": [20],
                "threshold_factor": [0.3],
            }
        )

        result = optimizer.optimize(
            sample_image, ground_truth=None, parameter_grid=small_grid, debug=False
        )

        assert isinstance(result, OptimizationResult)
        assert result.best_score == float("-inf")  # No ground truth = no valid score
        assert (
            result.best_params is not None
        )  # Should still have parameters from first combination

    def test_optimize_default_grid(self, optimizer, sample_image):
        """Test optimization with default parameter grid."""
        # This test uses the full default grid, so we'll limit it for speed
        # We can mock the processing to make it fast
        with patch.object(
            optimizer,
            "_process_single_combination",
            return_value=(0.5, [], {"pixels_per_mm": 4.0}),
        ):
            result = optimizer.optimize(sample_image, debug=False)

            assert isinstance(result, OptimizationResult)
            assert (
                result.total_combinations_tested > 1
            )  # Should test multiple combinations

    def test_get_top_results(self, optimizer):
        """Test getting top results from optimization."""
        # Create mock optimization result
        all_results = [
            {"params": {"param1": 1}, "score": 0.5, "num_detected": 3, "metadata": {}},
            {"params": {"param1": 2}, "score": 0.8, "num_detected": 4, "metadata": {}},
            {"params": {"param1": 3}, "score": 0.3, "num_detected": 2, "metadata": {}},
            {"params": {"param1": 4}, "score": 0.9, "num_detected": 4, "metadata": {}},
        ]

        mock_result = OptimizationResult(
            best_params={"param1": 4},
            best_score=0.9,
            best_measurements=[],
            all_results=all_results,
            optimization_metadata={},
            total_combinations_tested=4,
            optimization_time=10.0,
        )

        top_results = optimizer.get_top_results(mock_result, top_k=2)

        assert len(top_results) == 2
        assert top_results[0]["score"] == 0.9  # Highest score first
        assert top_results[1]["score"] == 0.8  # Second highest

    def test_analyze_parameter_importance(self, optimizer):
        """Test parameter importance analysis."""
        # Create mock results with clear parameter-score relationships
        all_results = [
            {
                "params": {"param1": 1, "param2": "a"},
                "score": 0.1,
                "num_detected": 1,
                "metadata": {},
            },
            {
                "params": {"param1": 2, "param2": "a"},
                "score": 0.3,
                "num_detected": 2,
                "metadata": {},
            },
            {
                "params": {"param1": 3, "param2": "a"},
                "score": 0.5,
                "num_detected": 3,
                "metadata": {},
            },
            {
                "params": {"param1": 1, "param2": "b"},
                "score": 0.2,
                "num_detected": 1,
                "metadata": {},
            },
            {
                "params": {"param1": 2, "param2": "b"},
                "score": 0.4,
                "num_detected": 2,
                "metadata": {},
            },
            {
                "params": {"param1": 3, "param2": "b"},
                "score": 0.6,
                "num_detected": 3,
                "metadata": {},
            },
        ]

        mock_result = OptimizationResult(
            best_params={"param1": 3, "param2": "b"},
            best_score=0.6,
            best_measurements=[],
            all_results=all_results,
            optimization_metadata={},
            total_combinations_tested=6,
            optimization_time=10.0,
        )

        importance = optimizer.analyze_parameter_importance(mock_result)

        assert isinstance(importance, dict)
        assert "param1" in importance
        assert "param2" in importance

        # param1 should have high importance (clear linear relationship with score)
        # param2 should have some importance (categorical, but categories have different score patterns)
        assert importance["param1"] > 0.8  # Strong correlation
        assert importance["param2"] >= 0.0  # Some categorical importance


class TestLegacyCompatibility:
    """Test legacy function compatibility."""

    @pytest.fixture
    def sample_image(self):
        """Create sample image."""
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.circle(img, (30, 30), 10, (100, 100, 100), -1)
        cv2.circle(img, (70, 70), 12, (110, 110, 110), -1)
        return img

    @pytest.fixture
    def sample_ground_truth(self):
        """Create sample ground truth."""
        return pd.DataFrame({"length": [40.0, 48.0], "width": [30.0, 36.0]})

    def test_legacy_function_signature(self, sample_image, sample_ground_truth):
        """Test that legacy function maintains original signature."""
        # Mock the optimization to make it fast
        with patch(
            "coffee_bean_analyzer.core.optimizer.ParameterOptimizer"
        ) as mock_optimizer_class:
            mock_optimizer = Mock()
            mock_result = OptimizationResult(
                best_params={"gaussian_kernel": 5},
                best_score=0.8,
                best_measurements=[],
                all_results=[
                    {
                        "params": {"gaussian_kernel": 5},
                        "metadata": {"pixels_per_mm": 4.0},
                    }
                ],
                optimization_metadata={},
                total_combinations_tested=1,
                optimization_time=1.0,
            )
            mock_optimizer.optimize.return_value = mock_result
            mock_optimizer_class.return_value = mock_optimizer

            best_params, best_result, best_score = grid_search_optimization_legacy(
                sample_image, sample_ground_truth, debug=False
            )

            # Check return format matches original
            assert isinstance(best_params, dict)
            assert isinstance(best_result, dict)
            assert isinstance(best_score, float)

            # Check result structure
            assert "image_name" in best_result
            assert "measurements" in best_result
            assert "pixels_per_mm" in best_result

    def test_legacy_with_image_path(self, tmp_path, sample_ground_truth):
        """Test legacy function with image path."""
        # Create a temporary image file
        img = np.zeros((50, 50, 3), dtype=np.uint8)
        cv2.circle(img, (25, 25), 10, (100, 100, 100), -1)

        image_path = tmp_path / "test_image.jpg"
        cv2.imwrite(str(image_path), img)

        # Mock the optimization
        with patch(
            "coffee_bean_analyzer.core.optimizer.ParameterOptimizer"
        ) as mock_optimizer_class:
            mock_optimizer = Mock()
            mock_result = OptimizationResult(
                best_params={"gaussian_kernel": 5},
                best_score=0.8,
                best_measurements=[],
                all_results=[],
                optimization_metadata={},
                total_combinations_tested=1,
                optimization_time=1.0,
            )
            mock_optimizer.optimize.return_value = mock_result
            mock_optimizer_class.return_value = mock_optimizer

            best_params, best_result, best_score = grid_search_optimization_legacy(
                str(image_path), sample_ground_truth, debug=False
            )

            assert isinstance(best_params, dict)
            assert isinstance(best_result, dict)
            assert isinstance(best_score, float)

    def test_legacy_invalid_image_path(self, sample_ground_truth):
        """Test legacy function with invalid image path."""
        with pytest.raises(ValueError, match="Could not load image"):
            grid_search_optimization_legacy(
                "nonexistent_image.jpg", sample_ground_truth, debug=False
            )


class TestOptimizerFactory:
    """Test the optimizer factory function."""

    def test_default_preset(self):
        """Test creating optimizer with default preset."""
        optimizer = create_optimizer("default")

        assert isinstance(optimizer, ParameterOptimizer)
        assert optimizer.config["scorer"] == "count_accuracy"
        assert optimizer.config["count_weight"] == 0.7
        assert optimizer.config["measurement_weight"] == 0.3

    def test_count_focused_preset(self):
        """Test count focused preset."""
        optimizer = create_optimizer("count_focused")

        assert optimizer.config["count_weight"] == 0.9
        assert optimizer.config["measurement_weight"] == 0.1

    def test_measurement_focused_preset(self):
        """Test measurement focused preset."""
        optimizer = create_optimizer("measurement_focused")

        assert optimizer.config["count_weight"] == 0.5
        assert optimizer.config["measurement_weight"] == 0.5

    def test_quick_preset(self):
        """Test quick preset."""
        optimizer = create_optimizer("quick")

        assert optimizer.config["use_quick_grid"] is True
        assert optimizer.config["count_weight"] == 0.7

    def test_parameter_override(self):
        """Test overriding preset parameters."""
        optimizer = create_optimizer(
            "default", count_weight=0.8, measurement_weight=0.2
        )

        assert optimizer.config["count_weight"] == 0.8
        assert optimizer.config["measurement_weight"] == 0.2
        # Other parameters should remain from preset
        assert optimizer.config["scorer"] == "count_accuracy"

    def test_invalid_preset(self):
        """Test creating optimizer with invalid preset."""
        with pytest.raises(ValueError, match="Unknown preset"):
            create_optimizer("invalid_preset")


class TestOptimizationResult:
    """Test OptimizationResult dataclass."""

    def test_optimization_result_creation(self):
        """Test OptimizationResult can be created with valid data."""
        result = OptimizationResult(
            best_params={"param1": 5},
            best_score=0.85,
            best_measurements=[],
            all_results=[],
            optimization_metadata={"test": "value"},
            total_combinations_tested=100,
            optimization_time=45.0,
        )

        assert result.best_params == {"param1": 5}
        assert result.best_score == 0.85
        assert result.best_measurements == []
        assert result.all_results == []
        assert result.optimization_metadata == {"test": "value"}
        assert result.total_combinations_tested == 100
        assert result.optimization_time == 45.0


@pytest.mark.integration
class TestRealOptimization:
    """Integration tests for real optimization scenarios."""

    def test_small_scale_optimization(self):
        """Test optimization with a very small parameter grid."""
        # Create a synthetic image
        img = np.zeros((100, 100, 3), dtype=np.uint8)

        # Add a coin
        cv2.circle(img, (25, 25), 15, (200, 200, 200), -1)

        # Add some bean-like objects
        cv2.ellipse(img, (60, 30), (8, 5), 0, 0, 360, (100, 100, 100), -1)
        cv2.ellipse(img, (60, 70), (9, 6), 0, 0, 360, (110, 110, 110), -1)

        # Create ground truth
        ground_truth = pd.DataFrame(
            {
                "length": [32.0, 36.0],  # In pixels
                "width": [20.0, 24.0],  # In pixels
            }
        )

        # Create optimizer
        optimizer = create_optimizer("default")

        # Create very small grid for speed
        small_grid = ParameterGrid(
            {
                "gaussian_kernel": [3, 5],
                "clahe_clip": [2.0],
                "morph_kernel_size": [3],
                "close_iterations": [2],
                "open_iterations": [1],
                "min_distance": [15, 20],
                "threshold_factor": [0.3],
            }
        )

        # Run optimization
        result = optimizer.optimize(img, ground_truth, small_grid, debug=False)

        # Check results
        assert isinstance(result, OptimizationResult)
        assert result.total_combinations_tested == 4  # 2 * 1 * 1 * 1 * 1 * 2 * 1
        assert result.best_params is not None
        assert result.optimization_time > 0
        assert len(result.all_results) == 4


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
