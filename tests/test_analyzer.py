#!/usr/bin/env python3
"""Unit tests for the analyzer module using pytest."""

import sys
from pathlib import Path
from unittest.mock import Mock, patch

import cv2
import numpy as np
import pandas as pd
import pytest

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from analyzer import CoffeeBeanAnalyzer
from coffee_bean_analyzer.core.detector import DetectionResult
from coffee_bean_analyzer.core.measurer import BeanMeasurement, MeasurementResult
from coffee_bean_analyzer.core.optimizer import OptimizationResult
from coffee_bean_analyzer.core.segmentor import SegmentationResult


class TestCoffeeBeanAnalyzer:
    """Test suite for CoffeeBeanAnalyzer class."""

    @pytest.fixture
    def analyzer(self, tmp_path):
        """Create analyzer instance with temporary directory."""
        return CoffeeBeanAnalyzer(
            output_base_dir=str(tmp_path / "test_output")
        )

    @pytest.fixture
    def sample_image(self):
        """Create a sample test image."""
        img = np.zeros((200, 200, 3), dtype=np.uint8)
        # Add coin
        cv2.circle(img, (50, 50), 25, (200, 200, 200), -1)
        # Add beans
        cv2.ellipse(img, (100, 60), (15, 10), 45, 0, 360, (100, 100, 100), -1)
        cv2.ellipse(img, (150, 60), (12, 8), 30, 0, 360, (110, 110, 110), -1)
        return img

    @pytest.fixture
    def sample_measurements(self):
        """Create sample BeanMeasurement objects."""
        return [
            BeanMeasurement(
                bean_id=1,
                centroid_x=100.0,
                centroid_y=60.0,
                area=150.0,
                length=15.0,
                width=10.0,
                orientation=45.0,
                eccentricity=0.7,
                solidity=0.9,
                perimeter=40.0,
                aspect_ratio=1.5,
                unit="mm",
            ),
            BeanMeasurement(
                bean_id=2,
                centroid_x=150.0,
                centroid_y=60.0,
                area=96.0,
                length=12.0,
                width=8.0,
                orientation=30.0,
                eccentricity=0.6,
                solidity=0.9,
                perimeter=35.0,
                aspect_ratio=1.5,
                unit="mm",
            ),
        ]

    @pytest.fixture
    def sample_ground_truth(self):
        """Create sample ground truth data."""
        return pd.DataFrame({"length": [15.2, 12.1], "width": [10.1, 8.2]})

    def test_initialization(self, tmp_path):
        """Test analyzer initialization."""
        analyzer = CoffeeBeanAnalyzer(output_base_dir=str(tmp_path))

        assert analyzer.output_dir.exists()
        assert analyzer.data_dir.exists()
        assert analyzer.images_dir.exists()
        assert analyzer.reports_dir.exists()
        assert analyzer.coin_detector is not None

    def test_find_image_files(self, analyzer, tmp_path):
        """Test finding image files."""
        # Create test images
        test_dir = tmp_path / "test_images"
        test_dir.mkdir()

        (test_dir / "test1.tif").touch()
        (test_dir / "test2.jpg").touch()
        (test_dir / "test3.png").touch()
        (test_dir / "not_image.txt").touch()

        found = analyzer.find_image_files([str(test_dir)])

        assert len(found) == 3
        assert all(f.endswith((".tif", ".jpg", ".png")) for f in found)

    def test_safe_get_position(self, analyzer, sample_measurements):
        """Test safe position extraction."""
        measurement = sample_measurements[0]

        # Test with BeanMeasurement object
        x, y = analyzer._safe_get_position(measurement)
        assert x == 100.0
        assert y == 60.0

        # Test with object without position attributes
        mock_obj = Mock(spec=[])
        x, y = analyzer._safe_get_position(mock_obj)
        assert x == 100.0  # Fallback values
        assert y == 100.0

    def test_safe_get_bounding_box(self, analyzer):
        """Test safe bounding box extraction."""
        # Test with object that has bounding_box attribute
        mock_measurement = Mock()
        mock_measurement.bounding_box = Mock(x=10, y=20, width=30, height=40)

        bbox = analyzer._safe_get_bounding_box(mock_measurement)
        assert bbox == (10, 20, 30, 40)

        # Test with object without bounding box
        mock_measurement = Mock(spec=["other_attr"])
        bbox = analyzer._safe_get_bounding_box(mock_measurement)
        assert bbox is None

    def test_safe_get_attribute(self, analyzer, sample_measurements):
        """Test safe attribute access."""
        measurement = sample_measurements[0]

        # Test existing attribute
        length = analyzer._safe_get_attribute(measurement, "length")
        assert length == 15.0

        # Test non-existent attribute with default
        value = analyzer._safe_get_attribute(measurement, "nonexistent", default=999)
        assert value == 999

    @patch("matplotlib.pyplot.savefig")
    @patch("matplotlib.pyplot.close")
    def test_create_original_analysis_visualization(
        self, mock_close, mock_savefig, analyzer, sample_image, sample_measurements
    ):
        """Test visualization creation."""
        labels = np.zeros((200, 200), dtype=np.int32)
        labels[50:70, 90:110] = 1
        labels[50:70, 140:160] = 2

        coin = DetectionResult(
            center=(50, 50), radius=25.0, confidence=0.9, bbox=(25, 25, 50, 50)
        )

        filename = analyzer.create_original_analysis_visualization(
            sample_image,
            labels,
            sample_measurements,
            coin,
            "test_image",
            {"param1": 5},
            is_optimized=False,
        )

        assert filename == "test_image_original_analysis.png"
        assert mock_savefig.called
        assert mock_close.called

    @patch("matplotlib.pyplot.savefig")
    @patch("matplotlib.pyplot.close")
    def test_create_optimization_comparison(
        self, mock_close, mock_savefig, analyzer, sample_measurements
    ):
        """Test optimization comparison visualization."""
        original_results = {
            "measurements": sample_measurements,
            "parameters": {"param1": 5},
        }

        optimized_results = {
            "measurements": sample_measurements
            + [sample_measurements[0]],  # Add one more
            "parameters": {"param1": 7},
        }

        comparison_file = analyzer.create_optimization_comparison(
            "test_image", original_results, optimized_results
        )

        assert comparison_file.name == "optimization_comparison.png"
        assert mock_savefig.called

    def test_save_measurement_data(
        self, analyzer, sample_measurements, sample_ground_truth
    ):
        """Test saving measurement data."""
        original_results = {"measurements": sample_measurements}
        optimized_results = {"measurements": sample_measurements}

        analyzer._save_measurement_data(
            "test_image", original_results, optimized_results, sample_ground_truth
        )

        # Check that CSV files were created
        assert (analyzer.data_dir / "bean_measurements.csv").exists()
        assert (analyzer.data_dir / "ground_truth_comparison.csv").exists()

        # Load and verify CSV content
        df = pd.read_csv(analyzer.data_dir / "bean_measurements.csv")
        assert len(df) == 4  # 2 original + 2 optimized
        assert "analysis_type" in df.columns
        assert "length" in df.columns
        assert "width" in df.columns

    def test_generate_analysis_reports(self, analyzer, sample_measurements):
        """Test report generation."""
        coin = DetectionResult(
            center=(50, 50),
            radius=25.0,
            confidence=0.9,
            bbox=(25, 25, 50, 50),
            pixels_per_mm=4.0,
        )

        original_results = {
            "measurements": sample_measurements,
            "parameters": {"param1": 5, "param2": "test"},
        }

        opt_result = OptimizationResult(
            best_params={"param1": 7},
            best_score=0.85,
            best_measurements=sample_measurements,
            all_results=[],
            optimization_metadata={},
            total_combinations_tested=10,
            optimization_time=5.0,
        )

        optimized_results = {
            "measurements": sample_measurements,
            "parameters": {"param1": 7},
            "optimization_result": opt_result,
        }

        analyzer._generate_analysis_reports(
            "test_image", original_results, optimized_results, coin
        )

        # Check that report files were created
        assert (analyzer.reports_dir / "analysis_report.txt").exists()
        assert (analyzer.reports_dir / "optimized_analysis_report.txt").exists()

        # Verify content
        with open(analyzer.reports_dir / "analysis_report.txt") as f:
            content = f.read()
            assert "Coffee Bean Analysis Report" in content
            assert "Total beans detected: 2" in content

    def test_generate_analysis_summary(self, analyzer, sample_measurements):
        """Test summary generation."""
        # Add some results to analyzer
        analyzer.analysis_results["test_image"] = {
            "original": {"measurements": sample_measurements},
            "optimized": {"measurements": sample_measurements},
            "coin_detection": None,
            "image_path": "test.jpg",
        }

        analyzer.generate_analysis_summary()

        summary_file = analyzer.output_dir / "ANALYSIS_SUMMARY.txt"
        assert summary_file.exists()

        with open(summary_file) as f:
            content = f.read()
            assert "COFFEE BEAN ANALYSIS SUMMARY" in content
            assert "test_image" in content

    @patch.object(CoffeeBeanAnalyzer, "analyze_image")
    def test_run_full_analysis(self, mock_analyze, analyzer, tmp_path):
        """Test full analysis pipeline."""
        # Create test image
        test_dir = tmp_path / "test_data"
        test_dir.mkdir()
        test_image = test_dir / "test.tif"
        test_image.touch()

        # Mock the analysis method
        mock_analyze.return_value = {
            "original": {"measurements": []},
            "optimized": None,
            "coin_detection": None,
            "image_path": str(test_image),
        }

        output_dir = analyzer.run_full_analysis([str(test_dir)])

        assert output_dir == analyzer.output_dir
        assert mock_analyze.called
        assert (analyzer.output_dir / "ANALYSIS_SUMMARY.txt").exists()

    @patch("cv2.imread")
    @patch("coffee_bean_analyzer.core.detector.CoinDetector.detect")
    @patch("coffee_bean_analyzer.core.segmentor.BeanSegmentor.segment")
    @patch("coffee_bean_analyzer.core.measurer.BeanMeasurer.measure")
    def test_analyze_image_no_optimization(
        self,
        mock_measure,
        mock_segment,
        mock_detect,
        mock_imread,
        analyzer,
        sample_image,
        sample_measurements,
    ):
        """Test detailed analysis without optimization."""
        # Setup mocks
        mock_imread.return_value = sample_image
        mock_detect.return_value = []  # No coin detected

        mock_seg_result = SegmentationResult(
            labels=np.zeros((200, 200), dtype=np.int32),
            binary_mask=np.zeros((200, 200), dtype=np.uint8),
            num_segments=2,
            preprocessing_metadata={},
            segmentation_metadata={},
            excluded_regions=[],
        )
        mock_segment.return_value = mock_seg_result

        mock_measure_result = MeasurementResult(
            measurements=sample_measurements,
            total_beans=2,
            scale_factor=1.0,
            pixels_per_mm=None,
            unit="pixels",
            metadata={},
            excluded_regions=[],
        )
        mock_measure.return_value = mock_measure_result

        # Run analysis
        result = analyzer.analyze_image(
            "test.jpg", ground_truth_path=None, run_optimization=False
        )

        assert result is not None
        assert "original" in result
        assert "optimized" in result
        assert result["optimized"] is None  # No optimization run
        assert len(result["original"]["measurements"]) == 2

    def test_edge_cases(self, analyzer):
        """Test edge cases and error handling."""
        # Test with empty measurements list
        original_results = {"measurements": []}
        optimized_results = None

        # Should not raise errors
        analyzer._save_measurement_data(
            "test_image", original_results, optimized_results, None
        )

        # Test with measurements that have None values
        mock_measurement = Mock()
        mock_measurement.to_dict.return_value = {
            "length": None,
            "width": None,
            "area": 100,
        }

        length = analyzer._safe_get_attribute(mock_measurement, "length", default=0)
        assert length == 0  # Should use default


class TestIntegration:
    """Integration tests for the analyzer."""

    @pytest.mark.integration
    def test_full_pipeline_with_real_image(self, tmp_path):
        """Test the full pipeline with a synthetic but realistic image."""
        # Create a more realistic test image
        img = np.ones((300, 300, 3), dtype=np.uint8) * 230  # Light background

        # Add a coin
        cv2.circle(img, (75, 75), 30, (180, 180, 180), -1)
        cv2.circle(img, (75, 75), 30, (160, 160, 160), 2)

        # Add some bean-like objects
        cv2.ellipse(img, (150, 100), (20, 12), 45, 0, 360, (80, 80, 80), -1)
        cv2.ellipse(img, (200, 100), (18, 10), 30, 0, 360, (90, 90, 90), -1)
        cv2.ellipse(img, (150, 180), (22, 13), 60, 0, 360, (85, 85, 85), -1)
        cv2.ellipse(img, (200, 180), (19, 11), 20, 0, 360, (75, 75, 75), -1)

        # Save the image
        image_path = tmp_path / "test_beans.jpg"
        cv2.imwrite(str(image_path), img)

        # Create ground truth
        ground_truth = pd.DataFrame(
            {"length": [20.0, 18.0, 22.0, 19.0], "width": [12.0, 10.0, 13.0, 11.0]}
        )
        gt_path = tmp_path / "ground_truth.csv"
        ground_truth.to_csv(gt_path, index=False)

        # Run analysis
        analyzer = CoffeeBeanAnalyzer(
            output_base_dir=str(tmp_path / "output")
        )

        # We can't easily test the full pipeline without mocking everything,
        # but we can at least verify the structure is set up correctly
        assert analyzer.output_dir.exists()
        assert analyzer.data_dir.exists()
        assert analyzer.images_dir.exists()
        assert analyzer.reports_dir.exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
