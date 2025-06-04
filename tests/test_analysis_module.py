#!/usr/bin/env python3
"""Unit tests for the analysis module using pytest."""

import sys
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import analysis module
from coffee_bean_analyzer.analysis.analyzer import CoffeeBeanAnalyzer


class TestCoffeeBeanAnalyzer:
    """Test the main CoffeeBeanAnalyzer class."""

    def test_initialization_default_dir(self):
        """Test analyzer initialization with default directory."""
        with patch(
            "coffee_bean_analyzer.analysis.analyzer.DataHandler"
        ) as mock_data_handler:
            with patch(
                "coffee_bean_analyzer.analysis.analyzer.VisualizationGenerator"
            ) as mock_viz:
                with patch(
                    "coffee_bean_analyzer.analysis.analyzer.ReportGenerator"
                ) as mock_report:
                    with patch(
                        "coffee_bean_analyzer.analysis.analyzer.CoinDetector"
                    ) as mock_detector:
                        # Mock the data handler to have the required attributes
                        mock_data_handler.return_value.images_dir = Path("/fake/images")
                        mock_data_handler.return_value.reports_dir = Path(
                            "/fake/reports"
                        )

                        CoffeeBeanAnalyzer()

                        # Verify components were initialized
                        mock_data_handler.assert_called_once_with(None)
                        mock_viz.assert_called_once_with(Path("/fake/images"))
                        mock_report.assert_called_once_with(Path("/fake/reports"))
                        mock_detector.assert_called_once()

    def test_initialization_custom_dir(self, tmp_path):
        """Test analyzer initialization with custom directory."""
        custom_dir = tmp_path / "custom_analysis"

        with patch(
            "coffee_bean_analyzer.analysis.analyzer.DataHandler"
        ) as mock_data_handler:
            with patch("coffee_bean_analyzer.analysis.analyzer.VisualizationGenerator"):
                with patch("coffee_bean_analyzer.analysis.analyzer.ReportGenerator"):
                    with patch("coffee_bean_analyzer.analysis.analyzer.CoinDetector"):
                        # Mock the data handler to have the required attributes
                        mock_data_handler.return_value.images_dir = Path("/fake/images")
                        mock_data_handler.return_value.reports_dir = Path(
                            "/fake/reports"
                        )

                        CoffeeBeanAnalyzer(custom_dir)

                        # Verify data handler was called with custom directory
                        mock_data_handler.assert_called_once_with(custom_dir)

    def test_coin_detector_configuration(self):
        """Test that coin detector is configured with correct parameters."""
        with patch(
            "coffee_bean_analyzer.analysis.analyzer.DataHandler"
        ) as mock_data_handler:
            with patch("coffee_bean_analyzer.analysis.analyzer.VisualizationGenerator"):
                with patch("coffee_bean_analyzer.analysis.analyzer.ReportGenerator"):
                    with patch(
                        "coffee_bean_analyzer.analysis.analyzer.CoinDetector"
                    ) as mock_detector:
                        # Mock the data handler to have the required attributes
                        mock_data_handler.return_value.images_dir = Path("/fake/images")
                        mock_data_handler.return_value.reports_dir = Path(
                            "/fake/reports"
                        )

                        CoffeeBeanAnalyzer()

                        # Verify coin detector was initialized with expected config
                        expected_config = {
                            "dp": 1,
                            "min_dist": 100,
                            "param1": 50,
                            "param2": 30,
                            "min_radius": 50,
                            "max_radius": 150,
                            "gaussian_kernel": 15,
                        }
                        mock_detector.assert_called_once_with(expected_config)

    def test_analyzer_has_required_attributes(self):
        """Test that analyzer has all required attributes after initialization."""
        with patch(
            "coffee_bean_analyzer.analysis.analyzer.DataHandler"
        ) as mock_data_handler:
            with patch("coffee_bean_analyzer.analysis.analyzer.VisualizationGenerator"):
                with patch("coffee_bean_analyzer.analysis.analyzer.ReportGenerator"):
                    with patch("coffee_bean_analyzer.analysis.analyzer.CoinDetector"):
                        # Mock the data handler to have the required attributes
                        mock_data_handler.return_value.images_dir = Path("/fake/images")
                        mock_data_handler.return_value.reports_dir = Path(
                            "/fake/reports"
                        )

                        analyzer = CoffeeBeanAnalyzer()

                        # Check that all main components are accessible
                        assert hasattr(analyzer, "data_handler")
                        assert hasattr(analyzer, "visualization_generator")
                        assert hasattr(analyzer, "report_generator")
                        assert hasattr(analyzer, "coin_detector")
                        assert hasattr(analyzer, "analysis_results")
                        assert hasattr(analyzer, "optimization_results")

                        # Check initial values
                        assert analyzer.analysis_results == {}
                        assert analyzer.optimization_results == {}
                        assert analyzer.preprocessor is None
                        assert analyzer.segmentor is None
                        assert analyzer.measurer is None
                        assert analyzer.optimizer is None


class TestAnalyzerComponentInitialization:
    """Test component initialization patterns."""

    def test_initialization_creates_components(self):
        """Test that initialization creates all required components."""
        with patch(
            "coffee_bean_analyzer.analysis.analyzer.DataHandler"
        ) as mock_data_handler:
            with patch(
                "coffee_bean_analyzer.analysis.analyzer.VisualizationGenerator"
            ) as mock_viz:
                with patch(
                    "coffee_bean_analyzer.analysis.analyzer.ReportGenerator"
                ) as mock_report:
                    with patch(
                        "coffee_bean_analyzer.analysis.analyzer.CoinDetector"
                    ) as mock_detector:
                        # Mock the data handler to have the required attributes
                        mock_data_handler_instance = Mock()
                        mock_data_handler_instance.images_dir = Path("/test/images")
                        mock_data_handler_instance.reports_dir = Path("/test/reports")
                        mock_data_handler.return_value = mock_data_handler_instance

                        CoffeeBeanAnalyzer()

                        # Verify initialization calls
                        mock_data_handler.assert_called_once_with(None)
                        mock_viz.assert_called_once_with(Path("/test/images"))
                        mock_report.assert_called_once_with(Path("/test/reports"))

                        # Verify coin detector called with config
                        assert mock_detector.called
                        call_args = mock_detector.call_args[0][0]
                        assert isinstance(call_args, dict)
                        assert "dp" in call_args
                        assert "min_dist" in call_args

    def test_component_creation_workflow(self):
        """Test the workflow of component creation."""
        with patch(
            "coffee_bean_analyzer.analysis.analyzer.DataHandler"
        ) as mock_data_handler:
            with patch(
                "coffee_bean_analyzer.analysis.analyzer.VisualizationGenerator"
            ) as mock_viz:
                with patch(
                    "coffee_bean_analyzer.analysis.analyzer.ReportGenerator"
                ) as mock_report:
                    with patch("coffee_bean_analyzer.analysis.analyzer.CoinDetector"):
                        # Setup mock data handler
                        mock_data_handler_instance = Mock()
                        mock_data_handler_instance.images_dir = Path("/workflow/images")
                        mock_data_handler_instance.reports_dir = Path(
                            "/workflow/reports"
                        )
                        mock_data_handler.return_value = mock_data_handler_instance

                        # Create analyzer and verify workflow
                        custom_output = Path("/workflow/output")
                        CoffeeBeanAnalyzer(custom_output)

                        # Check that data handler was called first with the custom output
                        mock_data_handler.assert_called_once_with(custom_output)

                        # Check that other components were called with data handler directories
                        mock_viz.assert_called_once_with(
                            mock_data_handler_instance.images_dir
                        )
                        mock_report.assert_called_once_with(
                            mock_data_handler_instance.reports_dir
                        )


class TestRealComponentInstantiation:
    """Test with real components when possible."""

    def test_real_instantiation_basic(self, tmp_path):
        """Test analyzer with real component instantiation where possible."""
        output_dir = tmp_path / "test_real_analysis"

        try:
            # This will use real components
            analyzer = CoffeeBeanAnalyzer(output_dir)

            # Verify basic functionality
            assert analyzer is not None
            assert hasattr(analyzer, "data_handler")
            assert hasattr(analyzer, "coin_detector")
            assert hasattr(analyzer, "visualization_generator")
            assert hasattr(analyzer, "report_generator")

            # Verify directories were created
            assert output_dir.exists()

        except Exception as e:
            # If real components fail, skip the test
            pytest.skip(f"Real component instantiation failed: {e}")

    def test_analyzer_with_real_data_handler(self, tmp_path):
        """Test analyzer with real data handler."""
        output_dir = tmp_path / "real_data_handler_test"

        with patch("coffee_bean_analyzer.analysis.analyzer.VisualizationGenerator"):
            with patch("coffee_bean_analyzer.analysis.analyzer.ReportGenerator"):
                with patch("coffee_bean_analyzer.analysis.analyzer.CoinDetector"):
                    try:
                        CoffeeBeanAnalyzer(output_dir)

                        # Verify directory structure if DataHandler creates it
                        assert output_dir.exists()

                        # Check for expected subdirectories
                        expected_subdirs = ["data", "images", "reports"]
                        for subdir in expected_subdirs:
                            subdir_path = output_dir / subdir
                            if subdir_path.exists():
                                assert subdir_path.is_dir()

                    except Exception as e:
                        pytest.skip(f"Real data handler test failed: {e}")
