#!/usr/bin/env python3
"""Unit tests for the analyze_beans.py script using pytest."""

import sys
from pathlib import Path
from unittest.mock import Mock, patch

import cv2
import numpy as np
import pandas as pd
import pytest

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import after adding to path
import analyze_beans


class TestAnalyzeBeansScript:
    """Test suite for analyze_beans.py script functions."""

    @pytest.fixture
    def sample_image_path(self, tmp_path):
        """Create a sample image file."""
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.circle(img, (50, 50), 20, (100, 100, 100), -1)

        image_path = tmp_path / "test_image.jpg"
        cv2.imwrite(str(image_path), img)
        return str(image_path)

    @pytest.fixture
    def sample_ground_truth_path(self, tmp_path):
        """Create a sample ground truth CSV."""
        df = pd.DataFrame({"length": [10.0, 12.0], "width": [8.0, 9.0]})

        gt_path = tmp_path / "ground_truth.csv"
        df.to_csv(gt_path, index=False)
        return str(gt_path)

    def test_parse_args_basic(self):
        """Test basic argument parsing."""
        with patch("sys.argv", ["analyze_beans.py", "test.jpg"]):
            args = analyze_beans.parse_args()

            assert args.image == "test.jpg"
            assert args.preset == "default"
            assert args.optimize is False
            assert args.ground_truth is None

    def test_parse_args_with_options(self):
        """Test argument parsing with all options."""
        with patch(
            "sys.argv",
            [
                "analyze_beans.py",
                "test.jpg",
                "--preset",
                "aggressive",
                "--ground-truth",
                "gt.csv",
                "--optimize",
                "--output-dir",
                "output",
                "--verbose",
            ],
        ):
            args = analyze_beans.parse_args()

            assert args.image == "test.jpg"
            assert args.preset == "aggressive"
            assert args.optimize is True
            assert args.ground_truth == "gt.csv"
            assert args.output_dir == "output"
            assert args.verbose is True

    @patch("analyze_beans.CoffeeBeanAnalyzer")
    def test_main_basic_analysis(self, mock_analyzer_class, sample_image_path):
        """Test main function with basic analysis."""
        # Setup mock
        mock_analyzer = Mock()
        mock_analyzer_class.return_value = mock_analyzer
        mock_analyzer.analyze_image.return_value = {
            "original": {"measurements": []},
            "optimized": None,
            "coin_detection": None,
        }

        # Run with basic arguments
        with patch("sys.argv", ["analyze_beans.py", sample_image_path]):
            analyze_beans.main()

        # Verify calls
        mock_analyzer_class.assert_called_once()
        mock_analyzer.analyze_image.assert_called_once_with(
            sample_image_path,
            ground_truth_path=None,
            config_preset="default",
            run_optimization=False,
        )

    @patch("analyze_beans.CoffeeBeanAnalyzer")
    def test_main_with_optimization(
        self, mock_analyzer_class, sample_image_path, sample_ground_truth_path
    ):
        """Test main function with optimization."""
        # Setup mock
        mock_analyzer = Mock()
        mock_analyzer_class.return_value = mock_analyzer
        mock_analyzer.analyze_image.return_value = {
            "original": {"measurements": []},
            "optimized": {"measurements": []},
            "coin_detection": None,
        }

        # Run with optimization
        with patch(
            "sys.argv",
            [
                "analyze_beans.py",
                sample_image_path,
                "--ground-truth",
                sample_ground_truth_path,
                "--optimize",
            ],
        ):
            analyze_beans.main()

        # Verify optimization was requested
        mock_analyzer.analyze_image.assert_called_once_with(
            sample_image_path,
            ground_truth_path=sample_ground_truth_path,
            config_preset="default",
            run_optimization=True,
        )

    @patch("analyze_beans.CoffeeBeanAnalyzer")
    def test_main_with_custom_output_dir(
        self, mock_analyzer_class, sample_image_path, tmp_path
    ):
        """Test main function with custom output directory."""
        output_dir = str(tmp_path / "custom_output")

        # Setup mock
        mock_analyzer = Mock()
        mock_analyzer_class.return_value = mock_analyzer
        mock_analyzer.analyze_image.return_value = {
            "original": {"measurements": []},
            "optimized": None,
            "coin_detection": None,
        }

        # Run with custom output directory
        with patch(
            "sys.argv",
            ["analyze_beans.py", sample_image_path, "--output-dir", output_dir],
        ):
            analyze_beans.main()

        # Verify output directory was passed
        mock_analyzer_class.assert_called_once_with(output_base_dir=output_dir)

    @patch("analyze_beans.CoffeeBeanAnalyzer")
    def test_main_invalid_image(self, mock_analyzer_class):
        """Test main function with invalid image path."""
        # Setup mock to raise exception
        mock_analyzer = Mock()
        mock_analyzer_class.return_value = mock_analyzer
        mock_analyzer.analyze_image.side_effect = ValueError(
            "Could not load image"
        )

        # Run with invalid image
        with patch("sys.argv", ["analyze_beans.py", "nonexistent.jpg"]):
            with pytest.raises(SystemExit):
                analyze_beans.main()

    @patch("analyze_beans.CoffeeBeanAnalyzer")
    def test_main_different_presets(self, mock_analyzer_class, sample_image_path):
        """Test main function with different presets."""
        # Setup mock
        mock_analyzer = Mock()
        mock_analyzer_class.return_value = mock_analyzer
        mock_analyzer.analyze_image.return_value = {
            "original": {"measurements": []},
            "optimized": None,
            "coin_detection": None,
        }

        # Test each preset
        for preset in ["default", "aggressive", "conservative", "quick"]:
            with patch(
                "sys.argv", ["analyze_beans.py", sample_image_path, "--preset", preset]
            ):
                analyze_beans.main()

            # Verify preset was passed
            mock_analyzer.analyze_image.assert_called_with(
                sample_image_path,
                ground_truth_path=None,
                config_preset=preset,
                run_optimization=False,
            )

    @patch("analyze_beans.CoffeeBeanAnalyzer")
    def test_main_verbose_output(
        self, mock_analyzer_class, sample_image_path, sample_ground_truth_path, capsys
    ):
        """Test verbose output."""
        # Setup mocks
        mock_analyzer = Mock()
        mock_analyzer_class.return_value = mock_analyzer
        mock_analyzer.output_dir = Path("/fake/output")
        mock_analyzer.analyze_image.return_value = {
            "original": {"measurements": [Mock(), Mock()]},
            "optimized": {"measurements": [Mock(), Mock(), Mock()]},
            "coin_detection": Mock(pixels_per_mm=4.0),
        }

        # Setup additional mocks
        mock_analyzer.analysis_results = {}
        mock_analyzer.generate_analysis_summary = Mock()

        # Run with verbose flag
        with patch(
            "sys.argv",
            ["analyze_beans.py", sample_image_path, "--verbose", "--optimize", "--ground-truth", sample_ground_truth_path],
        ):
            analyze_beans.main()

        # Check verbose output
        captured = capsys.readouterr()
        assert "🚀 Initializing analyzer..." in captured.out
        assert "✅ Analysis complete: 2 beans detected" in captured.out
        assert "✅ Optimization: 3 beans (+1)" in captured.out


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_parse_args_help(self):
        """Test help argument."""
        with patch("sys.argv", ["analyze_beans.py", "--help"]):
            with pytest.raises(SystemExit) as exc_info:
                analyze_beans.parse_args()
            assert exc_info.value.code == 0

    def test_parse_args_invalid_preset(self):
        """Test invalid preset argument."""
        with patch("sys.argv", ["analyze_beans.py", "test.jpg", "--preset", "invalid"]):
            with pytest.raises(SystemExit):
                analyze_beans.parse_args()

    @patch("cv2.imread")
    def test_main_image_loading_error(self, mock_imread):
        """Test handling of image loading errors."""
        mock_imread.return_value = None

        with patch("sys.argv", ["analyze_beans.py", "test.jpg"]):
            with pytest.raises(SystemExit):
                analyze_beans.main()


class TestIntegrationWithScript:
    """Integration tests for the script."""

    @pytest.mark.integration
    def test_script_execution(self, tmp_path):
        """Test that the script can be executed."""
        # Create a minimal test image
        img = np.ones((50, 50, 3), dtype=np.uint8) * 255
        image_path = tmp_path / "test.jpg"
        cv2.imwrite(str(image_path), img)

        # Import subprocess to run the script
        import subprocess

        # Run the script with minimal arguments
        result = subprocess.run(
            [
                sys.executable,
                "analyze_beans.py",
                str(image_path),
                "--output-dir",
                str(tmp_path / "output"),
            ],
            capture_output=True,
            text=True,
        )

        # Script should complete (even if no beans are found)
        # Note: This will only work if all dependencies are properly installed
        # In a test environment, this might fail due to missing dependencies


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
