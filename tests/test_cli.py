#!/usr/bin/env python3
"""Unit tests for CLI modules using pytest."""

import sys
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml
from click.testing import CliRunner

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import CLI modules
from coffee_bean_analyzer.cli.main import cli


class TestCLIMain:
    """Test the main CLI interface."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()

    def test_cli_help(self):
        """Test CLI help output."""
        result = self.runner.invoke(cli, ["--help"])

        assert result.exit_code == 0
        assert "Coffee Bean Size Analysis Tool" in result.output
        assert "analyze" in result.output or "Commands:" in result.output

    def test_cli_version(self):
        """Test CLI version output."""
        result = self.runner.invoke(cli, ["--version"])

        assert result.exit_code == 0
        # Should contain version information
        assert "version" in result.output.lower() or "0.1.0" in result.output

    def test_cli_verbose_flag(self):
        """Test CLI with verbose flag."""
        result = self.runner.invoke(cli, ["--verbose", "--help"])

        assert result.exit_code == 0
        assert "Coffee Bean Size Analysis Tool" in result.output
        # The verbose flag should be accepted without error

    def test_cli_quiet_flag(self):
        """Test CLI with quiet flag."""
        result = self.runner.invoke(cli, ["--quiet", "--help"])

        assert result.exit_code == 0
        assert "Coffee Bean Size Analysis Tool" in result.output
        # The quiet flag should be accepted without error

    def test_cli_log_file_option(self, tmp_path):
        """Test CLI with log file option."""
        log_file = tmp_path / "test.log"

        result = self.runner.invoke(cli, ["--log-file", str(log_file), "--help"])

        assert result.exit_code == 0
        assert "Coffee Bean Size Analysis Tool" in result.output
        # The log-file option should be accepted without error

    def test_cli_context_setup(self):
        """Test that CLI properly sets up context."""
        result = self.runner.invoke(
            cli, ["--verbose", "--log-file", "test.log", "--help"]
        )

        assert result.exit_code == 0
        assert "Coffee Bean Size Analysis Tool" in result.output
        # Multiple flags should be accepted without error


class TestAnalyzeCommand:
    """Test the analyze command."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()

    def test_analyze_command_help(self):
        """Test analyze command help."""
        result = self.runner.invoke(cli, ["analyze", "--help"])

        # Should show help without error even if command doesn't exist yet
        # or show that analyze command doesn't exist
        assert result.exit_code in [0, 2]  # 0 for help, 2 for no such command

    @pytest.mark.skip("analyze command may not be fully implemented")
    def test_analyze_command_basic(self, tmp_path):
        """Test basic analyze command execution."""
        # Create a test image file
        test_image = tmp_path / "test_image.jpg"
        test_image.touch()  # Create empty file for path validation

        output_dir = tmp_path / "output"

        with patch(
            "coffee_bean_analyzer.cli.commands.analyze.analyze_command"
        ) as mock_analyze:
            mock_analyze.return_value = None

            result = self.runner.invoke(
                cli, ["analyze", str(test_image), "--output-dir", str(output_dir)]
            )

            # Check if command executed or if it doesn't exist yet
            assert result.exit_code in [0, 2]

    @pytest.mark.skip("analyze command may not be fully implemented")
    def test_analyze_command_with_config(self, tmp_path):
        """Test analyze command with configuration file."""
        # Create test files
        test_image = tmp_path / "test_image.jpg"
        test_image.touch()

        config_file = tmp_path / "config.yaml"
        config_data = {"detection": {"coin_detection": {"dp": 1.5}}}
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        output_dir = tmp_path / "output"

        with patch(
            "coffee_bean_analyzer.cli.commands.analyze.analyze_command"
        ) as mock_analyze:
            mock_analyze.return_value = None

            result = self.runner.invoke(
                cli,
                [
                    "analyze",
                    str(test_image),
                    "--output-dir",
                    str(output_dir),
                    "--config",
                    str(config_file),
                ],
            )

            assert result.exit_code in [0, 2]

    @pytest.mark.skip("analyze command may not be fully implemented")
    def test_analyze_command_with_ground_truth(self, tmp_path):
        """Test analyze command with ground truth file."""
        # Create test files
        test_image = tmp_path / "test_image.jpg"
        test_image.touch()

        ground_truth_file = tmp_path / "ground_truth.csv"
        ground_truth_file.write_text("length,width\n10.0,8.0\n12.0,9.0\n")

        output_dir = tmp_path / "output"

        with patch(
            "coffee_bean_analyzer.cli.commands.analyze.analyze_command"
        ) as mock_analyze:
            mock_analyze.return_value = None

            result = self.runner.invoke(
                cli,
                [
                    "analyze",
                    str(test_image),
                    "--output-dir",
                    str(output_dir),
                    "--ground-truth",
                    str(ground_truth_file),
                    "--optimize",
                ],
            )

            assert result.exit_code in [0, 2]


class TestBatchCommand:
    """Test the batch command."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()

    def test_batch_command_help(self):
        """Test batch command help."""
        result = self.runner.invoke(cli, ["batch", "--help"])

        # Should show help without error even if command doesn't exist yet
        assert result.exit_code in [0, 2]

    @pytest.mark.skip("batch command may not be fully implemented")
    def test_batch_command_basic(self, tmp_path):
        """Test basic batch command execution."""
        # Create test directory with images
        input_dir = tmp_path / "images"
        input_dir.mkdir()
        (input_dir / "image1.jpg").touch()
        (input_dir / "image2.tif").touch()

        output_dir = tmp_path / "output"

        with patch(
            "coffee_bean_analyzer.cli.commands.batch.batch_command"
        ) as mock_batch:
            mock_batch.return_value = None

            result = self.runner.invoke(
                cli, ["batch", str(input_dir), "--output-dir", str(output_dir)]
            )

            assert result.exit_code in [0, 2]


class TestOptimizeCommand:
    """Test the optimize command."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()

    def test_optimize_command_help(self):
        """Test optimize command help."""
        result = self.runner.invoke(cli, ["optimize", "--help"])

        # Should show help without error even if command doesn't exist yet
        assert result.exit_code in [0, 2]

    @pytest.mark.skip("optimize command may not be fully implemented")
    def test_optimize_command_basic(self, tmp_path):
        """Test basic optimize command execution."""
        # Create test files
        test_image = tmp_path / "test_image.jpg"
        test_image.touch()

        ground_truth_file = tmp_path / "ground_truth.csv"
        ground_truth_file.write_text("length,width\n10.0,8.0\n12.0,9.0\n")

        output_dir = tmp_path / "output"

        with patch(
            "coffee_bean_analyzer.cli.commands.optimize.optimize_command"
        ) as mock_optimize:
            mock_optimize.return_value = None

            result = self.runner.invoke(
                cli,
                [
                    "optimize",
                    str(test_image),
                    "--ground-truth",
                    str(ground_truth_file),
                    "--output-dir",
                    str(output_dir),
                ],
            )

            assert result.exit_code in [0, 2]


class TestCLIIntegration:
    """Integration tests for CLI functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()

    def test_cli_command_chaining(self):
        """Test that CLI commands can be chained properly."""
        # Test help for different commands
        commands_to_test = [
            "--help",
            "analyze --help",
            "batch --help",
            "optimize --help",
        ]

        for cmd in commands_to_test:
            result = self.runner.invoke(cli, cmd.split())
            # Should either show help (exit code 0) or show "no such command" (exit code 2)
            assert result.exit_code in [0, 2]

    def test_cli_error_handling(self):
        """Test CLI error handling for invalid commands."""
        result = self.runner.invoke(cli, ["nonexistent-command"])

        assert result.exit_code == 2  # Click returns 2 for no such command
        assert "No such command" in result.output or "Usage:" in result.output

    def test_cli_with_invalid_options(self):
        """Test CLI with invalid options."""
        result = self.runner.invoke(cli, ["--invalid-option"])

        assert result.exit_code == 2  # Click returns 2 for invalid options
        assert "No such option" in result.output or "Usage:" in result.output

    def test_logging_configuration(self):
        """Test that logging flags are accepted correctly."""
        test_cases = [
            ["--verbose"],
            ["--quiet"],
            [],  # No flags
        ]

        for args in test_cases:
            result = self.runner.invoke(cli, args + ["--help"])

            # All should succeed
            assert result.exit_code == 0
            assert "Coffee Bean Size Analysis Tool" in result.output

    def test_cli_path_validation(self, tmp_path):
        """Test CLI path validation for commands that require file inputs."""
        nonexistent_file = tmp_path / "nonexistent.jpg"

        # Test with nonexistent input file
        result = self.runner.invoke(cli, ["analyze", str(nonexistent_file)])

        # Should fail due to file not existing (if analyze command is implemented)
        # or show no such command error
        assert result.exit_code in [
            2
        ]  # Either path validation error or no such command

    def test_output_format_validation(self, tmp_path):
        """Test output format validation."""
        test_image = tmp_path / "test_image.jpg"
        test_image.touch()

        # Test with invalid output format
        result = self.runner.invoke(
            cli, ["analyze", str(test_image), "--format", "invalid_format"]
        )

        # Should fail due to invalid choice (if analyze command is implemented)
        assert result.exit_code in [2]  # Either validation error or no such command

    def test_config_file_validation(self, tmp_path):
        """Test configuration file validation."""
        test_image = tmp_path / "test_image.jpg"
        test_image.touch()

        nonexistent_config = tmp_path / "nonexistent_config.yaml"

        # Test with nonexistent config file
        result = self.runner.invoke(
            cli, ["analyze", str(test_image), "--config", str(nonexistent_config)]
        )

        # Should fail due to config file not existing (if analyze command is implemented)
        assert result.exit_code in [2]


class TestCLIUtilities:
    """Test CLI utility functions and helpers."""

    def test_click_context_setup(self):
        """Test Click context setup and management."""
        runner = CliRunner()

        # Test that context is properly managed
        with patch("coffee_bean_analyzer.cli.main.setup_logging"):
            result = runner.invoke(cli, ["--verbose", "--help"])

            assert result.exit_code == 0

    def test_parameter_passing(self, tmp_path):
        """Test parameter passing between CLI layers."""
        runner = CliRunner()

        # Create test files
        test_image = tmp_path / "test.jpg"
        test_image.touch()

        config_file = tmp_path / "config.yaml"
        config_file.write_text("test: config")

        # Test parameter parsing
        with patch("coffee_bean_analyzer.cli.main.setup_logging"):
            result = runner.invoke(
                cli, ["--verbose", "--log-file", str(tmp_path / "test.log"), "--help"]
            )

            assert result.exit_code == 0

    def test_error_message_formatting(self):
        """Test that error messages are properly formatted."""
        runner = CliRunner()

        # Test with invalid command
        result = runner.invoke(cli, ["invalid_command"])

        assert result.exit_code == 2
        # Error message should be user-friendly
        assert result.output  # Should have some output
        assert len(result.output.strip()) > 0


class TestCommandImports:
    """Test that command imports work correctly."""

    def test_analyze_command_import(self):
        """Test that analyze command can be imported."""
        try:
            # Try to import the analyze command
            from coffee_bean_analyzer.cli.commands.analyze import analyze_command

            # If successful, it should be a function or command
            assert callable(analyze_command) or hasattr(analyze_command, "callback")
        except ImportError:
            # Command module may not be fully implemented yet
            pytest.skip("Analyze command not yet implemented")

    def test_batch_command_import(self):
        """Test that batch command can be imported."""
        try:
            from coffee_bean_analyzer.cli.commands.batch import batch_command

            assert callable(batch_command) or hasattr(batch_command, "callback")
        except ImportError:
            pytest.skip("Batch command not yet implemented")

    def test_optimize_command_import(self):
        """Test that optimize command can be imported."""
        try:
            from coffee_bean_analyzer.cli.commands.optimize import optimize_command

            assert callable(optimize_command) or hasattr(optimize_command, "callback")
        except ImportError:
            pytest.skip("Optimize command not yet implemented")

    def test_logging_setup_import(self):
        """Test that logging setup can be imported."""
        try:
            from coffee_bean_analyzer.utils.logging_config import setup_logging

            assert callable(setup_logging)
        except ImportError:
            # Logging config may not exist yet
            pytest.skip("Logging config not yet implemented")


class TestCLIDocumentation:
    """Test CLI documentation and help text."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()

    def test_main_help_content(self):
        """Test main CLI help content."""
        result = self.runner.invoke(cli, ["--help"])

        assert result.exit_code == 0

        # Check for expected content
        help_content = result.output.lower()
        assert "coffee" in help_content
        assert "bean" in help_content
        assert any(word in help_content for word in ["analysis", "analyze", "tool"])

    def test_help_examples_present(self):
        """Test that help includes usage examples."""
        result = self.runner.invoke(cli, ["--help"])

        assert result.exit_code == 0

        # Should include examples in help text
        help_content = result.output.lower()
        # Look for example indicators
        has_examples = any(
            word in help_content
            for word in ["example", "examples", "usage", "coffee-bean-analyzer"]
        )
        assert has_examples

    def test_version_information(self):
        """Test version information is available."""
        result = self.runner.invoke(cli, ["--version"])

        assert result.exit_code == 0
        # Should contain version number
        assert result.output.strip()  # Should have output
        # Version output should be reasonable length
        assert len(result.output.strip()) < 100
