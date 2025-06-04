"""Unit tests for the preprocessor module using pytest."""

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from coffee_bean_analyzer.core.preprocessor import (
    CLAHEEnhancer,
    GaussianBlurStep,
    GrayscaleConverter,
    ImagePreprocessor,
    PreprocessingResult,
    create_preprocessor,
    preprocess_image_legacy,
)


class TestPreprocessingSteps:
    """Test individual preprocessing steps."""

    @pytest.fixture
    def color_image(self):
        """Create a sample color image."""
        return np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

    @pytest.fixture
    def grayscale_image(self):
        """Create a sample grayscale image."""
        return np.random.randint(0, 255, (100, 100), dtype=np.uint8)

    def test_grayscale_converter_color_image(self, color_image):
        """Test grayscale conversion with color image."""
        converter = GrayscaleConverter()
        metadata = {}

        result = converter.apply(color_image, metadata)

        assert len(result.shape) == 2  # Should be grayscale
        assert result.shape[:2] == color_image.shape[:2]  # Same width/height
        assert metadata["converted_to_grayscale"] is True
        assert metadata["original_channels"] == 3

    def test_grayscale_converter_grayscale_image(self, grayscale_image):
        """Test grayscale conversion with already grayscale image."""
        converter = GrayscaleConverter()
        metadata = {}

        result = converter.apply(grayscale_image, metadata)

        assert len(result.shape) == 2  # Should remain grayscale
        assert np.array_equal(result, grayscale_image)  # Should be unchanged
        assert metadata["converted_to_grayscale"] is False

    def test_gaussian_blur_step(self, grayscale_image):
        """Test Gaussian blur step."""
        blur_step = GaussianBlurStep(kernel_size=5)
        metadata = {}

        result = blur_step.apply(grayscale_image, metadata)

        assert result.shape == grayscale_image.shape
        assert metadata["gaussian_kernel_size"] == 5
        assert blur_step.get_step_name() == "gaussian_blur_5"

        # Image should be different (blurred)
        assert not np.array_equal(result, grayscale_image)

    def test_gaussian_blur_even_kernel(self, grayscale_image):
        """Test that even kernel sizes are made odd."""
        blur_step = GaussianBlurStep(kernel_size=4)  # Even number
        assert blur_step.kernel_size == 5  # Should be made odd

    def test_clahe_enhancer(self, grayscale_image):
        """Test CLAHE enhancement step."""
        clahe_step = CLAHEEnhancer(clip_limit=2.0, tile_grid_size=(8, 8))
        metadata = {}

        result = clahe_step.apply(grayscale_image, metadata)

        assert result.shape == grayscale_image.shape
        assert metadata["clahe_clip_limit"] == 2.0
        assert metadata["clahe_tile_grid_size"] == (8, 8)
        assert clahe_step.get_step_name() == "clahe_2.0"


class TestImagePreprocessor:
    """Test the main ImagePreprocessor class."""

    @pytest.fixture
    def default_config(self):
        """Default preprocessing configuration."""
        return {"gaussian_kernel": 5, "clahe_clip": 2.0, "clahe_grid_size": (8, 8)}

    @pytest.fixture
    def preprocessor(self, default_config):
        """Create an ImagePreprocessor instance."""
        return ImagePreprocessor(default_config)

    @pytest.fixture
    def sample_color_image(self):
        """Create a sample color image for testing."""
        # Create an image with some structure
        img = np.zeros((200, 200, 3), dtype=np.uint8)
        cv2.rectangle(img, (50, 50), (150, 150), (100, 100, 100), -1)
        cv2.circle(img, (100, 100), 30, (200, 200, 200), -1)
        return img

    def test_preprocessor_initialization(self, default_config):
        """Test preprocessor initialization."""
        preprocessor = ImagePreprocessor(default_config)

        assert preprocessor.config == default_config
        assert len(preprocessor.steps) == 3  # Grayscale, Gaussian, CLAHE

    def test_preprocess_full_pipeline(self, preprocessor, sample_color_image):
        """Test complete preprocessing pipeline."""
        result = preprocessor.preprocess(sample_color_image, debug=False)

        assert isinstance(result, PreprocessingResult)
        assert result.processed_image.shape[:2] == sample_color_image.shape[:2]
        assert len(result.processed_image.shape) == 2  # Should be grayscale
        assert result.original_shape == sample_color_image.shape
        assert len(result.preprocessing_steps) == 3
        assert "converted_to_grayscale" in result.metadata

    def test_preprocess_empty_image(self, preprocessor):
        """Test preprocessing with empty image."""
        empty_image = np.array([])

        with pytest.raises(ValueError, match="Input image is empty or None"):
            preprocessor.preprocess(empty_image)

    def test_preprocess_none_image(self, preprocessor):
        """Test preprocessing with None image."""
        with pytest.raises(ValueError, match="Input image is empty or None"):
            preprocessor.preprocess(None)

    def test_add_custom_step(self, preprocessor, sample_color_image):
        """Test adding custom preprocessing step."""
        initial_steps = len(preprocessor.steps)
        custom_step = GaussianBlurStep(kernel_size=7)

        preprocessor.add_custom_step(custom_step)

        assert len(preprocessor.steps) == initial_steps + 1
        assert preprocessor.steps[-1] == custom_step

    def test_remove_step_by_name(self, preprocessor):
        """Test removing preprocessing step by name."""
        initial_steps = len(preprocessor.steps)

        # Remove Gaussian blur step
        removed = preprocessor.remove_step_by_name("gaussian_blur_5")

        assert removed is True
        assert len(preprocessor.steps) == initial_steps - 1

    def test_remove_nonexistent_step(self, preprocessor):
        """Test removing non-existent step."""
        removed = preprocessor.remove_step_by_name("nonexistent_step")
        assert removed is False

    def test_get_pipeline_summary(self, preprocessor):
        """Test getting pipeline summary."""
        summary = preprocessor.get_pipeline_summary()

        assert "num_steps" in summary
        assert "step_names" in summary
        assert "config" in summary
        assert summary["num_steps"] == len(preprocessor.steps)
        assert len(summary["step_names"]) == summary["num_steps"]


class TestLegacyCompatibility:
    """Test legacy function compatibility."""

    @pytest.fixture
    def sample_image(self):
        """Create a sample image for testing."""
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.circle(img, (50, 50), 20, (150, 150, 150), -1)
        return img

    def test_legacy_function_signature(self, sample_image):
        """Test that legacy function maintains original signature."""
        # Test with default parameters
        result1 = preprocess_image_legacy(sample_image)

        # Test with custom parameters (matching your original function)
        result2 = preprocess_image_legacy(
            sample_image, gaussian_kernel=7, clahe_clip=3.0
        )

        assert isinstance(result1, np.ndarray)
        assert isinstance(result2, np.ndarray)
        assert len(result1.shape) == 2  # Should be grayscale
        assert len(result2.shape) == 2  # Should be grayscale
        assert result1.shape == sample_image.shape[:2]
        assert result2.shape == sample_image.shape[:2]

    def test_legacy_vs_new_consistency(self, sample_image):
        """Test that legacy and new methods give identical results."""
        # Process with legacy function
        legacy_result = preprocess_image_legacy(
            sample_image, gaussian_kernel=5, clahe_clip=2.0
        )

        # Process with new preprocessor
        config = {"gaussian_kernel": 5, "clahe_clip": 2.0, "clahe_grid_size": (8, 8)}
        preprocessor = ImagePreprocessor(config)
        new_result = preprocessor.preprocess(sample_image)

        # Results should be identical
        np.testing.assert_array_equal(legacy_result, new_result.processed_image)


class TestPreprocessorFactory:
    """Test the preprocessor factory function."""

    def test_default_preset(self):
        """Test creating preprocessor with default preset."""
        preprocessor = create_preprocessor("default")

        assert isinstance(preprocessor, ImagePreprocessor)
        assert preprocessor.config["gaussian_kernel"] == 5
        assert preprocessor.config["clahe_clip"] == 2.0

    def test_aggressive_enhancement_preset(self):
        """Test aggressive enhancement preset."""
        preprocessor = create_preprocessor("aggressive_enhancement")

        assert preprocessor.config["gaussian_kernel"] == 7
        assert preprocessor.config["clahe_clip"] == 3.0

    def test_minimal_preset(self):
        """Test minimal preset."""
        preprocessor = create_preprocessor("minimal")

        assert preprocessor.config["gaussian_kernel"] == 3
        assert preprocessor.config["clahe_clip"] == 1.0

    def test_no_blur_preset(self):
        """Test no blur preset."""
        preprocessor = create_preprocessor("no_blur")

        assert preprocessor.config["gaussian_kernel"] == 0
        assert len(preprocessor.steps) == 2  # Should skip Gaussian blur

    def test_parameter_override(self):
        """Test overriding preset parameters."""
        preprocessor = create_preprocessor("default", gaussian_kernel=9, clahe_clip=4.0)

        assert preprocessor.config["gaussian_kernel"] == 9
        assert preprocessor.config["clahe_clip"] == 4.0

    def test_invalid_preset(self):
        """Test creating preprocessor with invalid preset."""
        with pytest.raises(ValueError, match="Unknown preset"):
            create_preprocessor("invalid_preset")


@pytest.mark.integration
class TestRealImagePreprocessing:
    """Integration tests with real images (requires test data)."""

    def test_real_image_preprocessing(self):
        """Test preprocessing on a real image if available."""
        test_data_dir = Path(__file__).parent / "data"

        if not test_data_dir.exists():
            pytest.skip("No test data directory found")

        # Find a real image
        real_image = None
        for pattern in ["*.tif", "*.jpg", "*.png"]:
            images = list(test_data_dir.glob(pattern))
            if images:
                img = cv2.imread(str(images[0]))
                if img is not None:
                    real_image = img
                    break

        if real_image is None:
            pytest.skip("No valid test images found")

        # Test preprocessing with different presets
        for preset in ["default", "aggressive_enhancement", "minimal", "no_blur"]:
            preprocessor = create_preprocessor(preset)
            result = preprocessor.preprocess(real_image)

            # Should return valid result
            assert isinstance(result, PreprocessingResult)
            assert result.processed_image.shape[:2] == real_image.shape[:2]
            assert len(result.processed_image.shape) == 2  # Grayscale
            assert len(result.preprocessing_steps) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
