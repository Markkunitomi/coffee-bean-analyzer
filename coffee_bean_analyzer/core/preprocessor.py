"""Coffee Bean Analyzer - Image Preprocessing Module

Adapted from the original coffee_bean_analyzer.py script.
Handles image preprocessing operations for coffee bean segmentation.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional

import cv2
import numpy as np


@dataclass
class PreprocessingResult:
    """Container for preprocessing results"""

    processed_image: np.ndarray
    original_shape: tuple
    preprocessing_steps: list
    metadata: Dict[str, Any]


class PreprocessingStep(ABC):
    """Abstract base class for preprocessing steps."""

    @abstractmethod
    def apply(self, image: np.ndarray, metadata: Dict[str, Any]) -> np.ndarray:
        """Apply this preprocessing step to the image."""
        pass

    @abstractmethod
    def get_step_name(self) -> str:
        """Get the name of this preprocessing step."""
        pass


class GrayscaleConverter(PreprocessingStep):
    """Convert image to grayscale if needed."""

    def apply(self, image: np.ndarray, metadata: Dict[str, Any]) -> np.ndarray:
        """Convert to grayscale if image has multiple channels."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            metadata["converted_to_grayscale"] = True
            metadata["original_channels"] = image.shape[2]
            return gray
        metadata["converted_to_grayscale"] = False
        return image

    def get_step_name(self) -> str:
        return "grayscale_conversion"


class GaussianBlurStep(PreprocessingStep):
    """Apply Gaussian blur to reduce noise."""

    def __init__(self, kernel_size: int = 5):
        """Initialize Gaussian blur step.

        Args:
            kernel_size: Size of the Gaussian kernel (must be odd)
        """
        if kernel_size % 2 == 0:
            kernel_size += 1  # Ensure odd kernel size
        self.kernel_size = kernel_size

    def apply(self, image: np.ndarray, metadata: Dict[str, Any]) -> np.ndarray:
        """Apply Gaussian blur to the image."""
        blurred = cv2.GaussianBlur(image, (self.kernel_size, self.kernel_size), 0)
        metadata["gaussian_kernel_size"] = self.kernel_size
        return blurred

    def get_step_name(self) -> str:
        return f"gaussian_blur_{self.kernel_size}"


class CLAHEEnhancer(PreprocessingStep):
    """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)."""

    def __init__(self, clip_limit: float = 2.0, tile_grid_size: tuple = (8, 8)):
        """Initialize CLAHE enhancer.

        Args:
            clip_limit: Threshold for contrast limiting
            tile_grid_size: Size of the grid for adaptive equalization
        """
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size

    def apply(self, image: np.ndarray, metadata: Dict[str, Any]) -> np.ndarray:
        """Apply CLAHE enhancement to the image."""
        clahe = cv2.createCLAHE(
            clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size
        )
        enhanced = clahe.apply(image)
        metadata["clahe_clip_limit"] = self.clip_limit
        metadata["clahe_tile_grid_size"] = self.tile_grid_size
        return enhanced

    def get_step_name(self) -> str:
        return f"clahe_{self.clip_limit}"


class ImagePreprocessor:
    """Image preprocessor for coffee bean analysis.

    Adapted from the original preprocess_image() function with added flexibility
    and modularity for different preprocessing pipelines.
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize preprocessor with configuration.

        Args:
            config: Dictionary containing preprocessing parameters
        """
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Build preprocessing pipeline based on config
        self.steps = self._build_pipeline()

    def _build_pipeline(self) -> list:
        """Build preprocessing pipeline from configuration."""
        steps = []

        # Always convert to grayscale first
        steps.append(GrayscaleConverter())

        # Add Gaussian blur if specified
        gaussian_kernel = self.config.get("gaussian_kernel", 5)
        if gaussian_kernel > 0:
            steps.append(GaussianBlurStep(gaussian_kernel))

        # Add CLAHE enhancement if specified
        clahe_clip = self.config.get("clahe_clip", 2.0)
        if clahe_clip > 0:
            clahe_grid_size = self.config.get("clahe_grid_size", (8, 8))
            steps.append(CLAHEEnhancer(clahe_clip, clahe_grid_size))

        return steps

    def preprocess(self, image: np.ndarray, debug: bool = False) -> PreprocessingResult:
        """Preprocess the input image through the configured pipeline.

        This is your original preprocess_image() function adapted to the new structure.

        Args:
            image: Input image as numpy array
            debug: Enable debug output

        Returns:
            PreprocessingResult containing processed image and metadata
        """
        if image is None or image.size == 0:
            raise ValueError("Input image is empty or None")

        original_shape = image.shape
        current_image = image.copy()
        metadata = {"original_shape": original_shape, "original_dtype": image.dtype}
        applied_steps = []

        if debug:
            self.logger.info(
                f"Starting preprocessing pipeline with {len(self.steps)} steps"
            )
            self.logger.info(f"Input image shape: {original_shape}")

        # Apply each preprocessing step
        for step in self.steps:
            step_name = step.get_step_name()

            if debug:
                self.logger.info(f"Applying step: {step_name}")

            try:
                current_image = step.apply(current_image, metadata)
                applied_steps.append(step_name)

                if debug:
                    self.logger.info(
                        f"Step {step_name} completed. New shape: {current_image.shape}"
                    )

            except Exception as e:
                self.logger.error(f"Error in preprocessing step {step_name}: {e}")
                raise

        if debug:
            self.logger.info(
                f"Preprocessing completed. Applied {len(applied_steps)} steps."
            )

        return PreprocessingResult(
            processed_image=current_image,
            original_shape=original_shape,
            preprocessing_steps=applied_steps,
            metadata=metadata,
        )

    def add_custom_step(self, step: PreprocessingStep, position: Optional[int] = None):
        """Add a custom preprocessing step to the pipeline.

        Args:
            step: Custom preprocessing step
            position: Position to insert the step (None = append at end)
        """
        if position is None:
            self.steps.append(step)
        else:
            self.steps.insert(position, step)

        self.logger.info(f"Added custom step: {step.get_step_name()}")

    def remove_step_by_name(self, step_name: str) -> bool:
        """Remove a preprocessing step by name.

        Args:
            step_name: Name of the step to remove

        Returns:
            True if step was found and removed, False otherwise
        """
        for i, step in enumerate(self.steps):
            if step.get_step_name() == step_name:
                removed_step = self.steps.pop(i)
                self.logger.info(f"Removed step: {removed_step.get_step_name()}")
                return True
        return False

    def get_pipeline_summary(self) -> Dict[str, Any]:
        """Get a summary of the current preprocessing pipeline."""
        return {
            "num_steps": len(self.steps),
            "step_names": [step.get_step_name() for step in self.steps],
            "config": self.config.copy(),
        }


# Legacy compatibility function
def preprocess_image_legacy(image, gaussian_kernel=5, clahe_clip=2.0):
    """Legacy wrapper to maintain compatibility with your existing code.

    This function provides the same interface as your original preprocess_image()
    but uses the new ImagePreprocessor class internally.

    Args:
        image: Input image
        gaussian_kernel: Kernel size for Gaussian blur
        clahe_clip: CLAHE clip limit

    Returns:
        Preprocessed image (numpy array)
    """
    config = {
        "gaussian_kernel": gaussian_kernel,
        "clahe_clip": clahe_clip,
        "clahe_grid_size": (8, 8),
    }

    preprocessor = ImagePreprocessor(config)
    result = preprocessor.preprocess(image)

    # Return just the processed image to match original function signature
    return result.processed_image


# Factory function for common preprocessing configurations
def create_preprocessor(preset: str = "default", **kwargs) -> ImagePreprocessor:
    """Factory function to create preprocessors with common configurations.

    Args:
        preset: Preset configuration name
        **kwargs: Override specific parameters

    Returns:
        Configured ImagePreprocessor instance
    """
    presets = {
        "default": {"gaussian_kernel": 5, "clahe_clip": 2.0, "clahe_grid_size": (8, 8)},
        "aggressive_enhancement": {
            "gaussian_kernel": 7,
            "clahe_clip": 3.0,
            "clahe_grid_size": (8, 8),
        },
        "minimal": {"gaussian_kernel": 3, "clahe_clip": 1.0, "clahe_grid_size": (4, 4)},
        "no_blur": {
            "gaussian_kernel": 0,  # Skip Gaussian blur
            "clahe_clip": 2.0,
            "clahe_grid_size": (8, 8),
        },
        "high_contrast": {
            "gaussian_kernel": 5,
            "clahe_clip": 4.0,
            "clahe_grid_size": (16, 16),
        },
    }

    if preset not in presets:
        raise ValueError(f"Unknown preset: {preset}. Available: {list(presets.keys())}")

    config = presets[preset].copy()
    config.update(kwargs)  # Allow parameter overrides

    return ImagePreprocessor(config)
