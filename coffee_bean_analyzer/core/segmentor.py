"""Coffee Bean Analyzer - Bean Segmentation Module

Adapted from the original coffee_bean_analyzer.py script.
Handles segmentation of individual coffee beans using watershed algorithm.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from skimage import segmentation
from skimage.feature import peak_local_max

from .detector import DetectionResult

# Import our own modules
from .preprocessor import ImagePreprocessor


@dataclass
class SegmentationResult:
    """Container for segmentation results"""

    labels: np.ndarray
    binary_mask: np.ndarray
    num_segments: int
    preprocessing_metadata: Dict[str, Any]
    segmentation_metadata: Dict[str, Any]
    excluded_regions: List[Dict[str, Any]]


class MorphologicalProcessor:
    """Handles morphological operations for segmentation."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize morphological processor.

        Args:
            config: Configuration containing morphological parameters
        """
        self.config = config
        self.logger = logging.getLogger(__name__)

    def create_kernel(self, kernel_size: int, shape: str = "ellipse") -> np.ndarray:
        """Create morphological kernel."""
        if shape.lower() == "ellipse":
            return cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
            )
        if shape.lower() == "rect":
            return cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        return cv2.getStructuringElement(cv2.MORPH_CROSS, (kernel_size, kernel_size))

    def apply_closing(
        self, binary_image: np.ndarray, kernel_size: int, iterations: int
    ) -> np.ndarray:
        """Apply morphological closing operation."""
        kernel = self.create_kernel(kernel_size)
        return cv2.morphologyEx(
            binary_image, cv2.MORPH_CLOSE, kernel, iterations=iterations
        )

    def apply_opening(
        self, binary_image: np.ndarray, kernel_size: int, iterations: int
    ) -> np.ndarray:
        """Apply morphological opening operation."""
        kernel = self.create_kernel(kernel_size)
        return cv2.morphologyEx(
            binary_image, cv2.MORPH_OPEN, kernel, iterations=iterations
        )

    def clean_binary_mask(self, binary_image: np.ndarray) -> np.ndarray:
        """Clean binary mask using configured morphological operations.

        This implements your original noise removal and hole filling logic.
        """
        kernel_size = self.config.get("morph_kernel_size", 3)
        close_iterations = self.config.get("close_iterations", 2)
        open_iterations = self.config.get("open_iterations", 1)

        # Remove noise and fill holes (your original logic)
        cleaned = self.apply_closing(binary_image, kernel_size, close_iterations)
        cleaned = self.apply_opening(cleaned, kernel_size, open_iterations)

        return cleaned


class WatershedSegmentor:
    """Handles watershed segmentation for bean detection."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize watershed segmentor.

        Args:
            config: Configuration containing watershed parameters
        """
        self.config = config
        self.logger = logging.getLogger(__name__)

    def find_peaks(self, distance_transform: np.ndarray) -> np.ndarray:
        """Find local maxima for watershed markers.

        This implements your original peak detection logic.
        """
        min_distance = self.config.get("min_distance", 20)
        threshold_factor = self.config.get("threshold_factor", 0.3)

        # Find local maxima (your original code)
        local_maxima = peak_local_max(
            distance_transform,
            min_distance=min_distance,
            threshold_abs=threshold_factor * distance_transform.max(),
        )

        return local_maxima

    def create_markers(
        self, local_maxima: np.ndarray, image_shape: Tuple[int, int]
    ) -> np.ndarray:
        """Create markers for watershed from local maxima."""
        markers = np.zeros(image_shape, dtype=np.int32)
        for i, (y, x) in enumerate(local_maxima):
            markers[y, x] = i + 1
        return markers

    def apply_watershed(
        self, distance_transform: np.ndarray, markers: np.ndarray, mask: np.ndarray
    ) -> np.ndarray:
        """Apply watershed segmentation algorithm."""
        # Apply watershed (your original implementation)
        labels = segmentation.watershed(-distance_transform, markers, mask=mask)
        return labels


class CoinExcluder:
    """Handles exclusion of coin regions from segmentation."""

    def __init__(self, expansion_factor: float = 1.2):
        """Initialize coin excluder.

        Args:
            expansion_factor: Factor to expand coin radius for exclusion
        """
        self.expansion_factor = expansion_factor
        self.logger = logging.getLogger(__name__)

    def create_coin_mask(
        self, image_shape: Tuple[int, int], coin_detection: Optional[DetectionResult]
    ) -> np.ndarray:
        """Create mask to exclude coin region.

        This implements your original coin exclusion logic.
        """
        if coin_detection is None:
            # No coin to exclude, return all-ones mask
            return np.ones(image_shape, dtype=np.uint8) * 255

        # Create mask to exclude the quarter from analysis (your original logic)
        coin_mask = np.ones(image_shape, dtype=np.uint8) * 255

        x, y = coin_detection.center
        radius = int(coin_detection.radius)

        # Create a slightly larger mask around the coin (your original logic)
        expanded_radius = int(radius * self.expansion_factor)
        cv2.circle(coin_mask, (x, y), expanded_radius, 0, -1)

        self.logger.debug(
            f"Excluded coin region: center=({x}, {y}), radius={expanded_radius}"
        )

        return coin_mask

    def apply_coin_exclusion(
        self, binary_image: np.ndarray, coin_detection: Optional[DetectionResult]
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Apply coin exclusion to binary image."""
        if coin_detection is None:
            return binary_image, {"coin_excluded": False}

        coin_mask = self.create_coin_mask(binary_image.shape, coin_detection)
        excluded_binary = cv2.bitwise_and(binary_image, coin_mask)

        exclusion_info = {
            "coin_excluded": True,
            "coin_center": coin_detection.center,
            "coin_radius": coin_detection.radius,
            "expanded_radius": int(coin_detection.radius * self.expansion_factor),
        }

        return excluded_binary, exclusion_info


class BeanSegmentor:
    """Main bean segmentation class using watershed algorithm.

    Adapted from your original segment_beans() function with added modularity
    and configurability.
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize bean segmentor with configuration.

        Args:
            config: Dictionary containing segmentation parameters
        """
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Initialize sub-components
        self.preprocessor = self._create_preprocessor()
        self.morph_processor = MorphologicalProcessor(config)
        self.watershed_segmentor = WatershedSegmentor(config)
        self.coin_excluder = CoinExcluder(config.get("coin_expansion_factor", 1.2))

    def _create_preprocessor(self) -> ImagePreprocessor:
        """Create preprocessor from segmentation config."""
        # Extract preprocessing parameters from segmentation config
        preprocess_config = {
            "gaussian_kernel": self.config.get("gaussian_kernel", 5),
            "clahe_clip": self.config.get("clahe_clip", 2.0),
            "clahe_grid_size": self.config.get("clahe_grid_size", (8, 8)),
        }
        return ImagePreprocessor(preprocess_config)

    def segment(
        self,
        image: np.ndarray,
        coin_detection: Optional[DetectionResult] = None,
        debug: bool = False,
    ) -> SegmentationResult:
        """Segment individual coffee beans using watershed algorithm.

        This is your original segment_beans() function adapted to the new structure.

        Args:
            image: Input image as numpy array
            coin_detection: Optional coin detection result for exclusion
            debug: Enable debug output

        Returns:
            SegmentationResult containing segmentation data
        """
        if debug:
            self.logger.info("Starting bean segmentation")
            self.logger.info(f"Input image shape: {image.shape}")

        # Step 1: Preprocess image (using our new preprocessor)
        preprocess_result = self.preprocessor.preprocess(image, debug=debug)
        preprocessed = preprocess_result.processed_image

        if debug:
            self.logger.info("Preprocessing completed")

        # Step 2: Otsu thresholding (your original logic)
        _, binary = cv2.threshold(
            preprocessed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        # Step 3: Invert binary image so beans are white (your original logic)
        binary = cv2.bitwise_not(binary)

        if debug:
            self.logger.info("Binary thresholding completed")

        # Step 4: Apply coin exclusion (your original logic)
        binary_excluded, exclusion_info = self.coin_excluder.apply_coin_exclusion(
            binary, coin_detection
        )

        if debug and exclusion_info["coin_excluded"]:
            self.logger.info(f"Coin exclusion applied: {exclusion_info}")

        # Step 5: Clean binary mask with morphological operations (your original logic)
        cleaned_binary = self.morph_processor.clean_binary_mask(binary_excluded)

        if debug:
            self.logger.info("Morphological cleaning completed")

        # Step 6: Distance transform (your original logic)
        dist_transform = cv2.distanceTransform(cleaned_binary, cv2.DIST_L2, 5)

        # Step 7: Find local maxima for watershed markers (your original logic)
        local_maxima = self.watershed_segmentor.find_peaks(dist_transform)

        if debug:
            self.logger.info(f"Found {len(local_maxima)} potential bean centers")

        # Step 8: Create markers for watershed (your original logic)
        markers = self.watershed_segmentor.create_markers(
            local_maxima, cleaned_binary.shape
        )

        # Step 9: Apply watershed segmentation (your original logic)
        labels = self.watershed_segmentor.apply_watershed(
            dist_transform, markers, cleaned_binary
        )

        num_segments = len(np.unique(labels)) - 1  # Subtract 1 for background

        if debug:
            self.logger.info(
                f"Watershed segmentation completed: {num_segments} segments"
            )

        # Prepare metadata
        segmentation_metadata = {
            "num_local_maxima": len(local_maxima),
            "otsu_threshold": cv2.threshold(
                preprocessed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )[0],
            "distance_transform_max": dist_transform.max(),
            "watershed_parameters": {
                "min_distance": self.config.get("min_distance", 20),
                "threshold_factor": self.config.get("threshold_factor", 0.3),
            },
            "morphological_parameters": {
                "kernel_size": self.config.get("morph_kernel_size", 3),
                "close_iterations": self.config.get("close_iterations", 2),
                "open_iterations": self.config.get("open_iterations", 1),
            },
        }

        excluded_regions = [exclusion_info] if exclusion_info["coin_excluded"] else []

        return SegmentationResult(
            labels=labels,
            binary_mask=cleaned_binary,
            num_segments=num_segments,
            preprocessing_metadata=preprocess_result.metadata,
            segmentation_metadata=segmentation_metadata,
            excluded_regions=excluded_regions,
        )

    def get_configuration(self) -> Dict[str, Any]:
        """Get current segmentation configuration."""
        return self.config.copy()

    def update_configuration(self, new_config: Dict[str, Any]):
        """Update segmentation configuration."""
        self.config.update(new_config)

        # Recreate components with new config
        self.preprocessor = self._create_preprocessor()
        self.morph_processor = MorphologicalProcessor(self.config)
        self.watershed_segmentor = WatershedSegmentor(self.config)
        self.coin_excluder = CoinExcluder(self.config.get("coin_expansion_factor", 1.2))

        self.logger.info("Configuration updated and components recreated")


# Legacy compatibility function
def segment_beans_legacy(image, coin=None, params=None, debug=False):
    """Legacy wrapper to maintain compatibility with your existing code.

    This function provides the same interface as your original segment_beans()
    but uses the new BeanSegmentor class internally.

    Args:
        image: Input image
        coin: Coin tuple (x, y, radius) for exclusion (your original format)
        params: Parameter dictionary (your original format)
        debug: Enable debug output

    Returns:
        Tuple of (labels, binary) to match your original function signature
    """
    if params is None:
        params = {
            "gaussian_kernel": 5,
            "clahe_clip": 2.0,
            "morph_kernel_size": 3,
            "close_iterations": 2,
            "open_iterations": 1,
            "min_distance": 20,
            "threshold_factor": 0.3,
        }

    # Create segmentor with parameters
    segmentor = BeanSegmentor(params)

    # Convert coin tuple to DetectionResult if provided
    coin_detection = None
    if coin is not None:
        x, y, r = coin
        coin_detection = DetectionResult(
            center=(x, y),
            radius=float(r),
            confidence=1.0,  # Assume high confidence for legacy compatibility
            bbox=(x - r, y - r, 2 * r, 2 * r),
        )

    # Perform segmentation
    result = segmentor.segment(image, coin_detection, debug=debug)

    # Return in original format (labels, binary)
    return result.labels, result.binary_mask


# Factory function for common segmentation configurations
def create_segmentor(preset: str = "default", **kwargs) -> BeanSegmentor:
    """Factory function to create segmentors with common configurations.

    Args:
        preset: Preset configuration name
        **kwargs: Override specific parameters

    Returns:
        Configured BeanSegmentor instance
    """
    presets = {
        "default": {
            "gaussian_kernel": 5,
            "clahe_clip": 2.0,
            "morph_kernel_size": 3,
            "close_iterations": 2,
            "open_iterations": 1,
            "min_distance": 20,
            "threshold_factor": 0.3,
            "coin_expansion_factor": 1.2,
        },
        "aggressive": {
            "gaussian_kernel": 7,
            "clahe_clip": 3.0,
            "morph_kernel_size": 5,
            "close_iterations": 3,
            "open_iterations": 2,
            "min_distance": 15,
            "threshold_factor": 0.2,
            "coin_expansion_factor": 1.3,
        },
        "conservative": {
            "gaussian_kernel": 3,
            "clahe_clip": 1.5,
            "morph_kernel_size": 3,
            "close_iterations": 1,
            "open_iterations": 1,
            "min_distance": 25,
            "threshold_factor": 0.4,
            "coin_expansion_factor": 1.1,
        },
        "fine_detail": {
            "gaussian_kernel": 3,
            "clahe_clip": 2.5,
            "morph_kernel_size": 2,
            "close_iterations": 1,
            "open_iterations": 1,
            "min_distance": 15,
            "threshold_factor": 0.25,
            "coin_expansion_factor": 1.2,
        },
    }

    if preset not in presets:
        raise ValueError(f"Unknown preset: {preset}. Available: {list(presets.keys())}")

    config = presets[preset].copy()
    config.update(kwargs)  # Allow parameter overrides

    return BeanSegmentor(config)
