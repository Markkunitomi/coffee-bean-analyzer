"""Coffee Bean Analyzer - Core Detection Module

Adapted from the original coffee_bean_analyzer.py script.
Handles detection of coins (for scale calibration) and coffee beans.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class DetectionResult:
    """Container for detection results"""

    center: Tuple[int, int]
    radius: float
    confidence: float
    bbox: Tuple[int, int, int, int]  # x, y, width, height


class CoinDetector:
    """Detects US quarter coins in images for scale calibration.

    Adapted from the original detect_coin() function.
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize coin detector with configuration parameters.

        Args:
            config: Dictionary containing detection parameters
        """
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Extract parameters from config (with defaults from your original code)
        self.hough_params = {
            "dp": config.get("dp", 1),
            "min_dist": config.get("min_dist", 100),
            "param1": config.get("param1", 50),
            "param2": config.get("param2", 30),
            "min_radius": config.get("min_radius", 50),
            "max_radius": config.get("max_radius", 150),
        }

        # Gaussian blur parameters (from your preprocess_image)
        self.gaussian_kernel = config.get("gaussian_kernel", 15)

    def detect(self, image: np.ndarray, debug: bool = False) -> List[DetectionResult]:
        """Detect quarter coins in the input image.

        This is your original detect_coin() function adapted to return structured results.

        Args:
            image: Input image as numpy array (BGR format)
            debug: Enable debug output

        Returns:
            List of DetectionResult objects for detected coins
        """
        # Convert to grayscale (from your original code)
        gray = (
            cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        )

        # Apply Gaussian blur to reduce noise (from your original code)
        blurred = cv2.GaussianBlur(
            gray, (self.gaussian_kernel, self.gaussian_kernel), 0
        )

        # Use HoughCircles to detect circular objects (your original implementation)
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=self.hough_params["dp"],
            minDist=self.hough_params["min_dist"],
            param1=self.hough_params["param1"],
            param2=self.hough_params["param2"],
            minRadius=self.hough_params["min_radius"],
            maxRadius=self.hough_params["max_radius"],
        )

        detections = []

        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")

            # Find the most circular object (your original logic)
            best_circle = None
            best_score = 0

            for x, y, r in circles:
                # Create mask for the circle (your original code)
                mask = np.zeros(gray.shape, dtype=np.uint8)
                cv2.circle(mask, (x, y), r, 255, -1)

                # Calculate edge strength within the circle (your original code)
                edges = cv2.Canny(gray, 50, 150)
                edge_score = np.sum(edges[mask > 0])

                # Score based on circularity and edge strength (your original code)
                score = edge_score / (np.pi * r * r)

                if score > best_score:
                    best_score = score
                    best_circle = (x, y, r)

            if best_circle is not None:
                x, y, r = best_circle

                # Create DetectionResult from your best circle
                detection = DetectionResult(
                    center=(x, y),
                    radius=float(r),
                    confidence=min(best_score / 10.0, 1.0),  # Normalize score to 0-1
                    bbox=(x - r, y - r, 2 * r, 2 * r),
                )
                detections.append(detection)

                if debug:
                    self.logger.info(f"Coin detected: center=({x}, {y}), radius={r}")

        if debug and not detections:
            self.logger.warning("No coin detected for scale calibration")

        return detections

    def get_best_coin(
        self, detections: List[DetectionResult]
    ) -> Optional[DetectionResult]:
        """Get the best coin detection (highest confidence).

        Args:
            detections: List of coin detections

        Returns:
            Best coin detection or None
        """
        if not detections:
            return None

        return max(detections, key=lambda d: d.confidence)


class BeanDetector:
    """Detects individual coffee beans in segmented images.

    This will be adapted from your segment_beans() function in the next step.
    For now, it's a placeholder that works with your existing segmentation.
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize bean detector with configuration parameters.

        Args:
            config: Dictionary containing detection parameters
        """
        self.config = config
        self.logger = logging.getLogger(__name__)

        # These will be used when we adapt your segment_beans function
        self.min_area = config.get("min_area", 100)

    def detect_from_labels(
        self, labels: np.ndarray, original_image: Optional[np.ndarray] = None
    ) -> List[DetectionResult]:
        """Convert segmentation labels to bean detections.

        This is a bridge function to work with your existing segment_beans output.

        Args:
            labels: Segmentation labels from your watershed algorithm
            original_image: Original image for additional analysis

        Returns:
            List of DetectionResult objects for detected beans
        """
        from skimage import measure

        detections = []

        # Get region properties (from your measure_beans function)
        props = measure.regionprops(labels)

        for i, prop in enumerate(props):
            # Filter out very small regions (your original logic)
            if prop.area < self.min_area:
                continue

            # Calculate center and radius
            centroid_y, centroid_x = prop.centroid
            equivalent_radius = np.sqrt(prop.area / np.pi)

            # Calculate bounding box
            min_row, min_col, max_row, max_col = prop.bbox
            bbox_width = max_col - min_col
            bbox_height = max_row - min_row

            # Simple confidence based on area and solidity
            confidence = min(prop.solidity, 1.0)

            detection = DetectionResult(
                center=(int(centroid_x), int(centroid_y)),
                radius=equivalent_radius,
                confidence=confidence,
                bbox=(min_col, min_row, bbox_width, bbox_height),
            )
            detections.append(detection)

        self.logger.info(f"Detected {len(detections)} potential beans")
        return detections


# Utility function to maintain compatibility with your existing code
def detect_coin_legacy(image, debug=False):
    """Legacy wrapper to maintain compatibility with your existing code.

    This function provides the same interface as your original detect_coin()
    but uses the new CoinDetector class internally.

    Returns: (best_circle, pixels_per_mm) tuple like your original function
    """
    # Use default configuration
    config = {
        "dp": 1,
        "min_dist": 100,
        "param1": 50,
        "param2": 30,
        "min_radius": 50,
        "max_radius": 150,
        "gaussian_kernel": 15,
    }

    detector = CoinDetector(config)
    detections = detector.detect(image, debug=debug)

    if detections:
        best_detection = detector.get_best_coin(detections)
        if best_detection:
            # Convert back to your original format
            x, y = best_detection.center
            r = best_detection.radius
            best_circle = (x, y, int(r))

            # Calculate pixels_per_mm (your original logic)
            QUARTER_DIAMETER_MM = 24.26
            pixels_per_mm = (2 * r) / QUARTER_DIAMETER_MM

            if debug:
                print(f"Calibration: {pixels_per_mm:.2f} pixels/mm")

            return best_circle, pixels_per_mm

    return None, None
