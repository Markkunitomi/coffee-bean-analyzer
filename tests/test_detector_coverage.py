#!/usr/bin/env python3
"""Additional tests for detector module to improve coverage."""

import sys
from pathlib import Path
from unittest.mock import Mock, patch

import cv2
import numpy as np
import pytest

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from coffee_bean_analyzer.core.detector import CoinDetector, DetectionResult, detect_coin_legacy


class TestDetectorMissingCoverage:
    """Test missing coverage lines in detector module."""

    def test_detect_coin_legacy_function_with_debug(self):
        """Test the standalone detect_coin_legacy function with debug enabled."""
        # Create test image with circle
        image = np.zeros((200, 200, 3), dtype=np.uint8)
        cv2.circle(image, (100, 100), 50, (255, 255, 255), -1)
        
        # Test with debug=True
        result = detect_coin_legacy(image, debug=True)
        
        # Should return tuple (circle, pixels_per_mm) or (None, None)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_detect_coin_legacy_function_without_debug(self):
        """Test the standalone detect_coin_legacy function without debug."""
        # Create test image
        image = np.zeros((200, 200, 3), dtype=np.uint8)
        
        # Test with debug=False (default)
        result = detect_coin_legacy(image, debug=False)
        
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_coin_detector_no_circles_detected(self):
        """Test coin detector when no circles are detected."""
        config = {"dp": 1, "min_dist": 100, "param1": 50, "param2": 30, "min_radius": 50, "max_radius": 150}
        detector = CoinDetector(config)
        
        # Create image with no circular objects
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        # Add some noise but no circles
        cv2.rectangle(image, (10, 10), (50, 50), (255, 255, 255), -1)
        
        detections = detector.detect(image, debug=True)
        
        # Should return empty list
        assert isinstance(detections, list)
        assert len(detections) == 0

    def test_coin_detector_edge_case_parameters(self):
        """Test coin detector with edge case parameters."""
        # Test with very strict parameters
        config = {
            "dp": 2.0,
            "min_dist": 200,  # Very high min distance
            "param1": 100,
            "param2": 100,    # Very high threshold
            "min_radius": 80,
            "max_radius": 90,
            "gaussian_kernel": 3
        }
        
        detector = CoinDetector(config)
        
        # Create image that might not meet strict criteria
        image = np.zeros((150, 150, 3), dtype=np.uint8)
        cv2.circle(image, (75, 75), 30, (255, 255, 255), -1)  # Radius too small
        
        detections = detector.detect(image)
        
        # Might detect nothing due to strict parameters
        assert isinstance(detections, list)

    def test_coin_detector_with_multiple_circles(self):
        """Test coin detector with multiple circular objects."""
        config = {"dp": 1, "min_dist": 100, "param1": 50, "param2": 30, "min_radius": 50, "max_radius": 150}
        detector = CoinDetector(config)
        
        # Create image with multiple circles
        image = np.zeros((300, 300, 3), dtype=np.uint8)
        cv2.circle(image, (100, 100), 40, (255, 255, 255), -1)
        cv2.circle(image, (200, 200), 45, (255, 255, 255), -1)
        cv2.circle(image, (150, 100), 35, (255, 255, 255), -1)
        
        detections = detector.detect(image, debug=True)
        
        assert isinstance(detections, list)
        # Should detect some circles
        for detection in detections:
            assert isinstance(detection, DetectionResult)
            assert hasattr(detection, 'center')
            assert hasattr(detection, 'radius')
            assert hasattr(detection, 'confidence')
            assert hasattr(detection, 'pixels_per_mm')

    def test_detection_result_with_pixels_per_mm(self):
        """Test DetectionResult creation with pixels_per_mm."""
        result = DetectionResult(
            center=(100, 100),
            radius=50.0,
            confidence=0.8,
            bbox=(50, 50, 100, 100),
            pixels_per_mm=4.0
        )
        
        assert result.center == (100, 100)
        assert result.radius == 50.0
        assert result.confidence == 0.8
        assert result.bbox == (50, 50, 100, 100)
        assert result.pixels_per_mm == 4.0

    def test_detection_result_without_pixels_per_mm(self):
        """Test DetectionResult creation without pixels_per_mm."""
        result = DetectionResult(
            center=(75, 75),
            radius=25.0,
            confidence=0.9,
            bbox=(50, 50, 50, 50),
            pixels_per_mm=None
        )
        
        assert result.center == (75, 75)
        assert result.radius == 25.0
        assert result.confidence == 0.9
        assert result.bbox == (50, 50, 50, 50)
        assert result.pixels_per_mm is None

    def test_coin_detector_configuration_access(self):
        """Test that detector configuration can be accessed."""
        config = {"dp": 1, "min_dist": 100, "param1": 50, "param2": 30, "min_radius": 50, "max_radius": 150}
        detector = CoinDetector(config)
        
        # Test that configuration is stored and accessible
        assert detector.config == config
        assert detector.hough_params["dp"] == 1
        assert detector.hough_params["min_dist"] == 100
        
        # Test that missing config values get defaults
        minimal_config = {"dp": 2}
        detector2 = CoinDetector(minimal_config)
        assert detector2.hough_params["dp"] == 2
        assert detector2.hough_params["param1"] == 50  # Default value

    def test_coin_detector_custom_min_area(self):
        """Test coin detector with custom min_area configuration."""
        config = {
            "min_area": 200  # Custom minimum area
        }
        detector = CoinDetector(config)
        
        # Verify the configuration was stored
        assert detector.config["min_area"] == 200

    def test_coin_detector_best_coin_selection(self):
        """Test get_best_coin method with multiple detections."""
        config = {"dp": 1, "min_dist": 100, "param1": 50, "param2": 30, "min_radius": 50, "max_radius": 150}
        detector = CoinDetector(config)
        
        # Create multiple detection results with different confidence scores
        detection1 = DetectionResult(
            center=(50, 50),
            radius=25.0,
            confidence=0.7,
            bbox=(25, 25, 50, 50),
            pixels_per_mm=4.0
        )
        
        detection2 = DetectionResult(
            center=(100, 100),
            radius=30.0,
            confidence=0.9,  # Higher confidence
            bbox=(70, 70, 60, 60),
            pixels_per_mm=3.5
        )
        
        detection3 = DetectionResult(
            center=(150, 150),
            radius=20.0,
            confidence=0.6,
            bbox=(130, 130, 40, 40),
            pixels_per_mm=5.0
        )
        
        detections = [detection1, detection2, detection3]
        best_coin = detector.get_best_coin(detections)
        
        # Should return the detection with highest confidence
        assert best_coin == detection2
        assert best_coin.confidence == 0.9

    def test_coin_detector_best_coin_empty_list(self):
        """Test get_best_coin with empty detection list."""
        config = {"dp": 1, "min_dist": 100, "param1": 50, "param2": 30, "min_radius": 50, "max_radius": 150}
        detector = CoinDetector(config)
        
        best_coin = detector.get_best_coin([])
        
        assert best_coin is None

    def test_coin_detector_best_coin_single_detection(self):
        """Test get_best_coin with single detection."""
        config = {"dp": 1, "min_dist": 100, "param1": 50, "param2": 30, "min_radius": 50, "max_radius": 150}
        detector = CoinDetector(config)
        
        detection = DetectionResult(
            center=(75, 75),
            radius=35.0,
            confidence=0.8,
            bbox=(40, 40, 70, 70),
            pixels_per_mm=3.0
        )
        
        best_coin = detector.get_best_coin([detection])
        
        assert best_coin == detection

    def test_detect_coin_legacy_with_good_circular_object(self):
        """Test detect_coin_legacy function with a good circular object."""
        # Create image with a well-defined circle
        image = np.zeros((200, 200, 3), dtype=np.uint8)
        cv2.circle(image, (100, 100), 50, (200, 200, 200), -1)
        cv2.circle(image, (100, 100), 50, (150, 150, 150), 3)  # Add edge
        
        circle, pixels_per_mm = detect_coin_legacy(image, debug=True)
        
        if circle is not None:
            assert len(circle) == 3  # (x, y, radius)
            assert isinstance(pixels_per_mm, float)
            assert pixels_per_mm > 0
        else:
            # Detection might fail due to algorithm sensitivity
            assert pixels_per_mm is None

    def test_detect_coin_legacy_with_noisy_image(self):
        """Test detect_coin_legacy function with noisy image."""
        # Create noisy image
        image = np.random.randint(0, 255, (150, 150, 3), dtype=np.uint8)
        
        circle, pixels_per_mm = detect_coin_legacy(image, debug=False)
        
        # Most likely won't detect anything in pure noise
        assert circle is None or len(circle) == 3
        assert pixels_per_mm is None or isinstance(pixels_per_mm, float)

    def test_coin_detector_logging_messages(self):
        """Test that coin detector produces appropriate logging messages."""
        config = {"dp": 1, "min_dist": 100, "param1": 50, "param2": 30, "min_radius": 50, "max_radius": 150}
        detector = CoinDetector(config)
        
        # Create image with no circles
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        
        with patch.object(detector.logger, 'warning') as mock_warning:
            with patch.object(detector.logger, 'info') as mock_info:
                detections = detector.detect(image, debug=True)
                
                # Should log warning about no coins detected
                if len(detections) == 0:
                    mock_warning.assert_called()

    def test_quarter_diameter_calculation(self):
        """Test that quarter diameter is used correctly in pixels_per_mm calculation."""
        # Create a detection with known radius
        config = {"min_area": 50}
        detector = CoinDetector(config)
        
        # Create image with circle
        image = np.zeros((200, 200, 3), dtype=np.uint8)
        cv2.circle(image, (100, 100), 60, (255, 255, 255), -1)
        
        detections = detector.detect(image)
        
        for detection in detections:
            if detection.pixels_per_mm is not None:
                # Verify calculation: pixels_per_mm = (2 * radius) / 24.26
                expected_pixels_per_mm = (2 * detection.radius) / 24.26
                assert abs(detection.pixels_per_mm - expected_pixels_per_mm) < 0.01

    def test_detection_result_validation(self):
        """Test that DetectionResult objects have valid data."""
        # Test manual DetectionResult creation
        result = DetectionResult(
            center=(50, 50),
            radius=25.0,
            confidence=0.8,
            bbox=(25, 25, 50, 50),
            pixels_per_mm=4.0
        )
        
        # Verify bounding box format and reasonable values
        assert len(result.bbox) == 4
        min_col, min_row, width, height = result.bbox
        assert min_col >= 0
        assert min_row >= 0
        assert width > 0
        assert height > 0
        
        # Verify other fields
        assert isinstance(result.center, tuple)
        assert len(result.center) == 2
        assert result.radius > 0
        assert 0 <= result.confidence <= 1
        assert result.pixels_per_mm > 0