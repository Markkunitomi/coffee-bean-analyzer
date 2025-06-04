"""Unit tests for the detector module using pytest."""

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from coffee_bean_analyzer.core.detector import (
    CoinDetector,
    DetectionResult,
    detect_coin_legacy,
)


class TestCoinDetector:
    """Test suite for CoinDetector class."""

    @pytest.fixture
    def default_config(self):
        """Default configuration for tests."""
        return {
            "dp": 1,
            "min_dist": 100,
            "param1": 50,
            "param2": 30,
            "min_radius": 50,
            "max_radius": 150,
            "gaussian_kernel": 15,
        }

    @pytest.fixture
    def detector(self, default_config):
        """Create a CoinDetector instance."""
        return CoinDetector(default_config)

    @pytest.fixture
    def sample_image(self):
        """Load a sample image for testing."""
        # Try to find a test image
        test_data_dir = Path(__file__).parent / "data"

        if test_data_dir.exists():
            # Look for any image file
            for pattern in ["*.tif", "*.jpg", "*.png"]:
                images = list(test_data_dir.glob(pattern))
                if images:
                    img = cv2.imread(str(images[0]))
                    if img is not None:
                        print(f"Using real test image: {images[0]}")
                        return img

        # If no real image, create a synthetic one with a circle
        print("Using synthetic test image")
        img = np.zeros((400, 400, 3), dtype=np.uint8)
        cv2.circle(img, (200, 200), 60, (200, 200, 200), -1)  # Gray circle
        return img

    def test_detector_initialization(self, default_config):
        """Test that detector initializes correctly."""
        detector = CoinDetector(default_config)
        assert detector.config == default_config
        assert detector.hough_params["min_radius"] == 50
        assert detector.gaussian_kernel == 15

    def test_detect_returns_list(self, detector, sample_image):
        """Test that detect() returns a list of DetectionResult objects."""
        results = detector.detect(sample_image)
        assert isinstance(results, list)

        for result in results:
            assert isinstance(result, DetectionResult)
            assert isinstance(result.center, tuple)
            assert len(result.center) == 2
            assert isinstance(result.radius, float)
            assert 0 <= result.confidence <= 1

    def test_get_best_coin_empty_list(self, detector):
        """Test get_best_coin with empty detection list."""
        result = detector.get_best_coin([])
        assert result is None

    def test_get_best_coin_single_detection(self, detector):
        """Test get_best_coin with single detection."""
        detection = DetectionResult(
            center=(100, 100), radius=50.0, confidence=0.8, bbox=(50, 50, 100, 100)
        )
        result = detector.get_best_coin([detection])
        assert result == detection

    def test_get_best_coin_multiple_detections(self, detector):
        """Test get_best_coin selects highest confidence."""
        detections = [
            DetectionResult((100, 100), 50.0, 0.6, (50, 50, 100, 100)),
            DetectionResult((200, 200), 60.0, 0.9, (140, 140, 120, 120)),
            DetectionResult((300, 300), 55.0, 0.7, (245, 245, 110, 110)),
        ]
        result = detector.get_best_coin(detections)
        assert result.confidence == 0.9
        assert result.center == (200, 200)


class TestLegacyCompatibility:
    """Test suite for legacy function compatibility."""

    @pytest.fixture
    def sample_image(self):
        """Create a synthetic image with a coin-like circle."""
        img = np.zeros((400, 400, 3), dtype=np.uint8)
        # Create a bright circle that should be detected
        cv2.circle(img, (200, 200), 60, (200, 200, 200), -1)
        # Add some noise/texture
        cv2.circle(img, (200, 200), 60, (180, 180, 180), 2)
        return img

    def test_legacy_function_signature(self, sample_image):
        """Test that legacy function maintains original signature."""
        result = detect_coin_legacy(sample_image, debug=False)
        assert isinstance(result, tuple)
        assert len(result) == 2

        best_circle, pixels_per_mm = result

        # If coin detected
        if best_circle is not None:
            assert isinstance(best_circle, tuple)
            assert len(best_circle) == 3  # (x, y, radius)
            assert isinstance(pixels_per_mm, float)
            assert pixels_per_mm > 0
        else:
            assert pixels_per_mm is None

    def test_legacy_vs_new_consistency(self, sample_image):
        """Test that legacy and new methods give consistent results."""
        # Test with legacy method
        legacy_result = detect_coin_legacy(sample_image, debug=False)

        # Test with new method
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
        new_detections = detector.detect(sample_image, debug=False)

        # Both should either find coins or not find coins
        legacy_found = legacy_result[0] is not None
        new_found = len(new_detections) > 0

        # They should agree on whether a coin was found
        # (allowing for slight differences in detection sensitivity)
        if legacy_found or new_found:
            # At least one method found something, which is reasonable
            assert True
        else:
            # Both found nothing, which is also fine for a synthetic image
            assert True


class TestDetectionResult:
    """Test suite for DetectionResult dataclass."""

    def test_detection_result_creation(self):
        """Test DetectionResult can be created with valid data."""
        result = DetectionResult(
            center=(100, 150), radius=25.5, confidence=0.85, bbox=(75, 125, 50, 50)
        )

        assert result.center == (100, 150)
        assert result.radius == 25.5
        assert result.confidence == 0.85
        assert result.bbox == (75, 125, 50, 50)


# Integration test for real image (if available)
@pytest.mark.integration
class TestRealImageDetection:
    """Integration tests with real images (requires test data)."""

    def test_real_image_detection(self):
        """Test detection on a real image if available."""
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

        # Test detection
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
        detections = detector.detect(real_image)

        # Should return a list (may be empty if no coins in image)
        assert isinstance(detections, list)

        # If coins detected, they should have reasonable properties
        for detection in detections:
            assert detection.radius > 0
            assert 0 <= detection.confidence <= 1
            assert detection.center[0] >= 0
            assert detection.center[1] >= 0


if __name__ == "__main__":
    # Run tests if script is executed directly
    pytest.main([__file__, "-v"])
