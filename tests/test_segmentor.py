"""Unit tests for the segmentation module using pytest."""

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from coffee_bean_analyzer.core.detector import DetectionResult
from coffee_bean_analyzer.core.segmentor import (
    BeanSegmentor,
    CoinExcluder,
    MorphologicalProcessor,
    SegmentationResult,
    WatershedSegmentor,
    create_segmentor,
    segment_beans_legacy,
)


class TestMorphologicalProcessor:
    """Test morphological processing components."""

    @pytest.fixture
    def config(self):
        """Default morphological configuration."""
        return {"morph_kernel_size": 3, "close_iterations": 2, "open_iterations": 1}

    @pytest.fixture
    def processor(self, config):
        """Create MorphologicalProcessor instance."""
        return MorphologicalProcessor(config)

    @pytest.fixture
    def binary_image(self):
        """Create a sample binary image with noise."""
        img = np.zeros((100, 100), dtype=np.uint8)
        # Add some objects
        cv2.circle(img, (30, 30), 15, 255, -1)
        cv2.circle(img, (70, 70), 12, 255, -1)
        # Add some noise
        img[20, 20] = 255
        img[80, 80] = 255
        return img

    def test_create_kernel(self, processor):
        """Test kernel creation."""
        ellipse_kernel = processor.create_kernel(5, "ellipse")
        rect_kernel = processor.create_kernel(5, "rect")

        assert ellipse_kernel.shape == (5, 5)
        assert rect_kernel.shape == (5, 5)
        assert not np.array_equal(ellipse_kernel, rect_kernel)

    def test_apply_closing(self, processor, binary_image):
        """Test morphological closing."""
        result = processor.apply_closing(binary_image, kernel_size=3, iterations=1)

        assert result.shape == binary_image.shape
        assert result.dtype == binary_image.dtype
        # Closing should fill holes and connect nearby objects
        assert np.sum(result) >= np.sum(binary_image)

    def test_apply_opening(self, processor, binary_image):
        """Test morphological opening."""
        result = processor.apply_opening(binary_image, kernel_size=3, iterations=1)

        assert result.shape == binary_image.shape
        assert result.dtype == binary_image.dtype
        # Opening should remove noise and separate objects
        assert np.sum(result) <= np.sum(binary_image)

    def test_clean_binary_mask(self, processor, binary_image):
        """Test complete binary mask cleaning."""
        result = processor.clean_binary_mask(binary_image)

        assert result.shape == binary_image.shape
        assert result.dtype == binary_image.dtype
        # Should be different from original (noise removed, holes filled)
        assert not np.array_equal(result, binary_image)


class TestWatershedSegmentor:
    """Test watershed segmentation components."""

    @pytest.fixture
    def config(self):
        """Default watershed configuration."""
        return {"min_distance": 20, "threshold_factor": 0.3}

    @pytest.fixture
    def segmentor(self, config):
        """Create WatershedSegmentor instance."""
        return WatershedSegmentor(config)

    @pytest.fixture
    def distance_transform(self):
        """Create a sample distance transform."""
        # Create binary image with two objects
        binary = np.zeros((100, 100), dtype=np.uint8)
        cv2.circle(binary, (30, 30), 15, 255, -1)
        cv2.circle(binary, (70, 70), 12, 255, -1)

        # Compute distance transform
        dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
        return dist, binary

    def test_find_peaks(self, segmentor, distance_transform):
        """Test peak detection."""
        dist, _ = distance_transform
        peaks = segmentor.find_peaks(dist)

        assert isinstance(peaks, np.ndarray)
        assert peaks.shape[1] == 2  # Should be (y, x) coordinates
        # Should find at least one peak (likely two for our test image)
        assert len(peaks) >= 1

    def test_create_markers(self, segmentor, distance_transform):
        """Test marker creation."""
        dist, binary = distance_transform
        peaks = segmentor.find_peaks(dist)
        markers = segmentor.create_markers(peaks, binary.shape)

        assert markers.shape == binary.shape
        assert markers.dtype == np.int32
        # Should have as many unique values as peaks + background
        unique_markers = np.unique(markers)
        assert len(unique_markers) == len(peaks) + 1  # +1 for background (0)

    def test_apply_watershed(self, segmentor, distance_transform):
        """Test watershed application."""
        dist, binary = distance_transform
        peaks = segmentor.find_peaks(dist)
        markers = segmentor.create_markers(peaks, binary.shape)
        labels = segmentor.apply_watershed(dist, markers, binary)

        assert labels.shape == binary.shape
        assert labels.dtype in [np.int32, np.int64]
        # Should have background (0) plus labeled regions
        unique_labels = np.unique(labels)
        assert 0 in unique_labels  # Background
        assert len(unique_labels) >= 2  # At least background + one region


class TestCoinExcluder:
    """Test coin exclusion functionality."""

    @pytest.fixture
    def excluder(self):
        """Create CoinExcluder instance."""
        return CoinExcluder(expansion_factor=1.2)

    @pytest.fixture
    def coin_detection(self):
        """Create sample coin detection."""
        return DetectionResult(
            center=(50, 50), radius=20.0, confidence=0.9, bbox=(30, 30, 40, 40)
        )

    @pytest.fixture
    def binary_image(self):
        """Create sample binary image."""
        img = np.ones((100, 100), dtype=np.uint8) * 255
        return img

    def test_create_coin_mask_no_coin(self, excluder):
        """Test mask creation when no coin is provided."""
        mask = excluder.create_coin_mask((100, 100), None)

        assert mask.shape == (100, 100)
        assert np.all(mask == 255)  # Should be all white (no exclusion)

    def test_create_coin_mask_with_coin(self, excluder, coin_detection):
        """Test mask creation with coin detection."""
        mask = excluder.create_coin_mask((100, 100), coin_detection)

        assert mask.shape == (100, 100)
        assert np.any(mask == 0)  # Should have some black (excluded) area
        # Check that center of coin region is excluded
        x, y = coin_detection.center
        assert mask[y, x] == 0

    def test_apply_coin_exclusion_no_coin(self, excluder, binary_image):
        """Test exclusion when no coin is provided."""
        result, info = excluder.apply_coin_exclusion(binary_image, None)

        assert np.array_equal(result, binary_image)  # Should be unchanged
        assert info["coin_excluded"] is False

    def test_apply_coin_exclusion_with_coin(
        self, excluder, binary_image, coin_detection
    ):
        """Test exclusion with coin detection."""
        result, info = excluder.apply_coin_exclusion(binary_image, coin_detection)

        assert result.shape == binary_image.shape
        assert not np.array_equal(result, binary_image)  # Should be changed
        assert info["coin_excluded"] is True
        assert info["coin_center"] == coin_detection.center
        assert info["coin_radius"] == coin_detection.radius


class TestBeanSegmentor:
    """Test the main BeanSegmentor class."""

    @pytest.fixture
    def default_config(self):
        """Default segmentation configuration."""
        return {
            "gaussian_kernel": 5,
            "clahe_clip": 2.0,
            "morph_kernel_size": 3,
            "close_iterations": 2,
            "open_iterations": 1,
            "min_distance": 20,
            "threshold_factor": 0.3,
            "coin_expansion_factor": 1.2,
        }

    @pytest.fixture
    def segmentor(self, default_config):
        """Create BeanSegmentor instance."""
        return BeanSegmentor(default_config)

    @pytest.fixture
    def sample_image(self):
        """Create a sample image with bean-like objects."""
        # Create a color image with bean-like objects
        img = np.zeros((200, 200, 3), dtype=np.uint8)

        # Add some oval objects that could be beans
        cv2.ellipse(img, (60, 60), (15, 10), 45, 0, 360, (100, 100, 100), -1)
        cv2.ellipse(img, (140, 60), (12, 8), 30, 0, 360, (110, 110, 110), -1)
        cv2.ellipse(img, (60, 140), (14, 9), 60, 0, 360, (105, 105, 105), -1)
        cv2.ellipse(img, (140, 140), (13, 10), 20, 0, 360, (95, 95, 95), -1)

        return img

    @pytest.fixture
    def coin_detection(self):
        """Create sample coin detection."""
        return DetectionResult(
            center=(100, 100), radius=25.0, confidence=0.9, bbox=(75, 75, 50, 50)
        )

    def test_segmentor_initialization(self, default_config):
        """Test segmentor initialization."""
        segmentor = BeanSegmentor(default_config)

        assert segmentor.config == default_config
        assert segmentor.preprocessor is not None
        assert segmentor.morph_processor is not None
        assert segmentor.watershed_segmentor is not None
        assert segmentor.coin_excluder is not None

    def test_segment_basic(self, segmentor, sample_image):
        """Test basic segmentation without coin."""
        result = segmentor.segment(sample_image, coin_detection=None, debug=False)

        assert isinstance(result, SegmentationResult)
        assert result.labels.shape == sample_image.shape[:2]
        assert result.binary_mask.shape == sample_image.shape[:2]
        assert result.num_segments >= 0
        assert isinstance(result.preprocessing_metadata, dict)
        assert isinstance(result.segmentation_metadata, dict)
        assert isinstance(result.excluded_regions, list)

    def test_segment_with_coin(self, segmentor, sample_image, coin_detection):
        """Test segmentation with coin exclusion."""
        result = segmentor.segment(
            sample_image, coin_detection=coin_detection, debug=False
        )

        assert isinstance(result, SegmentationResult)
        assert result.labels.shape == sample_image.shape[:2]
        assert result.binary_mask.shape == sample_image.shape[:2]
        assert len(result.excluded_regions) == 1
        assert result.excluded_regions[0]["coin_excluded"] is True

    def test_segment_empty_image(self, segmentor):
        """Test segmentation with empty image."""
        empty_image = np.zeros((50, 50, 3), dtype=np.uint8)

        result = segmentor.segment(empty_image, debug=False)

        # Should handle empty image gracefully
        assert isinstance(result, SegmentationResult)
        assert result.num_segments >= 0

    def test_get_configuration(self, segmentor, default_config):
        """Test getting configuration."""
        config = segmentor.get_configuration()

        assert config == default_config
        assert config is not segmentor.config  # Should be a copy

    def test_update_configuration(self, segmentor):
        """Test updating configuration."""
        new_params = {"min_distance": 30, "threshold_factor": 0.4}

        segmentor.update_configuration(new_params)

        assert segmentor.config["min_distance"] == 30
        assert segmentor.config["threshold_factor"] == 0.4
        # Other parameters should remain unchanged
        assert segmentor.config["gaussian_kernel"] == 5


class TestLegacyCompatibility:
    """Test legacy function compatibility."""

    @pytest.fixture
    def sample_image(self):
        """Create a sample image for testing."""
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.circle(img, (30, 30), 15, (100, 100, 100), -1)
        cv2.circle(img, (70, 70), 12, (110, 110, 110), -1)
        return img

    def test_legacy_function_signature(self, sample_image):
        """Test that legacy function maintains original signature."""
        # Test with default parameters (matching your original function)
        labels, binary = segment_beans_legacy(sample_image, debug=False)

        assert isinstance(labels, np.ndarray)
        assert isinstance(binary, np.ndarray)
        assert labels.shape == sample_image.shape[:2]
        assert binary.shape == sample_image.shape[:2]
        assert labels.dtype in [np.int32, np.int64]
        assert binary.dtype == np.uint8

    def test_legacy_with_coin(self, sample_image):
        """Test legacy function with coin parameter."""
        coin = (50, 50, 20)  # Your original format: (x, y, radius)

        labels, binary = segment_beans_legacy(sample_image, coin=coin, debug=False)

        assert isinstance(labels, np.ndarray)
        assert isinstance(binary, np.ndarray)
        assert labels.shape == sample_image.shape[:2]
        assert binary.shape == sample_image.shape[:2]

    def test_legacy_with_params(self, sample_image):
        """Test legacy function with custom parameters."""
        params = {
            "gaussian_kernel": 7,
            "clahe_clip": 3.0,
            "morph_kernel_size": 5,
            "close_iterations": 3,
            "open_iterations": 2,
            "min_distance": 25,
            "threshold_factor": 0.4,
        }

        labels, binary = segment_beans_legacy(sample_image, params=params, debug=False)

        assert isinstance(labels, np.ndarray)
        assert isinstance(binary, np.ndarray)
        assert labels.shape == sample_image.shape[:2]
        assert binary.shape == sample_image.shape[:2]

    def test_legacy_vs_new_consistency(self, sample_image):
        """Test that legacy and new methods give consistent results."""
        # Test with legacy function
        coin = (50, 50, 20)
        params = {
            "gaussian_kernel": 5,
            "clahe_clip": 2.0,
            "morph_kernel_size": 3,
            "close_iterations": 2,
            "open_iterations": 1,
            "min_distance": 20,
            "threshold_factor": 0.3,
        }

        legacy_labels, legacy_binary = segment_beans_legacy(
            sample_image, coin=coin, params=params, debug=False
        )

        # Test with new segmentor
        segmentor = BeanSegmentor(params)
        coin_detection = DetectionResult(
            center=(50, 50), radius=20.0, confidence=1.0, bbox=(30, 30, 40, 40)
        )

        new_result = segmentor.segment(sample_image, coin_detection, debug=False)

        # Results should be identical (or very close due to floating point precision)
        assert legacy_labels.shape == new_result.labels.shape
        assert legacy_binary.shape == new_result.binary_mask.shape

        # Check that the general structure is similar
        legacy_unique = len(np.unique(legacy_labels))
        new_unique = len(np.unique(new_result.labels))

        # Should have similar number of segments (within reasonable tolerance)
        assert abs(legacy_unique - new_unique) <= 2


class TestSegmentorFactory:
    """Test the segmentor factory function."""

    def test_default_preset(self):
        """Test creating segmentor with default preset."""
        segmentor = create_segmentor("default")

        assert isinstance(segmentor, BeanSegmentor)
        assert segmentor.config["gaussian_kernel"] == 5
        assert segmentor.config["min_distance"] == 20

    def test_aggressive_preset(self):
        """Test aggressive preset."""
        segmentor = create_segmentor("aggressive")

        assert segmentor.config["gaussian_kernel"] == 7
        assert segmentor.config["clahe_clip"] == 3.0
        assert segmentor.config["min_distance"] == 15

    def test_conservative_preset(self):
        """Test conservative preset."""
        segmentor = create_segmentor("conservative")

        assert segmentor.config["gaussian_kernel"] == 3
        assert segmentor.config["clahe_clip"] == 1.5
        assert segmentor.config["min_distance"] == 25

    def test_fine_detail_preset(self):
        """Test fine detail preset."""
        segmentor = create_segmentor("fine_detail")

        assert segmentor.config["gaussian_kernel"] == 3
        assert segmentor.config["min_distance"] == 15
        assert segmentor.config["threshold_factor"] == 0.25

    def test_parameter_override(self):
        """Test overriding preset parameters."""
        segmentor = create_segmentor("default", min_distance=30, clahe_clip=4.0)

        assert segmentor.config["min_distance"] == 30
        assert segmentor.config["clahe_clip"] == 4.0
        # Other parameters should remain from preset
        assert segmentor.config["gaussian_kernel"] == 5

    def test_invalid_preset(self):
        """Test creating segmentor with invalid preset."""
        with pytest.raises(ValueError, match="Unknown preset"):
            create_segmentor("invalid_preset")


@pytest.mark.integration
class TestRealImageSegmentation:
    """Integration tests with real images (requires test data)."""

    def test_real_image_segmentation(self):
        """Test segmentation on a real image if available."""
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

        # Test segmentation with different presets
        for preset in ["default", "aggressive", "conservative", "fine_detail"]:
            segmentor = create_segmentor(preset)
            result = segmentor.segment(real_image)

            # Should return valid result
            assert isinstance(result, SegmentationResult)
            assert result.labels.shape == real_image.shape[:2]
            assert result.binary_mask.shape == real_image.shape[:2]
            assert result.num_segments >= 0
            assert len(result.preprocessing_metadata) > 0
            assert len(result.segmentation_metadata) > 0

    def test_segmentation_with_detected_coin(self):
        """Test segmentation combined with coin detection."""
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

        # Try to detect coin first
        from coffee_bean_analyzer.core.detector import CoinDetector

        coin_config = {
            "dp": 1,
            "min_dist": 100,
            "param1": 50,
            "param2": 30,
            "min_radius": 50,
            "max_radius": 150,
            "gaussian_kernel": 15,
        }
        coin_detector = CoinDetector(coin_config)
        coin_detections = coin_detector.detect(real_image)

        # Perform segmentation
        segmentor = create_segmentor("default")

        if coin_detections:
            # Test with coin exclusion
            best_coin = coin_detector.get_best_coin(coin_detections)
            result = segmentor.segment(real_image, best_coin)

            assert len(result.excluded_regions) == 1
            assert result.excluded_regions[0]["coin_excluded"] is True
        else:
            # Test without coin
            result = segmentor.segment(real_image)

            assert len(result.excluded_regions) == 0

        # Common assertions
        assert isinstance(result, SegmentationResult)
        assert result.num_segments >= 0


class TestSegmentationResult:
    """Test SegmentationResult dataclass."""

    def test_segmentation_result_creation(self):
        """Test SegmentationResult can be created with valid data."""
        labels = np.array([[0, 1], [2, 3]])
        binary_mask = np.array([[0, 255], [255, 0]], dtype=np.uint8)

        result = SegmentationResult(
            labels=labels,
            binary_mask=binary_mask,
            num_segments=3,
            preprocessing_metadata={"test": "value"},
            segmentation_metadata={"param": 42},
            excluded_regions=[],
        )

        assert np.array_equal(result.labels, labels)
        assert np.array_equal(result.binary_mask, binary_mask)
        assert result.num_segments == 3
        assert result.preprocessing_metadata == {"test": "value"}
        assert result.segmentation_metadata == {"param": 42}
        assert result.excluded_regions == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
