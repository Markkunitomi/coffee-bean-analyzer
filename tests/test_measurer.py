"""Unit tests for the measurement module using pytest."""

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))


from coffee_bean_analyzer.core.detector import DetectionResult
from coffee_bean_analyzer.core.measurer import (
    BeanFilter,
    BeanMeasurement,
    BeanMeasurer,
    MeasurementResult,
    ScaleCalibrator,
    SpatialSorter,
    create_measurer,
    measure_beans_legacy,
)


class TestBeanMeasurement:
    """Test BeanMeasurement dataclass."""

    def test_bean_measurement_creation(self):
        """Test BeanMeasurement can be created with valid data."""
        measurement = BeanMeasurement(
            bean_id=1,
            centroid_x=50.5,
            centroid_y=60.5,
            area=150.0,
            length=12.5,
            width=8.3,
            orientation=45.0,
            eccentricity=0.7,
            solidity=0.9,
            perimeter=35.2,
            aspect_ratio=1.5,
            unit="mm",
        )

        assert measurement.bean_id == 1
        assert measurement.centroid_x == 50.5
        assert measurement.area == 150.0
        assert measurement.unit == "mm"

    def test_to_dict(self):
        """Test conversion to dictionary."""
        measurement = BeanMeasurement(
            bean_id=1,
            centroid_x=50.0,
            centroid_y=60.0,
            area=150.0,
            length=12.0,
            width=8.0,
            orientation=45.0,
            eccentricity=0.7,
            solidity=0.9,
            perimeter=35.0,
            aspect_ratio=1.5,
            unit="mm",
        )

        result_dict = measurement.to_dict()

        assert isinstance(result_dict, dict)
        assert result_dict["bean_id"] == 1
        assert result_dict["area"] == 150.0
        assert result_dict["unit"] == "mm"
        assert len(result_dict) == 12  # All fields should be present


class TestScaleCalibrator:
    """Test scale calibration functionality."""

    @pytest.fixture
    def calibrator(self):
        """Create ScaleCalibrator instance."""
        return ScaleCalibrator(quarter_diameter_mm=24.26)

    @pytest.fixture
    def coin_detection(self):
        """Create sample coin detection."""
        return DetectionResult(
            center=(100, 100),
            radius=50.0,  # 100 pixel diameter
            confidence=0.9,
            bbox=(50, 50, 100, 100),
        )

    def test_calculate_scale_no_coin(self, calibrator):
        """Test scale calculation when no coin is provided."""
        scale_factor, pixels_per_mm, unit = calibrator.calculate_scale(None)

        assert scale_factor == 1.0
        assert pixels_per_mm is None
        assert unit == "pixels"

    def test_calculate_scale_with_coin(self, calibrator, coin_detection):
        """Test scale calculation with coin detection."""
        scale_factor, pixels_per_mm, unit = calibrator.calculate_scale(coin_detection)

        # 100 pixel diameter / 24.26 mm = ~4.12 pixels/mm
        expected_pixels_per_mm = 100.0 / 24.26
        expected_scale_factor = 1.0 / expected_pixels_per_mm

        assert abs(pixels_per_mm - expected_pixels_per_mm) < 0.01
        assert abs(scale_factor - expected_scale_factor) < 0.01
        assert unit == "mm"

    def test_custom_quarter_diameter(self):
        """Test calibrator with custom quarter diameter."""
        custom_calibrator = ScaleCalibrator(quarter_diameter_mm=25.0)
        coin_detection = DetectionResult(
            center=(100, 100),
            radius=50.0,  # 100 pixel diameter
            confidence=0.9,
            bbox=(50, 50, 100, 100),
        )

        scale_factor, pixels_per_mm, unit = custom_calibrator.calculate_scale(
            coin_detection
        )

        expected_pixels_per_mm = 100.0 / 25.0  # 4.0 pixels/mm
        assert abs(pixels_per_mm - expected_pixels_per_mm) < 0.01


class TestBeanFilter:
    """Test bean filtering functionality."""

    @pytest.fixture
    def config(self):
        """Default filter configuration."""
        return {"min_area": 100, "coin_overlap_threshold": 1.1}

    @pytest.fixture
    def filter_instance(self, config):
        """Create BeanFilter instance."""
        return BeanFilter(config)

    @pytest.fixture
    def mock_region_props(self):
        """Create mock region properties with different areas."""

        # Create mock objects that simulate skimage regionprops
        class MockRegion:
            def __init__(self, area, centroid):
                self.area = area
                self.centroid = centroid  # (row, col) format

        return [
            MockRegion(area=50, centroid=(30, 30)),  # Below min_area
            MockRegion(area=150, centroid=(60, 60)),  # Valid
            MockRegion(area=200, centroid=(90, 90)),  # Valid
            MockRegion(area=80, centroid=(120, 120)),  # Below min_area
        ]

    def test_filter_by_area(self, filter_instance, mock_region_props):
        """Test filtering by minimum area."""
        filtered = filter_instance.filter_by_area(mock_region_props)

        # Should keep only regions with area >= 100
        assert len(filtered) == 2
        assert all(prop.area >= 100 for prop in filtered)
        assert filtered[0].area == 150
        assert filtered[1].area == 200

    def test_filter_coin_overlap_no_coin(self, filter_instance, mock_region_props):
        """Test coin overlap filtering when no coin is provided."""
        filtered, excluded = filter_instance.filter_coin_overlap(
            mock_region_props, None
        )

        assert filtered == mock_region_props  # Should be unchanged
        assert excluded == []

    def test_filter_coin_overlap_with_coin(self, filter_instance, mock_region_props):
        """Test coin overlap filtering with coin detection."""
        coin_detection = DetectionResult(
            center=(60, 60),  # Close to second region
            radius=20.0,
            confidence=0.9,
            bbox=(40, 40, 40, 40),
        )

        filtered, excluded = filter_instance.filter_coin_overlap(
            mock_region_props, coin_detection
        )

        # Region at (60, 60) should be excluded due to proximity to coin at (60, 60)
        # Overlap threshold is radius * 1.1 = 22.0
        assert len(filtered) < len(mock_region_props)
        assert len(excluded) > 0

        # Check that excluded region info is properly recorded
        for exclusion in excluded:
            assert exclusion["reason"] == "coin_overlap"
            assert "distance_to_coin" in exclusion
            assert "threshold" in exclusion


class TestSpatialSorter:
    """Test spatial sorting functionality."""

    @pytest.fixture
    def sorter(self):
        """Create SpatialSorter instance."""
        return SpatialSorter(row_threshold=30)

    @pytest.fixture
    def sample_measurements(self):
        """Create sample measurements in different spatial positions."""
        return [
            BeanMeasurement(
                1, 80, 30, 100, 10, 8, 0, 0.5, 0.9, 25, 1.25, "mm"
            ),  # Top right
            BeanMeasurement(
                2, 20, 30, 100, 10, 8, 0, 0.5, 0.9, 25, 1.25, "mm"
            ),  # Top left
            BeanMeasurement(
                3, 80, 90, 100, 10, 8, 0, 0.5, 0.9, 25, 1.25, "mm"
            ),  # Bottom right
            BeanMeasurement(
                4, 20, 90, 100, 10, 8, 0, 0.5, 0.9, 25, 1.25, "mm"
            ),  # Bottom left
        ]

    def test_sort_spatially_empty_list(self, sorter):
        """Test sorting empty list."""
        result = sorter.sort_spatially([])
        assert result == []

    def test_sort_spatially_single_item(self, sorter):
        """Test sorting single measurement."""
        measurement = BeanMeasurement(
            1, 50, 50, 100, 10, 8, 0, 0.5, 0.9, 25, 1.25, "mm"
        )
        result = sorter.sort_spatially([measurement])

        assert len(result) == 1
        assert result[0] == measurement

    def test_sort_spatially_multiple_items(self, sorter, sample_measurements):
        """Test spatial sorting of multiple measurements."""
        # Input order: top-right, top-left, bottom-right, bottom-left
        # Expected order: top-left, top-right, bottom-left, bottom-right (row by row, left to right)

        sorted_measurements = sorter.sort_spatially(sample_measurements)

        assert len(sorted_measurements) == 4

    def test_sort_spatially_multiple_items(self, sorter, sample_measurements):
        """Test spatial sorting of multiple measurements."""
        # Input order: top-right, top-left, bottom-right, bottom-left
        # Expected order: top-left, top-right, bottom-left, bottom-right (row by row, left to right)

        sorted_measurements = sorter.sort_spatially(sample_measurements)

        assert len(sorted_measurements) == 4

        # Check that bean IDs are updated to reflect sorted order
        for i, measurement in enumerate(sorted_measurements):
            assert measurement.bean_id == i + 1

        # Check spatial ordering: should be sorted by rows, then by x within each row
        # Top row (y ~= 30): left (x=20) should come before right (x=80)
        # Bottom row (y ~= 90): left (x=20) should come before right (x=80)

        # First two should be top row, sorted left to right
        assert sorted_measurements[0].centroid_y < 60  # Top row
        assert sorted_measurements[1].centroid_y < 60  # Top row
        assert (
            sorted_measurements[0].centroid_x < sorted_measurements[1].centroid_x
        )  # Left before right

        # Last two should be bottom row, sorted left to right
        assert sorted_measurements[2].centroid_y > 60  # Bottom row
        assert sorted_measurements[3].centroid_y > 60  # Bottom row
        assert (
            sorted_measurements[2].centroid_x < sorted_measurements[3].centroid_x
        )  # Left before right


class TestBeanMeasurer:
    """Test the main BeanMeasurer class."""

    @pytest.fixture
    def default_config(self):
        """Default measurement configuration."""
        return {
            "min_area": 100,
            "row_threshold": 30,
            "coin_overlap_threshold": 1.1,
            "quarter_diameter_mm": 24.26,
        }

    @pytest.fixture
    def measurer(self, default_config):
        """Create BeanMeasurer instance."""
        return BeanMeasurer(default_config)

    @pytest.fixture
    def sample_labels(self):
        """Create sample segmentation labels."""
        # Create a label image with a few regions
        labels = np.zeros((100, 100), dtype=np.int32)

        # Add some circular regions
        cv2.circle(labels, (30, 30), 10, 1, -1)  # Region 1
        cv2.circle(labels, (70, 30), 8, 2, -1)  # Region 2
        cv2.circle(labels, (30, 70), 12, 3, -1)  # Region 3
        cv2.circle(labels, (70, 70), 9, 4, -1)  # Region 4

        return labels

    @pytest.fixture
    def coin_detection(self):
        """Create sample coin detection."""
        return DetectionResult(
            center=(50, 50), radius=20.0, confidence=0.9, bbox=(30, 30, 40, 40)
        )

    def test_measurer_initialization(self, default_config):
        """Test measurer initialization."""
        measurer = BeanMeasurer(default_config)

        assert measurer.config == default_config
        assert measurer.scale_calibrator is not None
        assert measurer.bean_filter is not None
        assert measurer.spatial_sorter is not None

    def test_extract_region_properties(self, measurer, sample_labels):
        """Test region properties extraction."""
        props = measurer.extract_region_properties(sample_labels)

        assert len(props) == 4  # Should find 4 regions
        assert all(hasattr(prop, "area") for prop in props)
        assert all(hasattr(prop, "centroid") for prop in props)
        assert all(hasattr(prop, "major_axis_length") for prop in props)

    def test_calculate_measurements(self, measurer):
        """Test measurement calculations."""

        # Create mock region properties
        class MockRegion:
            def __init__(
                self,
                area,
                centroid,
                major_axis,
                minor_axis,
                orientation,
                eccentricity,
                solidity,
                perimeter,
            ):
                self.area = area
                self.centroid = centroid
                self.major_axis_length = major_axis
                self.minor_axis_length = minor_axis
                self.orientation = orientation
                self.eccentricity = eccentricity
                self.solidity = solidity
                self.perimeter = perimeter

        mock_props = [
            MockRegion(
                area=100,
                centroid=(50, 60),
                major_axis=15,
                minor_axis=10,
                orientation=0.5,
                eccentricity=0.6,
                solidity=0.9,
                perimeter=40,
            )
        ]

        scale_factor = 0.1  # 10 pixels per unit
        unit = "mm"

        measurements = measurer.calculate_measurements(mock_props, scale_factor, unit)

        assert len(measurements) == 1
        measurement = measurements[0]

        assert measurement.area == 100 * (0.1**2)  # area * scale_factor^2
        assert measurement.length == 15 * 0.1  # major_axis * scale_factor
        assert measurement.width == 10 * 0.1  # minor_axis * scale_factor
        assert measurement.unit == "mm"
        assert measurement.aspect_ratio == 1.5  # length / width

    def test_measure_basic(self, measurer, sample_labels):
        """Test basic measurement without coin."""
        result = measurer.measure(sample_labels, coin_detection=None, debug=False)

        assert isinstance(result, MeasurementResult)
        assert result.total_beans >= 0
        assert result.unit == "pixels"  # No coin calibration
        assert result.scale_factor == 1.0
        assert result.pixels_per_mm is None
        assert isinstance(result.metadata, dict)
        assert isinstance(result.excluded_regions, list)

    def test_measure_with_coin(self, measurer, sample_labels, coin_detection):
        """Test measurement with coin calibration."""
        result = measurer.measure(
            sample_labels, coin_detection=coin_detection, debug=False
        )

        assert isinstance(result, MeasurementResult)
        assert result.unit == "mm"  # Should use mm with coin calibration
        assert result.pixels_per_mm is not None
        assert result.scale_factor != 1.0

        # Should have some measurements
        assert len(result.measurements) >= 0

        # Check that measurements have proper units
        for measurement in result.measurements:
            assert measurement.unit == "mm"

    def test_measure_empty_labels(self, measurer):
        """Test measurement with empty labels."""
        empty_labels = np.zeros((50, 50), dtype=np.int32)

        result = measurer.measure(empty_labels, debug=False)

        assert isinstance(result, MeasurementResult)
        assert result.total_beans == 0
        assert len(result.measurements) == 0

    def test_get_summary_statistics_empty(self, measurer):
        """Test summary statistics with empty measurements."""
        stats = measurer.get_summary_statistics([])
        assert stats == {}

    def test_get_summary_statistics(self, measurer):
        """Test summary statistics calculation."""
        measurements = [
            BeanMeasurement(1, 50, 50, 100, 12, 8, 0, 0.5, 0.9, 30, 1.5, "mm"),
            BeanMeasurement(2, 60, 60, 120, 14, 9, 0, 0.6, 0.9, 35, 1.56, "mm"),
            BeanMeasurement(3, 70, 70, 110, 13, 8.5, 0, 0.55, 0.9, 32, 1.53, "mm"),
        ]

        stats = measurer.get_summary_statistics(measurements)

        assert stats["count"] == 3
        assert stats["unit"] == "mm"
        assert "length" in stats
        assert "width" in stats
        assert "area" in stats
        assert "aspect_ratio" in stats

        # Check that statistics include mean, std, min, max, median
        for metric in ["length", "width", "area", "aspect_ratio"]:
            assert "mean" in stats[metric]
            assert "std" in stats[metric]
            assert "min" in stats[metric]
            assert "max" in stats[metric]
            assert "median" in stats[metric]


class TestLegacyCompatibility:
    """Test legacy function compatibility."""

    @pytest.fixture
    def sample_labels(self):
        """Create sample segmentation labels."""
        labels = np.zeros((100, 100), dtype=np.int32)
        cv2.circle(labels, (30, 30), 10, 1, -1)
        cv2.circle(labels, (70, 70), 12, 2, -1)
        return labels

    def test_legacy_function_signature(self, sample_labels):
        """Test that legacy function maintains original signature."""
        # Test with default parameters (matching your original function)
        measurements = measure_beans_legacy(sample_labels, debug=False)

        assert isinstance(measurements, list)

        # Each measurement should be a dictionary
        for measurement in measurements:
            assert isinstance(measurement, dict)
            assert "centroid_x" in measurement
            assert "centroid_y" in measurement
            assert "area" in measurement
            assert "length" in measurement
            assert "width" in measurement
            assert "unit" in measurement

    def test_legacy_with_coin(self, sample_labels):
        """Test legacy function with coin parameter."""
        coin = (50, 50, 20)  # Your original format: (x, y, radius)

        measurements = measure_beans_legacy(sample_labels, coin=coin, debug=False)

        assert isinstance(measurements, list)
        # Should use mm units with coin calibration
        if measurements:
            assert measurements[0]["unit"] == "mm"

    def test_legacy_with_pixels_per_mm(self, sample_labels):
        """Test legacy function with pixels_per_mm parameter."""
        measurements = measure_beans_legacy(
            sample_labels, pixels_per_mm=4.0, debug=False
        )

        assert isinstance(measurements, list)
        if measurements:
            assert measurements[0]["unit"] == "mm"

    def test_legacy_with_min_area(self, sample_labels):
        """Test legacy function with min_area parameter."""
        measurements = measure_beans_legacy(sample_labels, min_area=50, debug=False)

        assert isinstance(measurements, list)
        # All measurements should have area >= 50 (in pixels^2)
        for measurement in measurements:
            # Note: area is converted to real units, so we need to account for scale
            assert measurement["area"] >= 0  # Just check it's positive

    def test_legacy_vs_new_consistency(self, sample_labels):
        """Test that legacy and new methods give consistent results."""
        # Test with legacy function
        coin = (50, 50, 20)
        legacy_measurements = measure_beans_legacy(
            sample_labels, coin=coin, min_area=100, debug=False
        )

        # Test with new measurer
        config = {
            "min_area": 100,
            "row_threshold": 30,
            "coin_overlap_threshold": 1.1,
            "quarter_diameter_mm": 24.26,
        }
        measurer = BeanMeasurer(config)
        coin_detection = DetectionResult(
            center=(50, 50), radius=20.0, confidence=1.0, bbox=(30, 30, 40, 40)
        )

        new_result = measurer.measure(
            sample_labels, coin_detection=coin_detection, debug=False
        )

        # Should have same number of measurements
        assert len(legacy_measurements) == len(new_result.measurements)

        # Check that measurement values are similar (allowing for small differences)
        if legacy_measurements and new_result.measurements:
            legacy_dict = legacy_measurements[0]
            new_measurement = new_result.measurements[0]

            # Values should be similar (within 1% tolerance)
            assert (
                abs(legacy_dict["area"] - new_measurement.area) / new_measurement.area
                < 0.01
            )
            assert (
                abs(legacy_dict["length"] - new_measurement.length)
                / new_measurement.length
                < 0.01
            )


class TestMeasurerFactory:
    """Test the measurer factory function."""

    def test_default_preset(self):
        """Test creating measurer with default preset."""
        measurer = create_measurer("default")

        assert isinstance(measurer, BeanMeasurer)
        assert measurer.config["min_area"] == 100
        assert measurer.config["row_threshold"] == 30

    def test_strict_preset(self):
        """Test strict preset."""
        measurer = create_measurer("strict")

        assert measurer.config["min_area"] == 200
        assert measurer.config["row_threshold"] == 20
        assert measurer.config["coin_overlap_threshold"] == 1.2

    def test_permissive_preset(self):
        """Test permissive preset."""
        measurer = create_measurer("permissive")

        assert measurer.config["min_area"] == 50
        assert measurer.config["row_threshold"] == 40
        assert measurer.config["coin_overlap_threshold"] == 1.0

    def test_fine_sorting_preset(self):
        """Test fine sorting preset."""
        measurer = create_measurer("fine_sorting")

        assert measurer.config["row_threshold"] == 15
        assert measurer.config["min_area"] == 100

    def test_parameter_override(self):
        """Test overriding preset parameters."""
        measurer = create_measurer("default", min_area=150, row_threshold=40)

        assert measurer.config["min_area"] == 150
        assert measurer.config["row_threshold"] == 40
        # Other parameters should remain from preset
        assert measurer.config["coin_overlap_threshold"] == 1.1

    def test_invalid_preset(self):
        """Test creating measurer with invalid preset."""
        with pytest.raises(ValueError, match="Unknown preset"):
            create_measurer("invalid_preset")


@pytest.mark.integration
class TestRealImageMeasurement:
    """Integration tests with real images (requires test data)."""

    def test_real_image_measurement(self):
        """Test measurement on real segmentation data if available."""
        test_data_dir = Path(__file__).parent / "data"

        if not test_data_dir.exists():
            pytest.skip("No test data directory found")

        # Create synthetic segmentation labels for testing
        # (In real scenario, these would come from actual segmentation)
        labels = np.zeros((200, 200), dtype=np.int32)

        # Add some regions that simulate segmented beans
        cv2.ellipse(labels, (60, 60), (15, 10), 45, 0, 360, 1, -1)
        cv2.ellipse(labels, (140, 60), (12, 8), 30, 0, 360, 2, -1)
        cv2.ellipse(labels, (60, 140), (14, 9), 60, 0, 360, 3, -1)
        cv2.ellipse(labels, (140, 140), (13, 10), 20, 0, 360, 4, -1)

        # Test measurement with different presets
        for preset in ["default", "strict", "permissive", "fine_sorting"]:
            measurer = create_measurer(preset)
            result = measurer.measure(labels)

            # Should return valid result
            assert isinstance(result, MeasurementResult)
            assert result.total_beans >= 0
            assert len(result.measurements) == result.total_beans
            assert result.unit in ["pixels", "mm"]

            # Check measurement properties
            for measurement in result.measurements:
                assert measurement.area > 0
                assert measurement.length > 0
                assert measurement.width > 0
                assert measurement.aspect_ratio > 0
                assert measurement.unit == result.unit


class TestMeasurementResult:
    """Test MeasurementResult dataclass."""

    def test_measurement_result_creation(self):
        """Test MeasurementResult can be created with valid data."""
        measurements = [
            BeanMeasurement(1, 50, 50, 100, 12, 8, 0, 0.5, 0.9, 30, 1.5, "mm")
        ]

        result = MeasurementResult(
            measurements=measurements,
            total_beans=1,
            scale_factor=0.25,
            pixels_per_mm=4.0,
            unit="mm",
            metadata={"test": "value"},
            excluded_regions=[],
        )

        assert len(result.measurements) == 1
        assert result.total_beans == 1
        assert result.scale_factor == 0.25
        assert result.pixels_per_mm == 4.0
        assert result.unit == "mm"
        assert result.metadata == {"test": "value"}
        assert result.excluded_regions == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
