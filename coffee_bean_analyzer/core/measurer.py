"""Coffee Bean Analyzer - Bean Measurement Module.

Adapted from the original coffee_bean_analyzer.py script.
Handles measurement of individual coffee bean properties and spatial analysis.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from skimage import measure

# Import our own modules
from .detector import DetectionResult


@dataclass
class BeanMeasurement:
    """Container for individual bean measurements."""

    bean_id: int
    centroid_x: float
    centroid_y: float
    area: float
    length: float  # major axis length
    width: float  # minor axis length
    orientation: float  # in degrees
    eccentricity: float
    solidity: float
    perimeter: float
    aspect_ratio: float
    unit: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert measurement to dictionary."""
        return {
            "bean_id": self.bean_id,
            "centroid_x": self.centroid_x,
            "centroid_y": self.centroid_y,
            "area": self.area,
            "length": self.length,
            "width": self.width,
            "orientation": self.orientation,
            "eccentricity": self.eccentricity,
            "solidity": self.solidity,
            "perimeter": self.perimeter,
            "aspect_ratio": self.aspect_ratio,
            "unit": self.unit,
        }


@dataclass
class MeasurementResult:
    """Container for measurement results."""

    measurements: List[BeanMeasurement]
    total_beans: int
    scale_factor: float
    pixels_per_mm: Optional[float]
    unit: str
    metadata: Dict[str, Any]
    excluded_regions: List[Dict[str, Any]]


class ScaleCalibrator:
    """Handles scale calibration using coin detection."""

    def __init__(self, quarter_diameter_mm: float = 24.26):
        """Initialize scale calibrator.

        Args:
            quarter_diameter_mm: Diameter of US quarter in millimeters
        """
        self.quarter_diameter_mm = quarter_diameter_mm
        self.logger = logging.getLogger(__name__)

    def calculate_scale(
        self, coin_detection: Optional[DetectionResult]
    ) -> Tuple[float, Optional[float], str]:
        """Calculate scale factors from coin detection.

        Args:
            coin_detection: Coin detection result

        Returns:
            Tuple of (scale_factor, pixels_per_mm, unit)
        """
        if coin_detection is None:
            self.logger.warning("No coin detection provided. Using pixel measurements.")
            return 1.0, None, "pixels"

        # Calculate pixels per mm (your original logic)
        coin_radius_pixels = coin_detection.radius
        coin_diameter_pixels = 2 * coin_radius_pixels
        pixels_per_mm = coin_diameter_pixels / self.quarter_diameter_mm

        # Scale factor to convert from pixels to mm
        scale_factor = 1.0 / pixels_per_mm

        self.logger.info(f"Scale calibration: {pixels_per_mm:.2f} pixels/mm")

        return scale_factor, pixels_per_mm, "mm"


class BeanFilter:
    """Handles filtering of bean regions based on various criteria."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize bean filter.

        Args:
            config: Configuration containing filter parameters
        """
        self.config = config
        self.logger = logging.getLogger(__name__)

        self.min_area = config.get("min_area", 100)
        self.coin_overlap_threshold = config.get("coin_overlap_threshold", 1.1)

    def filter_by_area(self, region_props: List) -> List:
        """Filter regions by minimum area."""
        filtered = [prop for prop in region_props if prop.area >= self.min_area]

        removed_count = len(region_props) - len(filtered)
        if removed_count > 0:
            self.logger.info(
                f"Filtered out {removed_count} regions below minimum area ({self.min_area})"
            )

        return filtered

    def filter_coin_overlap(
        self, region_props: List, coin_detection: Optional[DetectionResult]
    ) -> Tuple[List, List[Dict[str, Any]]]:
        """Filter regions that overlap with coin.

        This implements your original coin overlap detection logic.
        """
        if coin_detection is None:
            return region_props, []

        filtered_props = []
        excluded_regions = []

        coin_x, coin_y = coin_detection.center
        coin_radius = coin_detection.radius
        overlap_threshold = coin_radius * self.coin_overlap_threshold

        for prop in region_props:
            centroid_x, centroid_y = (
                prop.centroid[1],
                prop.centroid[0],
            )  # Note: centroid is (row, col)
            distance_to_coin = np.sqrt(
                (centroid_x - coin_x) ** 2 + (centroid_y - coin_y) ** 2
            )

            # Check if centroid is too close to coin center (your original logic)
            if distance_to_coin < overlap_threshold:
                excluded_regions.append(
                    {
                        "reason": "coin_overlap",
                        "centroid": (centroid_x, centroid_y),
                        "distance_to_coin": distance_to_coin,
                        "threshold": overlap_threshold,
                        "area": prop.area,
                    }
                )
                self.logger.debug(
                    f"Excluding region too close to coin: distance={distance_to_coin:.1f}, threshold={overlap_threshold:.1f}"
                )
            else:
                filtered_props.append(prop)

        if excluded_regions:
            self.logger.info(
                f"Excluded {len(excluded_regions)} regions due to coin overlap"
            )

        return filtered_props, excluded_regions


class SpatialSorter:
    """Handles spatial sorting of beans for consistent ordering."""

    def __init__(self, row_threshold: int = 30):
        """Initialize spatial sorter.

        Args:
            row_threshold: Threshold for grouping beans into rows (pixels)
        """
        self.row_threshold = row_threshold
        self.logger = logging.getLogger(__name__)

    def sort_spatially(
        self, measurements: List[BeanMeasurement]
    ) -> List[BeanMeasurement]:
        """Sort beans from left to right, top to bottom to match ground truth order.

        This implements your original sort_beans_spatially logic.
        """
        if not measurements:
            return measurements

        if len(measurements) <= 1:
            return measurements

        # Extract coordinates with measurements
        coords = [
            (m.centroid_x, m.centroid_y, i, m) for i, m in enumerate(measurements)
        ]

        # Group beans into rows based on y-coordinate similarity (your original logic)
        coords_by_y = sorted(coords, key=lambda x: x[1])  # Sort by y-coordinate

        rows = []
        current_row = [coords_by_y[0]]

        for coord in coords_by_y[1:]:
            # If y-coordinate is close to the current row, add to current row (your original logic)
            if abs(coord[1] - current_row[-1][1]) <= self.row_threshold:
                current_row.append(coord)
            else:
                # Start a new row
                rows.append(current_row)
                current_row = [coord]

        # Don't forget the last row
        if current_row:
            rows.append(current_row)

        # Sort each row by x-coordinate (left to right) (your original logic)
        sorted_measurements = []
        for row in rows:
            row_sorted = sorted(row, key=lambda x: x[0])  # Sort by x-coordinate
            sorted_measurements.extend(
                [coord[3] for coord in row_sorted]
            )  # Extract measurements

        # Update bean IDs to reflect sorted order
        for i, measurement in enumerate(sorted_measurements):
            measurement.bean_id = i + 1

        self.logger.info(
            f"Spatially sorted {len(measurements)} beans into {len(rows)} rows"
        )

        return sorted_measurements


class BeanMeasurer:
    """Main bean measurement class for extracting physical properties.

    Adapted from your original measure_beans() function with added modularity
    and detailed measurement capabilities.
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize bean measurer with configuration.

        Args:
            config: Dictionary containing measurement parameters
        """
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Initialize sub-components
        self.scale_calibrator = ScaleCalibrator(
            config.get("quarter_diameter_mm", 24.26)
        )
        self.bean_filter = BeanFilter(config)
        self.spatial_sorter = SpatialSorter(config.get("row_threshold", 30))

    def extract_region_properties(self, labels: np.ndarray) -> List:
        """Extract region properties from segmentation labels."""
        # Get region properties (your original code)
        props = measure.regionprops(labels)
        self.logger.info(f"Extracted properties for {len(props)} regions")
        return props

    def calculate_measurements(
        self, region_props: List, scale_factor: float, unit: str
    ) -> List[BeanMeasurement]:
        """Calculate detailed measurements for each bean region.

        This implements your original measurement calculations.
        """
        measurements = []

        for i, prop in enumerate(region_props):
            # Calculate measurements (your original logic)
            area_real = prop.area * (scale_factor**2)

            # Major and minor axis lengths (converted to real units) (your original logic)
            length_real = prop.major_axis_length * scale_factor
            width_real = prop.minor_axis_length * scale_factor

            # Additional properties (your original logic plus enhancements)
            centroid = prop.centroid
            orientation = np.degrees(prop.orientation)
            eccentricity = prop.eccentricity
            solidity = prop.solidity
            perimeter_real = prop.perimeter * scale_factor

            # Calculate aspect ratio
            aspect_ratio = length_real / width_real if width_real > 0 else 0.0

            measurement = BeanMeasurement(
                bean_id=i + 1,  # Will be updated during spatial sorting
                centroid_x=centroid[1],  # Note: centroid is (row, col), we want (x, y)
                centroid_y=centroid[0],
                area=area_real,
                length=length_real,
                width=width_real,
                orientation=orientation,
                eccentricity=eccentricity,
                solidity=solidity,
                perimeter=perimeter_real,
                aspect_ratio=aspect_ratio,
                unit=unit,
            )

            measurements.append(measurement)

        self.logger.info(f"Calculated measurements for {len(measurements)} beans")
        return measurements

    def measure(
        self,
        labels: np.ndarray,
        original_image: Optional[np.ndarray] = None,
        coin_detection: Optional[DetectionResult] = None,
        debug: bool = False,
    ) -> MeasurementResult:
        """Measure bean properties and convert to real-world units.

        This is your original measure_beans() function adapted to the new structure.

        Args:
            labels: Segmentation labels from watershed
            original_image: Original image for additional analysis
            coin_detection: Coin detection for scale calibration
            debug: Enable debug output

        Returns:
            MeasurementResult containing all measurements and metadata
        """
        if debug:
            self.logger.info("Starting bean measurement")
            if original_image is not None:
                self.logger.info(f"Original image shape: {original_image.shape}")

        # Step 1: Calculate scale factors (your original logic)
        scale_factor, pixels_per_mm, unit = self.scale_calibrator.calculate_scale(
            coin_detection
        )

        if debug:
            if pixels_per_mm:
                self.logger.info(f"Scale calibration: {pixels_per_mm:.2f} pixels/mm")
            else:
                self.logger.info(
                    "No scale calibration available. Using pixel measurements."
                )

        # Step 2: Extract region properties (your original logic)
        props = self.extract_region_properties(labels)

        # Step 3: Filter regions (your original logic)
        filtered_props = self.bean_filter.filter_by_area(props)

        # Step 4: Exclude coin overlaps (your original logic)
        final_props, excluded_regions = self.bean_filter.filter_coin_overlap(
            filtered_props, coin_detection
        )

        if debug:
            self.logger.info(f"Filtered to {len(final_props)} valid bean regions")

        # Step 5: Calculate measurements (your original logic)
        measurements = self.calculate_measurements(final_props, scale_factor, unit)

        # Step 6: Sort spatially (your original logic)
        sorted_measurements = self.spatial_sorter.sort_spatially(measurements)

        if debug:
            self.logger.info(
                f"Measurement completed: {len(sorted_measurements)} beans in {unit}"
            )

        # Prepare metadata
        metadata = {
            "total_regions_detected": len(props),
            "regions_after_area_filter": len(filtered_props),
            "regions_after_coin_filter": len(final_props),
            "scale_factor": scale_factor,
            "measurement_parameters": {
                "min_area": self.config.get("min_area", 100),
                "row_threshold": self.config.get("row_threshold", 30),
                "coin_overlap_threshold": self.config.get(
                    "coin_overlap_threshold", 1.1
                ),
            },
        }

        return MeasurementResult(
            measurements=sorted_measurements,
            total_beans=len(sorted_measurements),
            scale_factor=scale_factor,
            pixels_per_mm=pixels_per_mm,
            unit=unit,
            metadata=metadata,
            excluded_regions=excluded_regions,
        )

    def get_summary_statistics(
        self, measurements: List[BeanMeasurement]
    ) -> Dict[str, Any]:
        """Calculate summary statistics for measurements."""
        if not measurements:
            return {}

        lengths = [m.length for m in measurements]
        widths = [m.width for m in measurements]
        areas = [m.area for m in measurements]
        aspect_ratios = [m.aspect_ratio for m in measurements]

        unit = measurements[0].unit

        return {
            "count": len(measurements),
            "unit": unit,
            "length": {
                "mean": np.mean(lengths),
                "std": np.std(lengths),
                "min": np.min(lengths),
                "max": np.max(lengths),
                "median": np.median(lengths),
            },
            "width": {
                "mean": np.mean(widths),
                "std": np.std(widths),
                "min": np.min(widths),
                "max": np.max(widths),
                "median": np.median(widths),
            },
            "area": {
                "mean": np.mean(areas),
                "std": np.std(areas),
                "min": np.min(areas),
                "max": np.max(areas),
                "median": np.median(areas),
            },
            "aspect_ratio": {
                "mean": np.mean(aspect_ratios),
                "std": np.std(aspect_ratios),
                "min": np.min(aspect_ratios),
                "max": np.max(aspect_ratios),
                "median": np.median(aspect_ratios),
            },
        }


# Legacy compatibility function
def measure_beans_legacy(
    labels,
    original_image=None,
    pixels_per_mm=None,
    coin=None,
    min_area=100,
    debug=False,
):
    """Legacy wrapper to maintain compatibility with your existing code.

    This function provides the same interface as your original measure_beans()
    but uses the new BeanMeasurer class internally.

    Args:
        labels: Segmentation labels
        original_image: Original image
        pixels_per_mm: Scale calibration (your original format)
        coin: Coin tuple (x, y, radius) for exclusion (your original format)
        min_area: Minimum area filter
        debug: Enable debug output

    Returns:
        List of measurement dictionaries (your original format)
    """
    config = {
        "min_area": min_area,
        "row_threshold": 30,
        "coin_overlap_threshold": 1.1,
        "quarter_diameter_mm": 24.26,
    }

    measurer = BeanMeasurer(config)

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

    # If pixels_per_mm is provided directly, create a synthetic coin detection
    elif pixels_per_mm is not None:
        # Create a synthetic coin detection that will give the desired scale
        quarter_diameter_mm = 24.26
        coin_diameter_pixels = pixels_per_mm * quarter_diameter_mm
        coin_radius = coin_diameter_pixels / 2

        coin_detection = DetectionResult(
            center=(0, 0),  # Dummy position, won't be used for overlap detection
            radius=coin_radius,
            confidence=1.0,
            bbox=(0, 0, 0, 0),
        )

    # Perform measurement
    result = measurer.measure(labels, original_image, coin_detection, debug=debug)

    # Convert to original format (list of dictionaries)
    legacy_measurements = []
    for measurement in result.measurements:
        legacy_dict = measurement.to_dict()
        # Remove bean_id to match original format
        legacy_dict.pop("bean_id", None)
        legacy_measurements.append(legacy_dict)

    return legacy_measurements


# Factory function for common measurement configurations
def create_measurer(preset: str = "default", **kwargs) -> BeanMeasurer:
    """Factory function to create measurers with common configurations.

    Args:
        preset: Preset configuration name
        **kwargs: Override specific parameters

    Returns:
        Configured BeanMeasurer instance
    """
    presets = {
        "default": {
            "min_area": 100,
            "row_threshold": 30,
            "coin_overlap_threshold": 1.1,
            "quarter_diameter_mm": 24.26,
        },
        "strict": {
            "min_area": 200,
            "row_threshold": 20,
            "coin_overlap_threshold": 1.2,
            "quarter_diameter_mm": 24.26,
        },
        "permissive": {
            "min_area": 50,
            "row_threshold": 40,
            "coin_overlap_threshold": 1.0,
            "quarter_diameter_mm": 24.26,
        },
        "fine_sorting": {
            "min_area": 100,
            "row_threshold": 15,
            "coin_overlap_threshold": 1.1,
            "quarter_diameter_mm": 24.26,
        },
    }

    if preset not in presets:
        raise ValueError(f"Unknown preset: {preset}. Available: {list(presets.keys())}")

    config = presets[preset].copy()
    config.update(kwargs)  # Allow parameter overrides

    return BeanMeasurer(config)
