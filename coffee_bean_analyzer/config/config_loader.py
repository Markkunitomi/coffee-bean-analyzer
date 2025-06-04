"""Coffee Bean Analyzer - Configuration Management

Handles loading and validation of configuration files with defaults.
"""

import importlib.resources
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


@dataclass
class DetectionConfig:
    """Configuration for detection algorithms"""

    # Coin detection parameters
    coin_hough_dp: float = 1.0
    coin_hough_min_dist: int = 100
    coin_hough_param1: int = 50
    coin_hough_param2: int = 30
    coin_min_radius: int = 50
    coin_max_radius: int = 150

    # Bean detection parameters
    bean_min_area: int = 500
    bean_max_area: int = 5000
    bean_min_aspect_ratio: float = 1.2
    bean_max_aspect_ratio: float = 3.0
    bean_min_solidity: float = 0.7


@dataclass
class SegmentationConfig:
    """Configuration for segmentation algorithms"""

    # Watershed parameters
    gaussian_blur_kernel: int = 5
    morphology_kernel: int = 3
    distance_threshold: float = 0.5

    # Preprocessing
    adaptive_thresh_block_size: int = 11
    adaptive_thresh_c: int = 2

    # Post-processing
    min_contour_area: int = 100
    fill_holes: bool = True


@dataclass
class MeasurementConfig:
    """Configuration for measurement calculations"""

    # Reference measurements
    quarter_diameter_mm: float = 24.26

    # Output precision
    measurement_precision: int = 2
    area_precision: int = 1

    # Validation thresholds
    min_length_mm: float = 3.0
    max_length_mm: float = 15.0
    min_width_mm: float = 2.0
    max_width_mm: float = 8.0


@dataclass
class OptimizationConfig:
    """Configuration for parameter optimization"""

    # Grid search parameters
    n_trials: int = 100
    timeout_seconds: Optional[int] = None
    n_jobs: int = 1

    # Metrics
    primary_metric: str = "accuracy"
    validation_split: float = 0.2

    # Parameter ranges (will be loaded from separate file if provided)
    param_ranges: Dict[str, Dict[str, Any]] = None


@dataclass
class OutputConfig:
    """Configuration for output generation"""

    # Image outputs
    save_annotated_images: bool = True
    save_individual_beans: bool = False
    save_intermediate_steps: bool = False

    # Data outputs
    csv_delimiter: str = ","
    csv_encoding: str = "utf-8"
    report_format: str = "detailed"  # "detailed", "summary", "minimal"

    # Visualization
    annotation_color: tuple = (0, 255, 0)  # Green in BGR
    annotation_thickness: int = 2
    show_measurements: bool = True
    show_confidence: bool = False


@dataclass
class AppConfig:
    """Main application configuration"""

    detection: DetectionConfig
    segmentation: SegmentationConfig
    measurement: MeasurementConfig
    optimization: OptimizationConfig
    output: OutputConfig

    # Logging
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


def load_default_config() -> Dict[str, Any]:
    """Load default configuration from package resources.

    Returns:
        Dictionary containing default configuration
    """
    try:
        # Try to load from package resources first
        with importlib.resources.open_text(
            "coffee_bean_analyzer.config", "default_config.yaml"
        ) as f:
            config_data = yaml.safe_load(f)
    except (FileNotFoundError, ImportError):
        # Fallback to hardcoded defaults
        config_data = {
            "detection": {
                "coin_detection": {
                    "dp": 1,
                    "min_dist": 100,
                    "param1": 50,
                    "param2": 30,
                    "min_radius": 50,
                    "max_radius": 150,
                },
                "bean_detection": {
                    "min_area": 500,
                    "max_area": 5000,
                    "min_aspect_ratio": 1.2,
                    "max_aspect_ratio": 3.0,
                    "min_solidity": 0.7,
                },
            },
            "segmentation": {
                "watershed": {
                    "gaussian_blur_kernel": 5,
                    "morphology_kernel": 3,
                    "distance_threshold": 0.5,
                },
                "preprocessing": {
                    "adaptive_thresh_block_size": 11,
                    "adaptive_thresh_c": 2,
                },
            },
            "measurement": {
                "quarter_diameter_mm": 24.26,
                "measurement_precision": 2,
                "area_precision": 1,
                "min_length_mm": 3.0,
                "max_length_mm": 15.0,
                "min_width_mm": 2.0,
                "max_width_mm": 8.0,
            },
            "optimization": {
                "n_trials": 100,
                "timeout_seconds": None,
                "n_jobs": 1,
                "primary_metric": "accuracy",
                "validation_split": 0.2,
            },
            "output": {
                "save_annotated_images": True,
                "save_individual_beans": False,
                "save_intermediate_steps": False,
                "csv_delimiter": ",",
                "csv_encoding": "utf-8",
                "report_format": "detailed",
                "annotation_color": [0, 255, 0],
                "annotation_thickness": 2,
                "show_measurements": True,
                "show_confidence": False,
            },
        }

    return config_data


def load_config(config_path: Optional[Path] = None) -> AppConfig:
    """Load configuration from file or use defaults.

    Args:
        config_path: Optional path to configuration file

    Returns:
        AppConfig object with loaded configuration
    """
    logger = logging.getLogger(__name__)

    # Start with default configuration
    config_data = load_default_config()

    # Override with user configuration if provided
    if config_path and config_path.exists():
        try:
            with open(config_path, encoding="utf-8") as f:
                user_config = yaml.safe_load(f)

            # Deep merge user config with defaults
            config_data = deep_merge(config_data, user_config)
            logger.info(f"Loaded configuration from {config_path}")

        except Exception as e:
            logger.warning(f"Failed to load config from {config_path}: {e}")
            logger.info("Using default configuration")

    # Convert to structured configuration objects
    try:
        app_config = AppConfig(
            detection=DetectionConfig(
                **flatten_dict(config_data.get("detection", {}), "detection")
            ),
            segmentation=SegmentationConfig(
                **flatten_dict(config_data.get("segmentation", {}), "segmentation")
            ),
            measurement=MeasurementConfig(
                **flatten_dict(config_data.get("measurement", {}), "measurement")
            ),
            optimization=OptimizationConfig(
                **flatten_dict(config_data.get("optimization", {}), "optimization")
            ),
            output=OutputConfig(
                **flatten_dict(config_data.get("output", {}), "output")
            ),
        )

        # Validate configuration
        validate_config(app_config)

        return app_config

    except Exception as e:
        logger.error(f"Failed to parse configuration: {e}")
        logger.info("Using default configuration objects")

        # Return default configuration objects
        return AppConfig(
            detection=DetectionConfig(),
            segmentation=SegmentationConfig(),
            measurement=MeasurementConfig(),
            optimization=OptimizationConfig(),
            output=OutputConfig(),
        )


def deep_merge(
    base_dict: Dict[str, Any], override_dict: Dict[str, Any]
) -> Dict[str, Any]:
    """Deep merge two dictionaries, with override_dict taking precedence.

    Args:
        base_dict: Base dictionary
        override_dict: Dictionary with override values

    Returns:
        Merged dictionary
    """
    result = base_dict.copy()

    for key, value in override_dict.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value

    return result


def flatten_dict(nested_dict: Dict[str, Any], section: str) -> Dict[str, Any]:
    """Flatten nested dictionary for dataclass initialization.

    Args:
        nested_dict: Nested dictionary to flatten
        section: Configuration section name

    Returns:
        Flattened dictionary with appropriate keys
    """
    flattened = {}

    # Map nested keys to flat keys based on section
    if section == "detection":
        coin_config = nested_dict.get("coin_detection", {})
        bean_config = nested_dict.get("bean_detection", {})

        # Map coin detection parameters
        flattened.update(
            {
                "coin_hough_dp": coin_config.get("dp", 1.0),
                "coin_hough_min_dist": coin_config.get("min_dist", 100),
                "coin_hough_param1": coin_config.get("param1", 50),
                "coin_hough_param2": coin_config.get("param2", 30),
                "coin_min_radius": coin_config.get("min_radius", 50),
                "coin_max_radius": coin_config.get("max_radius", 150),
            }
        )

        # Map bean detection parameters
        flattened.update(
            {
                "bean_min_area": bean_config.get("min_area", 500),
                "bean_max_area": bean_config.get("max_area", 5000),
                "bean_min_aspect_ratio": bean_config.get("min_aspect_ratio", 1.2),
                "bean_max_aspect_ratio": bean_config.get("max_aspect_ratio", 3.0),
                "bean_min_solidity": bean_config.get("min_solidity", 0.7),
            }
        )

    elif section == "segmentation":
        watershed_config = nested_dict.get("watershed", {})
        preprocess_config = nested_dict.get("preprocessing", {})

        flattened.update(
            {
                "gaussian_blur_kernel": watershed_config.get("gaussian_blur_kernel", 5),
                "morphology_kernel": watershed_config.get("morphology_kernel", 3),
                "distance_threshold": watershed_config.get("distance_threshold", 0.5),
                "adaptive_thresh_block_size": preprocess_config.get(
                    "adaptive_thresh_block_size", 11
                ),
                "adaptive_thresh_c": preprocess_config.get("adaptive_thresh_c", 2),
                "min_contour_area": nested_dict.get("min_contour_area", 100),
                "fill_holes": nested_dict.get("fill_holes", True),
            }
        )

    else:
        # For other sections, use direct mapping
        flattened = nested_dict

    return flattened


def validate_config(config: AppConfig) -> None:
    """Validate configuration parameters.

    Args:
        config: Configuration to validate

    Raises:
        ValueError: If configuration is invalid
    """
    logger = logging.getLogger(__name__)

    # Validate detection parameters
    if config.detection.coin_min_radius >= config.detection.coin_max_radius:
        raise ValueError("coin_min_radius must be less than coin_max_radius")

    if config.detection.bean_min_area >= config.detection.bean_max_area:
        raise ValueError("bean_min_area must be less than bean_max_area")

    # Validate measurement parameters
    if config.measurement.quarter_diameter_mm <= 0:
        raise ValueError("quarter_diameter_mm must be positive")

    if config.measurement.min_length_mm >= config.measurement.max_length_mm:
        raise ValueError("min_length_mm must be less than max_length_mm")

    # Validate optimization parameters
    if config.optimization.n_trials <= 0:
        raise ValueError("n_trials must be positive")

    if (
        config.optimization.validation_split <= 0
        or config.optimization.validation_split >= 1
    ):
        raise ValueError("validation_split must be between 0 and 1")

    logger.debug("Configuration validation passed")


def save_config(config: AppConfig, output_path: Path) -> None:
    """Save configuration to YAML file.

    Args:
        config: Configuration to save
        output_path: Path to save configuration file
    """
    # Convert dataclasses to dict
    config_dict = {
        "detection": {
            "coin_detection": {
                "dp": config.detection.coin_hough_dp,
                "min_dist": config.detection.coin_hough_min_dist,
                "param1": config.detection.coin_hough_param1,
                "param2": config.detection.coin_hough_param2,
                "min_radius": config.detection.coin_min_radius,
                "max_radius": config.detection.coin_max_radius,
            },
            "bean_detection": {
                "min_area": config.detection.bean_min_area,
                "max_area": config.detection.bean_max_area,
                "min_aspect_ratio": config.detection.bean_min_aspect_ratio,
                "max_aspect_ratio": config.detection.bean_max_aspect_ratio,
                "min_solidity": config.detection.bean_min_solidity,
            },
        },
        "segmentation": asdict(config.segmentation),
        "measurement": asdict(config.measurement),
        "optimization": asdict(config.optimization),
        "output": asdict(config.output),
    }

    # Save to file
    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
