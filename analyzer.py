#!/usr/bin/env python3
"""Coffee Bean Analyzer
Handles different attribute names safely for measurement objects.
"""

import datetime
import glob
import json
import os
from pathlib import Path

import cv2
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Import the new modular system
from coffee_bean_analyzer.core.detector import CoinDetector
from coffee_bean_analyzer.core.measurer import create_measurer
from coffee_bean_analyzer.core.optimizer import ParameterGrid, create_optimizer
from coffee_bean_analyzer.core.preprocessor import create_preprocessor
from coffee_bean_analyzer.core.segmentor import create_segmentor


class CoffeeBeanAnalyzer:
    """Full-featured coffee bean analyzer with safe attribute access."""

    def __init__(self, output_base_dir=None):
        """Initialize analyzer with output directory structure."""
        # Create timestamped output directory
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        if output_base_dir is None:
            self.output_dir = Path(f"coffee_analysis_{timestamp}")
        else:
            self.output_dir = Path(output_base_dir)

        # Create directory structure
        self.data_dir = self.output_dir / "data"
        self.images_dir = self.output_dir / "images"
        self.reports_dir = self.output_dir / "reports"

        for dir_path in [
            self.output_dir,
            self.data_dir,
            self.images_dir,
            self.reports_dir,
        ]:
            dir_path.mkdir(exist_ok=True, parents=True)

        # Initialize components
        self.coin_detector = CoinDetector(
            {
                "dp": 1,
                "min_dist": 100,
                "param1": 50,
                "param2": 30,
                "min_radius": 50,
                "max_radius": 150,
                "gaussian_kernel": 15,
            }
        )

        # Will be set based on analysis type
        self.preprocessor = None
        self.segmentor = None
        self.measurer = None
        self.optimizer = None

        # Analysis results storage
        self.analysis_results = {}
        self.optimization_results = {}

        print(f"📁 Output directory created: {self.output_dir}")

    def find_image_files(self, search_dirs=None):
        """Find coffee bean images in specified or common locations."""
        if search_dirs is None:
            search_dirs = ["tests/data", "data", "."]

        image_extensions = ["*.tif", "*.TIF", "*.jpg", "*.JPG", "*.png", "*.PNG"]
        found_images = []

        for search_dir in search_dirs:
            for ext in image_extensions:
                pattern = os.path.join(search_dir, ext)
                found_images.extend(glob.glob(pattern))

        return sorted(set(found_images))

    def _safe_get_position(self, measurement):
        """Safely extract position information from measurement object."""
        # Try different possible attribute names for position
        position_attrs = ["centroid", "center", "position"]

        for attr in position_attrs:
            if hasattr(measurement, attr):
                pos = getattr(measurement, attr)
                if pos is not None:
                    if hasattr(pos, "__len__") and len(pos) >= 2:
                        return float(pos[0]), float(pos[1])

        # Try getting from to_dict if available
        if hasattr(measurement, "to_dict"):
            try:
                data = measurement.to_dict()
                if "centroid_x" in data and "centroid_y" in data:
                    return float(data["centroid_x"]), float(data["centroid_y"])
                if "center_x" in data and "center_y" in data:
                    return float(data["center_x"]), float(data["center_y"])
                if "x" in data and "y" in data:
                    return float(data["x"]), float(data["y"])
            except:
                pass

        # Fallback to image center
        return 100.0, 100.0

    def _safe_get_bounding_box(self, measurement):
        """Safely extract bounding box information from measurement object."""
        # Try different possible bounding box attributes
        if (
            hasattr(measurement, "bounding_box")
            and measurement.bounding_box is not None
        ):
            bbox = measurement.bounding_box
            if (
                hasattr(bbox, "x")
                and hasattr(bbox, "y")
                and hasattr(bbox, "width")
                and hasattr(bbox, "height")
            ):
                return int(bbox.x), int(bbox.y), int(bbox.width), int(bbox.height)

        if hasattr(measurement, "bbox") and measurement.bbox is not None:
            bbox = measurement.bbox
            if hasattr(bbox, "__len__") and len(bbox) >= 4:
                return int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])

        # Try getting from to_dict
        if hasattr(measurement, "to_dict"):
            try:
                data = measurement.to_dict()
                if all(
                    key in data
                    for key in ["bbox_x", "bbox_y", "bbox_width", "bbox_height"]
                ):
                    return (
                        int(data["bbox_x"]),
                        int(data["bbox_y"]),
                        int(data["bbox_width"]),
                        int(data["bbox_height"]),
                    )
            except:
                pass

        return None

    def _safe_get_attribute(self, measurement, attr_name, default=None):
        """Safely get an attribute from measurement object."""
        # Try from to_dict first (more reliable for test mocks)
        if hasattr(measurement, "to_dict"):
            try:
                data = measurement.to_dict()
                if attr_name in data:
                    value = data[attr_name]
                    if value is not None:
                        return value
                    # Explicitly found None in to_dict, use default
                    return default
            except:
                pass

        # Direct attribute access as fallback
        if hasattr(measurement, attr_name):
            try:
                value = getattr(measurement, attr_name)
                if value is not None:
                    return value
            except:
                pass

        return default

    def create_original_analysis_visualization(
        self,
        image,
        labels,
        measurements,
        coin_detection,
        image_name,
        params,
        is_optimized=False,
    ):
        """Create visualization with safe attribute access."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(
            f"Coffee Bean Analysis: {image_name}"
            + (" (Optimized Parameters)" if is_optimized else " (Original Parameters)"),
            fontsize=16,
            fontweight="bold",
        )

        # Original image
        axes[0, 0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title("Original Image")
        axes[0, 0].axis("off")

        # Add coin annotation if detected
        if coin_detection:
            circle = patches.Circle(
                (coin_detection.center[0], coin_detection.center[1]),
                coin_detection.radius,
                linewidth=2,
                edgecolor="red",
                facecolor="none",
                linestyle="--",
            )
            axes[0, 0].add_patch(circle)
            axes[0, 0].text(
                coin_detection.center[0],
                coin_detection.center[1] - coin_detection.radius - 10,
                f"Coin: {coin_detection.radius:.1f}px",
                color="red",
                fontweight="bold",
                ha="center",
            )

        # Binary segmentation
        binary = (labels > 0).astype(np.uint8) * 255
        axes[0, 1].imshow(binary, cmap="gray")
        axes[0, 1].set_title("Binary Segmentation")
        axes[0, 1].axis("off")

        # Labeled segmentation
        axes[0, 2].imshow(labels, cmap="tab20")
        max_label = np.max(labels) if labels.size > 0 else 0
        axes[0, 2].set_title(f"Labeled Regions ({max_label} segments)")
        axes[0, 2].axis("off")

        # Annotated results - with safe attribute access
        annotated = image.copy()
        unit = "pixels"  # Default unit

        if measurements:
            # Safely get unit from first measurement
            first_measurement = measurements[0]
            if hasattr(first_measurement, "unit"):
                unit = first_measurement.unit
            elif hasattr(first_measurement, "to_dict"):
                try:
                    measurement_dict = first_measurement.to_dict()
                    unit = measurement_dict.get("unit", "pixels")
                except:
                    unit = "pixels"

        # Draw annotations for each measurement
        for i, measurement in enumerate(measurements):
            try:
                # Safely extract position information
                center_x, center_y = self._safe_get_position(measurement)

                # Draw centroid
                cv2.circle(
                    annotated, (int(center_x), int(center_y)), 5, (255, 0, 0), -1
                )

                # Try to get bounding box for rectangle
                bbox_coords = self._safe_get_bounding_box(measurement)

                if bbox_coords:
                    x, y, w, h = bbox_coords
                    cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    label_x, label_y = x, y - 5
                else:
                    # Fallback: draw circle around centroid
                    area = self._safe_get_attribute(measurement, "area", 100)
                    radius = max(5, int(np.sqrt(area / np.pi)))
                    cv2.circle(
                        annotated,
                        (int(center_x), int(center_y)),
                        radius,
                        (0, 255, 0),
                        2,
                    )
                    label_x, label_y = int(center_x - 10), int(center_y - radius - 5)

                # Add label with white outline for visibility
                label = f"#{i + 1}"
                cv2.putText(
                    annotated,
                    label,
                    (label_x, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    2,
                )
                cv2.putText(
                    annotated,
                    label,
                    (label_x, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 0),
                    1,
                )

            except Exception as e:
                print(f"Warning: Could not annotate measurement {i}: {e}")
                continue

        axes[1, 0].imshow(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
        axes[1, 0].set_title(f"Annotated Results ({len(measurements)} beans)")
        axes[1, 0].axis("off")

        # Measurement analysis
        if measurements:
            try:
                lengths = [self._safe_get_attribute(m, "length") for m in measurements]
                widths = [self._safe_get_attribute(m, "width") for m in measurements]

                # Filter out None values
                lengths = [l for l in lengths if l is not None]
                widths = [w for w in widths if w is not None]

                if lengths and widths:
                    axes[1, 1].hist(
                        lengths,
                        bins=15,
                        alpha=0.7,
                        label=f"Length ({unit})",
                        color="blue",
                    )
                    axes[1, 1].hist(
                        widths,
                        bins=15,
                        alpha=0.7,
                        label=f"Width ({unit})",
                        color="orange",
                    )
                    axes[1, 1].set_title("Size Distribution")
                    axes[1, 1].set_xlabel(f"Size ({unit})")
                    axes[1, 1].set_ylabel("Count")
                    axes[1, 1].legend()
                    axes[1, 1].grid(True, alpha=0.3)

                    # Scatter plot
                    axes[1, 2].scatter(lengths, widths, alpha=0.6, s=50)
                    axes[1, 2].set_xlabel(f"Length ({unit})")
                    axes[1, 2].set_ylabel(f"Width ({unit})")
                    axes[1, 2].set_title("Length vs Width")
                    axes[1, 2].grid(True, alpha=0.3)

                    # Add correlation coefficient
                    if len(lengths) > 1:
                        corr = np.corrcoef(lengths, widths)[0, 1]
                        axes[1, 2].text(
                            0.05,
                            0.95,
                            f"Correlation: {corr:.3f}",
                            transform=axes[1, 2].transAxes,
                            fontweight="bold",
                            bbox={
                                "boxstyle": "round,pad=0.3",
                                "facecolor": "white",
                                "alpha": 0.8,
                            },
                        )
                else:
                    axes[1, 1].text(
                        0.5,
                        0.5,
                        "No valid measurements for histogram",
                        ha="center",
                        va="center",
                        transform=axes[1, 1].transAxes,
                    )
                    axes[1, 2].text(
                        0.5,
                        0.5,
                        "No valid measurements for scatter plot",
                        ha="center",
                        va="center",
                        transform=axes[1, 2].transAxes,
                    )

            except Exception as e:
                print(f"Warning: Could not create measurement plots: {e}")
                axes[1, 1].text(
                    0.5,
                    0.5,
                    "Error creating plots",
                    ha="center",
                    va="center",
                    transform=axes[1, 1].transAxes,
                )
                axes[1, 2].text(
                    0.5,
                    0.5,
                    "Error creating plots",
                    ha="center",
                    va="center",
                    transform=axes[1, 2].transAxes,
                )
        else:
            axes[1, 1].text(
                0.5,
                0.5,
                "No measurements available",
                ha="center",
                va="center",
                transform=axes[1, 1].transAxes,
            )
            axes[1, 2].text(
                0.5,
                0.5,
                "No measurements available",
                ha="center",
                va="center",
                transform=axes[1, 2].transAxes,
            )

        plt.tight_layout()

        # Save the detailed visualization
        suffix = "_optimized" if is_optimized else "_original"
        filename = f"{image_name}{suffix}_analysis.png"
        plt.savefig(self.images_dir / filename, dpi=300, bbox_inches="tight")
        plt.close()

        # Save individual component images
        self._save_individual_component_images(
            image, binary, labels, annotated, image_name, is_optimized
        )

        return filename

    def _save_individual_component_images(
        self, image, binary, labels, annotated, image_name, is_optimized
    ):
        """Save individual component images like your original script."""
        suffix = "_optimized" if is_optimized else "_original"

        # Save annotated image
        cv2.imwrite(
            str(self.images_dir / f"{image_name}{suffix}_annotated.png"), annotated
        )

        # Save binary segmentation
        cv2.imwrite(
            str(self.images_dir / f"{image_name}{suffix}_binary_segmentation.png"),
            binary,
        )

        # Save labeled segmentation (convert to color)
        max_label = np.max(labels)
        if max_label > 0:
            normalized_labels = (labels * 255 // max_label).astype(np.uint8)
        else:
            normalized_labels = labels.astype(np.uint8)
        labeled_color = cv2.applyColorMap(normalized_labels, cv2.COLORMAP_VIRIDIS)
        cv2.imwrite(
            str(self.images_dir / f"{image_name}{suffix}_labeled_segmentation.png"),
            labeled_color,
        )

    def create_optimization_comparison(
        self, image_name, original_results, optimized_results
    ):
        """Create optimization comparison visualization with safe attribute access."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(
            f"Optimization Comparison: {image_name}", fontsize=16, fontweight="bold"
        )

        original_measurements = original_results["measurements"]
        optimized_measurements = optimized_results["measurements"]

        # Bean count comparison
        counts = [len(original_measurements), len(optimized_measurements)]
        methods = ["Original", "Optimized"]
        colors = ["lightblue", "lightgreen"]

        bars = axes[0, 0].bar(methods, counts, color=colors)
        axes[0, 0].set_title("Bean Count Comparison")
        axes[0, 0].set_ylabel("Number of Beans")

        # Add value labels on bars
        for bar, count in zip(bars, counts):
            axes[0, 0].text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.1,
                str(count),
                ha="center",
                va="bottom",
                fontweight="bold",
            )

        # Size distribution comparison
        if original_measurements and optimized_measurements:
            # Get unit safely
            unit = self._safe_get_attribute(original_measurements[0], "unit", "pixels")

            # Safely extract lengths
            orig_lengths = [
                self._safe_get_attribute(m, "length") for m in original_measurements
            ]
            opt_lengths = [
                self._safe_get_attribute(m, "length") for m in optimized_measurements
            ]

            # Filter out None values
            orig_lengths = [l for l in orig_lengths if l is not None]
            opt_lengths = [l for l in opt_lengths if l is not None]

            if orig_lengths and opt_lengths:
                axes[0, 1].hist(
                    orig_lengths, bins=15, alpha=0.6, label="Original", color="blue"
                )
                axes[0, 1].hist(
                    opt_lengths, bins=15, alpha=0.6, label="Optimized", color="green"
                )
                axes[0, 1].set_title("Length Distribution Comparison")
                axes[0, 1].set_xlabel(f"Length ({unit})")
                axes[0, 1].set_ylabel("Count")
                axes[0, 1].legend()
                axes[0, 1].grid(True, alpha=0.3)

                # Safely extract widths
                orig_widths = [
                    self._safe_get_attribute(m, "width") for m in original_measurements
                ]
                opt_widths = [
                    self._safe_get_attribute(m, "width") for m in optimized_measurements
                ]

                # Filter out None values
                orig_widths = [w for w in orig_widths if w is not None]
                opt_widths = [w for w in opt_widths if w is not None]

                # Summary statistics comparison
                orig_stats = {
                    "mean_length": np.mean(orig_lengths) if orig_lengths else 0,
                    "std_length": np.std(orig_lengths) if orig_lengths else 0,
                    "mean_width": np.mean(orig_widths) if orig_widths else 0,
                    "std_width": np.std(orig_widths) if orig_widths else 0,
                }

                opt_stats = {
                    "mean_length": np.mean(opt_lengths) if opt_lengths else 0,
                    "std_length": np.std(opt_lengths) if opt_lengths else 0,
                    "mean_width": np.mean(opt_widths) if opt_widths else 0,
                    "std_width": np.std(opt_widths) if opt_widths else 0,
                }

                # Statistics table
                stats_data = [
                    [
                        "Mean Length",
                        f"{orig_stats['mean_length']:.2f}",
                        f"{opt_stats['mean_length']:.2f}",
                    ],
                    [
                        "Std Length",
                        f"{orig_stats['std_length']:.2f}",
                        f"{opt_stats['std_length']:.2f}",
                    ],
                    [
                        "Mean Width",
                        f"{orig_stats['mean_width']:.2f}",
                        f"{opt_stats['mean_width']:.2f}",
                    ],
                    [
                        "Std Width",
                        f"{orig_stats['std_width']:.2f}",
                        f"{opt_stats['std_width']:.2f}",
                    ],
                    [
                        "Bean Count",
                        str(len(original_measurements)),
                        str(len(optimized_measurements)),
                    ],
                ]

                axes[1, 0].axis("tight")
                axes[1, 0].axis("off")
                table = axes[1, 0].table(
                    cellText=stats_data,
                    colLabels=["Metric", "Original", "Optimized"],
                    cellLoc="center",
                    loc="center",
                )
                table.auto_set_font_size(False)
                table.set_fontsize(10)
                table.scale(1.2, 1.5)
                axes[1, 0].set_title("Summary Statistics")

                # Improvement metrics
                count_improvement = len(optimized_measurements) - len(
                    original_measurements
                )
                length_diff = opt_stats["mean_length"] - orig_stats["mean_length"]
                width_diff = opt_stats["mean_width"] - orig_stats["mean_width"]

                axes[1, 1].text(
                    0.5,
                    0.8,
                    "Optimization Impact",
                    ha="center",
                    va="center",
                    transform=axes[1, 1].transAxes,
                    fontsize=14,
                    fontweight="bold",
                )

                axes[1, 1].text(
                    0.5,
                    0.6,
                    f"Bean count change: {count_improvement:+d}",
                    ha="center",
                    va="center",
                    transform=axes[1, 1].transAxes,
                    fontsize=12,
                )

                axes[1, 1].text(
                    0.5,
                    0.4,
                    f"Length change: {length_diff:+.2f} {unit}",
                    ha="center",
                    va="center",
                    transform=axes[1, 1].transAxes,
                    fontsize=12,
                )

                axes[1, 1].text(
                    0.5,
                    0.2,
                    f"Width change: {width_diff:+.2f} {unit}",
                    ha="center",
                    va="center",
                    transform=axes[1, 1].transAxes,
                    fontsize=12,
                )

                axes[1, 1].axis("off")
            else:
                axes[0, 1].text(
                    0.5,
                    0.5,
                    "No valid measurements for comparison",
                    ha="center",
                    va="center",
                    transform=axes[0, 1].transAxes,
                )
                axes[1, 0].axis("off")
                axes[1, 1].axis("off")

        plt.tight_layout()
        comparison_file = self.images_dir / "optimization_comparison.png"
        plt.savefig(comparison_file, dpi=300, bbox_inches="tight")
        plt.close()

        return comparison_file

    def analyze_image(
        self,
        image_path,
        ground_truth_path=None,
        config_preset="default",
        run_optimization=None,
    ):
        """Run detailed analysis matching your original script's functionality."""
        image_name = Path(image_path).stem
        print(f"\n{'=' * 80}")
        print(f"🔬 DETAILED ANALYSIS: {image_name}")
        print(f"{'=' * 80}")

        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")

        print(f"📸 Image loaded: {image.shape[1]}x{image.shape[0]} pixels")

        # Load ground truth
        ground_truth = None
        if ground_truth_path and Path(ground_truth_path).exists():
            ground_truth = pd.read_csv(ground_truth_path)
            print(f"📋 Ground truth loaded: {len(ground_truth)} reference beans")

            # Save ground truth reference
            ground_truth.to_csv(
                self.data_dir / "ground_truth_reference.csv", index=False
            )

        # Decide whether to run optimization
        if run_optimization is None:
            run_optimization = ground_truth is not None

        # Initialize components
        self.preprocessor = create_preprocessor(config_preset)
        self.segmentor = create_segmentor(config_preset)
        self.measurer = create_measurer(config_preset)

        # Step 1: Coin detection
        print("\n🪙 Step 1: Coin Detection")
        coin_detections = self.coin_detector.detect(image, debug=True)
        coin_detection = (
            self.coin_detector.get_best_coin(coin_detections)
            if coin_detections
            else None
        )

        if coin_detection:
            print(
                f"✅ Coin detected: center=({coin_detection.center[0]:.1f}, {coin_detection.center[1]:.1f}), radius={coin_detection.radius:.1f}"
            )

        # Step 2: Original analysis (with default parameters)
        print(f"\n🔍 Step 2: Original Analysis (preset: {config_preset})")
        original_params = self.segmentor.get_configuration()

        segmentation_result = self.segmentor.segment(image, coin_detection, debug=True)
        measurement_result = self.measurer.measure(
            segmentation_result.labels, image, coin_detection, debug=True
        )

        original_results = {
            "measurements": measurement_result.measurements,
            "segmentation_result": segmentation_result,
            "measurement_result": measurement_result,
            "parameters": original_params,
        }

        print(
            f"✅ Original analysis: {len(measurement_result.measurements)} beans detected"
        )

        # Create original analysis visualization
        self.create_original_analysis_visualization(
            image,
            segmentation_result.labels,
            measurement_result.measurements,
            coin_detection,
            image_name,
            original_params,
            is_optimized=False,
        )

        # Step 3: Optimization (if requested)
        optimized_results = None
        if run_optimization:
            print("\n🔧 Step 3: Parameter Optimization")

            self.optimizer = create_optimizer(
                "count_focused"
            )  # Use count-focused for optimization
            param_grid = ParameterGrid.create_default_grid()

            optimization_result = self.optimizer.optimize(
                image, ground_truth, param_grid, debug=True
            )

            print("✅ Optimization complete:")
            print(f"   Best score: {optimization_result.best_score:.3f}")
            print(
                f"   Combinations tested: {optimization_result.total_combinations_tested}"
            )
            print(f"   Time: {optimization_result.optimization_time:.1f}s")

            # Save optimization results
            opt_data = {
                "best_score": optimization_result.best_score,
                "best_parameters": optimization_result.best_params,
                "total_combinations_tested": optimization_result.total_combinations_tested,
                "optimization_time": optimization_result.optimization_time,
                "optimization_metadata": optimization_result.optimization_metadata,
            }

            with open(self.data_dir / "optimized_parameters.json", "w") as f:
                json.dump(opt_data, f, indent=2, default=str)

            # Run analysis with optimized parameters
            print("\n🎯 Step 4: Optimized Analysis")

            self.segmentor.update_configuration(optimization_result.best_params)
            opt_segmentation_result = self.segmentor.segment(
                image, coin_detection, debug=True
            )
            opt_measurement_result = self.measurer.measure(
                opt_segmentation_result.labels, image, coin_detection, debug=True
            )

            optimized_results = {
                "measurements": opt_measurement_result.measurements,
                "segmentation_result": opt_segmentation_result,
                "measurement_result": opt_measurement_result,
                "parameters": optimization_result.best_params,
                "optimization_result": optimization_result,
            }

            print(
                f"✅ Optimized analysis: {len(opt_measurement_result.measurements)} beans detected"
            )

            # Create optimized analysis visualization
            self.create_original_analysis_visualization(
                image,
                opt_segmentation_result.labels,
                opt_measurement_result.measurements,
                coin_detection,
                image_name,
                optimization_result.best_params,
                is_optimized=True,
            )

            # Create optimization comparison
            self.create_optimization_comparison(
                image_name, original_results, optimized_results
            )

            self.optimization_results[image_name] = optimization_result

        # Save measurement data
        self._save_measurement_data(
            image_name, original_results, optimized_results, ground_truth
        )

        # Generate reports
        self._generate_analysis_reports(
            image_name, original_results, optimized_results, coin_detection
        )

        self.analysis_results[image_name] = {
            "original": original_results,
            "optimized": optimized_results,
            "coin_detection": coin_detection,
            "image_path": image_path,
        }

        return self.analysis_results[image_name]

    def _save_measurement_data(
        self, image_name, original_results, optimized_results, ground_truth
    ):
        """Save measurement data in CSV format with safe attribute access."""

        # Helper function to safely convert measurements to dict
        def safe_measurements_to_df(measurements, analysis_type):
            rows = []
            for m in measurements:
                if hasattr(m, "to_dict"):
                    row = m.to_dict()
                else:
                    # Manually extract attributes
                    row = {
                        "length": self._safe_get_attribute(m, "length"),
                        "width": self._safe_get_attribute(m, "width"),
                        "area": self._safe_get_attribute(m, "area"),
                        "aspect_ratio": self._safe_get_attribute(m, "aspect_ratio"),
                        "perimeter": self._safe_get_attribute(m, "perimeter"),
                        "unit": self._safe_get_attribute(m, "unit", "pixels"),
                    }
                row["analysis_type"] = analysis_type
                row["image_name"] = image_name
                rows.append(row)
            return pd.DataFrame(rows)

        # Save original measurements
        if original_results["measurements"]:
            orig_df = safe_measurements_to_df(
                original_results["measurements"], "original"
            )
            orig_df.to_csv(self.data_dir / "bean_measurements.csv", index=False)

        # Save optimized measurements and comparison
        if optimized_results and optimized_results["measurements"]:
            opt_df = safe_measurements_to_df(
                optimized_results["measurements"], "optimized"
            )

            # Combine original and optimized
            combined_df = pd.concat([orig_df, opt_df], ignore_index=True)
            combined_df.to_csv(self.data_dir / "bean_measurements.csv", index=False)

            # Create ground truth comparison if available
            if ground_truth is not None:
                comparison_data = {
                    "metric": [
                        "bean_count",
                        "mean_length",
                        "mean_width",
                        "std_length",
                        "std_width",
                    ],
                    "ground_truth": [
                        len(ground_truth),
                        ground_truth["length"].mean()
                        if "length" in ground_truth.columns
                        else "N/A",
                        ground_truth["width"].mean()
                        if "width" in ground_truth.columns
                        else "N/A",
                        ground_truth["length"].std()
                        if "length" in ground_truth.columns
                        else "N/A",
                        ground_truth["width"].std()
                        if "width" in ground_truth.columns
                        else "N/A",
                    ],
                    "original_analysis": [
                        len(original_results["measurements"]),
                        np.mean(
                            [
                                self._safe_get_attribute(m, "length")
                                for m in original_results["measurements"]
                                if self._safe_get_attribute(m, "length") is not None
                            ]
                        )
                        if original_results["measurements"]
                        else "N/A",
                        np.mean(
                            [
                                self._safe_get_attribute(m, "width")
                                for m in original_results["measurements"]
                                if self._safe_get_attribute(m, "width") is not None
                            ]
                        )
                        if original_results["measurements"]
                        else "N/A",
                        np.std(
                            [
                                self._safe_get_attribute(m, "length")
                                for m in original_results["measurements"]
                                if self._safe_get_attribute(m, "length") is not None
                            ]
                        )
                        if original_results["measurements"]
                        else "N/A",
                        np.std(
                            [
                                self._safe_get_attribute(m, "width")
                                for m in original_results["measurements"]
                                if self._safe_get_attribute(m, "width") is not None
                            ]
                        )
                        if original_results["measurements"]
                        else "N/A",
                    ],
                    "optimized_analysis": [
                        len(optimized_results["measurements"])
                        if optimized_results
                        else "N/A",
                        np.mean(
                            [
                                self._safe_get_attribute(m, "length")
                                for m in optimized_results["measurements"]
                                if self._safe_get_attribute(m, "length") is not None
                            ]
                        )
                        if optimized_results and optimized_results["measurements"]
                        else "N/A",
                        np.mean(
                            [
                                self._safe_get_attribute(m, "width")
                                for m in optimized_results["measurements"]
                                if self._safe_get_attribute(m, "width") is not None
                            ]
                        )
                        if optimized_results and optimized_results["measurements"]
                        else "N/A",
                        np.std(
                            [
                                self._safe_get_attribute(m, "length")
                                for m in optimized_results["measurements"]
                                if self._safe_get_attribute(m, "length") is not None
                            ]
                        )
                        if optimized_results and optimized_results["measurements"]
                        else "N/A",
                        np.std(
                            [
                                self._safe_get_attribute(m, "width")
                                for m in optimized_results["measurements"]
                                if self._safe_get_attribute(m, "width") is not None
                            ]
                        )
                        if optimized_results and optimized_results["measurements"]
                        else "N/A",
                    ],
                }

                comparison_df = pd.DataFrame(comparison_data)
                comparison_df.to_csv(
                    self.data_dir / "ground_truth_comparison.csv", index=False
                )

    def _generate_analysis_reports(
        self, image_name, original_results, optimized_results, coin_detection
    ):
        """Generate detailed text reports with safe attribute access."""
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Original analysis report
        with open(self.reports_dir / "analysis_report.txt", "w") as f:
            f.write("Coffee Bean Analysis Report\n")
            f.write("=" * 50 + "\n")
            f.write(f"Image: {image_name}\n")
            f.write(f"Analysis Date: {timestamp}\n")
            f.write("Analysis Type: Original Parameters\n\n")

            f.write("COIN DETECTION:\n")
            f.write("-" * 20 + "\n")
            if coin_detection:
                f.write("Coin detected: Yes\n")
                f.write(
                    f"Center: ({coin_detection.center[0]:.1f}, {coin_detection.center[1]:.1f})\n"
                )
                f.write(f"Radius: {coin_detection.radius:.1f} pixels\n")
                f.write(f"Scale: {coin_detection.pixels_per_mm:.2f} pixels/mm\n")
            else:
                f.write("Coin detected: No\n")
                f.write("Measurements in pixels\n")

            f.write("\nBEAN DETECTION:\n")
            f.write("-" * 20 + "\n")
            f.write(f"Total beans detected: {len(original_results['measurements'])}\n")

            if original_results["measurements"]:
                unit = self._safe_get_attribute(
                    original_results["measurements"][0], "unit", "pixels"
                )

                # Safely extract measurements
                lengths = [
                    self._safe_get_attribute(m, "length")
                    for m in original_results["measurements"]
                ]
                widths = [
                    self._safe_get_attribute(m, "width")
                    for m in original_results["measurements"]
                ]
                areas = [
                    self._safe_get_attribute(m, "area")
                    for m in original_results["measurements"]
                ]

                # Filter out None values
                lengths = [l for l in lengths if l is not None]
                widths = [w for w in widths if w is not None]
                areas = [a for a in areas if a is not None]

                f.write(f"Measurement unit: {unit}\n\n")

                if lengths and widths and areas:
                    f.write("SUMMARY STATISTICS:\n")
                    f.write("-" * 20 + "\n")
                    f.write(
                        f"Length - Mean: {np.mean(lengths):.2f} {unit}, Std: {np.std(lengths):.2f} {unit}\n"
                    )
                    f.write(
                        f"Width - Mean: {np.mean(widths):.2f} {unit}, Std: {np.std(widths):.2f} {unit}\n"
                    )
                    f.write(
                        f"Area - Mean: {np.mean(areas):.2f} {unit}², Std: {np.std(areas):.2f} {unit}²\n"
                    )

                f.write("\nINDIVIDUAL MEASUREMENTS:\n")
                f.write("-" * 30 + "\n")
                f.write(
                    f"{'Bean':<6} {'Length':<8} {'Width':<8} {'Area':<10} {'Aspect':<8}\n"
                )
                f.write("-" * 42 + "\n")

                for i, m in enumerate(original_results["measurements"], 1):
                    length = self._safe_get_attribute(m, "length", 0)
                    width = self._safe_get_attribute(m, "width", 0)
                    area = self._safe_get_attribute(m, "area", 0)
                    aspect_ratio = self._safe_get_attribute(m, "aspect_ratio", 0)
                    f.write(
                        f"{i:<6} {length:<8.2f} {width:<8.2f} {area:<10.1f} {aspect_ratio:<8.2f}\n"
                    )

            f.write("\nPARAMETERS USED:\n")
            f.write("-" * 20 + "\n")
            for key, value in original_results["parameters"].items():
                f.write(f"{key}: {value}\n")

        # Optimized analysis report (if available)
        if optimized_results:
            with open(self.reports_dir / "optimized_analysis_report.txt", "w") as f:
                f.write("Optimized Coffee Bean Analysis Report\n")
                f.write("=" * 50 + "\n")
                f.write(f"Image: {image_name}\n")
                f.write(f"Analysis Date: {timestamp}\n")
                f.write("Analysis Type: Optimized Parameters\n\n")

                f.write("OPTIMIZATION RESULTS:\n")
                f.write("-" * 25 + "\n")
                opt_result = optimized_results["optimization_result"]
                f.write(f"Best score: {opt_result.best_score:.3f}\n")
                f.write(
                    f"Combinations tested: {opt_result.total_combinations_tested}\n"
                )
                f.write(
                    f"Optimization time: {opt_result.optimization_time:.1f} seconds\n\n"
                )

                f.write("BEAN DETECTION (OPTIMIZED):\n")
                f.write("-" * 30 + "\n")
                f.write(
                    f"Total beans detected: {len(optimized_results['measurements'])}\n"
                )

                # Include same detailed analysis as original
                if optimized_results["measurements"]:
                    unit = self._safe_get_attribute(
                        optimized_results["measurements"][0], "unit", "pixels"
                    )

                    # Safely extract measurements
                    lengths = [
                        self._safe_get_attribute(m, "length")
                        for m in optimized_results["measurements"]
                    ]
                    widths = [
                        self._safe_get_attribute(m, "width")
                        for m in optimized_results["measurements"]
                    ]
                    areas = [
                        self._safe_get_attribute(m, "area")
                        for m in optimized_results["measurements"]
                    ]

                    # Filter out None values
                    lengths = [l for l in lengths if l is not None]
                    widths = [w for w in widths if w is not None]
                    areas = [a for a in areas if a is not None]

                    f.write(f"Measurement unit: {unit}\n\n")

                    if lengths and widths and areas:
                        f.write("SUMMARY STATISTICS:\n")
                        f.write("-" * 20 + "\n")
                        f.write(
                            f"Length - Mean: {np.mean(lengths):.2f} {unit}, Std: {np.std(lengths):.2f} {unit}\n"
                        )
                        f.write(
                            f"Width - Mean: {np.mean(widths):.2f} {unit}, Std: {np.std(widths):.2f} {unit}\n"
                        )
                        f.write(
                            f"Area - Mean: {np.mean(areas):.2f} {unit}², Std: {np.std(areas):.2f} {unit}²\n"
                        )

                    f.write("\nINDIVIDUAL MEASUREMENTS:\n")
                    f.write("-" * 30 + "\n")
                    f.write(
                        f"{'Bean':<6} {'Length':<8} {'Width':<8} {'Area':<10} {'Aspect':<8}\n"
                    )
                    f.write("-" * 42 + "\n")

                    for i, m in enumerate(optimized_results["measurements"], 1):
                        length = self._safe_get_attribute(m, "length", 0)
                        width = self._safe_get_attribute(m, "width", 0)
                        area = self._safe_get_attribute(m, "area", 0)
                        aspect_ratio = self._safe_get_attribute(m, "aspect_ratio", 0)
                        f.write(
                            f"{i:<6} {length:<8.2f} {width:<8.2f} {area:<10.1f} {aspect_ratio:<8.2f}\n"
                        )

                f.write("\nOPTIMIZED PARAMETERS:\n")
                f.write("-" * 25 + "\n")
                for key, value in optimized_results["parameters"].items():
                    f.write(f"{key}: {value}\n")

                # Comparison with original
                orig_count = len(original_results["measurements"])
                opt_count = len(optimized_results["measurements"])
                improvement = opt_count - orig_count

                f.write("\nCOMPARISON WITH ORIGINAL:\n")
                f.write("-" * 30 + "\n")
                f.write(f"Original bean count: {orig_count}\n")
                f.write(f"Optimized bean count: {opt_count}\n")
                f.write(
                    f"Improvement: {improvement:+d} beans ({improvement / orig_count * 100:.1f}%)\n"
                    if orig_count > 0
                    else ""
                )

    def generate_analysis_summary(self):
        """Generate the main ANALYSIS_SUMMARY.txt file."""
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        with open(self.output_dir / "ANALYSIS_SUMMARY.txt", "w") as f:
            f.write("COFFEE BEAN ANALYSIS SUMMARY\n")
            f.write("=" * 60 + "\n")
            f.write(f"Analysis completed: {timestamp}\n")
            f.write(f"Output directory: {self.output_dir}\n")
            f.write(f"Total images processed: {len(self.analysis_results)}\n\n")

            f.write("DIRECTORY STRUCTURE:\n")
            f.write("-" * 25 + "\n")
            f.write(f"📁 {self.output_dir.name}/\n")
            f.write("  ├── 📄 ANALYSIS_SUMMARY.txt (this file)\n")
            f.write("  ├── 📁 data/\n")
            f.write("  │   ├── 📊 bean_measurements.csv\n")
            f.write("  │   ├── 📊 ground_truth_reference.csv\n")
            f.write("  │   ├── 📊 ground_truth_comparison.csv\n")
            f.write("  │   └── ⚙️ optimized_parameters.json\n")
            f.write("  ├── 📁 images/\n")
            f.write("  │   ├── 🖼️ *_original_analysis.png\n")
            f.write("  │   ├── 🖼️ *_optimized_analysis.png\n")
            f.write("  │   ├── 🖼️ *_annotated.png\n")
            f.write("  │   ├── 🖼️ *_binary_segmentation.png\n")
            f.write("  │   ├── 🖼️ *_labeled_segmentation.png\n")
            f.write("  │   └── 📊 optimization_comparison.png\n")
            f.write("  └── 📁 reports/\n")
            f.write("      ├── 📝 analysis_report.txt\n")
            f.write("      └── 📝 optimized_analysis_report.txt\n\n")

            f.write("ANALYSIS RESULTS SUMMARY:\n")
            f.write("-" * 30 + "\n")

            total_beans_original = 0
            total_beans_optimized = 0
            images_with_optimization = 0

            for image_name, results in self.analysis_results.items():
                f.write(f"\n📸 {image_name}:\n")

                orig_count = len(results["original"]["measurements"])
                total_beans_original += orig_count
                f.write(f"  Original analysis: {orig_count} beans\n")

                if results["optimized"]:
                    opt_count = len(results["optimized"]["measurements"])
                    total_beans_optimized += opt_count
                    images_with_optimization += 1
                    improvement = opt_count - orig_count
                    f.write(
                        f"  Optimized analysis: {opt_count} beans ({improvement:+d})\n"
                    )

                    if image_name in self.optimization_results:
                        opt_result = self.optimization_results[image_name]
                        f.write(f"  Optimization score: {opt_result.best_score:.3f}\n")
                        f.write(
                            f"  Combinations tested: {opt_result.total_combinations_tested}\n"
                        )

                coin = results["coin_detection"]
                if coin:
                    f.write(
                        f"  Coin detected: ✅ (scale: {coin.pixels_per_mm:.2f} px/mm)\n"
                    )
                else:
                    f.write("  Coin detected: ❌ (pixel measurements)\n")

            f.write("\nOVERALL STATISTICS:\n")
            f.write("-" * 25 + "\n")
            f.write(f"Total beans found (original): {total_beans_original}\n")

            if images_with_optimization > 0:
                f.write(f"Total beans found (optimized): {total_beans_optimized}\n")
                f.write(
                    f"Net improvement: {total_beans_optimized - total_beans_original:+d} beans\n"
                )
                f.write(
                    f"Images with optimization: {images_with_optimization}/{len(self.analysis_results)}\n"
                )

            f.write("\nFILES GENERATED:\n")
            f.write("-" * 20 + "\n")

            # List all generated files
            for subdir in ["data", "images", "reports"]:
                subdir_path = self.output_dir / subdir
                if subdir_path.exists():
                    files = list(subdir_path.iterdir())
                    f.write(f"\n{subdir}/ ({len(files)} files):\n")
                    for file_path in sorted(files):
                        f.write(f"  📄 {file_path.name}\n")

            f.write("\n" + "=" * 60 + "\n")
            f.write(f"Analysis complete! All results saved in {self.output_dir}\n")

    def run_full_analysis(self, search_dirs=None, config_preset="default"):
        """Run the complete analysis pipeline on all found images.
        This recreates your original script's full functionality!
        """
        print("☕ DETAILED COFFEE BEAN ANALYSIS")
        print("=" * 60)

        # Find images
        image_files = self.find_image_files(search_dirs)

        if not image_files:
            print("❌ No image files found!")
            print("💡 Place your images in: tests/data/, data/, or current directory")
            print("💡 Supported formats: .tif, .jpg, .png")
            return None

        print(f"🔍 Found {len(image_files)} image file(s):")
        for i, img_file in enumerate(image_files, 1):
            print(f"   {i}. {img_file}")

        # Look for ground truth files
        ground_truth_files = []
        for gt_pattern in ["*truth*.csv", "*beans*.csv", "*reference*.csv"]:
            ground_truth_files.extend(glob.glob(gt_pattern))
            ground_truth_files.extend(glob.glob(f"tests/data/{gt_pattern}"))
            ground_truth_files.extend(glob.glob(f"data/{gt_pattern}"))

        ground_truth_path = ground_truth_files[0] if ground_truth_files else None

        if ground_truth_path:
            print(f"📋 Found ground truth file: {ground_truth_path}")
        else:
            print(
                "📋 No ground truth file found - analysis will proceed without optimization"
            )

        # Process each image
        print("\n🚀 Starting detailed analysis...")

        for i, image_path in enumerate(image_files, 1):
            print("\n" + "🔄" * 20)
            print(f"Processing image {i}/{len(image_files)}: {Path(image_path).name}")
            print("🔄" * 20)

            try:
                self.analyze_image(
                    image_path,
                    ground_truth_path,
                    config_preset=config_preset,
                    run_optimization=(ground_truth_path is not None),
                )

                print(f"✅ Analysis complete for {Path(image_path).name}")

            except Exception as e:
                print(f"❌ Error processing {image_path}: {e}")
                import traceback

                traceback.print_exc()

        # Generate final summary
        print("\n📊 Generating analysis summary...")
        self.generate_analysis_summary()

        print("\n🎉 ANALYSIS COMPLETE!")
        print("=" * 60)
        print(f"📁 All results saved to: {self.output_dir}")
        print(f"📄 Summary report: {self.output_dir}/ANALYSIS_SUMMARY.txt")
        print(f"📊 Data files: {self.data_dir}")
        print(f"🖼️ Visualizations: {self.images_dir}")
        print(f"📝 Reports: {self.reports_dir}")

        # Show file counts
        data_files = len(list(self.data_dir.iterdir())) if self.data_dir.exists() else 0
        image_files_count = (
            len(list(self.images_dir.iterdir())) if self.images_dir.exists() else 0
        )
        report_files = (
            len(list(self.reports_dir.iterdir())) if self.reports_dir.exists() else 0
        )

        print("\n📈 Generated:")
        print(f"   {data_files} data files")
        print(f"   {image_files_count} visualization images")
        print(f"   {report_files} detailed reports")

        return self.output_dir


def main():
    """Main function - recreates your original script's detailed analysis
    with the new modular system underneath!
    """
    # You can customize these settings
    config_preset = (
        "default"  # Options: "default", "aggressive", "conservative", "quick"
    )
    search_directories = ["tests/data", "data", "."]  # Where to look for images

    # Create the detailed analyzer
    analyzer = CoffeeBeanAnalyzer()

    # Run the full analysis pipeline
    try:
        output_directory = analyzer.run_full_analysis(
            search_dirs=search_directories, config_preset=config_preset
        )

        if output_directory:
            print("\n✨ SUCCESS! Your analysis is complete.")
            print(f"📁 Check the output directory: {output_directory}")
            print(f"📄 Start with: {output_directory}/ANALYSIS_SUMMARY.txt")

    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
