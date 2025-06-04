# coffee_bean_analyzer/utils/visualization.py
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np


class VisualizationGenerator:
    """Handles all visualization generation."""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir

    def create_analysis_visualization(
        self,
        image: np.ndarray,
        results: Dict[str, Any],
        coin_detection,
        image_name: str,
        is_optimized: bool = False,
    ):
        """Create detailed analysis visualization."""
        # Implementation here - moved from detailed_analyzer
        pass

    def create_optimization_comparison(
        self,
        image_name: str,
        original_results: Dict[str, Any],
        optimized_results: Dict[str, Any],
    ):
        """Create optimization comparison visualization."""
        # Implementation here
        pass


def create_analysis_visualization(
    image: np.ndarray,
    labels: np.ndarray,
    measurements: List,
    coin_detection,
    output_path: Path,
    title: str = "Analysis Results",
) -> Path:
    """Create analysis visualization with annotations - standalone function."""
    # Create a copy of the image to annotate
    annotated = image.copy()

    # Draw coin detection if available
    if coin_detection:
        center = (int(coin_detection.center[0]), int(coin_detection.center[1]))
        radius = int(coin_detection.radius)
        cv2.circle(annotated, center, radius, (0, 255, 255), 3)  # Yellow circle
        cv2.putText(annotated, "COIN", (center[0] - 20, center[1] - radius - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # Draw bean measurements
    for i, measurement in enumerate(measurements):
        if hasattr(measurement, 'centroid'):
            center = (int(measurement.centroid[0]), int(measurement.centroid[1]))
            # Draw centroid
            cv2.circle(annotated, center, 3, (0, 255, 0), -1)  # Green dot
            # Draw bean number
            cv2.putText(annotated, str(i + 1), (center[0] + 5, center[1] - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Save the annotated image
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), annotated)

    return output_path


def create_optimization_comparison(
    image_name: str,
    original_results: Dict[str, Any],
    optimized_results: Dict[str, Any],
    output_path: Path,
) -> Path:
    """Create optimization comparison visualization - standalone function."""
    # Create a simple text comparison for now
    # In a real implementation, this would create side-by-side visualizations

    comparison_text = [
        f"Optimization Comparison for {image_name}",
        "=" * 50,
        f"Original: {len(original_results.get('measurements', []))} beans detected",
        f"Optimized: {len(optimized_results.get('measurements', []))} beans detected",
        "",
        "Parameters used:",
        f"Original: {original_results.get('parameters', 'N/A')}",
        f"Optimized: {optimized_results.get('parameters', 'N/A')}",
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path.with_suffix('.txt'), 'w') as f:
        f.write('\n'.join(comparison_text))

    # For now, just return the text file path
    # In a real implementation, this would generate an actual image comparison
    return output_path.with_suffix('.txt')
