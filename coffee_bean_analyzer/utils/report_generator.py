# coffee_bean_analyzer/utils/report_generator.py
from pathlib import Path
from typing import Dict, Any, Optional
import datetime

class ReportGenerator:
    """Handles report generation."""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir

    def generate_analysis_report(
        self,
        image_name: str,
        original_results: Dict[str, Any],
        optimized_results: Optional[Dict[str, Any]],
        coin_detection,
    ):
        """Generate detailed analysis report."""
        # Implementation here
        pass

    def generate_summary_report(
        self,
        analysis_results: Dict[str, Any],
        optimization_results: Dict[str, Any],
        output_dir: Path,
    ):
        """Generate summary report for all analyses."""
        # Implementation here
        pass


def generate_analysis_report(
    image_name: str,
    original_results: Dict[str, Any],
    optimized_results: Optional[Dict[str, Any]],
    coin_detection,
    ground_truth=None,
    output_path: Optional[Path] = None,
) -> Path:
    """Generate detailed analysis report - standalone function."""
    if output_path is None:
        output_path = Path(f"{image_name}_analysis_report.txt")
    
    # Create basic report content
    report_lines = [
        f"Coffee Bean Analysis Report",
        f"=" * 50,
        f"Image: {image_name}",
        f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"",
        f"Coin Detection:",
        f"  Detected: {'Yes' if coin_detection else 'No'}",
    ]
    
    if coin_detection:
        report_lines.extend([
            f"  Center: ({coin_detection.center[0]:.1f}, {coin_detection.center[1]:.1f})",
            f"  Radius: {coin_detection.radius:.1f}px",
            f"  Pixels per mm: {coin_detection.pixels_per_mm:.2f}" if hasattr(coin_detection, 'pixels_per_mm') and coin_detection.pixels_per_mm else "  Pixels per mm: Not available",
        ])
    
    # Add original results
    if original_results and 'measurements' in original_results:
        measurements = original_results['measurements']
        report_lines.extend([
            f"",
            f"Original Analysis:",
            f"  Bean count: {len(measurements)}",
        ])
    
    # Add optimized results if available
    if optimized_results and 'measurements' in optimized_results:
        opt_measurements = optimized_results['measurements']
        report_lines.extend([
            f"",
            f"Optimized Analysis:",
            f"  Bean count: {len(opt_measurements)}",
        ])
    
    # Write report
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write('\n'.join(report_lines))
    
    return output_path
