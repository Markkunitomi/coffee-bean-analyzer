# coffee_bean_analyzer/utils/visualization.py
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
        """Create comprehensive analysis visualization."""
        # Implementation here - moved from comprehensive_analyzer
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
