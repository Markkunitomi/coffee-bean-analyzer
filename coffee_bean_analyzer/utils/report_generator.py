# coffee_bean_analyzer/utils/report_generator.py
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
