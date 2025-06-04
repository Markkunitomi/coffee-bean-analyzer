# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Coffee Bean Analyzer is a computer vision system for automated coffee bean analysis from images. It uses coin-based scaling for real-world measurements and provides detailed morphological analysis with parameter optimization capabilities.

## Architecture

The system follows a modular 5-stage processing pipeline:

1. **Coin Detection** (`core/detector.py`) - Hough Circle Transform for scale calibration using US quarters
2. **Image Preprocessing** (`core/preprocessor.py`) - Grayscale, blur, CLAHE processing
3. **Bean Segmentation** (`core/segmentor.py`) - Watershed algorithm with distance transform
4. **Measurement** (`core/measurer.py`) - Contour analysis, ellipse fitting, spatial sorting
5. **Parameter Optimization** (`core/optimizer.py`) - Grid search using ground truth data

### Key Components

- **`analyzer.py`** - Main analyzer class combining all components
- **`analyze_beans.py`** - Legacy script interface for simple usage
- **`cli/main.py`** - Full CLI with batch processing and configuration presets
- **`config/config_loader.py`** - YAML configuration management

## Common Commands

### Testing
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=coffee_bean_analyzer

# Skip slow tests (useful during development)
pytest -m "not slow"

# Run only integration tests
pytest -m integration
```

### Linting and Type Checking
```bash
# Lint with ruff (configured in pyproject.toml)
ruff check .

# Format code
ruff format .

# Type checking (if mypy is available)
mypy coffee_bean_analyzer/
```

### Running Analysis
```bash
# Simple analysis (legacy script)
python analyze_beans.py image.tif

# With optimization using ground truth
python analyze_beans.py image.tif --ground-truth ground_truth.csv

# CLI tool (if installed)
coffee-bean-analyzer analyze image.jpg -o results/

# Batch processing
coffee-bean-analyzer batch images/ --parallel
```

## Configuration Presets

The system includes predefined configuration presets in `config/config_loader.py`:
- **default**: Balanced parameters for general use
- **aggressive**: More sensitive detection for difficult images  
- **conservative**: Stricter parameters to reduce false positives
- **quick**: Faster processing with reduced accuracy

## Output Structure

Analysis generates timestamped directories with:
- `ANALYSIS_SUMMARY.txt` - Summary statistics
- `data/bean_measurements.csv` - Individual bean measurements
- `data/optimized_parameters.json` - Tuned parameters if optimization run
- `images/` - Annotated images and visualizations
- `reports/` - Detailed analysis reports

## Key Dependencies

- **OpenCV 4.0+** for computer vision operations
- **scikit-image** for advanced image processing
- **scikit-learn** for optimization algorithms
- **Click** for CLI framework

## Testing Notes

- Integration tests in `tests/integration/` require sample data
- Ground truth data available in `tests/data/beans_ground_truth.csv`
- Test configuration in `pytest.ini` with custom markers
- Sample images and expected outputs in `data/sample/`