# Coffee Bean Analyzer

A computer vision system for automated coffee bean analysis using watershed segmentation and scale calibration. Analyzes bean dimensions, morphology, and quality metrics from images with coin-based scaling.

## Features

- **Automated Coin Detection** - Uses coins as reference for real-world measurements
- **Bean Segmentation** - Watershed segmentation to identify individual beans
- **Morphological Analysis** - Length, width, area, aspect ratio measurements
- **Parameter Optimization** - Automatically tunes detection parameters using ground truth
- **CLI Interface** - Easy-to-use command line tool
- **Batch Processing** - Process multiple images efficiently
- **Modular Architecture** - Clean, extensible codebase

## Quick Start

### Environment Setup

```bash
# Create conda environment
conda create -n coffee-bean-analyzer python=3.10
conda activate coffee-bean-analyzer

# Install package with dependencies
pip install -e .

# Install development dependencies (optional)
pip install -e ".[dev]"
```

### Basic Usage

```bash
# Analyze a single image
coffee-bean-analyzer analyze image.jpg -o results/

# View help
coffee-bean-analyzer --help
```

### Programmatic Usage

```python
from coffee_bean_analyzer.analysis.analyzer import CoffeeBeanAnalyzer

# Initialize analyzer
analyzer = CoffeeBeanAnalyzer()

# Analyze image
results = analyzer.analyze_image("path/to/image.jpg")
print(f"Detected {len(results['measurements'])} beans")
```

## Project Structure

```
coffee-bean-analyzer/
├── README.md
├── pyproject.toml              # Project configuration and dependencies
├── requirements.txt            # Empty (dependencies in pyproject.toml)
├── analyze_beans.py            # Legacy CLI script
├── analyzer.py                 # Legacy analyzer class
├── coffee_bean_analyzer/       # Main package
│   ├── __init__.py
│   ├── analysis/
│   │   └── analyzer.py         # High-level analyzer
│   ├── cli/
│   │   ├── main.py            # CLI entry point
│   │   └── commands/
│   │       └── analyze.py     # Analyze command
│   ├── config/
│   │   └── config_loader.py   # Configuration management
│   ├── core/                  # Core processing modules
│   │   ├── detector.py        # Coin detection
│   │   ├── preprocessor.py    # Image preprocessing
│   │   ├── segmentor.py       # Bean segmentation
│   │   ├── measurer.py        # Morphological measurements
│   │   └── optimizer.py       # Parameter optimization
│   ├── io/
│   │   └── data_handler.py    # File I/O operations
│   └── utils/
│       ├── file_finder.py     # File utilities
│       ├── logging_config.py  # Logging setup
│       ├── report_generator.py # Report generation
│       └── visualization.py   # Plotting utilities
└── tests/                     # Test suite
    ├── test_*.py              # Unit tests
    ├── integration/           # Integration tests
    └── data/                  # Test data
```

## Development

### Code Quality

```bash
# Linting and formatting
ruff check .
ruff format .

# Type checking
mypy .
```

### Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=coffee_bean_analyzer

# Run specific test categories
pytest -m "not slow"          # Skip slow tests
pytest -m integration         # Only integration tests
```

## Configuration

The system uses YAML configuration files with presets for different use cases:

- **default** - Balanced parameters for general use
- **aggressive** - More sensitive detection
- **conservative** - Stricter parameters to reduce false positives

## Algorithm Overview

### Processing Pipeline

1. **Preprocessing** - Gaussian blur and CLAHE enhancement
2. **Coin Detection** - Hough Circle Transform for scale reference
3. **Bean Segmentation** - Adaptive thresholding and watershed
4. **Measurement** - Contour analysis and morphological features
5. **Optimization** - Parameter tuning with ground truth (optional)

### Key Techniques

- **Coin Detection**: Multi-scale Hough Circle Transform
- **Segmentation**: Watershed algorithm for touching bean separation
- **Scaling**: Real-world measurements using coin diameter reference
- **Feature Extraction**: Ellipse fitting and contour-based measurements

## Requirements

- Python 3.8+
- OpenCV (opencv-python-headless)
- NumPy
- Pandas
- Matplotlib
- scikit-image
- Click (CLI framework)
- PyYAML (configuration)

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Run the test suite
5. Submit a pull request

---

Made with ☕ for coffee research and analysis