# coffee_bean_analyzer/utils/file_finder.py
from pathlib import Path
from typing import List, Optional


class FileFinder:
    """Utility for finding image files."""

    @staticmethod
    def find_image_files(search_dirs: Optional[List[str]] = None) -> List[Path]:
        """Find all image files in given directories."""
        import glob

        if search_dirs is None:
            search_dirs = ["tests/data", "data", "."]

        image_extensions = ["*.tif", "*.TIF", "*.jpg", "*.JPG", "*.png", "*.PNG"]
        found_images = []

        for search_dir in search_dirs:
            for ext in image_extensions:
                pattern = str(Path(search_dir) / ext)
                found_images.extend(Path(p) for p in glob.glob(pattern))

        return sorted(list(set(found_images)))
