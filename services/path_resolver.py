"""
Path Resolution Service - Single source of truth for dataset paths.

This module provides centralized path resolution for all dataset operations,
ensuring consistent behavior across the application.
"""
from pathlib import Path
from typing import List, Tuple, Optional


class PathResolver:
    """
    Centralized path resolution for dataset operations.

    Design principles:
    1. User's artifact selection is always honored
    2. When no selection, use deterministic order (alphabetically sorted)
    3. Captions go inside the project folder (cropped_images_*)
    4. dataset.toml goes inside the project folder
    """

    IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
    ARTIFACT_PREFIX = "cropped_images"

    def __init__(self):
        self._dataset_path: Optional[Path] = None
        self._active_artifact_path: Optional[Path] = None

    def set_dataset_path(self, path: Path):
        """Set the root dataset folder"""
        self._dataset_path = Path(path) if path else None
        self._active_artifact_path = None  # Reset artifact when dataset changes

    def set_active_artifact(self, artifact_name: str):
        """Set the currently selected artifact by name"""
        if self._dataset_path and artifact_name:
            self._active_artifact_path = self._dataset_path / artifact_name
        else:
            self._active_artifact_path = None

    def set_active_artifact_path(self, path: Path):
        """Set the currently selected artifact by full path"""
        self._active_artifact_path = Path(path) if path else None

    def clear_active_artifact(self):
        """Clear the active artifact selection"""
        self._active_artifact_path = None

    @property
    def dataset_path(self) -> Optional[Path]:
        """The root dataset folder"""
        return self._dataset_path

    @property
    def active_artifact_path(self) -> Optional[Path]:
        """The currently selected artifact folder"""
        return self._active_artifact_path

    @property
    def effective_path(self) -> Optional[Path]:
        """
        Returns the path that should be used for operations.
        Priority: active_artifact_path > dataset_path
        """
        return self._active_artifact_path or self._dataset_path

    def has_images(self, path: Path) -> bool:
        """Check if directory contains image files"""
        if not path or not path.exists():
            return False
        for ext in self.IMAGE_EXTENSIONS:
            # Check both lowercase and as-is (Windows is case-insensitive)
            if list(path.glob(f"*{ext}")) or list(path.glob(f"*{ext.upper()}")):
                return True
        return False

    def count_images(self, path: Path) -> int:
        """Count image files in directory"""
        if not path or not path.exists():
            return 0
        count = 0
        seen = set()
        for ext in self.IMAGE_EXTENSIONS:
            for f in path.glob(f"*{ext}"):
                if f.name.lower() not in seen:
                    seen.add(f.name.lower())
                    count += 1
        return count

    def find_artifact_folders(self) -> List[str]:
        """
        Find all cropped_images* folders in dataset, sorted alphabetically.
        Returns list of folder names (not full paths).
        """
        if not self._dataset_path or not self._dataset_path.exists():
            return []

        artifacts = []
        for item in self._dataset_path.iterdir():
            if item.is_dir() and item.name.startswith(self.ARTIFACT_PREFIX):
                artifacts.append(item.name)

        # Sort alphabetically for deterministic order
        return sorted(artifacts)

    def get_images_directory(self) -> Optional[Path]:
        """
        Get the directory containing images to process.

        Resolution order:
        1. If active_artifact_path is set and has images -> use it
        2. If effective_path has images directly -> use it
        3. Find first artifact folder with images (sorted alphabetically)
        4. Return effective_path as fallback
        """
        effective = self.effective_path
        if not effective:
            return None

        # Case 1: Active artifact selected and has images
        if self._active_artifact_path and self.has_images(self._active_artifact_path):
            return self._active_artifact_path

        # Case 2: Effective path has images directly
        if self.has_images(effective):
            return effective

        # Case 3: Find first artifact folder with images (deterministic order)
        for artifact_name in self.find_artifact_folders():
            artifact_path = self._dataset_path / artifact_name
            if self.has_images(artifact_path):
                return artifact_path

        # Case 4: Fallback to effective path
        return effective

    def get_captions_directory(self) -> Optional[Path]:
        """
        Get the directory where captions should be saved.
        Captions MUST go inside the images directory (as 'captions' subfolder).
        """
        images_dir = self.get_images_directory()
        if not images_dir:
            return None
        return images_dir / "captions"

    def get_toml_directory(self) -> Optional[Path]:
        """
        Get the directory where dataset.toml should be saved.
        TOML files MUST go inside the project/artifact folder.
        """
        return self.get_images_directory()

    def resolve_for_operation(self) -> Tuple[Optional[Path], Optional[Path], Optional[Path]]:
        """
        Resolve all paths needed for a complete operation.

        Returns:
            Tuple of (images_dir, captions_dir, toml_dir)
        """
        images_dir = self.get_images_directory()
        captions_dir = self.get_captions_directory()
        toml_dir = self.get_toml_directory()
        return images_dir, captions_dir, toml_dir

    def validate_for_captioning(self) -> Tuple[bool, str]:
        """
        Validate that paths are ready for caption generation.

        Returns:
            Tuple of (is_valid, error_message)
        """
        if not self._dataset_path:
            return False, "Nenhuma pasta de dataset selecionada"

        images_dir = self.get_images_directory()
        if not images_dir:
            return False, "Não foi possível determinar o diretório de imagens"

        if not images_dir.exists():
            return False, f"Diretório de imagens não existe: {images_dir}"

        if not self.has_images(images_dir):
            # Provide helpful message about where we looked
            artifacts = self.find_artifact_folders()
            if artifacts:
                return False, f"Nenhuma imagem encontrada em: {images_dir}\nPastas de projeto encontradas: {', '.join(artifacts)}"
            return False, f"Nenhuma imagem encontrada em: {images_dir}"

        return True, ""

    def validate_for_training(self) -> Tuple[bool, str]:
        """
        Validate that paths are ready for training.

        Returns:
            Tuple of (is_valid, error_message)
        """
        is_valid, error = self.validate_for_captioning()
        if not is_valid:
            return is_valid, error

        toml_dir = self.get_toml_directory()
        toml_path = toml_dir / "dataset.toml"

        if not toml_path.exists():
            return False, f"dataset.toml não encontrado em: {toml_dir}\nGere o arquivo TOML antes de treinar."

        return True, ""

    def get_status_info(self) -> dict:
        """
        Get current path resolution status for debugging/display.

        Returns:
            Dictionary with path information
        """
        images_dir = self.get_images_directory()
        return {
            "dataset_path": str(self._dataset_path) if self._dataset_path else None,
            "active_artifact": str(self._active_artifact_path) if self._active_artifact_path else None,
            "effective_path": str(self.effective_path) if self.effective_path else None,
            "images_directory": str(images_dir) if images_dir else None,
            "captions_directory": str(self.get_captions_directory()) if images_dir else None,
            "image_count": self.count_images(images_dir) if images_dir else 0,
            "artifacts_found": self.find_artifact_folders()
        }
