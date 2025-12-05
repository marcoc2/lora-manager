"""
Caption Controller - Manages caption generation workflow
"""
from PyQt6.QtCore import QObject, pyqtSignal
from pathlib import Path
from typing import Dict, List, Optional
from controllers.caption_worker import CaptionWorker


class CaptionController(QObject):
    """
    Controller for managing caption generation process.
    Orchestrates worker lifecycle, validates config, and routes signals.
    """

    # Signals emitted to view
    caption_generated = pyqtSignal(str, str, str)  # image_path, caption, thumbnail_path
    progress_updated = pyqtSignal(int, int, str)   # current, total, status_msg
    processing_complete = pyqtSignal(int, int)     # processed_count, failed_count
    error_occurred = pyqtSignal(str, str)          # image_path, error_msg

    def __init__(self, parent=None):
        super().__init__(parent)
        self.worker: Optional[CaptionWorker] = None
        self.current_config: Dict = {}

    def start_caption_generation(self, config: Dict, dataset_path: Path):
        """
        Main entry point - starts background caption generation.

        Args:
            config: Configuration dictionary from CaptionConfigDialog
            dataset_path: Path to dataset folder
        """
        # 1. Validate config
        errors = self.validate_config(config)
        if errors:
            self.error_occurred.emit("", "Erro de configuração:\n" + "\n".join(errors))
            return

        try:
            print(f"[CONTROLLER] Starting caption generation with config: {config}")
            print(f"[CONTROLLER] Dataset path: {dataset_path}")

            # 2. Get appropriate generator
            generator = self.get_generator(config['method'], config)
            print(f"[CONTROLLER] Generator created: {type(generator).__name__}")

            # 3. Determine paths
            images_dir, captions_dir = self._get_paths(dataset_path)
            print(f"[CONTROLLER] Images dir: {images_dir}")
            print(f"[CONTROLLER] Captions dir: {captions_dir}")

            # 4. Check if images directory exists and has images
            if not images_dir.exists():
                print(f"[CONTROLLER] ERROR: Images directory does not exist: {images_dir}")
                self.error_occurred.emit("", f"Diretório de imagens não encontrado: {images_dir}")
                return

            # 5. Create worker
            print(f"[CONTROLLER] Creating worker...")
            self.worker = CaptionWorker(generator, images_dir, captions_dir, config)

            # 6. Connect signals
            print(f"[CONTROLLER] Connecting worker signals...")
            self.worker.caption_ready.connect(self.caption_generated)
            self.worker.progress.connect(self.progress_updated)
            self.worker.finished.connect(self._on_worker_finished)
            self.worker.error.connect(self.error_occurred)
            print(f"[CONTROLLER] Worker signals connected")

            # 7. Start processing
            print(f"[CONTROLLER] Starting worker thread...")
            self.worker.start()
            print(f"[CONTROLLER] Worker thread started")

        except Exception as e:
            self.error_occurred.emit("", f"Erro ao iniciar geração: {str(e)}")

    def cancel_processing(self):
        """Cancel current processing"""
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait(3000)  # Wait up to 3 seconds
            if self.worker.isRunning():
                self.worker.terminate()

    def get_generator(self, method: str, config: Dict):
        """
        Factory method to create appropriate generator based on method.

        Args:
            method: Caption generation method (Florence-2, Danbooru, Janus-7B, Qwen3-VL)
            config: Configuration dictionary

        Returns:
            Generator instance

        Raises:
            ValueError: If method is unknown
        """
        if method == "Florence-2":
            from models.caption_generator import CaptionGenerator
            return CaptionGenerator()

        elif method == "Danbooru":
            from models.danbooru_generator import DanbooruGenerator
            model_type = config.get('model_type', 'vit')
            return DanbooruGenerator(model_type=model_type)

        elif method == "Janus-7B":
            from models.janus_generator import JanusGenerator
            gen = JanusGenerator()

            # Handle custom prompt
            custom_prompt = config.get('custom_prompt', '').strip()
            if custom_prompt:
                if config.get('replace_prompt', False):
                    # Replace default prompt entirely
                    gen.prompt = custom_prompt
                else:
                    # Add as context
                    gen.add_context(custom_prompt)

            return gen

        elif method == "Qwen3-VL":
            from models.qwen_generator import QwenGenerator
            gen = QwenGenerator()

            # Handle custom prompt for Qwen
            custom_prompt = config.get('custom_prompt', '').strip()
            if custom_prompt:
                # Qwen uses user_prompt attribute
                gen.user_prompt = custom_prompt

            return gen

        else:
            raise ValueError(f"Método desconhecido: {method}")

    def validate_config(self, config: Dict) -> List[str]:
        """
        Validate configuration, return list of errors.

        Args:
            config: Configuration dictionary

        Returns:
            List of error messages (empty if valid)
        """
        errors = []

        if not config.get('method'):
            errors.append("Método de caption é obrigatório")

        method = config.get('method', '')
        if method == "Danbooru" and not config.get('model_type'):
            errors.append("Tipo de modelo Danbooru é obrigatório")

        return errors

    def _get_paths(self, dataset_path: Path):
        """
        Determine images directory and captions directory.

        Args:
            dataset_path: Base dataset path

        Returns:
            Tuple of (images_dir, captions_dir)
        """
        dataset_path = Path(dataset_path)

        # Check if cropped_images subdirectory exists
        cropped_images = dataset_path / "cropped_images"
        if cropped_images.exists() and cropped_images.is_dir():
            images_dir = cropped_images
            captions_dir = cropped_images / "captions"
        else:
            # Use root dataset path
            images_dir = dataset_path
            captions_dir = dataset_path / "captions"

        return images_dir, captions_dir

    def _on_worker_finished(self, processed: int, failed: int):
        """
        Handle worker finished signal.
        Clean up worker and emit processing_complete.
        """
        # Emit completion signal
        self.processing_complete.emit(processed, failed)

        # Clean up worker
        if self.worker:
            self.worker.deleteLater()
            self.worker = None
