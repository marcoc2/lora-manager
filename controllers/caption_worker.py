"""
Caption Worker - Background thread for caption generation
"""
from PyQt6.QtCore import QThread, pyqtSignal
from pathlib import Path
from typing import Dict
import time
import gc
import torch


class CaptionWorker(QThread):
    """
    Worker thread for generating image captions in the background.
    Processes images one by one and emits signals for progress updates.
    """

    # Signals
    caption_ready = pyqtSignal(str, str, str)  # image_path, caption, thumbnail_path
    progress = pyqtSignal(int, int, str)       # current, total, message
    finished = pyqtSignal(int, int)            # processed, failed
    error = pyqtSignal(str, str)               # image_path, error_msg

    def __init__(self, generator, images_dir: Path, captions_dir: Path, config: Dict):
        """
        Initialize the caption worker.

        Args:
            generator: Caption generator instance (Florence-2, Danbooru, Janus, Qwen)
            images_dir: Directory containing images to process
            captions_dir: Directory to save generated captions
            config: Configuration dictionary with trigger_word, custom_prompt, etc.
        """
        super().__init__()
        self.generator = generator
        self.images_dir = Path(images_dir)
        self.captions_dir = Path(captions_dir)
        self.config = config
        self.is_running = True

    def run(self):
        """Main worker thread execution"""
        try:
            print(f"[WORKER] Starting caption worker thread...")
            print(f"[WORKER] Images dir: {self.images_dir}")
            print(f"[WORKER] Captions dir: {self.captions_dir}")
            print(f"[WORKER] Config: {self.config}")

            # Ensure captions directory exists
            self.captions_dir.mkdir(parents=True, exist_ok=True)

            processed = 0
            failed = 0

            # Get all image files (use lowercase only - Windows is case-insensitive)
            image_files = []
            for ext in ('*.jpg', '*.jpeg', '*.png', '*.webp'):
                image_files.extend(self.images_dir.glob(ext))

            # Remove duplicates (in case glob returns same file twice)
            image_files = list(dict.fromkeys(image_files))

            total = len(image_files)
            print(f"[WORKER] Found {total} images to process")

            if total == 0:
                print(f"[WORKER] No images found, finishing...")
                self.finished.emit(0, 0)
                return

            start_time = time.time()

            # Process each image
            for idx, img_path in enumerate(image_files):
                # Check if we should stop
                if not self.is_running:
                    break

                try:
                    print(f"[WORKER] Processing image {idx + 1}/{total}: {img_path.name}")

                    # Generate caption
                    caption = self.generator.generate_caption(str(img_path))

                    # Prepend trigger word if specified
                    trigger_word = self.config.get('trigger_word', '').strip()
                    if trigger_word:
                        caption = f"{trigger_word} {caption}"

                    # Save caption to file
                    caption_path = self.captions_dir / f"{img_path.stem}.txt"
                    caption_path.write_text(caption, encoding='utf-8')

                    # Emit success signal
                    print(f"[WORKER] Emitting caption_ready signal for: {img_path.name}")
                    self.caption_ready.emit(str(img_path), caption, str(img_path))
                    processed += 1

                    # Calculate and emit progress
                    current = idx + 1
                    elapsed = time.time() - start_time
                    speed = current / elapsed if elapsed > 0 else 0
                    remaining = total - current
                    eta = remaining / speed if speed > 0 else 0

                    # Format status message
                    msg = f"Processando {img_path.name} ({current}/{total})"
                    if speed > 0:
                        speed_per_min = speed * 60
                        eta_min = eta / 60
                        msg += f" | Velocidade: {speed_per_min:.1f} img/min | ETA: {eta_min:.1f} min"

                    self.progress.emit(current, total, msg)

                    # Clean up GPU memory after each image if using CUDA
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        gc.collect()

                except Exception as e:
                    # Emit error signal but continue processing
                    error_msg = f"{type(e).__name__}: {str(e)}"
                    print(f"[WORKER] Error processing {img_path.name}: {error_msg}")
                    print(f"[WORKER] Emitting error signal for: {img_path.name}")
                    self.error.emit(str(img_path), error_msg)
                    failed += 1

                    # Still update progress
                    current = idx + 1
                    msg = f"Erro em {img_path.name} ({current}/{total})"
                    self.progress.emit(current, total, msg)

        except Exception as e:
            # Catastrophic error - emit to final error handler
            print(f"[WORKER] CRITICAL ERROR: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            self.error.emit("", f"Erro crítico no worker: {str(e)}")
            processed = 0
            failed = total if 'total' in locals() else 0
        finally:
            # Always emit finished signal
            print(f"[WORKER] Worker finishing. Processed: {processed}, Failed: {failed}")
            self.finished.emit(processed, failed)
            print(f"[WORKER] Finished signal emitted")

    def stop(self):
        """Request worker to stop processing"""
        self.is_running = False
        self.quit()
