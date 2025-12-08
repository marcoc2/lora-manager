import sys
import toml
from pathlib import Path
from PyQt6.QtWidgets import QFileDialog, QApplication
from PyQt6.QtCore import QObject

from views.main_window import DatasetManagerGUI
from views.dialogs.suffix_input_dialog import SuffixInputDialog
from views.dialogs.toml_config_dialog import TomlConfigDialog
from views.dialogs.qwen_toml_config_dialog import QwenTomlConfigDialog
from views.dialogs.caption_config_dialog import CaptionConfigDialog
from views.dialogs.caption_progress_dialog import CaptionProgressDialog
from views.dialogs.batch_caption_editor import BatchCaptionEditor
from models.image_processor import ImageProcessor
from models.caption_generator import CaptionGenerator
from models.danbooru_generator import DanbooruGenerator
from models.janus_generator import JanusGenerator
from controllers.caption_controller import CaptionController
from training_widgets import CommandOutputDialog
from services.path_resolver import PathResolver


class MainController(QObject):
    def __init__(self, view: DatasetManagerGUI):
        super().__init__()
        self.view = view
        self.dataset_path = None
        self.path_resolver = PathResolver()  # Centralized path resolution
        self.image_processor = ImageProcessor()
        self.caption_controller = CaptionController(self)

        # Share path_resolver with view
        self.view.path_resolver = self.path_resolver

        self.connect_signals()

    def connect_signals(self):
        self.view.select_dataset_folder_clicked.connect(self.select_dataset_folder)
        self.view.process_images_clicked.connect(self.process_images)
        self.view.generate_captions_clicked.connect(self.generate_captions)
        self.view.generate_toml_clicked.connect(self.generate_all_toml)
        self.view.rename_and_convert_images_clicked.connect(self.rename_and_convert_images)
        self.view.analyze_dataset_clicked.connect(self.analyze_dataset)
        self.view.artifact_selected.connect(self.load_artifact_info)

        # Connect caption panel directly (MVC pattern)
        if hasattr(self.view, 'caption_panel'):
            self.view.caption_panel.generate_clicked.connect(self.generate_captions)

    def select_dataset_folder(self):
        folder = QFileDialog.getExistingDirectory(self.view, "Select Dataset Folder")
        if folder:
            self.dataset_path = Path(folder).absolute()
            self.path_resolver.set_dataset_path(self.dataset_path)  # Update resolver
            self.view.dataset_path = self.dataset_path  # Update view
            self.view.active_artifact_path = None  # Reset artifact
            self.view.populate_image_grid(self.dataset_path)
            self.scan_artifacts()
            self.update_status()

    def scan_artifacts(self):
        """Scans for cropped_images folders and populates the combo box"""
        if not self.dataset_path:
            return

        self.view.dataset_view.artifact_combo.clear()

        # Use path_resolver to find artifacts (already sorted alphabetically)
        artifacts = self.path_resolver.find_artifact_folders()

        if artifacts:
            self.view.dataset_view.artifact_combo.addItems(artifacts)
            # Select the most likely "active" one (e.g., just "cropped_images" or the first one)
            if "cropped_images" in artifacts:
                self.view.dataset_view.artifact_combo.setCurrentText("cropped_images")
            else:
                self.view.dataset_view.artifact_combo.setCurrentIndex(0)

    def load_artifact_info(self, folder_name):
        """Loads and displays info from dataset.toml in the selected folder"""
        if not self.dataset_path:
            return

        # Update path_resolver with selected artifact
        self.path_resolver.set_active_artifact(folder_name)
        artifact_path = self.path_resolver.active_artifact_path
        self.view.active_artifact_path = artifact_path  # Update view
        toml_path = artifact_path / "dataset.toml"
        qwen_toml_path = artifact_path / "dataset_qwen.toml"
        
        # Update image grid to show images from this artifact
        self.view.populate_image_grid(artifact_path)
        
        info_text = f"Artifact: {folder_name}\n"
        
        if toml_path.exists():
            try:
                data = toml.load(toml_path)
                # Extract some key info
                if "datasets" in data and len(data["datasets"]) > 0:
                    res = data["datasets"][0].get("resolution", "Unknown")
                    batch = data["datasets"][0].get("batch_size", "Unknown")
                    info_text += f"SD-Scripts: Res={res}, Batch={batch}\n"
            except Exception as e:
                info_text += f"Error reading dataset.toml: {e}\n"
        else:
            info_text += "No dataset.toml found.\n"
            
        if qwen_toml_path.exists():
             try:
                data = toml.load(qwen_toml_path)
                if "general" in data:
                    res = data["general"].get("resolution", "Unknown")
                    info_text += f"Musubi: Res={res}\n"
             except Exception as e:
                info_text += f"Error reading dataset_qwen.toml: {e}\n"
        
        # Count images
        try:
            n_images = len(list(artifact_path.glob("*.[jp][pn][g]")))
            info_text += f"Images: {n_images}"
        except:
            pass
            
        self.view.dataset_view.toml_info.setText(info_text)

    def process_images(self, config):
        if not self.dataset_path:
            self.view.show_warning("Warning", "Please select a dataset folder first!")
            return
            
        try:
            input_dir = self.dataset_path
            
            # Smart folder naming based on resolution
            width, height = config['target_size']
            if width == height:
                folder_name = f"cropped_images_{width}"
            else:
                folder_name = f"cropped_images_{width}x{height}"
                
            output_dir = self.dataset_path / folder_name
            
            n_files = sum(1 for f in input_dir.iterdir() if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.webp', '.avif'])
            if n_files == 0:
                self.view.show_warning("Warning", "No images found in the input directory!")
                return
            
            target_size = config['target_size']
            self.image_processor.use_face_detection = config['use_face_detection']
            
            processed, failed = self.image_processor.process_directory(
                input_dir,
                output_dir,
                target_size
            )
            
            self.view.show_message("Success", 
                f"Processing complete!\n\nOutput Folder: {folder_name}\nSuccessfully processed: {processed}\nFailed: {failed}")
            
            self.view.populate_image_grid(self.dataset_path)
            self.scan_artifacts() # Refresh list
            self.view.dataset_view.artifact_combo.setCurrentText(folder_name) # Auto-select new folder
            self.update_status()
            
        except Exception as e:
            self.view.show_critical("Error", f"Error processing images: {str(e)}")

    def generate_captions(self, config: dict = None):
        """
        Generate captions using the new CaptionController.
        Can be called with config (from view signal) or without (legacy support).
        """
        # Validate using path_resolver
        is_valid, error = self.path_resolver.validate_for_captioning()
        if not is_valid:
            self.view.show_warning("Aviso", error)
            return

        # If config not provided, show config dialog (legacy support)
        if config is None:
            config_dialog = CaptionConfigDialog(self.view)
            if config_dialog.exec() != config_dialog.DialogCode.ACCEPTED:
                return
            config = config_dialog.get_values()

        try:
            # Get resolved paths from path_resolver
            images_dir, captions_dir, _ = self.path_resolver.resolve_for_operation()

            # Create and show progress dialog
            progress_dialog = CaptionProgressDialog(self.view)

            # Connect controller signals to progress dialog
            self.caption_controller.caption_generated.connect(progress_dialog.on_caption_ready)
            self.caption_controller.progress_updated.connect(progress_dialog.on_progress_updated)
            self.caption_controller.error_occurred.connect(progress_dialog.on_error)
            self.caption_controller.processing_complete.connect(progress_dialog.on_complete)

            # Connect to caption panel for real-time list updates
            if hasattr(self.view, 'caption_panel') and hasattr(self.view.caption_panel, 'add_caption_to_list'):
                self.caption_controller.caption_generated.connect(self.view.caption_panel.add_caption_to_list)

            # Connect progress dialog signals
            progress_dialog.cancel_requested.connect(self.caption_controller.cancel_processing)
            progress_dialog.batch_edit_requested.connect(self.open_batch_caption_editor)

            # Start caption generation with resolved paths
            self.caption_controller.start_caption_generation(config, images_dir, captions_dir)

            # Show progress dialog (blocks until complete or cancelled)
            progress_dialog.exec()

            # Disconnect signals to avoid duplicates on next run
            try:
                self.caption_controller.caption_generated.disconnect(progress_dialog.on_caption_ready)
                self.caption_controller.progress_updated.disconnect(progress_dialog.on_progress_updated)
                self.caption_controller.error_occurred.disconnect(progress_dialog.on_error)
                self.caption_controller.processing_complete.disconnect(progress_dialog.on_complete)
                if hasattr(self.view, 'caption_panel') and hasattr(self.view.caption_panel, 'add_caption_to_list'):
                    self.caption_controller.caption_generated.disconnect(self.view.caption_panel.add_caption_to_list)
            except (TypeError, RuntimeError):
                pass  # Signals may already be disconnected

            # Refresh UI after completion
            if hasattr(self.view, 'populate_image_grid'):
                self.view.populate_image_grid(self.dataset_path)

            # Refresh caption list in caption panel (final refresh to ensure consistency)
            if hasattr(self.view, 'caption_panel') and hasattr(self.view.caption_panel, 'refresh_captions_list'):
                self.view.caption_panel.refresh_captions_list()

            if hasattr(self, 'update_status'):
                self.update_status()

        except Exception as e:
            import traceback
            error_msg = f"Erro ao gerar captions:\n{str(e)}\n\nTraceback:\n{traceback.format_exc()}"
            self.view.show_critical("Erro", error_msg)

    def open_batch_caption_editor(self, captions_data: list):
        """Open the batch caption editor with the provided caption data"""
        try:
            editor = BatchCaptionEditor(captions_data, self.view)
            editor.exec()

            # Refresh UI after editing
            if hasattr(self.view, 'populate_image_grid'):
                self.view.populate_image_grid(self.dataset_path)

        except Exception as e:
            self.view.show_critical("Erro", f"Erro ao abrir editor de captions: {str(e)}")

    def generate_all_toml(self):
        # Validate using path_resolver
        is_valid, error = self.path_resolver.validate_for_captioning()
        if not is_valid:
            self.view.show_warning("Warning", error)
            return

        try:
            dialog = QwenTomlConfigDialog(self.view)
            if dialog.exec() == dialog.DialogCode.ACCEPTED:
                config = dialog.get_values()

                # Use path_resolver to get the correct directory
                cropped_dir = self.path_resolver.get_toml_directory()
                if not cropped_dir:
                    self.view.show_warning("Warning", "Could not determine project directory!")
                    return
                cropped_dir.mkdir(parents=True, exist_ok=True)
                
                sdscripts_toml = {
                    "general": {
                        "shuffle_caption": False,
                        "caption_extension": ".txt",
                        "keep_tokens": 1
                    },
                    "datasets": [{
                        "resolution": config['resolution'][0],
                        "batch_size": 1,
                        "keep_tokens": 1,
                        "subsets": [{
                            "image_dir": str(cropped_dir.resolve()),
                            "class_tokens": "",
                            "num_repeats": config['num_repeats']
                        }]
                    }]
                }
                
                toml_path_sdscripts = cropped_dir / "dataset.toml"
                with open(toml_path_sdscripts, "w", encoding="utf-8") as f:
                    toml.dump(sdscripts_toml, f)
                
                musubi_toml = {
                    "general": {
                        "resolution": config['resolution'],
                        "caption_extension": config['caption_extension'],
                        "batch_size": config['batch_size'],
                        "enable_bucket": config['enable_bucket'],
                        "bucket_no_upscale": config['bucket_no_upscale']
                    },
                    "datasets": [{
                        "image_directory": str(cropped_dir.resolve()),
                        "cache_directory": str((cropped_dir / "cache_imgs").resolve()),
                        "num_repeats": config['num_repeats']
                    }]
                }
                
                toml_path_musubi = cropped_dir / "dataset_qwen.toml"
                with open(toml_path_musubi, "w", encoding="utf-8") as f:
                    toml.dump(musubi_toml, f)
                
                cache_dir = cropped_dir / "cache_imgs"
                cache_dir.mkdir(parents=True, exist_ok=True)
                
                self.view.show_message("Success", 
                    "Dataset TOML files generated successfully!\n\n" +
                    "• dataset.toml - For SDXL/Flux training (sd-scripts)\n" +
                    "• dataset_qwen.toml - For Qwen-Image training (Musubi)")
                
                self.view.populate_tree_view(self.dataset_path)
                self.update_status()
                
        except Exception as e:
            self.view.show_critical("Error", f"Error generating dataset TOML files: {str(e)}")

    def rename_and_convert_images(self):
        # Validate using path_resolver
        is_valid, error = self.path_resolver.validate_for_captioning()
        if not is_valid:
            self.view.show_warning("Warning", error)
            return

        dialog = SuffixInputDialog(self.view)
        if dialog.exec() != dialog.DialogCode.ACCEPTED:
            return
        suffix = dialog.get_suffix()
        if not suffix:
            self.view.show_warning("Warning", "Suffix cannot be empty!")
            return

        # Use path_resolver to get images directory
        image_dir = self.path_resolver.get_images_directory()
        if not image_dir or not image_dir.exists():
            self.view.show_warning("Warning", "Images directory does not exist!")
            return

        image_files = list(image_dir.glob("*.[jp][pn][g]")) + list(image_dir.glob("*.webp"))
        if not image_files:
            self.view.show_warning("Warning", "No images found to rename and convert!")
            return

        converted_count = 0

        for idx, image_path in enumerate(sorted(image_files), 1):
            try:
                new_name = f"{image_path.stem}{suffix}_{str(idx).zfill(3)}.png"
                new_path = image_dir / new_name

                with Image.open(image_path) as img:
                    img = img.convert("RGB")
                    img.save(new_path, "PNG")

                if image_path.suffix.lower() != ".png":
                    image_path.unlink()

                converted_count += 1
            except Exception as e:
                self.view.show_warning("Error", f"Failed to process {image_path.name}: {e}")

        self.view.show_message("Success", f"Renamed and converted {converted_count} images successfully!")
        self.view.populate_tree_view(self.dataset_path)

    def analyze_dataset(self):
        if not self.dataset_path:
            self.view.show_warning("Warning", "Please select a dataset folder first!")
            return

        try:
            stats = {
                "total_images": 0,
                "total_captions": 0,
                "missing_captions": []
            }

            # Use path_resolver to get directories
            images_dir = self.path_resolver.get_images_directory()
            captions_dir = self.path_resolver.get_captions_directory()

            if images_dir and images_dir.exists():
                stats["total_images"] = self.path_resolver.count_images(images_dir)

            if captions_dir and captions_dir.exists():
                stats["total_captions"] = len(list(captions_dir.glob("*.txt")))

                if images_dir:
                    for ext in ['.jpg', '.jpeg', '.png', '.webp', '.bmp']:
                        for img_path in images_dir.glob(f"*{ext}"):
                            caption_path = captions_dir / f"{img_path.stem}.txt"
                            if not caption_path.exists():
                                stats["missing_captions"].append(img_path.name)

            msg = f"""Dataset Analysis:

Images Directory: {images_dir}
Total Images: {stats['total_images']}
Total Captions: {stats['total_captions']}
Missing Captions: {len(stats['missing_captions'])}"""

            if stats["missing_captions"]:
                msg += "\n\nFiles missing captions:"
                for file in stats["missing_captions"][:10]:
                    msg += f"\n- {file}"
                if len(stats["missing_captions"]) > 10:
                    msg += f"\n... and {len(stats['missing_captions']) - 10} more"

            self.view.show_message("Dataset Analysis", msg)

        except Exception as e:
            self.view.show_critical("Error", f"Error analyzing dataset: {str(e)}")

    def update_status(self):
        if self.dataset_path:
            status_text = f"Dataset: {self.dataset_path}"

            # Use path_resolver to get directories
            images_dir = self.path_resolver.get_images_directory()
            captions_dir = self.path_resolver.get_captions_directory()

            if images_dir and images_dir.exists():
                n_images = self.path_resolver.count_images(images_dir)
                status_text += f"\nProject: {images_dir.name}"
                status_text += f"\nImages: {n_images}"

            if captions_dir and captions_dir.exists():
                n_captions = len(list(captions_dir.glob("*.txt")))
                status_text += f"\nCaptions: {n_captions}"

            self.view.update_status(status_text)
        else:
            self.view.update_status("No dataset selected")
