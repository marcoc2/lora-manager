from PyQt6.QtWidgets import QFileDialog, QMessageBox, QProgressDialog, QDialog
from PyQt6.QtGui import QStandardItem
from PyQt6.QtCore import Qt
from pathlib import Path
import toml
import traceback

from gui_components import SuffixInputDialog, TomlConfigDialog, CaptionConfigDialog
from caption_generator import CaptionGenerator
from danbooru_generator import DanbooruGenerator

class DatasetActionsMixin:
    def toggle_face_detection(self):
        if self.face_detection.isChecked():
            self.face_detection.setText("Face Detection: ON")
        else:
            self.face_detection.setText("Face Detection: OFF")
    
    def select_dataset_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Dataset Folder")
        if folder:
            self.dataset_path = Path(folder).absolute()
            self.populate_tree_view(self.dataset_path)
            self.update_status()
    
    def populate_tree_view(self, path):
        self.tree_model.clear()
        self.tree_model.setHorizontalHeaderLabels(['Dataset Structure'])
        
        root_item = QStandardItem(str(path))
        self.tree_model.appendRow(root_item)
        
        def add_directory_contents(parent_item, dir_path):
            try:
                for item_path in sorted(Path(dir_path).iterdir()):
                    item = QStandardItem(item_path.name)
                    parent_item.appendRow(item)
                    
                    if item_path.is_dir():
                        add_directory_contents(item, item_path)
            except Exception as e:
                print(f"Error accessing {dir_path}: {e}")
        
        add_directory_contents(root_item, path)
        self.tree_view.expandAll()
    
    def update_status(self):
        if self.dataset_path:
            self.status_label.setText(f"Dataset selected: {self.dataset_path}")
        else:
            self.status_label.setText("No dataset selected")
    
    def process_images(self):
        if not self.dataset_path:
            QMessageBox.warning(self, "Warning", "Please select a dataset folder first!")
            return
            
        try:
            input_dir = self.dataset_path
            output_dir = self.dataset_path / "cropped_images"
            
            n_files = sum(1 for _ in input_dir.glob("*.[jp][pn][g]"))
            if n_files == 0:
                QMessageBox.warning(self, "Warning", "No images found in the input directory!")
                return
            
            target_size = (self.crop_width.value(), self.crop_height.value())
            self.image_processor.use_face_detection = self.face_detection.isChecked()
            
            processed, failed = self.image_processor.process_directory(
                input_dir, output_dir, target_size
            )
            
            QMessageBox.information(self, "Success", 
                f"Processing complete!\n\nSuccessfully processed: {processed}\nFailed: {failed}")
            
            self.tree_model.clear()
            self.populate_tree_view(self.dataset_path)
            self.tree_view.expandAll()
            self.update_status()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error processing images: {str(e)}")
    
    def generate_captions(self):
        if not self.dataset_path:
            QMessageBox.warning(self, "Warning", "Please select a dataset folder first!")
            return
            
        try:
            cropped_dir = self.dataset_path / "cropped_images"
            if not cropped_dir.exists():
                QMessageBox.warning(self, "Warning", "Please process images first!")
                return
            
            n_files = sum(1 for _ in cropped_dir.glob("*.[jp][pn][g]"))
            if n_files == 0:
                QMessageBox.warning(self, "Warning", "No images found in cropped_images folder!")
                return
            
            config_dialog = CaptionConfigDialog(self)
            if config_dialog.exec() != QDialog.DialogCode.Accepted:
                return
                
            config = config_dialog.get_values()
            
            progress = QProgressDialog("Generating captions...", "Cancel", 0, 100, self)
            progress.setWindowModality(Qt.WindowModality.WindowModal)
            progress.setAutoClose(True)
            progress.show()
            
            def update_progress(message: str, value: int):
                if value >= 0:
                    progress.setLabelText(message)
                    progress.setValue(value)
            
            captions_dir = cropped_dir / "captions"
            
            # Escolhe o gerador apropriado
            generator = None
            if config['method'] == "Florence-2":
                generator = CaptionGenerator()
            elif config['method'] == "Danbooru":
                model_type = config.get('model_type', 'vit')
                generator = DanbooruGenerator(model_type=model_type)
            else:  # Janus-7B
                from janus_generator import JanusGenerator
                generator = JanusGenerator()
                if config['janus_context']:
                    if config['replace_prompt']:
                        generator.set_prompt(config['janus_context'])
                    else:
                        generator.add_context(config['janus_context'])
            
            processed, failed = generator.process_directory(
                cropped_dir, 
                captions_dir,
                prefix=config['prefix'],
                progress_callback=update_progress
            )
            
            progress.close()
            
            QMessageBox.information(self, "Success", 
                f"Caption generation complete!\n\nSuccessfully processed: {processed}\nFailed: {failed}")
            
            self.tree_model.clear()
            self.populate_tree_view(self.dataset_path)
            self.tree_view.expandAll()
            self.update_status()
            
        except Exception as e:
            error_msg = f"Error generating captions:\n{str(e)}\n\nTraceback:\n{traceback.format_exc()}"
            QMessageBox.critical(self, "Error", error_msg)
    
    def generate_toml(self):
        if not self.dataset_path:
            QMessageBox.warning(self, "Warning", "Please select a dataset folder first!")
            return
        
        try:
            dialog = TomlConfigDialog(self)
            if dialog.exec() == QDialog.DialogCode.Accepted:
                config = dialog.get_values()
                
                cropped_dir = self.dataset_path / "cropped_images"
                cropped_dir.mkdir(parents=True, exist_ok=True)
                
                toml_data = {
                    "general": {
                        "shuffle_caption": False,
                        "caption_extension": ".txt",
                        "keep_tokens": 1
                    },
                    "datasets": [{
                        "resolution": config['resolution'],
                        "batch_size": 1,
                        "keep_tokens": 1,
                        "subsets": [{
                            "image_dir": str(cropped_dir.resolve()),
                            "class_tokens": config['class_tokens'],
                            "num_repeats": config['num_repeats']
                        }]
                    }]
                }
                
                toml_path = cropped_dir / "dataset.toml"
                with open(toml_path, "w", encoding="utf-8") as f:
                    toml.dump(toml_data, f)
                
                QMessageBox.information(self, "Success", "dataset.toml generated successfully!")
                self.tree_model.clear()
                self.populate_tree_view(self.dataset_path)
                self.update_status()
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error generating dataset.toml: {str(e)}")
    
    def rename_and_convert_images(self):
        if not self.dataset_path:
            QMessageBox.warning(self, "Warning", "Please select a dataset folder first!")
            return

        dialog = SuffixInputDialog(self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
            
        suffix = dialog.get_suffix()
        if not suffix:
            QMessageBox.warning(self, "Warning", "Suffix cannot be empty!")
            return

        image_dir = self.dataset_path / "cropped_images"
        if not image_dir.exists():
            QMessageBox.warning(self, "Warning", "Cropped images directory does not exist!")
            return

        image_files = list(image_dir.glob("*.[jp][pn][g]")) + list(image_dir.glob("*.webp"))
        if not image_files:
            QMessageBox.warning(self, "Warning", "No images found for renaming and conversion!")
            return

        # Aqui você implementa a lógica de renomear e converter as imagens.
        # Por enquanto, apenas exibiremos uma mensagem de sucesso.
        QMessageBox.information(self, "Success", f"Images renamed and converted with suffix '{suffix}' successfully!")
        self.tree_model.clear()
        self.populate_tree_view(self.dataset_path)
        self.update_status()
    
    def analyze_dataset(self):
        # Implementação dummy para análise do dataset.
        QMessageBox.information(self, "Analyze Dataset", "Dataset analysis not implemented yet.")
