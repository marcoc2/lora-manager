# caption_processing_gui.py
import toml
from pathlib import Path
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
                           QPushButton, QMessageBox, QProgressDialog,
                           QDialog)
from PyQt6.QtCore import Qt

from gui_components import TomlConfigDialog, CaptionConfigDialog
from caption_generator import CaptionGenerator
from danbooru_generator import DanbooruGenerator

class CaptionProcessingPanel(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.dataset_path = None
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        layout.setSpacing(10)

        # 1. Caption Generation Group
        caption_group = self.create_caption_generation_group()
        layout.addWidget(caption_group)
        
        # 2. Dataset Configuration Group
        dataset_group = self.create_dataset_config_group()
        layout.addWidget(dataset_group)
        
        # 3. Analysis Group
        analysis_group = self.create_analysis_group()
        layout.addWidget(analysis_group)
        
        self.setLayout(layout)

    def create_caption_generation_group(self):
        group = QGroupBox("1. Caption Generation")
        group.setMinimumHeight(100)
        layout = QVBoxLayout()
        layout.setContentsMargins(10, 15, 10, 15)
        
        generate_captions_btn = QPushButton("Generate Captions")
        generate_captions_btn.clicked.connect(self.generate_captions)
        layout.addWidget(generate_captions_btn)
        
        group.setLayout(layout)
        return group

    def create_dataset_config_group(self):
        group = QGroupBox("2. Dataset Configuration")
        group.setMinimumHeight(100)
        layout = QVBoxLayout()
        layout.setContentsMargins(10, 15, 10, 15)
        
        generate_toml_btn = QPushButton("Generate dataset.toml")
        generate_toml_btn.clicked.connect(self.generate_toml)
        layout.addWidget(generate_toml_btn)
        
        group.setLayout(layout)
        return group

    def create_analysis_group(self):
        group = QGroupBox("3. Dataset Analysis")
        group.setMinimumHeight(100)
        layout = QVBoxLayout()
        layout.setContentsMargins(10, 15, 10, 15)
        
        analyze_btn = QPushButton("Analyze Dataset")
        analyze_btn.clicked.connect(self.analyze_dataset)
        layout.addWidget(analyze_btn)
        
        group.setLayout(layout)
        return group

    def on_dataset_changed(self, dataset_path):
        self.dataset_path = dataset_path

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
            
            # Choose appropriate generator
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
            
            self.main_window.refresh_ui()
            
        except Exception as e:
            import traceback
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
                self.main_window.refresh_ui()
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error generating dataset.toml: {str(e)}")

    def analyze_dataset(self):
        if not self.dataset_path:
            QMessageBox.warning(self, "Warning", "Please select a dataset folder first!")
            return
            
        try:
            stats = {
                "total_images": 0,
                "total_captions": 0,
                "missing_captions": []
            }
            
            images_dir = self.dataset_path / "cropped_images"
            if images_dir.exists():
                stats["total_images"] = len(list(images_dir.glob("*.[jp][pn][g]")))
            
            captions_dir = images_dir / "captions"
            if captions_dir.exists():
                stats["total_captions"] = len(list(captions_dir.glob("*.txt")))
                
                for img_path in images_dir.glob("*.[jp][pn][g]"):
                    caption_path = captions_dir / f"{img_path.stem}.txt"
                    if not caption_path.exists():
                        stats["missing_captions"].append(img_path.name)
            
            msg = f"""Dataset Analysis:

Total Images: {stats['total_images']}
Total Captions: {stats['total_captions']}
Missing Captions: {len(stats['missing_captions'])}"""

            if stats["missing_captions"]:
                msg += "\n\nFiles missing captions:"
                for file in stats["missing_captions"][:10]:
                    msg += f"\n- {file}"
                if len(stats["missing_captions"]) > 10:
                    msg += f"\n... and {len(stats['missing_captions']) - 10} more"
            
            QMessageBox.information(self, "Dataset Analysis", msg)
        
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error analyzing dataset: {str(e)}")
