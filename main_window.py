import toml
from pathlib import Path
from PIL import Image
from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                           QTreeView, QGroupBox, QPushButton, QMessageBox, 
                           QLabel, QSpinBox, QFileDialog, QProgressDialog)
from PyQt6.QtGui import QStandardItemModel, QStandardItem
from PyQt6.QtCore import Qt

from image_processor import ImageProcessor
from training_widgets import CommandOutputDialog
from training_tabs import TrainingTabs
from gui_components import SuffixInputDialog, TomlConfigDialog, CaptionConfigDialog
from caption_generator import CaptionGenerator
from danbooru_generator import DanbooruGenerator

class DatasetManagerGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Dataset Manager")
        self.setGeometry(100, 100, 1300, 900)
        
        self.dataset_path = None
        self.image_processor = ImageProcessor()
        
        self.init_ui()

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        layout = QHBoxLayout()
        
        # Left panel - TreeView and folder selection
        left_panel = self.create_left_panel()
        
        # Center panel - Configuration and processing
        center_panel = self.create_center_panel()
        
        # Right panel - Training tabs
        self.training_tabs = TrainingTabs(self)
        
        # Set panel proportions (25/10/65)
        layout.addWidget(left_panel, 25)
        layout.addWidget(center_panel, 10)
        layout.addWidget(self.training_tabs, 65)
        
        central_widget.setLayout(layout)

    def create_left_panel(self):
        left_panel = QWidget()
        left_layout = QVBoxLayout()
        
        # Folder selection button
        select_button = QPushButton("Select Dataset Folder")
        select_button.clicked.connect(self.select_dataset_folder)
        left_layout.addWidget(select_button)
        
        # TreeView for dataset visualization
        self.tree_view = QTreeView()
        self.tree_model = QStandardItemModel()
        self.tree_model.setHorizontalHeaderLabels(['Dataset Structure'])
        self.tree_view.setModel(self.tree_model)
        self.tree_view.setColumnWidth(0, 500)
        left_layout.addWidget(self.tree_view)
        
        left_panel.setLayout(left_layout)
        return left_panel

    def create_center_panel(self):
        center_panel = QWidget()
        center_layout = QVBoxLayout()
        center_layout.setSpacing(10)

        # 1. Image Processing Group
        image_group = self.create_image_processing_group()
        center_layout.addWidget(image_group)
        
        # 2. Caption Generation Group
        caption_group = self.create_caption_generation_group()
        center_layout.addWidget(caption_group)
        
        # 3. Dataset Configuration Group
        dataset_group = self.create_dataset_config_group()
        center_layout.addWidget(dataset_group)
        
        # 4. Utilities Group
        utils_group = self.create_utilities_group()
        center_layout.addWidget(utils_group)
        
        # Status Group
        status_group = self.create_status_group()
        center_layout.addWidget(status_group)
        
        center_panel.setLayout(center_layout)
        return center_panel

    def create_image_processing_group(self):
        group = QGroupBox("1. Image Processing")
        group.setMinimumHeight(150)
        layout = QVBoxLayout()
        
        # Size settings
        size_layout = QHBoxLayout()
        size_layout.addWidget(QLabel("Target Size:"))
        self.crop_width = QSpinBox()
        self.crop_width.setRange(64, 2048)
        self.crop_width.setValue(512)
        self.crop_height = QSpinBox()
        self.crop_height.setRange(64, 2048)
        self.crop_height.setValue(512)
        size_layout.addWidget(self.crop_width)
        size_layout.addWidget(QLabel("x"))
        size_layout.addWidget(self.crop_height)
        layout.addLayout(size_layout)
        
        # Face detection toggle
        self.face_detection = QPushButton("Face Detection: ON")
        self.face_detection.setCheckable(True)
        self.face_detection.setChecked(True)
        self.face_detection.clicked.connect(self.toggle_face_detection)
        layout.addWidget(self.face_detection)
        
        # Process button
        process_button = QPushButton("Process Images")
        process_button.clicked.connect(self.process_images)
        layout.addWidget(process_button)
        
        group.setLayout(layout)
        return group

    def create_caption_generation_group(self):
        group = QGroupBox("2. Caption Generation")
        group.setMinimumHeight(100)
        layout = QVBoxLayout()
        layout.setContentsMargins(10, 15, 10, 15)
        
        generate_captions_btn = QPushButton("Generate Captions")
        generate_captions_btn.clicked.connect(self.generate_captions)
        layout.addWidget(generate_captions_btn)
        
        group.setLayout(layout)
        return group

    def create_dataset_config_group(self):
        group = QGroupBox("3. Dataset Configuration")
        group.setMinimumHeight(100)
        layout = QVBoxLayout()
        layout.setContentsMargins(10, 15, 10, 15)
        
        generate_toml_btn = QPushButton("Generate dataset.toml")
        generate_toml_btn.clicked.connect(self.generate_toml)
        layout.addWidget(generate_toml_btn)
        
        group.setLayout(layout)
        return group

    def create_utilities_group(self):
        group = QGroupBox("Utilities")
        group.setMinimumHeight(120)
        layout = QVBoxLayout()
        layout.setContentsMargins(10, 15, 10, 15)
        layout.setSpacing(8)
        
        rename_convert_btn = QPushButton("Rename and Convert Images")
        rename_convert_btn.clicked.connect(self.rename_and_convert_images)
        layout.addWidget(rename_convert_btn)
        
        analyze_btn = QPushButton("Analyze Dataset")
        analyze_btn.clicked.connect(self.analyze_dataset)
        layout.addWidget(analyze_btn)
        
        group.setLayout(layout)
        return group

    def create_status_group(self):
        group = QGroupBox("Status")
        layout = QVBoxLayout()
        self.status_label = QLabel("No dataset selected")
        layout.addWidget(self.status_label)
        group.setLayout(layout)
        return group

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
            
            self.tree_model.clear()
            self.populate_tree_view(self.dataset_path)
            self.tree_view.expandAll()
            self.update_status()
            
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
            QMessageBox.warning(self, "Warning