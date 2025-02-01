# image_processing_gui.py
from pathlib import Path
from PIL import Image
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
                           QPushButton, QMessageBox, QLabel, QSpinBox)

from image_processor import ImageProcessor
from gui_components import SuffixInputDialog

class ImageProcessingPanel(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.dataset_path = None
        self.image_processor = ImageProcessor()
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        layout.setSpacing(10)

        # 1. Image Processing Group
        image_group = self.create_image_processing_group()
        layout.addWidget(image_group)
        
        # 2. Image Utilities Group
        utils_group = self.create_utilities_group()
        layout.addWidget(utils_group)
        
        self.setLayout(layout)

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

    def create_utilities_group(self):
        group = QGroupBox("Image Utilities")
        group.setMinimumHeight(120)
        layout = QVBoxLayout()
        layout.setContentsMargins(10, 15, 10, 15)
        layout.setSpacing(8)
        
        rename_convert_btn = QPushButton("Rename and Convert Images")
        rename_convert_btn.clicked.connect(self.rename_and_convert_images)
        layout.addWidget(rename_convert_btn)
        
        group.setLayout(layout)
        return group

    def toggle_face_detection(self):
        if self.face_detection.isChecked():
            self.face_detection.setText("Face Detection: ON")
        else:
            self.face_detection.setText("Face Detection: OFF")

    def on_dataset_changed(self, dataset_path):
        self.dataset_path = dataset_path

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
            
            self.main_window.refresh_ui()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error processing images: {str(e)}")

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
            QMessageBox.warning(self, "Warning", "No images found to rename and convert!")
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
                QMessageBox.warning(self, "Error", f"Failed to process {image_path.name}: {e}")

        QMessageBox.information(self, "Success", f"Renamed and converted {converted_count} images successfully!")
        self.main_window.refresh_ui()