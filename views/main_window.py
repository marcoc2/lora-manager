import sys
from pathlib import Path
from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QTabWidget, QMessageBox, QLabel, 
                            QProgressDialog)
from PyQt6.QtCore import Qt, pyqtSignal

from views.dataset_view import DatasetView
from training_tabs import TrainingTabs
from queue_manager import QueueManager

DARK_STYLESHEET = """
QMainWindow, QWidget {
    background-color: #1e1e1e;
    color: #ffffff;
    font-family: 'Segoe UI', sans-serif;
    font-size: 14px;
}
QGroupBox {
    border: 1px solid #3e3e42;
    border-radius: 6px;
    margin-top: 24px;
    padding-top: 10px;
    font-weight: bold;
    color: #e0e0e0;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 0 5px;
    left: 10px;
    color: #007acc;
}
QPushButton {
    background-color: #2d2d30;
    border: 1px solid #3e3e42;
    border-radius: 4px;
    padding: 6px 12px;
    color: #ffffff;
}
QPushButton:hover {
    background-color: #3e3e42;
    border-color: #007acc;
}
QPushButton:pressed {
    background-color: #007acc;
}
QPushButton#primaryButton {
    background-color: #007acc;
    font-weight: bold;
}
QPushButton#primaryButton:hover {
    background-color: #0098ff;
}
QPushButton#actionButton {
    background-color: #2d2d30;
    border: 1px solid #007acc;
    color: #007acc;
}
QPushButton#actionButton:hover {
    background-color: #007acc;
    color: #ffffff;
}
QLineEdit, QSpinBox, QComboBox {
    background-color: #252526;
    border: 1px solid #3e3e42;
    border-radius: 4px;
    padding: 4px;
    color: #ffffff;
}
QLineEdit:focus, QSpinBox:focus, QComboBox:focus {
    border-color: #007acc;
}
QTabWidget::pane {
    border: 1px solid #3e3e42;
    background-color: #1e1e1e;
}
QTabBar::tab {
    background-color: #2d2d30;
    color: #ffffff;
    padding: 8px 16px;
    border-top-left-radius: 4px;
    border-top-right-radius: 4px;
    margin-right: 2px;
}
QTabBar::tab:selected {
    background-color: #1e1e1e;
    border-bottom: 2px solid #007acc;
    font-weight: bold;
}
QTreeView {
    background-color: #252526;
    border: 1px solid #3e3e42;
    color: #ffffff;
}
QHeaderView::section {
    background-color: #2d2d30;
    color: #ffffff;
    padding: 4px;
    border: none;
}
QScrollBar:vertical {
    background-color: #1e1e1e;
    width: 12px;
}
QScrollBar::handle:vertical {
    background-color: #424242;
    border-radius: 6px;
    min-height: 20px;
}
QScrollBar::handle:vertical:hover {
    background-color: #686868;
}
"""

class DatasetManagerGUI(QMainWindow):
    # Signals
    select_dataset_folder_clicked = pyqtSignal()
    process_images_clicked = pyqtSignal(dict)
    generate_captions_clicked = pyqtSignal()
    generate_toml_clicked = pyqtSignal()
    rename_and_convert_images_clicked = pyqtSignal()
    analyze_dataset_clicked = pyqtSignal()
    artifact_selected = pyqtSignal(str) # New signal

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Dataset Manager Pro")
        self.setGeometry(100, 100, 1400, 900)

        # Apply Dark Theme
        self.setStyleSheet(DARK_STYLESHEET)

        self.dataset_path = None
        self.active_artifact_path = None
        self.path_resolver = None  # Set by MainController

        self.init_ui()

    def get_effective_dataset_path(self):
        """Returns the active artifact path if selected, otherwise the root dataset path"""
        if self.active_artifact_path:
            return self.active_artifact_path
        return self.dataset_path

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        # Main Tab Widget
        self.tabs = QTabWidget()
        
        # 1. Dataset View
        self.dataset_view = DatasetView(self)
        self.tabs.addTab(self.dataset_view, "1. Dataset Preparation")

        # 2. Caption Processing
        from views.caption_processing_view import CaptionProcessingView
        self.caption_panel = CaptionProcessingView(self)
        self.tabs.addTab(self.caption_panel, "2. Caption Processing")

        # Create Queue Manager (needed for Training View)
        self.queue_manager = QueueManager()
        
        # 3. Training View
        self.training_tabs = TrainingTabs(self, queue_manager=self.queue_manager)
        self.tabs.addTab(self.training_tabs, "3. Training")
        
        # 4. Queue & Monitor View
        self.tabs.addTab(self.queue_manager, "4. Queue & Monitor")
        
        main_layout.addWidget(self.tabs)
        
        # Global Status Bar
        self.status_bar = QWidget()
        self.status_bar.setStyleSheet("background-color: #007acc; color: white;")
        self.status_bar.setFixedHeight(30)
        status_layout = QHBoxLayout()
        status_layout.setContentsMargins(10, 0, 10, 0)
        self.status_label = QLabel("Ready")
        status_layout.addWidget(self.status_label)
        self.status_bar.setLayout(status_layout)
        
        main_layout.addWidget(self.status_bar)
        
        central_widget.setLayout(main_layout)

    # Proxy methods for DatasetView actions
    def select_dataset_folder(self):
        self.select_dataset_folder_clicked.emit()

    def on_artifact_selected(self, index):
        folder_name = self.dataset_view.artifact_combo.currentText()
        if folder_name:
            self.artifact_selected.emit(folder_name)

    def process_images(self):
        path = self.get_effective_dataset_path()
        print(f"DEBUG: Process Image clicked. Input path: {path}")
        config = {
            'target_size': (self.dataset_view.crop_width.value(), self.dataset_view.crop_height.value()),
            'use_face_detection': self.dataset_view.face_detection.isChecked()
        }
        self.process_images_clicked.emit(config)

    def toggle_face_detection(self):
        if self.dataset_view.face_detection.isChecked():
            self.dataset_view.face_detection.setText("Face Detection: ON")
        else:
            self.dataset_view.face_detection.setText("Face Detection: OFF")

    def generate_captions(self):
        self.generate_captions_clicked.emit()

    def generate_toml(self):
        self.generate_toml_clicked.emit()

    def rename_and_convert_images(self):
        self.rename_and_convert_images_clicked.emit()

    def analyze_dataset(self):
        self.analyze_dataset_clicked.emit()

    # Methods called by MainController
    def populate_image_grid(self, path):
        self.dataset_view.populate_image_grid(path)

    def update_status(self, status_text):
        self.status_label.setText(status_text)

    def show_message(self, title, message, detailed_text=None):
        msg = QMessageBox(self)
        msg.setWindowTitle(title)
        msg.setText(message)
        if detailed_text:
            msg.setDetailedText(detailed_text)
        msg.exec()

    def show_warning(self, title, message):
        QMessageBox.warning(self, title, message)

    def show_critical(self, title, message):
        QMessageBox.critical(self, title, message)

    def show_progress_dialog(self, title, cancel_label, min_val, max_val):
        progress = QProgressDialog(title, cancel_label, min_val, max_val, self)
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setAutoClose(True)
        return progress
