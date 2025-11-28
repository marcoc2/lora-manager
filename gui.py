from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QTabWidget,
    QMessageBox, QLabel, QSplitter
)
from PyQt6.QtGui import QAction
from PyQt6.QtCore import Qt
from pathlib import Path
import toml

# Módulos externos
from image_processor import ImageProcessor
from training_tabs import TrainingTabs
from views.dataset_view import DatasetView
from actions import DatasetActionsMixin

# Dark Theme Stylesheet
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
    background-color: #3e3e42;
    border: none;
    border-radius: 4px;
    padding: 8px 16px;
    color: #ffffff;
}
QPushButton:hover {
    background-color: #505054;
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
    border: 1px solid #3e3e42;
}
QLineEdit, QSpinBox, QComboBox {
    background-color: #252526;
    border: 1px solid #3e3e42;
    border-radius: 4px;
    padding: 6px;
    color: #ffffff;
    selection-background-color: #007acc;
}
QTreeView {
    background-color: #252526;
    border: 1px solid #3e3e42;
    border-radius: 4px;
}
QTabWidget::pane {
    border: 1px solid #3e3e42;
    background-color: #1e1e1e;
}
QTabBar::tab {
    background-color: #2d2d30;
    color: #b0b0b0;
    padding: 10px 20px;
    border-top-left-radius: 4px;
    border-top-right-radius: 4px;
    margin-right: 2px;
}
QTabBar::tab:selected {
    background-color: #1e1e1e;
    color: #007acc;
    border-bottom: 2px solid #007acc;
}
QTabBar::tab:hover {
    background-color: #3e3e42;
}
QScrollBar:vertical {
    border: none;
    background: #1e1e1e;
    width: 10px;
    margin: 0px;
}
QScrollBar::handle:vertical {
    background: #424242;
    min-height: 20px;
    border-radius: 5px;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0px;
}
"""

class DatasetManagerGUI(DatasetActionsMixin, QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Dataset Manager Pro")
        self.setGeometry(100, 100, 1400, 900)
        
        self.dataset_path = None
        self.image_processor = ImageProcessor()
        
        # Apply Dark Theme
        self.setStyleSheet(DARK_STYLESHEET)
        
        self.init_ui()
    
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
        
        # Create Queue Manager (needed for Training View)
        from queue_manager import QueueManager
        self.queue_manager = QueueManager()
        
        # 2. Training View
        self.training_tabs = TrainingTabs(self, queue_manager=self.queue_manager)
        self.tabs.addTab(self.training_tabs, "2. Training")
        
        # 3. Queue & Monitor View
        self.tabs.addTab(self.queue_manager, "3. Queue & Monitor")
        
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

    # --- Property Delegates for DatasetActionsMixin Compatibility ---
    # These properties allow the mixin to access widgets that are now inside DatasetView
    
    @property
    def tree_view(self):
        return self.dataset_view.tree_view
        
    @property
    def tree_model(self):
        return self.dataset_view.tree_model
        
    @property
    def crop_width(self):
        return self.dataset_view.crop_width
        
    @property
    def crop_height(self):
        return self.dataset_view.crop_height
        
    @property
    def face_detection(self):
        return self.dataset_view.face_detection

