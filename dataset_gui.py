from pathlib import Path
from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                           QTreeView, QGroupBox, QPushButton, QMessageBox, 
                           QLabel, QFileDialog)
from PyQt6.QtGui import QStandardItemModel, QStandardItem

from image_processing_gui import ImageProcessingPanel
from caption_processing_gui import CaptionProcessingPanel
from training_gui import TrainingPanel

class DatasetManagerGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Dataset Manager")
        self.setGeometry(100, 100, 1300, 900)
        
        self.dataset_path = None
        self.init_ui()

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        layout = QHBoxLayout()
        
        # Left panel - TreeView and folder selection
        left_panel = self.create_left_panel()
        
        # Center panel - All processing operations
        center_panel = QWidget()
        center_layout = QVBoxLayout()
        
        # Image processing panel
        self.image_panel = ImageProcessingPanel(self)
        center_layout.addWidget(self.image_panel)
        
        # Caption processing panel
        self.caption_panel = CaptionProcessingPanel(self)
        center_layout.addWidget(self.caption_panel)
        
        center_panel.setLayout(center_layout)
        
        # Right panel - Training operations
        self.training_panel = TrainingPanel(self)
        
        # Set panel proportions (25/40/35)
        layout.addWidget(left_panel, 25)
        layout.addWidget(center_panel, 10)
        layout.addWidget(self.training_panel, 65)
        
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
        
        # Status group
        status_group = QGroupBox("Status")
        status_layout = QVBoxLayout()
        self.status_label = QLabel("No dataset selected")
        status_layout.addWidget(self.status_label)
        status_group.setLayout(status_layout)
        left_layout.addWidget(status_group)
        
        left_panel.setLayout(left_layout)
        return left_panel

    def select_dataset_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Dataset Folder")
        if folder:
            self.dataset_path = Path(folder).absolute()
            self.populate_tree_view(self.dataset_path)
            self.update_status()
            # Notify panels of dataset change
            self.image_panel.on_dataset_changed(self.dataset_path)
            self.caption_panel.on_dataset_changed(self.dataset_path)
            self.training_panel.on_dataset_changed(self.dataset_path)

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
            status_text = f"Dataset: {self.dataset_path}"
            
            cropped_path = self.dataset_path / "cropped_images"
            captions_path = self.dataset_path / "cropped_images/captions"
            
            if cropped_path.exists():
                n_images = len(list(cropped_path.glob("*.[jp][pn][g]")))
                status_text += f"\nImages: {n_images}"
            
            if captions_path.exists():
                n_captions = len(list(captions_path.glob("*.txt")))
                status_text += f"\nCaptions: {n_captions}"
                
            self.status_label.setText(status_text)
        else:
            self.status_label.setText("No dataset selected")

    def refresh_ui(self):
        """Refresh the UI after processing operations"""
        if self.dataset_path:
            self.populate_tree_view(self.dataset_path)
            self.update_status()