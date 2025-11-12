import sys
from pathlib import Path
from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QTreeView, QGroupBox, QPushButton, 
                            QMessageBox, QLabel, QSpinBox,
                            QFileDialog, QProgressDialog)
from PyQt6.QtGui import QStandardItemModel, QStandardItem
from PyQt6.QtCore import Qt, pyqtSignal

from training_tabs import TrainingTabs

class DatasetManagerGUI(QMainWindow):
    # Signals
    select_dataset_folder_clicked = pyqtSignal()
    process_images_clicked = pyqtSignal(dict)
    generate_captions_clicked = pyqtSignal()
    generate_toml_clicked = pyqtSignal()
    rename_and_convert_images_clicked = pyqtSignal()
    analyze_dataset_clicked = pyqtSignal()
    start_training_clicked = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Dataset Manager")
        self.setGeometry(100, 100, 1300, 900)
        
        self.init_ui()

    def init_ui(self):
        # Widget central
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Layout principal
        layout = QHBoxLayout()
        
        # Painel esquerdo - TreeView e seleção de pasta
        left_panel = QWidget()
        left_layout = QVBoxLayout()
        
        # Botão para selecionar pasta
        self.select_button = QPushButton("Select Dataset Folder")
        left_layout.addWidget(self.select_button)
        
        # TreeView para visualização do dataset
        self.tree_view = QTreeView()
        self.tree_model = QStandardItemModel()
        self.tree_model.setHorizontalHeaderLabels(['Dataset Structure'])
        self.tree_view.setModel(self.tree_model)
        self.tree_view.setColumnWidth(0, 500)
        left_layout.addWidget(self.tree_view)
        
        left_panel.setLayout(left_layout)
        
        # Painel central - Configurações e processamento
        center_panel = QWidget()
        center_layout = QVBoxLayout()
        center_layout.setSpacing(10)
        
        # 1. Grupo de Processamento de Imagens
        image_group = QGroupBox("1. Image Processing")
        image_group.setMinimumHeight(150)
        image_layout = QVBoxLayout()
        
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
        image_layout.addLayout(size_layout)
        
        self.face_detection = QPushButton("Face Detection: ON")
        self.face_detection.setCheckable(True)
        self.face_detection.setChecked(True)
        image_layout.addWidget(self.face_detection)
        
        self.process_button = QPushButton("Process Images")
        image_layout.addWidget(self.process_button)
        
        image_group.setLayout(image_layout)
        center_layout.addWidget(image_group)
        
        # 2. Grupo de Geração de Captions
        caption_group = QGroupBox("2. Caption Generation")
        caption_group.setMinimumHeight(100)
        caption_layout = QVBoxLayout()
        caption_layout.setContentsMargins(10, 15, 10, 15)
        
        self.generate_captions_btn = QPushButton("Generate Captions")
        caption_layout.addWidget(self.generate_captions_btn)
        
        caption_group.setLayout(caption_layout)
        center_layout.addWidget(caption_group)
        
        # 3. Grupo de Configuração do Dataset
        dataset_group = QGroupBox("3. Dataset Configuration")
        dataset_group.setMinimumHeight(100)
        dataset_layout = QVBoxLayout()
        dataset_layout.setContentsMargins(10, 15, 10, 15)
        
        self.generate_toml_btn = QPushButton("Generate dataset.toml files")
        dataset_layout.addWidget(self.generate_toml_btn)
        
        dataset_group.setLayout(dataset_layout)
        center_layout.addWidget(dataset_group)
        
        # 4. Grupo de Utilitários
        utils_group = QGroupBox("Utilities")
        utils_group.setMinimumHeight(120)
        utils_layout = QVBoxLayout()
        utils_layout.setContentsMargins(10, 15, 10, 15)
        utils_layout.setSpacing(8)
        
        self.rename_convert_btn = QPushButton("Rename and Convert Images")
        utils_layout.addWidget(self.rename_convert_btn)
        
        self.analyze_btn = QPushButton("Analyze Dataset")
        utils_layout.addWidget(self.analyze_btn)
        
        utils_group.setLayout(utils_layout)
        center_layout.addWidget(utils_group)
        
        # Status
        status_group = QGroupBox("Status")
        status_layout = QVBoxLayout()
        self.status_label = QLabel("No dataset selected")
        status_layout.addWidget(self.status_label)
        status_group.setLayout(status_layout)
        center_layout.addWidget(status_group)
        
        center_panel.setLayout(center_layout)
        
        # Painel direito - Widgets de treinamento
        self.training_tabs = TrainingTabs(self)
        
        layout.addWidget(left_panel, 25)
        layout.addWidget(center_panel, 10)
        layout.addWidget(self.training_tabs, 65)
        
        central_widget.setLayout(layout)

        # Connect signals
        self.select_button.clicked.connect(self.select_dataset_folder_clicked.emit)
        self.process_button.clicked.connect(self.on_process_images_clicked)
        self.generate_captions_btn.clicked.connect(self.generate_captions_clicked.emit)
        self.generate_toml_btn.clicked.connect(self.generate_toml_clicked.emit)
        self.rename_convert_btn.clicked.connect(self.rename_and_convert_images_clicked.emit)
        self.analyze_btn.clicked.connect(self.analyze_dataset_clicked.emit)
        self.training_tabs.train_button.clicked.connect(self.start_training_clicked.emit)
        self.face_detection.clicked.connect(self.toggle_face_detection)

    def on_process_images_clicked(self):
        config = {
            'target_size': (self.crop_width.value(), self.crop_height.value()),
            'use_face_detection': self.face_detection.isChecked()
        }
        self.process_images_clicked.emit(config)

    def toggle_face_detection(self):
        if self.face_detection.isChecked():
            self.face_detection.setText("Face Detection: ON")
        else:
            self.face_detection.setText("Face Detection: OFF")

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
                print(f"Erro ao acessar {dir_path}: {e}")
        
        add_directory_contents(root_item, path)
        self.tree_view.expandAll()

    def update_status(self, status_text):
        self.status_label.setText(status_text)

    def show_message(self, title, message, detailed_text=None):
        msg = QMessageBox()
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
