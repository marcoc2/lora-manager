from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QTreeView, QGroupBox, 
    QPushButton, QMessageBox, QLabel, QSpinBox, QFileDialog, QProgressDialog
)
from PyQt6.QtGui import QStandardItemModel, QStandardItem
from PyQt6.QtCore import Qt
from pathlib import Path
import toml

# Módulos externos (supondo que estes já existam em seu projeto)
from image_processor import ImageProcessor
from training_widgets import CommandOutputDialog
from training_tabs import TrainingTabs
from gui_components import SuffixInputDialog, TomlConfigDialog, CaptionConfigDialog
from caption_generator import CaptionGenerator
from danbooru_generator import DanbooruGenerator

# Importa o mixin com os métodos de ação
from actions import DatasetActionsMixin

class DatasetManagerGUI(DatasetActionsMixin, QMainWindow):
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
        
        # Painel à esquerda: TreeView e seleção de pasta
        left_panel = self.create_left_panel()
        
        # Painel central: grupos de configuração e processamento
        center_panel = self.create_center_panel()
        
        # Painel à direita: abas de treinamento
        self.training_tabs = TrainingTabs(self)
        
        # Define as proporções dos painéis (25/10/65)
        layout.addWidget(left_panel, 25)
        layout.addWidget(center_panel, 10)
        layout.addWidget(self.training_tabs, 65)
        
        central_widget.setLayout(layout)
    
    def create_left_panel(self):
        left_panel = QWidget()
        left_layout = QVBoxLayout()
        
        # Botão para selecionar a pasta do dataset
        select_button = QPushButton("Select Dataset Folder")
        select_button.clicked.connect(self.select_dataset_folder)
        left_layout.addWidget(select_button)
        
        # TreeView para visualizar a estrutura do dataset
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

        # 1. Grupo de Processamento de Imagens
        image_group = self.create_image_processing_group()
        center_layout.addWidget(image_group)
        
        # 2. Grupo de Geração de Legendas
        caption_group = self.create_caption_generation_group()
        center_layout.addWidget(caption_group)
        
        # 3. Grupo de Configuração do Dataset
        dataset_group = self.create_dataset_config_group()
        center_layout.addWidget(dataset_group)
        
        # 4. Grupo de Utilitários
        utils_group = self.create_utilities_group()
        center_layout.addWidget(utils_group)
        
        # Grupo de Status
        status_group = self.create_status_group()
        center_layout.addWidget(status_group)
        
        center_panel.setLayout(center_layout)
        return center_panel

    def create_image_processing_group(self):
        group = QGroupBox("1. Image Processing")
        group.setMinimumHeight(150)
        layout = QVBoxLayout()
        
        # Configuração do tamanho (largura x altura)
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
        
        # Botão para alternar detecção facial
        self.face_detection = QPushButton("Face Detection: ON")
        self.face_detection.setCheckable(True)
        self.face_detection.setChecked(True)
        self.face_detection.clicked.connect(self.toggle_face_detection)
        layout.addWidget(self.face_detection)
        
        # Botão para processar as imagens
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
