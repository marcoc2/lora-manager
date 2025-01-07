import sys
import toml
from pathlib import Path
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QTreeView, QMenu, QPushButton, 
                            QFileDialog, QMessageBox, QLabel, QSpinBox,
                            QDialog, QLineEdit, QFormLayout)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QStandardItemModel, QStandardItem
from PyQt6.QtGui import QAction, QContextMenuEvent

class DatasetManagerGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Dataset Manager")
        self.setGeometry(100, 100, 1200, 800)
        
        # Configuração inicial
        self.dataset_path = None
        self.init_ui()

    def init_ui(self):
        # Widget central
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Layout principal
        layout = QHBoxLayout()
        
        # Painel esquerdo - TreeView e controles
        left_panel = QWidget()
        left_layout = QVBoxLayout()
        
        # Botão para selecionar pasta
        select_button = QPushButton("Select Dataset Folder")
        select_button.clicked.connect(self.select_dataset_folder)
        left_layout.addWidget(select_button)
        
        # TreeView para visualização do dataset
        self.tree_view = QTreeView()
        self.tree_view.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.tree_view.customContextMenuRequested.connect(self.show_context_menu)
        
        # Modelo de dados customizado
        self.tree_model = QStandardItemModel()
        self.tree_model.setHorizontalHeaderLabels(['Dataset Structure'])
        
        # Configura o TreeView
        self.tree_view.setModel(self.tree_model)
        self.tree_view.setColumnWidth(0, 300)
        left_layout.addWidget(self.tree_view)
        
        left_panel.setLayout(left_layout)
        
        # Painel direito - Configurações e status
        right_panel = QWidget()
        right_layout = QVBoxLayout()
        
        # Configurações de processamento
        config_group = QWidget()
        config_layout = QVBoxLayout()
        
        # Tamanho do crop
        crop_widget = QWidget()
        crop_layout = QHBoxLayout()
        crop_layout.addWidget(QLabel("Crop Size:"))
        self.crop_width = QSpinBox()
        self.crop_width.setRange(64, 2048)
        self.crop_width.setValue(512)
        self.crop_height = QSpinBox()
        self.crop_height.setRange(64, 2048)
        self.crop_height.setValue(512)
        crop_layout.addWidget(self.crop_width)
        crop_layout.addWidget(QLabel("x"))
        crop_layout.addWidget(self.crop_height)
        crop_widget.setLayout(crop_layout)
        config_layout.addWidget(crop_widget)
        
        # Face detection toggle
        self.face_detection = QPushButton("Face Detection: ON")
        self.face_detection.setCheckable(True)
        self.face_detection.setChecked(True)
        self.face_detection.clicked.connect(self.toggle_face_detection)
        config_layout.addWidget(self.face_detection)
        
        config_group.setLayout(config_layout)
        right_layout.addWidget(config_group)
        
        # Status e informações
        self.status_label = QLabel("No dataset selected")
        right_layout.addWidget(self.status_label)
        
        # Adiciona expansor para empurrar widgets para cima
        right_layout.addStretch()
        
        right_panel.setLayout(right_layout)
        
        # Define proporção dos painéis (70/30)
        layout.addWidget(left_panel, 70)
        layout.addWidget(right_panel, 30)
        
        central_widget.setLayout(layout)

    def select_dataset_folder(self):
        """Seleciona a pasta do dataset"""
        folder = QFileDialog.getExistingDirectory(self, "Select Dataset Folder")
        if folder:
            self.dataset_path = Path(folder)
            self.populate_tree_view(self.dataset_path)
            self.update_status()
    
    def populate_tree_view(self, path):
        """Popula a TreeView com a estrutura do dataset"""
        self.tree_model.clear()
        self.tree_model.setHorizontalHeaderLabels(['Dataset Structure'])
        
        # Cria item raiz
        root_item = QStandardItem(str(path))
        self.tree_model.appendRow(root_item)
        
        # Função recursiva para adicionar arquivos e pastas
        def add_directory_contents(parent_item, dir_path):
            # Lista todos os itens do diretório
            try:
                for item_path in sorted(Path(dir_path).iterdir()):
                    item = QStandardItem(item_path.name)
                    parent_item.appendRow(item)
                    
                    # Se for diretório, processa recursivamente
                    if item_path.is_dir():
                        add_directory_contents(item, item_path)
            except Exception as e:
                print(f"Erro ao acessar {dir_path}: {e}")
        
        # Popula a árvore
        add_directory_contents(root_item, path)
        
        # Expande o primeiro nível
        self.tree_view.expand(self.tree_model.index(0, 0))

    def show_context_menu(self, position):
        """Mostra menu de contexto com operações disponíveis"""
        if not self.dataset_path:
            return
            
        menu = QMenu()
        
        # Adiciona ações ao menu
        process_action = QAction("Process Images", self)
        process_action.triggered.connect(self.process_images)
        menu.addAction(process_action)
        
        generate_captions = QAction("Generate Captions", self)
        generate_captions.triggered.connect(self.generate_captions)
        menu.addAction(generate_captions)
        
        generate_toml = QAction("Generate dataset.toml", self)
        generate_toml.triggered.connect(self.generate_toml)
        menu.addAction(generate_toml)
        
        analyze = QAction("Analyze Dataset", self)
        analyze.triggered.connect(self.analyze_dataset)
        menu.addAction(analyze)
        
        # Mostra menu
        menu.exec(self.tree_view.viewport().mapToGlobal(position))

    def toggle_face_detection(self):
        """Alterna detecção facial"""
        if self.face_detection.isChecked():
            self.face_detection.setText("Face Detection: ON")
        else:
            self.face_detection.setText("Face Detection: OFF")

    def process_images(self):
        """Processa imagens do dataset"""
        if not self.dataset_path:
            return
            
        try:
            QMessageBox.information(self, "Processing", "Starting image processing...")
            # Aqui integramos com o DatasetManager
            # manager.process_images()
            QMessageBox.information(self, "Success", "Images processed successfully!")
            self.update_status()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error processing images: {str(e)}")

    def generate_captions(self):
        """Gera captions para as imagens"""
        if not self.dataset_path:
            return
            
        try:
            QMessageBox.information(self, "Processing", "Starting caption generation...")
            # manager.generate_captions()
            QMessageBox.information(self, "Success", "Captions generated successfully!")
            self.update_status()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error generating captions: {str(e)}")

    def process_images(self):
        """Processa imagens do dataset"""
        if not self.dataset_path:
            return
            
        try:
            QMessageBox.information(self, "Processing", 
                "This will process all images in the dataset folder.\n"
                "Do you want to continue?")
            
            output_dir = self.dataset_path / "cropped_images"
            output_dir.mkdir(exist_ok=True)
            
            # TODO: Implementar processamento de imagens
            # Integrar com o módulo de processamento
            
            QMessageBox.information(self, "Success", "Images processed successfully!")
            self.update_status()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error processing images: {str(e)}")

    def generate_captions(self):
        """Gera captions para as imagens"""
        if not self.dataset_path:
            return
            
        try:
            QMessageBox.information(self, "Processing", 
                "This will generate captions for all images.\n"
                "This might take a while. Do you want to continue?")
            
            captions_dir = self.dataset_path / "captions"
            captions_dir.mkdir(exist_ok=True)
            
            # TODO: Implementar geração de captions
            # Integrar com o módulo BLIP
            
            QMessageBox.information(self, "Success", "Captions generated successfully!")
            self.update_status()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error generating captions: {str(e)}")

    def analyze_dataset(self):
        """Analisa estado atual do dataset"""
        if not self.dataset_path:
            return
            
        try:
            # Análise básica do dataset
            stats = {
                "total_images": 0,
                "total_captions": 0,
                "image_sizes": set(),
                "missing_captions": []
            }
            
            # Conta imagens e verifica tamanhos
            images_dir = self.dataset_path / "cropped_images"
            if images_dir.exists():
                for img_path in images_dir.glob("*.[jp][pn][g]"):
                    stats["total_images"] += 1
                    
            # Verifica captions
            captions_dir = self.dataset_path / "captions"
            if captions_dir.exists():
                stats["total_captions"] = len(list(captions_dir.glob("*.txt")))
            
            # Prepara mensagem
            msg = f"""Dataset Analysis:
            
Total Images: {stats['total_images']}
Total Captions: {stats['total_captions']}
Missing Captions: {stats['total_images'] - stats['total_captions']}"""
            
            QMessageBox.information(self, "Dataset Analysis", msg)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error analyzing dataset: {str(e)}")
    
    def generate_toml(self):
        """Gera arquivo dataset.toml"""
        if not self.dataset_path:
            return
            
        try:
            # Abre diálogo de configuração
            dialog = TomlConfigDialog(self)
            if dialog.exec() == QDialog.DialogCode.Accepted:
                config = dialog.get_values()
                
                # Prepara dados do TOML
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
                            "image_dir": str(self.dataset_path / "cropped_images"),
                            "class_tokens": config['class_tokens'],
                            "num_repeats": config['num_repeats']
                        }]
                    }]
                }
                
                # Salva arquivo TOML
                toml_path = self.dataset_path / "dataset.toml"
                with open(toml_path, "w") as f:
                    toml.dump(toml_data, f)
                
                QMessageBox.information(self, "Success", "dataset.toml generated successfully!")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error generating dataset.toml: {str(e)}")
    
    def update_status(self):
        """Atualiza o label de status com informações do dataset atual"""
        if self.dataset_path:
            status_text = f"Dataset: {self.dataset_path}"
            
            # Conta arquivos nas pastas relevantes
            cropped_path = self.dataset_path / "cropped_images"
            captions_path = self.dataset_path / "captions"
            
            if cropped_path.exists():
                n_images = len(list(cropped_path.glob("*.[jp][pn][g]")))
                status_text += f"\nImages: {n_images}"
            
            if captions_path.exists():
                n_captions = len(list(captions_path.glob("*.txt")))
                status_text += f"\nCaptions: {n_captions}"
                
            self.status_label.setText(status_text)
        else:
            self.status_label.setText("No dataset selected")

class TomlConfigDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Dataset Configuration")
        self.setModal(True)
        
        # Layout
        layout = QFormLayout()
        
        # Campos
        self.class_tokens = QLineEdit()
        self.num_repeats = QSpinBox()
        self.num_repeats.setRange(1, 100)
        self.num_repeats.setValue(1)
        
        self.resolution = QSpinBox()
        self.resolution.setRange(64, 2048)
        self.resolution.setValue(512)
        self.resolution.setSingleStep(64)
        
        # Adiciona campos ao layout
        layout.addRow("Class Tokens:", self.class_tokens)
        layout.addRow("Number of Repeats:", self.num_repeats)
        layout.addRow("Resolution:", self.resolution)
        
        # Botões
        buttons = QHBoxLayout()
        ok_button = QPushButton("OK")
        cancel_button = QPushButton("Cancel")
        
        ok_button.clicked.connect(self.accept)
        cancel_button.clicked.connect(self.reject)
        
        buttons.addWidget(ok_button)
        buttons.addWidget(cancel_button)
        
        # Adiciona botões ao layout principal
        final_layout = QVBoxLayout()
        final_layout.addLayout(layout)
        final_layout.addLayout(buttons)
        
        self.setLayout(final_layout)
    
    def get_values(self):
        return {
            'class_tokens': self.class_tokens.text(),
            'num_repeats': self.num_repeats.value(),
            'resolution': self.resolution.value()
        }

    def generate_toml(self):
        """Gera arquivo dataset.toml"""
        if not self.dataset_path:
            return
            
        try:
            # Abre diálogo de configuração
            dialog = TomlConfigDialog(self)
            if dialog.exec() == QDialog.DialogCode.Accepted:
                config = dialog.get_values()
                
                # Prepara dados do TOML
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
                            "image_dir": str(self.dataset_path / "cropped_images"),
                            "class_tokens": config['class_tokens'],
                            "num_repeats": config['num_repeats']
                        }]
                    }]
                }
                
                # Salva arquivo TOML
                toml_path = self.dataset_path / "dataset.toml"
                with open(toml_path, "w") as f:
                    toml.dump(toml_data, f)
                
                QMessageBox.information(self, "Success", "dataset.toml generated successfully!")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error generating dataset.toml: {str(e)}")

    def analyze_dataset(self):
        """Analisa estado atual do dataset"""
        if not self.dataset_path:
            return
            
        try:
            # stats = manager.analyze_dataset()
            stats = {
                "total_images": 100,
                "total_captions": 95,
                "image_sizes": {(512, 512)},
                "missing_captions": ["img1.png", "img2.png"]
            }
            
            msg = f"""Dataset Analysis:
Total Images: {stats['total_images']}
Total Captions: {stats['total_captions']}
Image Sizes: {', '.join(str(s) for s in stats['image_sizes'])}
Missing Captions: {len(stats['missing_captions'])}"""
            
            QMessageBox.information(self, "Dataset Analysis", msg)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error analyzing dataset: {str(e)}")

    def update_status(self):
        """Atualiza label de status"""
        if self.dataset_path:
            self.status_label.setText(f"Dataset: {self.dataset_path}")
        else:
            self.status_label.setText("No dataset selected")

def main():
    app = QApplication(sys.argv)
    window = DatasetManagerGUI()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()