import sys
import toml
from PIL import Image
from pathlib import Path
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QTreeView, QMenu, QPushButton, 
                            QFileDialog, QMessageBox, QLabel, QSpinBox,
                            QDialog, QLineEdit, QFormLayout, QProgressDialog)  # Adicionado QProgressDialog aqui
from PyQt6.QtGui import QStandardItemModel, QStandardItem, QAction
from PyQt6.QtCore import Qt
from typing import Tuple, List, Optional, Callable

from image_processor import ImageProcessor
from caption_generator import CaptionGenerator
from training_dialog import TrainingConfigDialog

class SuffixInputDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Enter Suffix")
        self.setModal(True)

        layout = QVBoxLayout()
        self.suffix_input = QLineEdit()
        self.suffix_input.setPlaceholderText("Enter suffix (e.g., _XXX)")
        layout.addWidget(QLabel("Suffix for renaming:"))
        layout.addWidget(self.suffix_input)

        buttons = QHBoxLayout()
        ok_button = QPushButton("OK")
        cancel_button = QPushButton("Cancel")
        ok_button.clicked.connect(self.accept)
        cancel_button.clicked.connect(self.reject)
        buttons.addWidget(ok_button)
        buttons.addWidget(cancel_button)

        layout.addLayout(buttons)
        self.setLayout(layout)

    def get_suffix(self):
        return self.suffix_input.text().strip()


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
        
        # Layout final
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

class CaptionConfigDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Caption Configuration")
        self.setModal(True)
        
        layout = QFormLayout()
        
        self.prefix = QLineEdit()
        layout.addRow("Caption Prefix:", self.prefix)
        
        # Botões
        buttons = QHBoxLayout()
        ok_button = QPushButton("OK")
        cancel_button = QPushButton("Cancel")
        
        ok_button.clicked.connect(self.accept)
        cancel_button.clicked.connect(self.reject)
        
        buttons.addWidget(ok_button)
        buttons.addWidget(cancel_button)
        
        final_layout = QVBoxLayout()
        final_layout.addLayout(layout)
        final_layout.addLayout(buttons)
        
        self.setLayout(final_layout)
        
    def get_values(self):
        return {
            'prefix': self.prefix.text()
        }

class DatasetManagerGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Dataset Manager")
        self.setGeometry(100, 100, 1200, 800)
        
        # State
        self.dataset_path = None
        self.image_processor = ImageProcessor()
        
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

    def rename_and_convert_images(self):
        if not self.dataset_path:
            QMessageBox.warning(self, "Warning", "Please select a dataset folder first!")
            return

        # Abre o diálogo para obter o sufixo
        dialog = SuffixInputDialog(self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        suffix = dialog.get_suffix()
        if not suffix:
            QMessageBox.warning(self, "Warning", "Suffix cannot be empty!")
            return

        # Diretório com as imagens
        image_dir = self.dataset_path / "cropped_images"
        if not image_dir.exists():
            QMessageBox.warning(self, "Warning", "Cropped images directory does not exist!")
            return

        # Processa imagens
        image_files = list(image_dir.glob("*.[jp][pn][g]")) + list(image_dir.glob("*.webp"))
        if not image_files:
            QMessageBox.warning(self, "Warning", "No images found to rename and convert!")
            return

        converted_count = 0

        for idx, image_path in enumerate(sorted(image_files), 1):
            try:
                # Novo nome com o sufixo
                new_name = f"{image_path.stem}{suffix}_{str(idx).zfill(3)}.png"
                new_path = image_dir / new_name

                # Converte a imagem para PNG
                with Image.open(image_path) as img:
                    img = img.convert("RGB")  # Converte para RGB se necessário
                    img.save(new_path, "PNG")

                # Remove a imagem original se for diferente do novo formato
                if image_path.suffix.lower() != ".png":
                    image_path.unlink()

                converted_count += 1
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Failed to process {image_path.name}: {e}")

        QMessageBox.information(self, "Success", f"Renamed and converted {converted_count} images successfully!")
        
        # Atualiza a árvore
        self.tree_model.clear()
        self.populate_tree_view(self.dataset_path)



    def select_dataset_folder(self):
        """Seleciona a pasta do dataset"""
        folder = QFileDialog.getExistingDirectory(self, "Select Dataset Folder")
        if folder:
            # Garante que temos o caminho absoluto
            self.dataset_path = Path(folder).absolute()
            print(f"Dataset path selecionado: {self.dataset_path}")
            
            self.populate_tree_view(self.dataset_path)
            self.update_status()


    def show_context_menu(self, position):
        """Mostra menu de contexto com operações disponíveis"""
        if not self.dataset_path:
            return
            
        menu = QMenu()

        menu.addSeparator()  # Adiciona separador
        train_action = QAction("Start Training", self)
        train_action.triggered.connect(self.show_training_config)
        menu.addAction(train_action)
        
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
        
        rename_convert_action = QAction("Rename and Convert Images", self)
        rename_convert_action.triggered.connect(self.rename_and_convert_images)
        menu.addAction(rename_convert_action)

        process_action.triggered.connect(self.process_images)

        # Mostra menu
        menu.exec(self.tree_view.viewport().mapToGlobal(position))


    def show_training_config(self):
        """Mostra diálogo de configuração do treinamento"""
        if not self.dataset_path:
            return
            
        # Verifica se dataset.toml existe
        toml_path = self.dataset_path / "cropped_images/dataset.toml"
        if not toml_path.exists():
            QMessageBox.warning(self, "Warning", "Please generate dataset.toml first!")
            return
        
        dialog = TrainingConfigDialog(self.dataset_path, self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            # Gera comando
            command = dialog.get_command()
            
            # Mostra comando para confirmação
            msg = QMessageBox()
            msg.setWindowTitle("Training Command")
            msg.setText("The following command will be executed:")
            msg.setDetailedText(command)
            msg.setStandardButtons(QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel)
            
            if msg.exec() == QMessageBox.StandardButton.Ok:
                try:
                    # TODO: Executar comando
                    # Por enquanto só mostra que seria executado
                    QMessageBox.information(self, "Training", "Training command would be executed here.")
                except Exception as e:
                    QMessageBox.critical(self, "Error", f"Error starting training: {str(e)}")

    def populate_tree_view(self, path):
        """Popula a TreeView com a estrutura do dataset"""
        self.tree_model.clear()
        self.tree_model.setHorizontalHeaderLabels(['Dataset Structure'])
        
        # Cria item raiz
        root_item = QStandardItem(str(path))
        self.tree_model.appendRow(root_item)
        
        # Função recursiva para adicionar arquivos e pastas
        def add_directory_contents(parent_item, dir_path):
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
        
        # Expande todos os níveis
        self.tree_view.expandAll()

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
            # Prepara diretórios - usando o próprio dataset_path como input
            input_dir = self.dataset_path  # <-- Aqui está a correção
            output_dir = self.dataset_path / "cropped_images"
            
            # Conta arquivos para a barra de progresso
            n_files = sum(1 for _ in input_dir.glob("*.[jp][pn][g]"))
            if n_files == 0:
                QMessageBox.warning(self, "Warning", "No images found in the input directory!")
                return
            
            # Processa imagens
            target_size = (self.crop_width.value(), self.crop_height.value())
            self.image_processor.use_face_detection = self.face_detection.isChecked()
            
            processed, failed = self.image_processor.process_directory(
                input_dir, output_dir, target_size
            )
            
            # Mostra resultado
            QMessageBox.information(self, "Success", 
                f"Processing complete!\n\n"
                f"Successfully processed: {processed}\n"
                f"Failed: {failed}")
            
            # Força uma atualização completa da árvore
            self.tree_model.clear()
            self.populate_tree_view(self.dataset_path)
            self.tree_view.expandAll()
            self.update_status()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error processing images: {str(e)}")


    def generate_captions(self):
        """Gera captions para as imagens"""
        if not self.dataset_path:
            return
            
        try:
            # Verifica se existe pasta cropped_images
            cropped_dir = self.dataset_path / "cropped_images"
            if not cropped_dir.exists():
                QMessageBox.warning(self, "Warning", "Please process images first!")
                return
            
            # Verifica se tem imagens para processar
            n_files = sum(1 for _ in cropped_dir.glob("*.[jp][pn][g]"))
            if n_files == 0:
                QMessageBox.warning(self, "Warning", "No images found in cropped_images folder!")
                return
            
            # Pede configuração do caption
            config_dialog = CaptionConfigDialog(self)
            if config_dialog.exec() != QDialog.DialogCode.Accepted:
                return
                
            config = config_dialog.get_values()
            
            # Cria e configura barra de progresso
            progress = QProgressDialog("Generating captions...", "Cancel", 0, 100, self)
            progress.setWindowModality(Qt.WindowModality.WindowModal)
            progress.setAutoClose(True)
            progress.show()
            
            def update_progress(message: str, value: int):
                if value >= 0:
                    progress.setLabelText(message)
                    progress.setValue(value)
            
            # Cria pasta de captions dentro de cropped_images
            captions_dir = cropped_dir / "captions"
            
            # Processa imagens
            caption_generator = CaptionGenerator()
            processed, failed = caption_generator.process_directory(
                cropped_dir, 
                captions_dir,
                prefix=config['prefix'],
                progress_callback=update_progress
            )
            
            progress.close()
            
            # Mostra resultado
            QMessageBox.information(self, "Success", 
                f"Caption generation complete!\n\n"
                f"Successfully processed: {processed}\n"
                f"Failed: {failed}")
            
            # Atualiza visualização
            self.tree_model.clear()
            self.populate_tree_view(self.dataset_path)
            self.tree_view.expandAll()
            self.update_status()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error generating captions: {str(e)}")

    def generate_toml(self):
        """Gera o arquivo dataset.toml"""
        if not self.dataset_path:
            return
        
        try:
            # Abre diálogo de configuração
            dialog = TomlConfigDialog(self)
            if dialog.exec() == QDialog.DialogCode.Accepted:
                config = dialog.get_values()
                
                # Cria pasta cropped_images se não existir
                cropped_dir = self.dataset_path / "cropped_images"
                cropped_dir.mkdir(parents=True, exist_ok=True)
                
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
                            # Corrigindo o caminho do image_dir
                            "image_dir": str(cropped_dir.resolve()),  # Caminho absoluto
                            "class_tokens": config['class_tokens'],
                            "num_repeats": config['num_repeats']
                        }]
                    }]
                }
                
                # Salva arquivo TOML dentro de cropped_images
                toml_path = cropped_dir / "dataset.toml"
                with open(toml_path, "w", encoding="utf-8") as f:
                    toml.dump(toml_data, f)
                
                QMessageBox.information(self, "Success", "dataset.toml generated successfully!")
                self.tree_model.clear()
                self.populate_tree_view(self.dataset_path)
                self.update_status()
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error generating dataset.toml: {str(e)}")


    def update_status(self):
        """Atualiza o label de status com informações do dataset atual"""
        if self.dataset_path:
            status_text = f"Dataset: {self.dataset_path}"
            
            # Conta arquivos nas pastas relevantes
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

    def analyze_dataset(self):
        """Analisa estado atual do dataset"""
        if not self.dataset_path:
            return
            
        try:
            # Análise básica do dataset
            stats = {
                "total_images": 0,
                "total_captions": 0,
                "missing_captions": []
            }
            
            # Conta imagens e verifica tamanhos
            images_dir = self.dataset_path / "cropped_images"
            if images_dir.exists():
                stats["total_images"] = len(list(images_dir.glob("*.[jp][pn][g]")))
            
            # Verifica captions
            captions_dir = images_dir / "captions"
            if captions_dir.exists():
                stats["total_captions"] = len(list(captions_dir.glob("*.txt")))
                
                # Verifica quais imagens estão sem caption
                for img_path in images_dir.glob("*.[jp][pn][g]"):
                    caption_path = captions_dir / f"{img_path.stem}.txt"
                    if not caption_path.exists():
                        stats["missing_captions"].append(img_path.name)
            
            # Prepara mensagem
            msg = f"""Dataset Analysis:

    Total Images: {stats['total_images']}
    Total Captions: {stats['total_captions']}
    Missing Captions: {len(stats['missing_captions'])}"""

            if stats["missing_captions"]:
                msg += "\n\nFiles missing captions:"
                for file in stats["missing_captions"][:10]:  # Mostra só os 10 primeiros
                    msg += f"\n- {file}"
                if len(stats["missing_captions"]) > 10:
                    msg += f"\n... and {len(stats['missing_captions']) - 10} more"
            
            QMessageBox.information(self, "Dataset Analysis", msg)
        
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error analyzing dataset: {str(e)}")


def main():
    app = QApplication(sys.argv)
    window = DatasetManagerGUI()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()