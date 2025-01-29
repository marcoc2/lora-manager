import sys
import toml
from PIL import Image
from pathlib import Path
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QTreeView, QGroupBox, QPushButton, 
                            QDialog, QMessageBox, QLabel, QSpinBox,
                            QFileDialog, QProgressDialog, QFormLayout, QLineEdit,
                            QComboBox, QTextEdit, QCheckBox)
from PyQt6.QtGui import QStandardItemModel, QStandardItem
from PyQt6.QtCore import Qt, QProcess

from image_processor import ImageProcessor
from training_widgets import CommandOutputDialog
from dialogs import TomlConfigDialog, CaptionConfigDialog, ProcessProgressDialog, SuffixInputDialog
from caption_generator import CaptionGenerator
from danbooru_generator import DanbooruGenerator
from janus_generator import JanusGenerator
from training_tabs import TrainingTabs


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

# Modifique a classe CaptionConfigDialog para incluir a seleção do método:
class CaptionConfigDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Caption Configuration")
        self.setModal(True)
        
        layout = QFormLayout()
        
        # Adiciona seleção do método
        self.method_combo = QComboBox()
        self.method_combo.addItems(["Florence-2", "Danbooru", "Janus-7B"])
        layout.addRow("Captioning Method:", self.method_combo)
        
        # Campo para prefixo
        self.prefix = QLineEdit()
        layout.addRow("Caption Prefix:", self.prefix)
        
        # Para método Danbooru, adiciona seleção do modelo
        self.model_combo = QComboBox()
        self.model_combo.addItems(["vit", "swinv2", "convnext"])
        self.model_combo.setVisible(False)
        layout.addRow("Danbooru Model:", self.model_combo)
        
        # Para Janus-7B, adiciona campo de contexto e checkbox
        self.janus_context = QTextEdit()
        self.janus_context.setPlaceholderText("Enter additional context for Janus prompt (optional)")
        self.janus_context.setMaximumHeight(100)
        self.janus_context.setVisible(False)
        layout.addRow("Janus Context:", self.janus_context)
        
        self.replace_prompt = QCheckBox("Replace Default Prompt")
        self.replace_prompt.setVisible(False)
        layout.addRow("", self.replace_prompt)
        
        # Conecta evento de mudança do método
        self.method_combo.currentTextChanged.connect(self.on_method_changed)
        
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
    
    def on_method_changed(self, text):
        """Mostra/esconde opções específicas de cada método"""
        # Danbooru options
        self.model_combo.setVisible(text == "Danbooru")
        
        # Janus options
        self.janus_context.setVisible(text == "Janus-7B")
        self.replace_prompt.setVisible(text == "Janus-7B")
        
    def get_values(self):
        return {
            'method': self.method_combo.currentText(),
            'prefix': self.prefix.text(),
            'model_type': self.model_combo.currentText() if self.method_combo.currentText() == "Danbooru" else None,
            'janus_context': self.janus_context.toPlainText() if self.method_combo.currentText() == "Janus-7B" else None,
            'replace_prompt': self.replace_prompt.isChecked() if self.method_combo.currentText() == "Janus-7B" else False
        }

class DatasetManagerGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Dataset Manager")
        self.setGeometry(100, 100, 1300, 900)  # Aumentei a largura para acomodar os widgets de treino
        
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
        
        # Painel esquerdo - TreeView e seleção de pasta
        left_panel = QWidget()
        left_layout = QVBoxLayout()
        
        # Botão para selecionar pasta
        select_button = QPushButton("Select Dataset Folder")
        select_button.clicked.connect(self.select_dataset_folder)
        left_layout.addWidget(select_button)
        
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
        center_layout.setSpacing(10)  # Espaço entre os grupos
        
        # 1. Grupo de Processamento de Imagens
        image_group = QGroupBox("1. Image Processing")
        image_group.setMinimumHeight(150)  # Altura mínima fixa
        image_layout = QVBoxLayout()
        
        # Configurações de tamanho
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
        
        # Face detection toggle
        self.face_detection = QPushButton("Face Detection: ON")
        self.face_detection.setCheckable(True)
        self.face_detection.setChecked(True)
        self.face_detection.clicked.connect(self.toggle_face_detection)
        image_layout.addWidget(self.face_detection)
        
        # Botão de processamento
        process_button = QPushButton("Process Images")
        process_button.clicked.connect(self.process_images)
        image_layout.addWidget(process_button)
        
        image_group.setLayout(image_layout)
        center_layout.addWidget(image_group)
        
        # 2. Grupo de Geração de Captions
        caption_group = QGroupBox("2. Caption Generation")
        caption_group.setMinimumHeight(100)  # Menor que o grupo de processamento
        caption_layout = QVBoxLayout()
        caption_layout.setContentsMargins(10, 15, 10, 15)  # Margens internas
        
        generate_captions_btn = QPushButton("Generate Captions")
        generate_captions_btn.clicked.connect(self.generate_captions)
        caption_layout.addWidget(generate_captions_btn)
        
        caption_group.setLayout(caption_layout)
        center_layout.addWidget(caption_group)
        
        # 3. Grupo de Configuração do Dataset
        dataset_group = QGroupBox("3. Dataset Configuration")
        dataset_group.setMinimumHeight(100)  # Mesmo tamanho do grupo de captions
        dataset_layout = QVBoxLayout()
        dataset_layout.setContentsMargins(10, 15, 10, 15)  # Margens internas
        
        generate_toml_btn = QPushButton("Generate dataset.toml")
        generate_toml_btn.clicked.connect(self.generate_toml)
        dataset_layout.addWidget(generate_toml_btn)
        
        dataset_group.setLayout(dataset_layout)
        center_layout.addWidget(dataset_group)
        
        # 4. Grupo de Utilitários
        utils_group = QGroupBox("Utilities")
        utils_group.setMinimumHeight(120)  # Um pouco maior por ter mais botões
        utils_layout = QVBoxLayout()
        utils_layout.setContentsMargins(10, 15, 10, 15)  # Margens internas
        utils_layout.setSpacing(8)  # Espaço entre os botões
        
        rename_convert_btn = QPushButton("Rename and Convert Images")
        rename_convert_btn.clicked.connect(self.rename_and_convert_images)
        utils_layout.addWidget(rename_convert_btn)
        
        analyze_btn = QPushButton("Analyze Dataset")
        analyze_btn.clicked.connect(self.analyze_dataset)
        utils_layout.addWidget(analyze_btn)
        
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
        #self.training_widgets.train_button.clicked.connect(self.start_training)
        
        # Define proporção dos painéis (25/40/35)
        layout.addWidget(left_panel, 25)
        layout.addWidget(center_panel, 10)
        layout.addWidget(self.training_tabs, 65)
        
        central_widget.setLayout(layout)

    def toggle_face_detection(self):
        """Alterna detecção facial"""
        if self.face_detection.isChecked():
            self.face_detection.setText("Face Detection: ON")
        else:
            self.face_detection.setText("Face Detection: OFF")
            
    def select_dataset_folder(self):
        """Seleciona a pasta do dataset"""
        folder = QFileDialog.getExistingDirectory(self, "Select Dataset Folder")
        if folder:
            self.dataset_path = Path(folder).absolute()
            self.populate_tree_view(self.dataset_path)
            self.update_status()

    def populate_tree_view(self, path):
        """Popula a TreeView com a estrutura do dataset"""
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

    def process_images(self):
        """Processa imagens do dataset"""
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
        """Gera captions para as imagens"""
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
            
            print(f"Selected method: {config['method']}")
            print(f"Selected model type: {config.get('model_type')}")
            
            progress = QProgressDialog("Generating captions...", "Cancel", 0, 100, self)
            progress.setWindowModality(Qt.WindowModality.WindowModal)
            progress.setAutoClose(True)
            progress.show()
            
            def update_progress(message: str, value: int):
                if value >= 0:
                    progress.setLabelText(message)
                    progress.setValue(value)
            
            captions_dir = cropped_dir / "captions"
            
            # Escolhe o gerador apropriado
            if config['method'] == "Florence-2":
                print("Using Florence-2 generator")
                generator = CaptionGenerator()
            elif config['method'] == "Danbooru":
                print("Using Danbooru generator")
                model_type = config.get('model_type')
                if model_type not in ['vit', 'swinv2', 'convnext']:
                    model_type = 'vit'  # default to 'vit' if invalid
                generator = DanbooruGenerator(model_type=model_type)
                print(f"Initialized Danbooru generator with model: {model_type}")
            else:  # Janus-7B
                print("Using Janus-7B generator")
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
        """Gera o arquivo dataset.toml"""
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
        self.tree_model.clear()
        self.populate_tree_view(self.dataset_path)

    def analyze_dataset(self):
        """Analisa estado atual do dataset"""
        if not self.dataset_path:
            QMessageBox.warning(self, "Warning", "Please select a dataset folder first!")
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

    def start_training(self):
        """Inicia o treinamento com as configurações atuais"""
        if not self.dataset_path:
            QMessageBox.warning(self, "Warning", "Please select a dataset folder first!")
            return
        
        try:
            # Salva configuração atual
            self.training_tabs.save_config()
            
            # Gera e mostra comando
            command = self.training_tabs.get_command(self.dataset_path)
            if command is None:
                return
            
            msg = QMessageBox()
            msg.setWindowTitle("Training Command")
            msg.setText("The following command will be executed:")
            msg.setDetailedText(command)
            msg.setStandardButtons(QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel)
            
            if msg.exec() == QMessageBox.StandardButton.Ok:
                # Executa treinamento
                output_dialog = CommandOutputDialog(command, self)
                output_dialog.exec()
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error starting training: {str(e)}")

def main():
    app = QApplication(sys.argv)
    window = DatasetManagerGUI()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()