from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QScrollArea, QSpinBox)
from PyQt6.QtCore import Qt
from pathlib import Path
import json

CONFIG_FILE = "qwen_config.json"

def load_qwen_config():
    if Path(CONFIG_FILE).exists():
        with open(CONFIG_FILE, "r") as file:
            return json.load(file)
    return {}

def save_config(config):
    with open(CONFIG_FILE, "w") as file:
        json.dump(config, file, indent=2)

class NoWheelSpinBox(QSpinBox):
    def wheelEvent(self, event):
        event.ignore()

class QwenTrainingWidgetsBase(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.config = load_qwen_config()
        
        # Criar QScrollArea
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        
        # Criar widget para conter todos os controles
        container = QWidget()
        
        # Layout principal para o container
        main_layout = QVBoxLayout(self)
        main_layout.addWidget(scroll)
        
        # Configurar o container como widget do scroll
        scroll.setWidget(container)
        
        # Layout para os controles (será usado em init_ui)
        self.control_layout = QVBoxLayout(container)

    def save_current_config(self):
        config = self.get_current_config()
        save_config(config)

    def get_current_config(self):
        """Should be implemented by subclasses"""
        return {}

    def get_command(self, dataset_path):
        """Should be implemented by subclasses"""
        return None