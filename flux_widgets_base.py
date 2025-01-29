from PyQt6.QtWidgets import QWidget, QVBoxLayout, QScrollArea, QSpinBox
from PyQt6.QtCore import Qt
from pathlib import Path
import json
from command_utils import ScriptManager  # Corrigido o nome do import

CONFIG_FILE = "flux_config.json"

def load_config():
    if Path(CONFIG_FILE).exists():
        with open(CONFIG_FILE, "r") as file:
            return json.load(file)
    return {}

def save_config(config):
    with open(CONFIG_FILE, "w") as file:
        json.dump(config, file)

class NoWheelSpinBox(QSpinBox):
    def wheelEvent(self, event):
        event.ignore()  # Sempre ignora eventos de rolagem

class FluxTrainingWidgetsBase(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.config = load_config()
        self.script_manager = ScriptManager()
        
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

    def cleanup_temp_files(self):
        """Limpa arquivos temporários"""
        self.script_manager.cleanup_temp_files()

    def get_command(self, dataset_path):
        """Este método deve ser sobrescrito pela classe filha"""
        raise NotImplementedError("Método get_command deve ser implementado pela classe filha")

    def save_current_config(self):
        """Este método deve ser sobrescrito pela classe filha"""
        raise NotImplementedError("Método save_current_config deve ser implementado pela classe filha")