from PyQt6.QtWidgets import QTabWidget, QMessageBox
from training_widgets import TrainingWidgets
from flux_widgets import FluxTrainingWidgets
from pathlib import Path

class TrainingTabs(QTabWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # SDXL tab
        self.sdxl_widgets = TrainingWidgets()
        self.addTab(self.sdxl_widgets, "SDXL Training")
        
        # Flux tab
        self.flux_widgets = FluxTrainingWidgets()
        self.addTab(self.flux_widgets, "Flux Training")

        # Conecta os botões ao método de treinamento do parent
        if parent:
            self.sdxl_widgets.train_button.clicked.connect(parent.start_training)
            self.flux_widgets.train_button.clicked.connect(parent.start_training)

    def get_active_widgets(self):
        """Retorna os widgets ativos na aba atual"""
        return self.currentWidget()

    def get_mode(self):
        """Retorna o modo atual (SDXL ou Flux)"""
        return "flux" if isinstance(self.currentWidget(), FluxTrainingWidgets) else "sdxl"

    def validate_paths(self, dataset_path):
        """Valida os caminhos necessários dependendo do modo"""
        widget = self.get_active_widgets()
        mode = self.get_mode()

        # Verifica dataset.toml
        toml_path = Path(dataset_path) / "cropped_images/dataset.toml"
        if not toml_path.exists():
            QMessageBox.warning(None, "Error", "Please generate dataset.toml first!")
            return False

        if mode == "flux":
            # Validações específicas do Flux
            if not Path(widget.flux_path.text()).exists():
                QMessageBox.warning(None, "Error", "Invalid Flux model path!")
                return False
            if not Path(widget.clip_l_path.text()).exists():
                QMessageBox.warning(None, "Error", "Invalid CLIP-L model path!")
                return False
            if not Path(widget.t5xxl_path.text()).exists():
                QMessageBox.warning(None, "Error", "Invalid T5XXL model path!")
                return False
        else:
            # Validações do SDXL (reusando lógica existente)
            if not Path(widget.model_path.text()).exists():
                QMessageBox.warning(None, "Error", "Invalid model path!")
                return False
            if not widget.output_name.text().strip():
                QMessageBox.warning(None, "Error", "Output name cannot be empty!")
                return False

        return True

    def get_command(self, dataset_path):
        """Obtém o comando de treinamento dos widgets ativos"""
        if self.validate_paths(dataset_path):
            return self.get_active_widgets().get_command(dataset_path)
        return None

    def save_config(self):
        """Salva as configurações dos widgets ativos"""
        self.get_active_widgets().save_current_config()