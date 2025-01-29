from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
                            QFormLayout, QLineEdit, QPushButton, QSpinBox,
                            QCheckBox, QFileDialog, QLabel, QMessageBox, QScrollArea)
from PyQt6.QtCore import Qt
from pathlib import Path
import json
from command_utils import CommandOutputDialog, ScriptManager, format_command_args

CONFIG_FILE = "training_config.json"

def load_config():
    if Path(CONFIG_FILE).exists():
        with open(CONFIG_FILE, "r") as file:
            return json.load(file)
    return {}

def save_config(config):
    with open(CONFIG_FILE, "w") as file:
        json.dump(config, file)

class TrainingWidgets(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.config = load_config()
        self.script_manager = ScriptManager()
        
        # Criar SpinBox que nunca aceita rolagem
        class NoWheelSpinBox(QSpinBox):
            def wheelEvent(self, event):
                event.ignore()  # Sempre ignora eventos de rolagem
        
        # Guardar a classe para usar no init_ui
        self.SpinBox = NoWheelSpinBox
        
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
        
        self.init_ui()

    def init_ui(self):
        layout = self.control_layout
        layout.setSpacing(10)

        # Modelo base
        model_group = QGroupBox("Base Model")
        model_layout = QHBoxLayout()
        self.model_path = QLineEdit()
        self.model_path.setPlaceholderText("Path to base model")
        self.model_path.setText(self.config.get("model_path", ""))
        select_model = QPushButton("Browse")
        select_model.clicked.connect(self.select_model_path)
        model_layout.addWidget(self.model_path)
        model_layout.addWidget(select_model)
        model_group.setLayout(model_layout)
        layout.addWidget(model_group)

        # Scripts directory
        scripts_group = QGroupBox("Scripts Directory")
        scripts_layout = QHBoxLayout()
        self.scripts_dir = QLineEdit()
        self.scripts_dir.setPlaceholderText("Path to sd_scripts folder")
        self.scripts_dir.setText(self.config.get("scripts_dir", ""))
        select_scripts = QPushButton("Browse")
        select_scripts.clicked.connect(self.select_scripts_path)
        scripts_layout.addWidget(self.scripts_dir)
        scripts_layout.addWidget(select_scripts)
        scripts_group.setLayout(scripts_layout)
        layout.addWidget(scripts_group)

        # Resume Training
        resume_group = QGroupBox("Resume Training")
        resume_layout = QFormLayout()
        
        self.resume_checkbox = QCheckBox("Resume from weights")
        self.resume_checkbox.setChecked(self.config.get("resume_training", False))
        self.resume_path = QLineEdit()
        self.resume_path.setEnabled(self.resume_checkbox.isChecked())
        self.resume_path.setPlaceholderText("Path to network weights file (.safetensors or .pt)")
        self.resume_path.setText(self.config.get("resume_path", ""))
        select_resume = QPushButton("Browse")
        select_resume.setEnabled(self.resume_checkbox.isChecked())
        
        resume_path_layout = QHBoxLayout()
        resume_path_layout.addWidget(self.resume_path)
        resume_path_layout.addWidget(select_resume)
        
        resume_layout.addRow(self.resume_checkbox)
        resume_layout.addRow("Weights:", resume_path_layout)
        
        self.resume_checkbox.stateChanged.connect(lambda state: [
            self.resume_path.setEnabled(state == Qt.CheckState.Checked.value),
            select_resume.setEnabled(state == Qt.CheckState.Checked.value)
        ])
        select_resume.clicked.connect(self.select_resume_path)
        
        resume_group.setLayout(resume_layout)
        layout.addWidget(resume_group)

        # Network parameters
        network_group = QGroupBox("Network Parameters")
        network_layout = QFormLayout()
        
        self.network_dim = self.SpinBox()
        self.network_dim.setRange(1, 128)
        self.network_dim.setValue(self.config.get("network_dim", 32))
        network_layout.addRow("Network Dimension:", self.network_dim)

        self.network_alpha = self.SpinBox()
        self.network_alpha.setRange(1, 128)
        self.network_alpha.setValue(self.config.get("network_alpha", 16))
        network_layout.addRow("Network Alpha:", self.network_alpha)

        self.network_args = QLineEdit(self.config.get("network_args", ""))
        network_layout.addRow("Network Arguments:", self.network_args)

        network_group.setLayout(network_layout)
        layout.addWidget(network_group)

        # Training parameters
        training_group = QGroupBox("Training Parameters")
        training_layout = QFormLayout()

        self.learning_rate = QLineEdit(self.config.get("learning_rate", "1e-4"))
        training_layout.addRow("Learning Rate:", self.learning_rate)

        self.epochs = self.SpinBox()
        self.epochs.setRange(1, 1000)
        self.epochs.setValue(self.config.get("epochs", 32))
        training_layout.addRow("Max Epochs:", self.epochs)

        self.save_every = self.SpinBox()
        self.save_every.setRange(1, 100)
        self.save_every.setValue(self.config.get("save_every", 32))
        training_layout.addRow("Save Every N Epochs:", self.save_every)

        self.seed = self.SpinBox()
        self.seed.setRange(1, 999999)
        self.seed.setValue(self.config.get("seed", 42))
        training_layout.addRow("Seed:", self.seed)

        self.optimizer_args = QLineEdit(self.config.get("optimizer_args", ""))
        training_layout.addRow("Optimizer Arguments:", self.optimizer_args)

        training_group.setLayout(training_layout)
        layout.addWidget(training_group)

        # Advanced options
        advanced_group = QGroupBox("Advanced Options")
        advanced_layout = QVBoxLayout()

        self.cache_latents = QCheckBox("Cache Latents to Disk")
        self.cache_latents.setChecked(self.config.get("cache_latents", True))
        advanced_layout.addWidget(self.cache_latents)

        self.cache_text_encoder = QCheckBox("Cache Text Encoder Outputs")
        self.cache_text_encoder.setChecked(self.config.get("cache_text_encoder", True))
        advanced_layout.addWidget(self.cache_text_encoder)

        self.gradient_checkpointing = QCheckBox("Gradient Checkpointing")
        self.gradient_checkpointing.setChecked(self.config.get("gradient_checkpointing", True))
        advanced_layout.addWidget(self.gradient_checkpointing)

        self.sdpa = QCheckBox("SDPA")
        self.sdpa.setChecked(self.config.get("sdpa", True))
        advanced_layout.addWidget(self.sdpa)

        self.flip_aug = QCheckBox("Flip Augmentation")
        self.flip_aug.setChecked(self.config.get("flip_aug", False))
        advanced_layout.addWidget(self.flip_aug)

        self.persistent_workers = QCheckBox("Persistent Workers")
        self.persistent_workers.setChecked(self.config.get("persistent_workers", False))
        advanced_layout.addWidget(self.persistent_workers)

        self.max_workers = self.SpinBox()
        self.max_workers.setRange(0, 16)
        self.max_workers.setValue(self.config.get("max_workers", 2))
        worker_layout = QHBoxLayout()
        worker_layout.addWidget(QLabel("Max Workers:"))
        worker_layout.addWidget(self.max_workers)
        advanced_layout.addLayout(worker_layout)

        advanced_group.setLayout(advanced_layout)
        layout.addWidget(advanced_group)

        # Output Configuration
        output_group = QGroupBox("Output Configuration")
        output_layout = QVBoxLayout()

        output_dir_layout = QHBoxLayout()
        self.output_dir = QLineEdit()
        self.output_dir.setPlaceholderText("Output directory")
        self.output_dir.setText(self.config.get("output_dir", ""))
        select_output = QPushButton("Browse")
        select_output.clicked.connect(self.select_output_path)
        output_dir_layout.addWidget(self.output_dir)
        output_dir_layout.addWidget(select_output)

        self.output_name = QLineEdit()
        self.output_name.setPlaceholderText("Output file name (without extension)")
        self.output_name.setText(self.config.get("output_name", ""))

        output_layout.addWidget(QLabel("Output Directory:"))
        output_layout.addLayout(output_dir_layout)
        output_layout.addWidget(QLabel("Output Name:"))
        output_layout.addWidget(self.output_name)

        output_group.setLayout(output_layout)
        layout.addWidget(output_group)

        # Additional Parameters
        params_group = QGroupBox("Additional Parameters")
        params_layout = QVBoxLayout()
        
        self.additional_params = QLineEdit()
        self.additional_params.setPlaceholderText("Additional command line parameters (e.g., --param1 value1 --param2 value2)")
        self.additional_params.setText(self.config.get("additional_params", ""))
        
        params_layout.addWidget(self.additional_params)
        params_group.setLayout(params_layout)
        layout.addWidget(params_group)

        # Start Training button
        self.train_button = QPushButton("Start Training")
        layout.addWidget(self.train_button)

    def select_model_path(self):
        path = QFileDialog.getOpenFileName(self, "Select Base Model", 
                                         filter="Model files (*.safetensors)")[0]
        if path:
            self.model_path.setText(path)
            self.save_current_config()

    def select_scripts_path(self):
        path = QFileDialog.getExistingDirectory(self, "Select Scripts Directory")
        if path:
            self.scripts_dir.setText(path)
            self.save_current_config()

    def select_resume_path(self):
        path = QFileDialog.getOpenFileName(self, "Select Network Weights File", 
                                         filter="Model files (*.safetensors *.pt)")[0]
        if path:
            self.resume_path.setText(path)
            self.save_current_config()

    def select_output_path(self):
        path = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if path:
            self.output_dir.setText(path)
            self.save_current_config()

    def cleanup_temp_files(self):
        """Limpa arquivos temporários"""
        self.script_manager.cleanup_temp_files()

    def save_current_config(self):
        """Salva a configuração atual no arquivo JSON"""
        config = {
            "model_path": self.model_path.text(),
            "scripts_dir": self.scripts_dir.text(),
            "output_dir": self.output_dir.text(),
            "output_name": self.output_name.text(),
            "network_dim": self.network_dim.value(),
            "network_alpha": self.network_alpha.value(),
            "learning_rate": self.learning_rate.text(),
            "epochs": self.epochs.value(),
            "save_every": self.save_every.value(),
            "seed": self.seed.value(),
            "cache_latents": self.cache_latents.isChecked(),
            "cache_text_encoder": self.cache_text_encoder.isChecked(),
            "gradient_checkpointing": self.gradient_checkpointing.isChecked(),
            "sdpa": self.sdpa.isChecked(),
            "flip_aug": self.flip_aug.isChecked(),
            "persistent_workers": self.persistent_workers.isChecked(),
            "max_workers": self.max_workers.value(),
            "network_args": self.network_args.text(),
            "optimizer_args": self.optimizer_args.text(),
            "resume_training": self.resume_checkbox.isChecked(),
            "resume_path": self.resume_path.text(),
            "additional_params": self.additional_params.text()
        }
        save_config(config)

    def validate_paths(self):
        """Valida os caminhos necessários"""
        if not Path(self.model_path.text()).is_file():
            QMessageBox.critical(self, "Error", "Invalid model path.")
            return False
        if not Path(self.output_dir.text()).is_dir():
            QMessageBox.critical(self, "Error", "Invalid output directory.")
            return False
        if not self.output_name.text().strip():
            QMessageBox.critical(self, "Error", "Output name cannot be empty.")
            return False
        script_path = Path(self.scripts_dir.text()) / "sdxl_train_network.py"
        if not script_path.is_file():
            QMessageBox.critical(self, "Error", f"Training script not found in: {script_path}")
            return False
        if self.resume_checkbox.isChecked() and not Path(self.resume_path.text()).is_file():
            QMessageBox.critical(self, "Error", "Invalid network weights file.")
            return False
        return True

    def get_command(self, dataset_path):
        """Gera o comando de treinamento com base nas configurações"""
        dataset_config = dataset_path / "cropped_images/dataset.toml"

        original_script_path = Path(self.scripts_dir.text()) / "sdxl_train_network.py"
        script_path = self.script_manager.create_temp_script(original_script_path)

        cmd = [
            "accelerate launch",
            "--num_cpu_threads_per_process 1",
            str(script_path),
            f"--pretrained_model_name_or_path {self.model_path.text()}",
            "--cache_latents_to_disk" if self.cache_latents.isChecked() else "",
            f"--save_model_as safetensors",
            "--sdpa" if self.sdpa.isChecked() else "",
            "--persistent_data_loader_workers" if self.persistent_workers.isChecked() and self.max_workers.value() > 0 else "",
            f"--max_data_loader_n_workers {self.max_workers.value()}",
            f"--seed {self.seed.value()}",
            "--gradient_checkpointing" if self.gradient_checkpointing.isChecked() else "",
            "--mixed_precision bf16",
            "--save_precision bf16",
            "--network_module networks.lora",
            f"--network_dim {self.network_dim.value()}",
            f"--network_alpha {self.network_alpha.value()}",
            "--optimizer_type adafactor",
            f"--learning_rate {self.learning_rate.text()}",
            "--network_train_unet_only",
            "--cache_text_encoder_outputs" if self.cache_text_encoder.isChecked() else "",
            "--cache_text_encoder_outputs_to_disk" if self.cache_text_encoder.isChecked() else "",
            "--flip_aug" if self.flip_aug.isChecked() else "",
            f"--max_train_epochs {self.epochs.value()}",
            f"--save_every_n_epochs {self.save_every.value()}",
            f"--dataset_config {dataset_config}",
            f"--output_dir {self.output_dir.text()}" if self.output_dir.text() else "",
            f"--output_name {self.output_name.text()}" if self.output_name.text() else ""
        ]

        # Adiciona optimizer args se houver
        optimizer_args = self.optimizer_args.text().strip()
        if optimizer_args:
            cmd.append("--optimizer_args")
            cmd.extend(format_command_args(optimizer_args))

        # Adiciona network args se houver
        network_args = self.network_args.text().strip()
        if network_args:
            cmd.append("--network_args")
            cmd.extend(format_command_args(network_args))

        # Adiciona opção de network_weights se marcado
        if self.resume_checkbox.isChecked() and self.resume_path.text().strip():
            resume_path = self.resume_path.text().strip()
            cmd.append(f"--network_weights {resume_path}")

        # Adiciona parâmetros adicionais se houver
        additional_params = self.additional_params.text().strip()
        if additional_params:
            cmd.extend(additional_params.split())

        filtered_cmd = filter(None, cmd)
        # Converte os itens para string e filtra os vazios
        cmd_str = " ".join(str(item) for item in filtered_cmd if str(item).strip())
        return cmd_str