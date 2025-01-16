from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
                            QFormLayout, QLineEdit, QPushButton, QSpinBox,
                            QCheckBox, QFileDialog, QLabel)
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
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
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

        # Network parameters
        network_group = QGroupBox("Network Parameters")
        network_layout = QFormLayout()
        
        self.network_dim = QSpinBox()
        self.network_dim.setRange(1, 128)
        self.network_dim.setValue(self.config.get("network_dim", 32))
        network_layout.addRow("Network Dimension:", self.network_dim)

        self.network_alpha = QSpinBox()
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

        self.epochs = QSpinBox()
        self.epochs.setRange(1, 1000)
        self.epochs.setValue(self.config.get("epochs", 32))
        training_layout.addRow("Max Epochs:", self.epochs)

        self.save_every = QSpinBox()
        self.save_every.setRange(1, 100)
        self.save_every.setValue(self.config.get("save_every", 32))
        training_layout.addRow("Save Every N Epochs:", self.save_every)

        self.seed = QSpinBox()
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

        self.persistent_workers = QCheckBox("Persistent Workers")
        self.persistent_workers.setChecked(self.config.get("persistent_workers", False))
        advanced_layout.addWidget(self.persistent_workers)

        self.max_workers = QSpinBox()
        self.max_workers.setRange(0, 16)
        self.max_workers.setValue(self.config.get("max_workers", 2))
        worker_layout = QHBoxLayout()
        worker_layout.addWidget(QLabel("Max Workers:"))
        worker_layout.addWidget(self.max_workers)
        advanced_layout.addLayout(worker_layout)

        advanced_group.setLayout(advanced_layout)
        layout.addWidget(advanced_group)

        # Output configuration
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

        # Start Training button
        self.train_button = QPushButton("Start Training")
        layout.addWidget(self.train_button)

        self.setLayout(layout)

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
            "persistent_workers": self.persistent_workers.isChecked(),
            "max_workers": self.max_workers.value(),
            "network_args": self.network_args.text(),
            "optimizer_args": self.optimizer_args.text()
        }
        save_config(config)

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

        filtered_cmd = filter(None, cmd)
        # Converte os itens para string e filtra os vazios
        cmd_str = " ".join(str(item) for item in filtered_cmd if str(item).strip())
        return cmd_str