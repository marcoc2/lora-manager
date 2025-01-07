from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QGroupBox,
                            QFormLayout, QLineEdit, QPushButton, QSpinBox,
                            QCheckBox, QFileDialog)
from pathlib import Path

class TrainingConfigDialog(QDialog):
    def __init__(self, dataset_path, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Training Configuration")
        self.setModal(True)
        self.dataset_path = dataset_path
        
        # Layout principal
        layout = QVBoxLayout()
        
        # Modelo base
        model_group = QGroupBox("Base Model")
        model_layout = QHBoxLayout()
        self.model_path = QLineEdit()
        self.model_path.setPlaceholderText("Path to base model")
        select_model = QPushButton("Browse")
        select_model.clicked.connect(self.select_model_path)
        model_layout.addWidget(self.model_path)
        model_layout.addWidget(select_model)
        model_group.setLayout(model_layout)
        layout.addWidget(model_group)
        
        # Parâmetros de rede
        network_group = QGroupBox("Network Parameters")
        network_layout = QFormLayout()
        
        self.network_dim = QSpinBox()
        self.network_dim.setRange(1, 128)
        self.network_dim.setValue(16)
        network_layout.addRow("Network Dimension:", self.network_dim)
        
        self.network_alpha = QSpinBox()
        self.network_alpha.setRange(1, 128)
        self.network_alpha.setValue(16)
        network_layout.addRow("Network Alpha:", self.network_alpha)
        
        network_group.setLayout(network_layout)
        layout.addWidget(network_group)
        
        # Parâmetros de treinamento
        training_group = QGroupBox("Training Parameters")
        training_layout = QFormLayout()
        
        self.learning_rate = QLineEdit("1e-3")
        training_layout.addRow("Learning Rate:", self.learning_rate)
        
        self.epochs = QSpinBox()
        self.epochs.setRange(1, 1000)
        self.epochs.setValue(32)
        training_layout.addRow("Max Epochs:", self.epochs)
        
        self.save_every = QSpinBox()
        self.save_every.setRange(1, 100)
        self.save_every.setValue(16)
        training_layout.addRow("Save Every N Epochs:", self.save_every)
        
        self.seed = QSpinBox()
        self.seed.setRange(1, 999999)
        self.seed.setValue(42)
        training_layout.addRow("Seed:", self.seed)
        
        training_group.setLayout(training_layout)
        layout.addWidget(training_group)
        
        # Opções avançadas
        advanced_group = QGroupBox("Advanced Options")
        advanced_layout = QVBoxLayout()
        
        self.cache_latents = QCheckBox("Cache Latents to Disk")
        self.cache_latents.setChecked(True)
        advanced_layout.addWidget(self.cache_latents)
        
        self.cache_text_encoder = QCheckBox("Cache Text Encoder Outputs")
        self.cache_text_encoder.setChecked(True)
        advanced_layout.addWidget(self.cache_text_encoder)
        
        self.gradient_checkpointing = QCheckBox("Gradient Checkpointing")
        self.gradient_checkpointing.setChecked(True)
        advanced_layout.addWidget(self.gradient_checkpointing)
        
        self.sdpa = QCheckBox("SDPA")
        self.sdpa.setChecked(True)
        advanced_layout.addWidget(self.sdpa)
        
        advanced_group.setLayout(advanced_layout)
        layout.addWidget(advanced_group)
        
        # Output
        output_group = QGroupBox("Output Configuration")
        output_layout = QHBoxLayout()
        self.output_dir = QLineEdit()
        self.output_dir.setPlaceholderText("Output directory")
        select_output = QPushButton("Browse")
        select_output.clicked.connect(self.select_output_path)
        output_layout.addWidget(self.output_dir)
        output_layout.addWidget(select_output)
        output_group.setLayout(output_layout)
        layout.addWidget(output_group)
        
        # Botões
        buttons = QHBoxLayout()
        ok_button = QPushButton("Start Training")
        cancel_button = QPushButton("Cancel")
        
        ok_button.clicked.connect(self.accept)
        cancel_button.clicked.connect(self.reject)
        
        buttons.addWidget(ok_button)
        buttons.addWidget(cancel_button)
        layout.addLayout(buttons)
        
        self.setLayout(layout)

    def select_model_path(self):
        path = QFileDialog.getOpenFileName(self, "Select Base Model", 
                                         filter="Model files (*.safetensors)")[0]
        if path:
            self.model_path.setText(path)
    
    def select_output_path(self):
        path = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if path:
            self.output_dir.setText(path)
            
    def get_command(self):
        """Gera o comando de treinamento com base nas configurações"""
        dataset_config = self.dataset_path / "cropped_images/dataset.toml"
        
        cmd = [
            "accelerate launch",
            "--mixed_precision bf16",
            "--num_cpu_threads_per_process 1",
            "sdxl_train_network.py",
            f"--pretrained_model_name_or_path \"{self.model_path.text()}\"",
            "--cache_latents_to_disk" if self.cache_latents.isChecked() else "",
            "--save_model_as safetensors",
            "--sdpa" if self.sdpa.isChecked() else "",
            "--persistent_data_loader_workers",
            "--max_data_loader_n_workers 2",
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
            f"--dataset_config \"{dataset_config}\"",
            f"--output_dir \"{self.output_dir.text()}\""
        ]
        
        return " ".join(filter(None, cmd))
