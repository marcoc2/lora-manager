from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
                            QFormLayout, QLineEdit, QPushButton, QSpinBox,
                            QCheckBox, QFileDialog, QLabel)
from PyQt6.QtCore import Qt
from pathlib import Path
import json

CONFIG_FILE = "flux_config.json"

def load_config():
    if Path(CONFIG_FILE).exists():
        with open(CONFIG_FILE, "r") as file:
            return json.load(file)
    return {}

def save_config(config):
    with open(CONFIG_FILE, "w") as file:
        json.dump(config, file)

class FluxTrainingWidgets(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.config = load_config()
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        layout.setSpacing(10)

        # Base Models
        model_group = QGroupBox("Base Models")
        model_layout = QVBoxLayout()
        
        # Flux model
        flux_layout = QHBoxLayout()
        self.flux_path = QLineEdit()
        self.flux_path.setPlaceholderText("Path to Flux model")
        self.flux_path.setText(self.config.get("flux_path", ""))
        select_flux = QPushButton("Browse")
        select_flux.clicked.connect(self.select_flux_path)
        flux_layout.addWidget(self.flux_path)
        flux_layout.addWidget(select_flux)
        model_layout.addWidget(QLabel("Flux Model:"))
        model_layout.addLayout(flux_layout)

        # CLIP-L model
        clip_layout = QHBoxLayout()
        self.clip_l_path = QLineEdit()
        self.clip_l_path.setPlaceholderText("Path to CLIP-L model")
        self.clip_l_path.setText(self.config.get("clip_l_path", ""))
        select_clip = QPushButton("Browse")
        select_clip.clicked.connect(self.select_clip_path)
        clip_layout.addWidget(self.clip_l_path)
        clip_layout.addWidget(select_clip)
        model_layout.addWidget(QLabel("CLIP-L Model:"))
        model_layout.addLayout(clip_layout)

        # T5XXL model
        t5_layout = QHBoxLayout()
        self.t5xxl_path = QLineEdit()
        self.t5xxl_path.setPlaceholderText("Path to T5XXL model")
        self.t5xxl_path.setText(self.config.get("t5xxl_path", ""))
        select_t5 = QPushButton("Browse")
        select_t5.clicked.connect(self.select_t5_path)
        t5_layout.addWidget(self.t5xxl_path)
        t5_layout.addWidget(select_t5)
        model_layout.addWidget(QLabel("T5XXL Model:"))
        model_layout.addLayout(t5_layout)
        
        model_group.setLayout(model_layout)
        layout.addWidget(model_group)

        # Resume Training
        resume_group = QGroupBox("Resume Training")
        resume_layout = QFormLayout()
        
        self.resume_checkbox = QCheckBox("Resume from checkpoint")
        self.resume_path = QLineEdit()
        self.resume_path.setEnabled(False)
        self.resume_path.setPlaceholderText("Path to checkpoint folder")
        select_resume = QPushButton("Browse")
        select_resume.setEnabled(False)
        
        resume_path_layout = QHBoxLayout()
        resume_path_layout.addWidget(self.resume_path)
        resume_path_layout.addWidget(select_resume)
        
        resume_layout.addRow(self.resume_checkbox)
        resume_layout.addRow("Checkpoint:", resume_path_layout)
        
        self.resume_checkbox.stateChanged.connect(lambda state: [
            self.resume_path.setEnabled(state == Qt.CheckState.Checked.value),
            select_resume.setEnabled(state == Qt.CheckState.Checked.value)
        ])
        select_resume.clicked.connect(self.select_resume_path)
        
        resume_group.setLayout(resume_layout)
        layout.addWidget(resume_group)

        # Flux Parameters
        flux_group = QGroupBox("Flux Parameters")
        flux_layout = QFormLayout()
        
        self.guidance_scale = QSpinBox()
        self.guidance_scale.setRange(1, 20)
        self.guidance_scale.setValue(self.config.get("guidance_scale", 7))
        flux_layout.addRow("Guidance Scale:", self.guidance_scale)
        
        self.discrete_flow_shift = QCheckBox()
        self.discrete_flow_shift.setChecked(self.config.get("discrete_flow_shift", False))
        flux_layout.addRow("Discrete Flow Shift:", self.discrete_flow_shift)
        
        self.apply_t5_attn_mask = QCheckBox()
        self.apply_t5_attn_mask.setChecked(self.config.get("apply_t5_attn_mask", False))
        flux_layout.addRow("Apply T5 Attention Mask:", self.apply_t5_attn_mask)

        self.t5xxl_max_token_length = QSpinBox()
        self.t5xxl_max_token_length.setRange(64, 1024)
        self.t5xxl_max_token_length.setValue(self.config.get("t5xxl_max_token_length", 256))
        flux_layout.addRow("T5XXL Max Token Length:", self.t5xxl_max_token_length)
        
        flux_group.setLayout(flux_layout)
        layout.addWidget(flux_group)

        # Memory Optimization
        memory_group = QGroupBox("Memory Optimization")
        memory_layout = QFormLayout()
        
        self.blocks_to_swap = QSpinBox()
        self.blocks_to_swap.setRange(0, 32)
        self.blocks_to_swap.setValue(self.config.get("blocks_to_swap", 0))
        memory_layout.addRow("Blocks to Swap:", self.blocks_to_swap)
        
        self.blockwise_fused_optimizers = QCheckBox()
        self.blockwise_fused_optimizers.setChecked(self.config.get("blockwise_fused_optimizers", False))
        memory_layout.addRow("Blockwise Fused Optimizers:", self.blockwise_fused_optimizers)
        
        self.cpu_offload = QCheckBox()
        self.cpu_offload.setChecked(self.config.get("cpu_offload", False))
        memory_layout.addRow("CPU Offload Checkpointing:", self.cpu_offload)
        
        memory_group.setLayout(memory_layout)
        layout.addWidget(memory_group)

        # Network parameters (similar to SDXL for compatibility)
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

        network_group.setLayout(network_layout)
        layout.addWidget(network_group)

        # Training Parameters
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

        training_group.setLayout(training_layout)
        layout.addWidget(training_group)

        # Start Training button
        self.train_button = QPushButton("Start Training")
        layout.addWidget(self.train_button)

        self.setLayout(layout)

    def select_flux_path(self):
        path = QFileDialog.getOpenFileName(self, "Select Flux Model", 
                                         filter="Model files (*.safetensors)")[0]
        if path:
            self.flux_path.setText(path)
            self.save_current_config()

    def select_clip_path(self):
        path = QFileDialog.getOpenFileName(self, "Select CLIP-L Model", 
                                         filter="Model files (*.safetensors)")[0]
        if path:
            self.clip_l_path.setText(path)
            self.save_current_config()

    def select_t5_path(self):
        path = QFileDialog.getOpenFileName(self, "Select T5XXL Model", 
                                         filter="Model files (*.safetensors)")[0]
        if path:
            self.t5xxl_path.setText(path)
            self.save_current_config()

    def select_resume_path(self):
        path = QFileDialog.getExistingDirectory(self, "Select Checkpoint Directory")
        if path:
            self.resume_path.setText(path)

    def save_current_config(self):
        """Salva a configuração atual no arquivo JSON"""
        config = {
            "flux_path": self.flux_path.text(),
            "clip_l_path": self.clip_l_path.text(),
            "t5xxl_path": self.t5xxl_path.text(),
            "guidance_scale": self.guidance_scale.value(),
            "discrete_flow_shift": self.discrete_flow_shift.isChecked(),
            "apply_t5_attn_mask": self.apply_t5_attn_mask.isChecked(),
            "t5xxl_max_token_length": self.t5xxl_max_token_length.value(),
            "blocks_to_swap": self.blocks_to_swap.value(),
            "blockwise_fused_optimizers": self.blockwise_fused_optimizers.isChecked(),
            "cpu_offload": self.cpu_offload.isChecked(),
            "network_dim": self.network_dim.value(),
            "network_alpha": self.network_alpha.value(),
            "learning_rate": self.learning_rate.text(),
            "epochs": self.epochs.value(),
            "save_every": self.save_every.value(),
            "seed": self.seed.value()
        }
        save_config(config)

    def get_command(self, dataset_path):
        """Gera o comando de treinamento para o Flux"""
        dataset_config = dataset_path / "cropped_images/dataset.toml"

        cmd = [
            "accelerate launch",
            "flux_train.py",
            f"--pretrained_model_name_or_path {self.flux_path.text()}",
            f"--clip_l {self.clip_l_path.text()}",
            f"--t5xxl {self.t5xxl_path.text()}",
            f"--guidance_scale {self.guidance_scale.value()}",
            "--discrete_flow_shift" if self.discrete_flow_shift.isChecked() else "",
            "--apply_t5_attn_mask" if self.apply_t5_attn_mask.isChecked() else "",
            f"--t5xxl_max_token_length {self.t5xxl_max_token_length.value()}",
            f"--blocks_to_swap {self.blocks_to_swap.value()}" if self.blocks_to_swap.value() > 0 else "",
            "--blockwise_fused_optimizers" if self.blockwise_fused_optimizers.isChecked() else "",
            "--cpu_offload_checkpointing" if self.cpu_offload.isChecked() else "",
            f"--network_dim {self.network_dim.value()}",
            f"--network_alpha {self.network_alpha.value()}",
            f"--learning_rate {self.learning_rate.text()}",
            f"--max_train_epochs {self.epochs.value()}",
            f"--save_every_n_epochs {self.save_every.value()}",
            f"--seed {self.seed.value()}",
            f"--dataset_config {dataset_config}"
        ]

        # Adiciona opção de resume se marcado
        if self.resume_checkbox.isChecked() and self.resume_path.text().strip():
            resume_path = self.resume_path.text().strip()
            cmd.append(f"--resume {resume_path}")

        return " ".join(filter(None, cmd))