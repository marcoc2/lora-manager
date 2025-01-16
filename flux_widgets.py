from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
                            QFormLayout, QLineEdit, QPushButton, QSpinBox,
                            QCheckBox, QFileDialog, QLabel, QComboBox, QScrollArea)
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
        # Remover referência ao layout principal, agora usamos self.control_layout
        layout = self.control_layout
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

        # AE model
        ae_layout = QHBoxLayout()
        self.ae_path = QLineEdit()
        self.ae_path.setPlaceholderText("Path to AutoEncoder model")
        self.ae_path.setText(self.config.get("ae_path", ""))
        select_ae = QPushButton("Browse")
        select_ae.clicked.connect(self.select_ae_path)
        ae_layout.addWidget(self.ae_path)
        ae_layout.addWidget(select_ae)
        model_layout.addWidget(QLabel("AutoEncoder Model:"))
        model_layout.addLayout(ae_layout)
        
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
        self.guidance_scale.setValue(self.config.get("guidance_scale", 1))
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

        # Cache and Optimization
        cache_group = QGroupBox("Cache and Optimization")
        cache_layout = QFormLayout()

        self.cache_latents = QCheckBox()
        self.cache_latents.setChecked(self.config.get("cache_latents", True))
        cache_layout.addRow("Cache Latents to Disk:", self.cache_latents)

        self.cache_text_encoder = QCheckBox()
        self.cache_text_encoder.setChecked(self.config.get("cache_text_encoder", True))
        cache_layout.addRow("Cache Text Encoder:", self.cache_text_encoder)

        self.cache_text_encoder_disk = QCheckBox()
        self.cache_text_encoder_disk.setChecked(self.config.get("cache_text_encoder_disk", True))
        cache_layout.addRow("Cache Text Encoder to Disk:", self.cache_text_encoder_disk)

        self.persistent_workers = QCheckBox()
        self.persistent_workers.setChecked(self.config.get("persistent_workers", True))
        cache_layout.addRow("Persistent Data Loader Workers:", self.persistent_workers)

        self.max_workers = QSpinBox()
        self.max_workers.setRange(0, 16)
        self.max_workers.setValue(self.config.get("max_workers", 2))
        cache_layout.addRow("Max Data Loader Workers:", self.max_workers)

        self.sdpa = QCheckBox()
        self.sdpa.setChecked(self.config.get("sdpa", True))
        cache_layout.addRow("SDPA:", self.sdpa)

        self.save_model_as = QComboBox()
        self.save_model_as.addItems(["safetensors", "pt", "ckpt"])
        self.save_model_as.setCurrentText(self.config.get("save_model_as", "safetensors"))
        cache_layout.addRow("Save Model As:", self.save_model_as)

        cache_group.setLayout(cache_layout)
        layout.addWidget(cache_group)

        # Output Configuration
        output_group = QGroupBox("Output Configuration")
        output_layout = QVBoxLayout()

        # Output Directory
        output_dir_layout = QHBoxLayout()
        self.output_dir = QLineEdit()
        self.output_dir.setPlaceholderText("Output directory")
        self.output_dir.setText(self.config.get("output_dir", ""))
        select_output = QPushButton("Browse")
        select_output.clicked.connect(self.select_output_path)
        output_dir_layout.addWidget(self.output_dir)
        output_dir_layout.addWidget(select_output)

        # Output Name
        self.output_name = QLineEdit()
        self.output_name.setPlaceholderText("Output model name (without extension)")
        self.output_name.setText(self.config.get("output_name", ""))

        output_layout.addWidget(QLabel("Output Directory:"))
        output_layout.addLayout(output_dir_layout)
        output_layout.addWidget(QLabel("Output Name:"))
        output_layout.addWidget(self.output_name)

        output_group.setLayout(output_layout)
        layout.addWidget(output_group)

        # Network Configuration
        network_group = QGroupBox("Network Configuration")
        network_layout = QFormLayout()

        self.network_dim = QSpinBox()
        self.network_dim.setRange(1, 128)
        self.network_dim.setValue(self.config.get("network_dim", 32))
        network_layout.addRow("Network Dimension:", self.network_dim)

        self.network_alpha = QSpinBox()
        self.network_alpha.setRange(1, 128)
        self.network_alpha.setValue(self.config.get("network_alpha", 16))
        network_layout.addRow("Network Alpha:", self.network_alpha)

        self.network_args = QLineEdit(self.config.get("network_args", "train_blocks=single"))
        network_layout.addRow("Network Arguments:", self.network_args)

        network_group.setLayout(network_layout)
        layout.addWidget(network_group)

        # Training Parameters
        training_group = QGroupBox("Training Parameters")
        training_layout = QFormLayout()

        self.mixed_precision = QComboBox()
        self.mixed_precision.addItems(["no", "fp16", "bf16"])
        self.mixed_precision.setCurrentText(self.config.get("mixed_precision", "bf16"))
        training_layout.addRow("Mixed Precision:", self.mixed_precision)

        self.save_precision = QComboBox()
        self.save_precision.addItems(["no", "fp16", "bf16"])
        self.save_precision.setCurrentText(self.config.get("save_precision", "bf16"))
        training_layout.addRow("Save Precision:", self.save_precision)

        self.network_module = QLineEdit(self.config.get("network_module", "networks.lora_flux"))
        training_layout.addRow("Network Module:", self.network_module)

        self.optimizer_type = QLineEdit(self.config.get("optimizer_type", "adafactor"))
        training_layout.addRow("Optimizer Type:", self.optimizer_type)

        self.learning_rate = QLineEdit(self.config.get("learning_rate", "1e-4"))
        training_layout.addRow("Learning Rate:", self.learning_rate)

        self.epochs = QSpinBox()
        self.epochs.setRange(1, 1000)
        self.epochs.setValue(self.config.get("epochs", 16))
        training_layout.addRow("Max Epochs:", self.epochs)

        self.save_every = QSpinBox()
        self.save_every.setRange(1, 100)
        self.save_every.setValue(self.config.get("save_every", 8))
        training_layout.addRow("Save Every N Epochs:", self.save_every)

        self.seed = QSpinBox()
        self.seed.setRange(1, 999999)
        self.seed.setValue(self.config.get("seed", 42))
        training_layout.addRow("Seed:", self.seed)

        self.timestep_sampling = QComboBox()
        self.timestep_sampling.addItems(["uniform", "sigmoid"])
        self.timestep_sampling.setCurrentText(self.config.get("timestep_sampling", "sigmoid"))
        training_layout.addRow("Timestep Sampling:", self.timestep_sampling)

        self.model_prediction_type = QComboBox()
        self.model_prediction_type.addItems(["epsilon", "v", "raw"])
        self.model_prediction_type.setCurrentText(self.config.get("model_prediction_type", "raw"))
        training_layout.addRow("Model Prediction Type:", self.model_prediction_type)

        self.loss_type = QComboBox()
        self.loss_type.addItems(["l1", "l2", "huber"])
        self.loss_type.setCurrentText(self.config.get("loss_type", "l2"))
        training_layout.addRow("Loss Type:", self.loss_type)

        self.optimizer_args = QLineEdit(self.config.get("optimizer_args", "relative_step=False scale_parameter=False warmup_init=False"))
        training_layout.addRow("Optimizer Args:", self.optimizer_args)

        # Switches/Checkboxes
        self.network_train_unet_only = QCheckBox()
        self.network_train_unet_only.setChecked(self.config.get("network_train_unet_only", True))
        training_layout.addRow("Train UNet Only:", self.network_train_unet_only)

        self.fp8_base = QCheckBox()
        self.fp8_base.setChecked(self.config.get("fp8_base", True))
        training_layout.addRow("FP8 Base:", self.fp8_base)

        self.highvram = QCheckBox()
        self.highvram.setChecked(self.config.get("highvram", True))
        training_layout.addRow("High VRAM:", self.highvram)

        self.split_mode = QCheckBox()
        self.split_mode.setChecked(self.config.get("split_mode", True))
        training_layout.addRow("Split Mode:", self.split_mode)

        training_group.setLayout(training_layout)
        layout.addWidget(training_group)

        # Start Training button
        self.train_button = QPushButton("Start Training")
        layout.addWidget(self.train_button)



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

    def select_scripts_path(self):
        path = QFileDialog.getExistingDirectory(self, "Select Scripts Directory")
        if path:
            self.scripts_dir.setText(path)
            self.save_current_config()

    def select_ae_path(self):
        path = QFileDialog.getOpenFileName(self, "Select AutoEncoder Model", 
                                         filter="Model files (*.safetensors *.pt *.sft)")[0]
        if path:
            self.ae_path.setText(path)
            self.save_current_config()

    def select_output_path(self):
        path = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if path:
            self.output_dir.setText(path)
            self.save_current_config()

    def save_current_config(self):
        """Salva a configuração atual no arquivo JSON"""
        # ... Adicionar scripts_dir ao config
        config = {
            "flux_path": self.flux_path.text(),
            "clip_l_path": self.clip_l_path.text(),
            "t5xxl_path": self.t5xxl_path.text(),
            "ae_path": self.ae_path.text(),
            "scripts_dir": self.scripts_dir.text(),  # Adicionado
            "output_dir": self.output_dir.text(),
            "output_name": self.output_name.text(),
            "mixed_precision": self.mixed_precision.currentText(),
            "save_precision": self.save_precision.currentText(),
            "network_module": self.network_module.text(),
            "optimizer_type": self.optimizer_type.text(),
            "learning_rate": self.learning_rate.text(),
            "epochs": self.epochs.value(),
            "save_every": self.save_every.value(),
            "seed": self.seed.value(),
            "timestep_sampling": self.timestep_sampling.currentText(),
            "model_prediction_type": self.model_prediction_type.currentText(),
            "loss_type": self.loss_type.currentText(),
            "optimizer_args": self.optimizer_args.text(),
            "network_train_unet_only": self.network_train_unet_only.isChecked(),
            "fp8_base": self.fp8_base.isChecked(),
            "highvram": self.highvram.isChecked(),
            "split_mode": self.split_mode.isChecked(),
            "cache_latents": self.cache_latents.isChecked(),
            "cache_text_encoder": self.cache_text_encoder.isChecked(),
            "cache_text_encoder_disk": self.cache_text_encoder_disk.isChecked(),
            "persistent_workers": self.persistent_workers.isChecked(),
            "max_workers": self.max_workers.value(),
            "sdpa": self.sdpa.isChecked(),
            "save_model_as": self.save_model_as.currentText(),
            "network_dim": self.network_dim.value(),
            "network_alpha": self.network_alpha.value(),
            "network_args": self.network_args.text()
        }
        save_config(config)

    def get_command(self, dataset_path):
        """Gera o comando de treinamento para o Flux"""
        dataset_config = dataset_path / "cropped_images/dataset.toml"

        script_path = Path(self.scripts_dir.text()) / "flux_train_network.py"
        cmd = [
            "accelerate launch",
            "--mixed_precision", self.mixed_precision.currentText(),
            "--num_cpu_threads_per_process 1",
            str(script_path),
            f"--pretrained_model_name_or_path {self.flux_path.text()}",
            f"--clip_l {self.clip_l_path.text()}",
            f"--t5xxl {self.t5xxl_path.text()}",
            f"--ae {self.ae_path.text()}",
            "--cache_latents_to_disk" if self.cache_latents.isChecked() else "",
            f"--save_model_as {self.save_model_as.currentText()}",
            "--sdpa" if self.sdpa.isChecked() else "",
            "--persistent_data_loader_workers" if self.persistent_workers.isChecked() else "",
            f"--max_data_loader_n_workers {self.max_workers.value()}",
            f"--seed {self.seed.value()}",
            "--gradient_checkpointing",
            f"--mixed_precision {self.mixed_precision.currentText()}",
            f"--save_precision {self.save_precision.currentText()}",
            f"--network_module {self.network_module.text()}",
            f"--network_dim {self.network_dim.value()}",
            f"--network_alpha {self.network_alpha.value()}",
            f"--optimizer_type {self.optimizer_type.text()}",
            f"--learning_rate {self.learning_rate.text()}",
            "--network_train_unet_only" if self.network_train_unet_only.isChecked() else "",
            "--cache_text_encoder_outputs" if self.cache_text_encoder.isChecked() else "",
            "--cache_text_encoder_outputs_to_disk" if self.cache_text_encoder_disk.isChecked() else "",
            "--fp8_base" if self.fp8_base.isChecked() else "",
            "--highvram" if self.highvram.isChecked() else "",
            f"--max_train_epochs {self.epochs.value()}",
            f"--save_every_n_epochs {self.save_every.value()}",
            f"--dataset_config {dataset_config}",
            f"--output_dir {self.output_dir.text()}" if self.output_dir.text() else "",
            f"--output_name {self.output_name.text()}" if self.output_name.text() else "",
            f"--timestep_sampling {self.timestep_sampling.currentText()}",
            f"--model_prediction_type {self.model_prediction_type.currentText()}",
            f"--guidance_scale 1.0",
            f"--loss_type {self.loss_type.currentText()}",
            "--split_mode" if self.split_mode.isChecked() else ""
        ]
        
        # Adiciona optimizer args se houver
        optimizer_args = self.optimizer_args.text().strip()
        if optimizer_args:
            cmd.append('--optimizer_args ' + ' '.join(f'"{arg}"' for arg in optimizer_args.split()))


        # Adiciona network args se houver
        network_args = self.network_args.text().strip()
        if network_args:
            cmd.append(f'--network_args "{network_args}"')

        # Adiciona opção de resume se marcado
        if self.resume_checkbox.isChecked() and self.resume_path.text().strip():
            resume_path = self.resume_path.text().strip()
            cmd.append(f"--resume {resume_path}")

        filtered_cmd = filter(None, cmd)
        # Converte os itens para string e filtra os vazios
        cmd_str = " ".join(str(item) for item in filtered_cmd if str(item).strip())
        return cmd_str