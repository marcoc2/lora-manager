from PyQt6.QtWidgets import (QHBoxLayout, QGroupBox, QFormLayout, QLineEdit, 
                           QPushButton, QCheckBox, QFileDialog, QLabel, QComboBox,
                           QVBoxLayout, QSpinBox, QMessageBox, QWidget)
from PyQt6.QtCore import Qt
from pathlib import Path
from qwen_widgets_base import QwenTrainingWidgetsBase, NoWheelSpinBox, save_config
import subprocess
import sys
import os

class QwenTrainingWidgets(QwenTrainingWidgetsBase):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()

    def init_ui(self):
        layout = self.control_layout
        layout.setSpacing(10)

        # --- Essential Settings ---

        # Base Models
        model_group = QGroupBox("Base Models")
        model_layout = QVBoxLayout()
        
        # DiT model
        dit_layout = QHBoxLayout()
        self.dit_path = QLineEdit()
        self.dit_path.setPlaceholderText("Path to Qwen-Image DiT model (qwen_image_bf16.safetensors)")
        self.dit_path.setText(self.config.get("dit_path", ""))
        select_dit = QPushButton("Browse")
        select_dit.clicked.connect(self.select_dit_path)
        dit_layout.addWidget(self.dit_path)
        dit_layout.addWidget(select_dit)
        model_layout.addWidget(QLabel("DiT Model:"))
        model_layout.addLayout(dit_layout)

        # Text Encoder model
        te_layout = QHBoxLayout()
        self.text_encoder_path = QLineEdit()
        self.text_encoder_path.setPlaceholderText("Path to Qwen2.5-VL Text Encoder (qwen_2.5_vl_7b_fp8_scaled.safetensors)")
        self.text_encoder_path.setText(self.config.get("text_encoder_path", ""))
        select_te = QPushButton("Browse")
        select_te.clicked.connect(self.select_text_encoder_path)
        te_layout.addWidget(self.text_encoder_path)
        te_layout.addWidget(select_te)
        model_layout.addWidget(QLabel("Text Encoder Model:"))
        model_layout.addLayout(te_layout)

        # VAE model
        vae_layout = QHBoxLayout()
        self.vae_path = QLineEdit()
        self.vae_path.setPlaceholderText("Path to Qwen-Image VAE (qwen_image_vae.safetensors)")
        self.vae_path.setText(self.config.get("vae_path", ""))
        select_vae = QPushButton("Browse")
        select_vae.clicked.connect(self.select_vae_path)
        vae_layout.addWidget(self.vae_path)
        vae_layout.addWidget(select_vae)
        model_layout.addWidget(QLabel("VAE Model:"))
        model_layout.addLayout(vae_layout)
        
        model_group.setLayout(model_layout)
        layout.addWidget(model_group)

        # Output Configuration
        output_group = QGroupBox("Output Configuration")
        output_layout = QFormLayout()
        
        # Output directory
        output_dir_layout = QHBoxLayout()
        self.output_dir = QLineEdit()
        self.output_dir.setPlaceholderText("Output directory for trained LoRA")
        self.output_dir.setText(self.config.get("output_dir", ""))
        select_output = QPushButton("Browse")
        select_output.clicked.connect(self.select_output_path)
        output_dir_layout.addWidget(self.output_dir)
        output_dir_layout.addWidget(select_output)
        output_layout.addRow("Output Directory:", output_dir_layout)
        
        # Output name
        self.output_name = QLineEdit()
        self.output_name.setText(self.config.get("output_name", "qwen_lora"))
        output_layout.addRow("Output Name:", self.output_name)
        
        output_group.setLayout(output_layout)
        layout.addWidget(output_group)

        # Training Parameters (Essential)
        params_group = QGroupBox("Training Parameters")
        params_layout = QFormLayout()
        
        # Learning rate
        self.learning_rate = QLineEdit()
        self.learning_rate.setText(self.config.get("learning_rate", "2e-4"))
        params_layout.addRow("Learning Rate:", self.learning_rate)
        
        # Network dimension
        self.network_dim = NoWheelSpinBox()
        self.network_dim.setRange(1, 512)
        self.network_dim.setValue(self.config.get("network_dim", 32))
        params_layout.addRow("Network Dimension:", self.network_dim)
        
        # Network alpha
        self.network_alpha = NoWheelSpinBox()
        self.network_alpha.setRange(1, 512)
        self.network_alpha.setValue(self.config.get("network_alpha", 16))
        params_layout.addRow("Network Alpha:", self.network_alpha)
        
        # Epochs
        self.epochs = NoWheelSpinBox()
        self.epochs.setRange(1, 1000)
        self.epochs.setValue(self.config.get("epochs", 32))
        params_layout.addRow("Epochs:", self.epochs)
        
        # Save every
        self.save_every = NoWheelSpinBox()
        self.save_every.setRange(1, 100)
        self.save_every.setValue(self.config.get("save_every", 16))
        params_layout.addRow("Save Every N Epochs:", self.save_every)
        
        # Batch size
        self.batch_size = NoWheelSpinBox()
        self.batch_size.setRange(1, 16)
        self.batch_size.setValue(self.config.get("batch_size", 1))
        params_layout.addRow("Batch Size:", self.batch_size)
        
        params_group.setLayout(params_layout)
        layout.addWidget(params_group)

        # --- Advanced Settings Toggle ---
        self.advanced_toggle = QCheckBox("Show Advanced Options")
        self.advanced_toggle.setStyleSheet("font-weight: bold; color: #007acc; margin-top: 10px;")
        layout.addWidget(self.advanced_toggle)

        # --- Advanced Settings Container ---
        self.advanced_container = QWidget()
        self.advanced_layout = QVBoxLayout(self.advanced_container)
        self.advanced_layout.setContentsMargins(0, 0, 0, 0)

        # Python/Venv Path
        python_group = QGroupBox("Python Environment")
        python_layout = QVBoxLayout()
        
        # Python path
        python_path_layout = QHBoxLayout()
        self.python_venv_path = QLineEdit()
        self.python_venv_path.setPlaceholderText("Path to python.exe with Musubi installed (e.g., C:/Apps/musubi_env/Scripts/python.exe)")
        self.python_venv_path.setText(self.config.get("python_venv_path", ""))
        select_python = QPushButton("Browse")
        select_python.clicked.connect(self.select_python_path)
        python_path_layout.addWidget(self.python_venv_path)
        python_path_layout.addWidget(select_python)
        python_layout.addWidget(QLabel("Python/Venv Path:"))
        python_layout.addLayout(python_path_layout)
        
        # Musubi directory
        musubi_layout = QHBoxLayout()
        self.musubi_dir = QLineEdit()
        self.musubi_dir.setPlaceholderText("Path to musubi-tuner folder")
        self.musubi_dir.setText(self.config.get("musubi_dir", ""))
        select_musubi = QPushButton("Browse")
        select_musubi.clicked.connect(self.select_musubi_path)
        musubi_layout.addWidget(self.musubi_dir)
        musubi_layout.addWidget(select_musubi)
        python_layout.addWidget(QLabel("Musubi Directory:"))
        python_layout.addLayout(musubi_layout)
        
        python_group.setLayout(python_layout)
        self.advanced_layout.addWidget(python_group)

        # Advanced Options
        advanced_group = QGroupBox("Advanced Options")
        advanced_layout = QFormLayout()
        
        # Mixed precision
        self.mixed_precision = QComboBox()
        self.mixed_precision.addItems(["bf16", "fp16", "no"])
        self.mixed_precision.setCurrentText(self.config.get("mixed_precision", "bf16"))
        advanced_layout.addRow("Mixed Precision:", self.mixed_precision)
        
        # Optimizer
        self.optimizer_type = QComboBox()
        self.optimizer_type.addItems(["adamw8bit", "adamw", "lion", "adafactor"])
        self.optimizer_type.setCurrentText(self.config.get("optimizer_type", "adamw8bit"))
        advanced_layout.addRow("Optimizer:", self.optimizer_type)
        
        # Timestep sampling
        self.timestep_sampling = QComboBox()
        self.timestep_sampling.addItems(["shift", "uniform"])
        self.timestep_sampling.setCurrentText(self.config.get("timestep_sampling", "shift"))
        advanced_layout.addRow("Timestep Sampling:", self.timestep_sampling)
        
        # Weighting scheme
        self.weighting_scheme = QComboBox()
        self.weighting_scheme.addItems(["none", "sigma_sqrt"])
        self.weighting_scheme.setCurrentText(self.config.get("weighting_scheme", "none"))
        advanced_layout.addRow("Weighting Scheme:", self.weighting_scheme)

        # Discrete flow shift
        self.discrete_flow_shift = QLineEdit()
        self.discrete_flow_shift.setText(str(self.config.get("discrete_flow_shift", 2.0)))
        advanced_layout.addRow("Discrete Flow Shift:", self.discrete_flow_shift)
        
        # Blocks to swap (para economia de VRAM)
        self.blocks_to_swap = NoWheelSpinBox()
        self.blocks_to_swap.setRange(0, 50)
        self.blocks_to_swap.setValue(self.config.get("blocks_to_swap", 0))
        advanced_layout.addRow("Blocks to Swap (VRAM saving):", self.blocks_to_swap)
        
        advanced_group.setLayout(advanced_layout)
        self.advanced_layout.addWidget(advanced_group)

        # Checkboxes for various options
        options_group = QGroupBox("Training Options")
        options_layout = QVBoxLayout()
        
        self.gradient_checkpointing = QCheckBox("Gradient Checkpointing")
        self.gradient_checkpointing.setChecked(self.config.get("gradient_checkpointing", True))
        options_layout.addWidget(self.gradient_checkpointing)
        
        self.sdpa = QCheckBox("Scaled Dot Product Attention (SDPA)")
        self.sdpa.setChecked(self.config.get("sdpa", True))
        options_layout.addWidget(self.sdpa)
        
        self.fp8_llm = QCheckBox("FP8 for Text Encoder (saves VRAM)")
        self.fp8_llm.setChecked(self.config.get("fp8_llm", False))
        options_layout.addWidget(self.fp8_llm)
        
        self.cache_latents = QCheckBox("Cache Latents")
        self.cache_latents.setChecked(self.config.get("cache_latents", True))
        options_layout.addWidget(self.cache_latents)
        
        self.cache_text_encoder = QCheckBox("Cache Text Encoder")
        self.cache_text_encoder.setChecked(self.config.get("cache_text_encoder", True))
        options_layout.addWidget(self.cache_text_encoder)
        
        self.flip_aug = QCheckBox("Flip Augmentation")
        self.flip_aug.setChecked(self.config.get("flip_aug", False))
        options_layout.addWidget(self.flip_aug)
        
        options_group.setLayout(options_layout)
        self.advanced_layout.addWidget(options_group)

        # Resume Training
        resume_group = QGroupBox("Resume Training")
        resume_layout = QFormLayout()
        
        self.resume_checkbox = QCheckBox("Resume from checkpoint")
        self.resume_checkbox.setChecked(self.config.get("resume_training", False))
        self.resume_path = QLineEdit()
        self.resume_path.setText(self.config.get("resume_path", ""))
        self.resume_path.setEnabled(False)
        self.resume_path.setPlaceholderText("Path to network weights file (.safetensors or .pt)")
        select_resume = QPushButton("Browse")
        select_resume.setEnabled(False)
        
        resume_path_layout = QHBoxLayout()
        resume_path_layout.addWidget(self.resume_path)
        resume_path_layout.addWidget(select_resume)
        
        resume_layout.addRow(self.resume_checkbox)
        resume_layout.addRow("Resume path:", resume_path_layout)
        
        # Connect signals
        self.resume_checkbox.toggled.connect(lambda checked: self.resume_path.setEnabled(checked))
        self.resume_checkbox.toggled.connect(lambda checked: select_resume.setEnabled(checked))
        select_resume.clicked.connect(self.select_resume_path)
        
        resume_group.setLayout(resume_layout)
        self.advanced_layout.addWidget(resume_group)

        # Cache and Convert buttons
        cache_group = QGroupBox("Cache and Convert")
        cache_layout = QVBoxLayout()
        
        self.cache_latents_button = QPushButton("Cache Latents")
        self.cache_latents_button.clicked.connect(self.cache_latents_action)
        cache_layout.addWidget(self.cache_latents_button)
        
        self.cache_text_encoder_button = QPushButton("Cache Text Encoder")
        self.cache_text_encoder_button.clicked.connect(self.cache_text_encoder_action)
        cache_layout.addWidget(self.cache_text_encoder_button)
        
        self.convert_lora_button = QPushButton("Convert LoRA")
        self.convert_lora_button.clicked.connect(self.convert_lora_action)
        cache_layout.addWidget(self.convert_lora_button)
        
        cache_group.setLayout(cache_layout)
        self.advanced_layout.addWidget(cache_group)

        # Add Advanced Container to Main Layout
        layout.addWidget(self.advanced_container)
        
        # Connect Toggle
        self.advanced_container.setVisible(False)
        self.advanced_toggle.toggled.connect(self.advanced_container.setVisible)

        # Training button
        self.train_button = QPushButton("Start Training")
        self.train_button.setObjectName("primaryButton")
        self.train_button.setMinimumHeight(50)
        self.train_button.setStyleSheet("font-size: 16px; font-weight: bold;")
        layout.addWidget(self.train_button)

    def select_dit_path(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select DiT Model", "", "Safetensors Files (*.safetensors)")
        if file_path:
            self.dit_path.setText(file_path)

    def select_text_encoder_path(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Text Encoder Model", "", "Safetensors Files (*.safetensors)")
        if file_path:
            self.text_encoder_path.setText(file_path)

    def select_vae_path(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select VAE Model", "", "Safetensors Files (*.safetensors)")
        if file_path:
            self.vae_path.setText(file_path)

    def select_python_path(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Python Executable", "", "Python Executable (python.exe)")
        if file_path:
            self.python_venv_path.setText(file_path)

    def select_musubi_path(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Musubi Tuner Directory")
        if folder:
            self.musubi_dir.setText(folder)

    def select_resume_path(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Resume Checkpoint", "", "Checkpoint Files (*.safetensors *.pt)")
        if file_path:
            self.resume_path.setText(file_path)

    def select_output_path(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if folder:
            self.output_dir.setText(folder)

    def get_current_config(self):
        return {
            "dit_path": self.dit_path.text(),
            "text_encoder_path": self.text_encoder_path.text(),
            "vae_path": self.vae_path.text(),
            "python_venv_path": self.python_venv_path.text(),
            "musubi_dir": self.musubi_dir.text(),
            "output_dir": self.output_dir.text(),
            "output_name": self.output_name.text(),
            "mixed_precision": self.mixed_precision.currentText(),
            "save_precision": self.mixed_precision.currentText(),
            "network_module": "networks.lora_qwen_image",
            "optimizer_type": self.optimizer_type.currentText(),
            "learning_rate": self.learning_rate.text(),
            "epochs": self.epochs.value(),
            "save_every": self.save_every.value(),
            "seed": 42,
            "timestep_sampling": self.timestep_sampling.currentText(),
            "weighting_scheme": self.weighting_scheme.currentText(),
            "discrete_flow_shift": float(self.discrete_flow_shift.text()) if self.discrete_flow_shift.text() else 2.0,
            "network_dim": self.network_dim.value(),
            "network_alpha": self.network_alpha.value(),
            "batch_size": self.batch_size.value(),
            "gradient_checkpointing": self.gradient_checkpointing.isChecked(),
            "sdpa": self.sdpa.isChecked(),
            "blocks_to_swap": self.blocks_to_swap.value(),
            "fp8_llm": self.fp8_llm.isChecked(),
            "cache_latents": self.cache_latents.isChecked(),
            "cache_text_encoder": self.cache_text_encoder.isChecked(),
            "cache_text_encoder_disk": True,
            "persistent_workers": True,
            "max_workers": 2,
            "save_model_as": "safetensors",
            "network_args": "",
            "optimizer_args": "",
            "flip_aug": self.flip_aug.isChecked(),
            "resume_training": self.resume_checkbox.isChecked(),
            "resume_path": self.resume_path.text(),
            "additional_params": ""
        }

    def validate_inputs(self):
        errors = []
        
        if not self.python_venv_path.text():
            errors.append("Python/Venv path is required - Please configure a Python environment with Musubi Tuner installed")
        elif not Path(self.python_venv_path.text()).exists():
            errors.append("Python executable does not exist at specified path")
        elif not self.python_venv_path.text().endswith(("python.exe", "python")):
            errors.append("Path must point to a python executable")
        
        if not self.dit_path.text():
            errors.append("DiT model path is required")
        elif not Path(self.dit_path.text()).exists():
            errors.append("DiT model file does not exist")
            
        if not self.text_encoder_path.text():
            errors.append("Text encoder path is required")
        elif not Path(self.text_encoder_path.text()).exists():
            errors.append("Text encoder file does not exist")
            
        if not self.vae_path.text():
            errors.append("VAE model path is required")
        elif not Path(self.vae_path.text()).exists():
            errors.append("VAE model file does not exist")
            
        if not self.musubi_dir.text():
            errors.append("Musubi Tuner directory is required")
        elif not Path(self.musubi_dir.text()).exists():
            errors.append("Musubi Tuner directory does not exist")
        
        # Criar cache_imgs dentro do musubi_dir
        musubi_cache_dir = musubi_dir / "cache_imgs"
        musubi_cache_dir.mkdir(exist_ok=True)
        
        # Copiar todas as imagens para musubi_dir
        image_extensions = [".png", ".jpg", ".jpeg", ".webp", ".bmp"]
        for img_file in cropped_dir.iterdir():
            if img_file.is_file() and img_file.suffix.lower() in image_extensions:
                dest_img = musubi_dir / img_file.name
                if not dest_img.exists():
                    shutil.copy2(img_file, dest_img)
        
        # Copiar captions de captions/ para musubi_dir com conversão automática para UTF-8
        if captions_dir.exists():
            for caption_file in captions_dir.glob("*.txt"):
                dest_caption = musubi_dir / caption_file.name
                if not dest_caption.exists():
                    # Ler com detecção automática de encoding e salvar como UTF-8
                    try:
                        # Primeiro tenta UTF-8
                        with open(caption_file, 'r', encoding='utf-8') as f:
                            content = f.read()
                    except UnicodeDecodeError:
                        # Se falhar, tenta latin-1/iso-8859-1
                        try:
                            with open(caption_file, 'r', encoding='latin-1') as f:
                                content = f.read()
                        except UnicodeDecodeError:
                            # Se ainda falhar, tenta cp1252 (Windows)
                            with open(caption_file, 'r', encoding='cp1252') as f:
                                content = f.read()
                    
                    # Sempre salva como UTF-8, garantindo quebra de linha no final
                    with open(dest_caption, 'w', encoding='utf-8') as f:
                        f.write(content.rstrip() + '\n')
        
        return musubi_dir

    def generate_musubi_toml(self, dataset_path, musubi_dir):
        """Gera TOML específico para estrutura Musubi"""
        import toml
        
        # Configuração básica para Musubi
        config = self.get_current_config()
        
        musubi_toml = {
            "general": {
                "resolution": 512,
                "caption_extension": ".txt",
                "batch_size": config.get('batch_size', 1),
                "enable_bucket": True,
                "bucket_no_upscale": False
            },
            "datasets": [{
                "image_directory": str(musubi_dir.resolve()),
                "cache_directory": str((musubi_dir / "cache_imgs").resolve()),
                "num_repeats": 1
            }]
        }
        
        toml_path = musubi_dir / "dataset_qwen.toml"
        with open(toml_path, "w") as f:
            toml.dump(musubi_toml, f)
        
        return toml_path

    def get_command(self, dataset_path):
        errors = self.validate_inputs()
        if errors:
            QMessageBox.critical(None, "Validation Error", "\\n".join(errors))
            return None

        # Salvar configuração atual
        self.save_current_config()
        
        # Preparar estrutura Musubi (cropped_images_musubi)
        QMessageBox.information(None, "Preparing Musubi Dataset", 
                              "Preparing Musubi-specific dataset structure in 'cropped_images_musubi' folder.\n"
                              "This keeps your original sd-scripts structure intact.")
        musubi_dir = self.prepare_musubi_dataset(dataset_path)
        
        # Prepare dataset config path (usar dataset_qwen.toml específico para Musubi)
        dataset_config = musubi_dir / "dataset_qwen.toml"
        if not dataset_config.exists():
            # Gerar TOML específico para Musubi na nova pasta
            self.generate_musubi_toml(dataset_path, musubi_dir)

        # Auto-cache: Verificar e criar caches automaticamente
        cache_dir = musubi_dir / "cache_imgs"
        
        # Verificar se precisamos fazer cache
        needs_latent_cache = not any(cache_dir.glob("*.npz"))  # Verifica se há arquivos .npz
        needs_text_cache = not any(cache_dir.glob("*.txt"))    # Verifica se há cache de text
        
        if needs_latent_cache or needs_text_cache:
            reply = QMessageBox.question(None, "Cache Required", 
                f"Training requires cache files that don't exist:\n\n"
                f"{'• Latent cache missing' if needs_latent_cache else ''}\n"
                f"{'• Text encoder cache missing' if needs_text_cache else ''}\n\n"
                f"Create cache automatically? (This may take a few minutes)",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            
            if reply != QMessageBox.StandardButton.Yes:
                return None
            
            # Criar batch script para executar caches sequencialmente
            commands = []
            
            if needs_latent_cache:
                latent_cache_script = Path(self.musubi_dir.text()) / "src" / "musubi_tuner" / "qwen_image_cache_latents.py"
                commands.append([
                    self.python_venv_path.text(),
                    str(latent_cache_script),
                    "--vae", self.vae_path.text(),
                    "--dataset_config", str(dataset_config)
                ])
            
            if needs_text_cache:
                text_cache_script = Path(self.musubi_dir.text()) / "src" / "musubi_tuner" / "qwen_image_cache_text_encoder_outputs.py"
                text_cmd = [
                    self.python_venv_path.text(),
                    str(text_cache_script),
                    "--text_encoder", self.text_encoder_path.text(),
                    "--dataset_config", str(dataset_config)
                ]
                
                if self.fp8_llm.isChecked():
                    text_cmd.extend(["--device", "cpu"])
                    
                commands.append(text_cmd)
            
            # Retornar comando de cache + treinamento combinado
            # O queue manager executará os caches primeiro, depois o treinamento
            cache_info = {
                "cache_commands": commands,
                "training_command": self._build_training_command(dataset_config)
            }
            return cache_info

        # Se cache já existe, executa treinamento diretamente
        return self._build_training_command(dataset_config)

    def _build_training_command(self, dataset_config):
        """Constrói o comando de treinamento do Musubi"""
        python_path = Path(self.python_venv_path.text())
        venv_dir = python_path.parent  # Scripts/ directory
        accelerate_exe = venv_dir / "accelerate.exe"
        
        # Verificar se accelerate está disponível no venv
        if accelerate_exe.exists():
            # Usar accelerate.exe diretamente
            command_start = [str(accelerate_exe), "launch"]
        else:
            # Verificar se accelerate está instalado via importação
            import subprocess
            try:
                result = subprocess.run([str(python_path), "-c", "import accelerate; print('OK')"], 
                                      capture_output=True, text=True, timeout=5)
                if result.returncode != 0:
                    QMessageBox.critical(None, "Error", 
                        f"Accelerate is not installed in the specified venv!\n\n"
                        f"Please install accelerate in your Musubi venv:\n"
                        f"1. Activate venv: {venv_dir.parent}\\Scripts\\activate\n"
                        f"2. Install: pip install accelerate")
                    return None
            except:
                QMessageBox.critical(None, "Error", "Cannot verify accelerate installation in venv")
                return None
                
            # Usar python -m accelerate
            command_start = [str(python_path), "-m", "accelerate", "launch"]
        
        musubi_script = Path(self.musubi_dir.text()) / "src" / "musubi_tuner" / "qwen_image_train_network.py"
        
        command = command_start + [
            "--num_cpu_threads_per_process", "1",
            "--mixed_precision", self.mixed_precision.currentText(),
            str(musubi_script),
            "--dit", self.dit_path.text(),
            "--vae", self.vae_path.text(),
            "--text_encoder", self.text_encoder_path.text(),
            "--dataset_config", str(dataset_config),
            "--output_dir", self.output_dir.text(),
            "--output_name", self.output_name.text(),
            "--network_module", "networks.lora_qwen_image",
            "--fp8_base",  # Crucial for VRAM reduction with Qwen-Image
            "--network_dim", str(self.network_dim.value()),
            "--learning_rate", self.learning_rate.text(),
            "--timestep_sampling", self.timestep_sampling.currentText(),
            "--weighting_scheme", self.weighting_scheme.currentText(),
            "--discrete_flow_shift", str(float(self.discrete_flow_shift.text())),
            "--optimizer_type", self.optimizer_type.currentText(),
            "--max_train_epochs", str(self.epochs.value()),
            "--save_every_n_epochs", str(self.save_every.value()),
            "--seed", "42"
        ]

        # Adicionar opções condicionais
        if self.gradient_checkpointing.isChecked():
            command.append("--gradient_checkpointing")
        
        if self.sdpa.isChecked():
            command.append("--sdpa")
        
        if self.fp8_llm.isChecked():
            command.append("--fp8_llm")
        
        if self.blocks_to_swap.value() > 0:
            command.extend(["--blocks_to_swap", str(self.blocks_to_swap.value())])

        if self.flip_aug.isChecked():
            command.append("--flip_aug")

        # Resume training
        if self.resume_checkbox.isChecked() and self.resume_path.text():
            command.extend(["--network_weights", self.resume_path.text()])

        return command

    def cache_latents_action(self):
        errors = self.validate_inputs()
        if errors:
            QMessageBox.critical(None, "Validation Error", "\\n".join(errors))
            return

        # Get dataset path from parent
        if not hasattr(self.parent(), 'dataset_path') or not self.parent().dataset_path:
            QMessageBox.warning(None, "Warning", "Please select a dataset folder first!")
            return

        # Preparar estrutura Musubi
        musubi_dir = self.prepare_musubi_dataset(self.parent().dataset_path)
        
        dataset_config = musubi_dir / "dataset_qwen.toml"
        if not dataset_config.exists():
            self.generate_musubi_toml(self.parent().dataset_path, musubi_dir)

        cache_script = Path(self.musubi_dir.text()) / "src" / "musubi_tuner" / "qwen_image_cache_latents.py"
        
        command = [
            self.python_venv_path.text(), str(cache_script),
            "--vae", self.vae_path.text(),
            "--dataset_config", str(dataset_config)
        ]

        self.run_command_with_dialog(command, "Cache Latents")

    def cache_text_encoder_action(self):
        errors = self.validate_inputs()
        if errors:
            QMessageBox.critical(None, "Validation Error", "\\n".join(errors))
            return

        if not hasattr(self.parent(), 'dataset_path') or not self.parent().dataset_path:
            QMessageBox.warning(None, "Warning", "Please select a dataset folder first!")
            return

        # Preparar estrutura Musubi
        musubi_dir = self.prepare_musubi_dataset(self.parent().dataset_path)
        
        dataset_config = musubi_dir / "dataset_qwen.toml"
        if not dataset_config.exists():
            self.generate_musubi_toml(self.parent().dataset_path, musubi_dir)

        cache_script = Path(self.musubi_dir.text()) / "src" / "musubi_tuner" / "qwen_image_cache_text_encoder_outputs.py"
        
        command = [
            self.python_venv_path.text(), str(cache_script),
            "--text_encoder", self.text_encoder_path.text(),
            "--dataset_config", str(dataset_config)
        ]

        # Adicionar device CPU se usar FP8
        if self.fp8_llm.isChecked():
            command.extend(["--device", "cpu"])

        self.run_command_with_dialog(command, "Cache Text Encoder")

    def convert_lora_action(self):
        if not self.output_dir.text():
            QMessageBox.warning(None, "Warning", "Please set output directory first!")
            return

        lora_file, _ = QFileDialog.getOpenFileName(None, "Select LoRA file to convert", 
                                                 self.output_dir.text(), "Safetensors Files (*.safetensors)")
        if not lora_file:
            return

        output_file, _ = QFileDialog.getSaveFileName(None, "Save converted LoRA as", 
                                                   lora_file.replace(".safetensors", "_converted.safetensors"),
                                                   "Safetensors Files (*.safetensors)")
        if not output_file:
            return

        convert_script = Path(self.musubi_dir.text()) / "src" / "musubi_tuner" / "convert_lora.py"
        
        command = [
            self.python_venv_path.text(), str(convert_script),
            "--input", lora_file,
            "--output", output_file,
            "--target", "other",
            "--diffusers_prefix", "transformers"
        ]

        self.run_command_with_dialog(command, "Convert LoRA")

    def run_command_with_dialog(self, command, title):
        from command_utils import CommandOutputDialog
        
        dialog = CommandOutputDialog(title, self)
        dialog.run_command(command)
        dialog.exec()