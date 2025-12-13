from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                            QLineEdit, QPushButton, QSpinBox, QDoubleSpinBox,
                            QCheckBox, QComboBox, QGroupBox, QFileDialog, QTextEdit,
                            QScrollArea, QFrame, QFormLayout)
from PyQt6.QtCore import Qt
import json
from pathlib import Path

CONFIG_FILE = "flux_config.json"

def save_config(config, filename=CONFIG_FILE):
    try:
        with open(filename, 'w') as f:
            json.dump(config, f, indent=4)
    except Exception as e:
        print(f"Error saving config: {e}")

def load_config(filename=CONFIG_FILE):
    try:
        if Path(filename).exists():
            with open(filename, 'r') as f:
                return json.load(f)
    except Exception as e:
        print(f"Error loading config: {e}")
    return {}

class NoWheelSpinBox(QSpinBox):
    def wheelEvent(self, event):
        event.ignore()

class NoWheelDoubleSpinBox(QDoubleSpinBox):
    def wheelEvent(self, event):
        event.ignore()

class FluxTrainingWidgets(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.init_ui()
        self.load_saved_config()

    def init_ui(self):
        main_layout = QVBoxLayout()

        # Scroll Area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)

        content_widget = QWidget()
        layout = QVBoxLayout(content_widget)

        # --- Model Configuration ---
        model_group = QGroupBox("Model Configuration")
        model_layout = QVBoxLayout()

        # Model Path (HuggingFace ID or local)
        model_path_layout = QHBoxLayout()
        self.model_path = QLineEdit()
        self.model_path.setPlaceholderText("HuggingFace ID or local path to Flux model")
        self.model_path.setText("black-forest-labs/FLUX.1-dev")
        btn_model = QPushButton("Select Local")
        btn_model.clicked.connect(lambda: self.select_path(self.model_path, is_file=True))
        model_path_layout.addWidget(QLabel("Model:"))
        model_path_layout.addWidget(self.model_path)
        model_path_layout.addWidget(btn_model)
        model_layout.addLayout(model_path_layout)

        # Quantize and Low VRAM options
        options_layout = QHBoxLayout()
        self.quantize = QCheckBox("Quantize (8bit)")
        self.quantize.setChecked(True)
        self.quantize.setToolTip("Enable 8-bit quantization for lower VRAM usage")
        options_layout.addWidget(self.quantize)

        self.low_vram = QCheckBox("Low VRAM Mode")
        self.low_vram.setChecked(False)
        self.low_vram.setToolTip("Enable if GPU is connected to monitor (slower but uses less VRAM)")
        options_layout.addWidget(self.low_vram)
        options_layout.addStretch()
        model_layout.addLayout(options_layout)

        model_group.setLayout(model_layout)
        layout.addWidget(model_group)

        # --- Training Parameters ---
        params_group = QGroupBox("Training Parameters")
        params_layout = QVBoxLayout()

        # Grid for basic params
        grid_layout = QHBoxLayout()

        # Steps
        steps_layout = QVBoxLayout()
        self.steps = NoWheelSpinBox()
        self.steps.setRange(100, 100000)
        self.steps.setValue(2000)
        steps_layout.addWidget(QLabel("Training Steps:"))
        steps_layout.addWidget(self.steps)
        grid_layout.addLayout(steps_layout)

        # Batch Size
        batch_layout = QVBoxLayout()
        self.batch_size = NoWheelSpinBox()
        self.batch_size.setRange(1, 64)
        self.batch_size.setValue(1)
        batch_layout.addWidget(QLabel("Batch Size:"))
        batch_layout.addWidget(self.batch_size)
        grid_layout.addLayout(batch_layout)

        # Learning Rate
        lr_layout = QVBoxLayout()
        self.learning_rate = QLineEdit()
        self.learning_rate.setText("1e-4")
        lr_layout.addWidget(QLabel("Learning Rate:"))
        lr_layout.addWidget(self.learning_rate)
        grid_layout.addLayout(lr_layout)

        # Seed
        seed_layout = QVBoxLayout()
        self.seed = NoWheelSpinBox()
        self.seed.setRange(1, 999999)
        self.seed.setValue(42)
        seed_layout.addWidget(QLabel("Seed:"))
        seed_layout.addWidget(self.seed)
        grid_layout.addLayout(seed_layout)

        params_layout.addLayout(grid_layout)

        # LoRA Config
        lora_layout = QHBoxLayout()

        self.network_dim = NoWheelSpinBox()
        self.network_dim.setRange(1, 256)
        self.network_dim.setValue(16)
        lora_layout.addWidget(QLabel("LoRA Rank:"))
        lora_layout.addWidget(self.network_dim)

        self.network_alpha = NoWheelSpinBox()
        self.network_alpha.setRange(1, 256)
        self.network_alpha.setValue(16)
        lora_layout.addWidget(QLabel("LoRA Alpha:"))
        lora_layout.addWidget(self.network_alpha)

        params_layout.addLayout(lora_layout)

        # Resolution (multi-select style display)
        res_layout = QHBoxLayout()
        res_layout.addWidget(QLabel("Resolution:"))
        self.res_512 = QCheckBox("512")
        self.res_512.setChecked(True)
        self.res_768 = QCheckBox("768")
        self.res_768.setChecked(True)
        self.res_1024 = QCheckBox("1024")
        self.res_1024.setChecked(True)
        res_layout.addWidget(self.res_512)
        res_layout.addWidget(self.res_768)
        res_layout.addWidget(self.res_1024)
        res_layout.addStretch()
        params_layout.addLayout(res_layout)

        # Optimizer and EMA
        opt_layout = QHBoxLayout()
        opt_layout.addWidget(QLabel("Optimizer:"))
        self.optimizer = QComboBox()
        self.optimizer.addItems(["adamw8bit", "adamw", "prodigy"])
        self.optimizer.setCurrentText("adamw8bit")
        opt_layout.addWidget(self.optimizer)

        self.use_ema = QCheckBox("Use EMA")
        self.use_ema.setChecked(True)
        self.use_ema.setToolTip("Smooths learning, recommended to leave on")
        opt_layout.addWidget(self.use_ema)

        opt_layout.addWidget(QLabel("EMA Decay:"))
        self.ema_decay = NoWheelDoubleSpinBox()
        self.ema_decay.setRange(0.9, 0.9999)
        self.ema_decay.setDecimals(4)
        self.ema_decay.setSingleStep(0.001)
        self.ema_decay.setValue(0.99)
        opt_layout.addWidget(self.ema_decay)
        opt_layout.addStretch()
        params_layout.addLayout(opt_layout)

        # Precision
        prec_layout = QHBoxLayout()
        prec_layout.addWidget(QLabel("Train Precision:"))
        self.mixed_precision = QComboBox()
        self.mixed_precision.addItems(["bf16", "fp16"])
        self.mixed_precision.setCurrentText("bf16")
        prec_layout.addWidget(self.mixed_precision)

        prec_layout.addWidget(QLabel("Save Precision:"))
        self.save_precision = QComboBox()
        self.save_precision.addItems(["float16", "bf16"])
        self.save_precision.setCurrentText("float16")
        prec_layout.addWidget(self.save_precision)
        prec_layout.addStretch()
        params_layout.addLayout(prec_layout)

        # Cache option
        self.cache_latents = QCheckBox("Cache latents to disk")
        self.cache_latents.setChecked(True)
        self.cache_latents.setToolTip("Recommended: caches latents to disk for faster training")
        params_layout.addWidget(self.cache_latents)

        params_group.setLayout(params_layout)
        layout.addWidget(params_group)

        # --- Output Configuration ---
        output_group = QGroupBox("Output Configuration")
        output_layout = QVBoxLayout()

        # Output Directory
        output_dir_layout = QHBoxLayout()
        self.output_dir = QLineEdit()
        self.output_dir.setPlaceholderText("Output directory")
        self.output_dir.setText("output")
        btn_output_dir = QPushButton("Browse")
        btn_output_dir.clicked.connect(lambda: self.select_path(self.output_dir, is_file=False))
        output_dir_layout.addWidget(self.output_dir)
        output_dir_layout.addWidget(btn_output_dir)

        output_layout.addWidget(QLabel("Output Directory:"))
        output_layout.addLayout(output_dir_layout)

        # Output Name
        name_layout = QHBoxLayout()
        self.output_name = QLineEdit()
        self.output_name.setPlaceholderText("my_flux_lora")
        name_layout.addWidget(QLabel("Output Name:"))
        name_layout.addWidget(self.output_name)
        output_layout.addLayout(name_layout)

        # Save Every
        save_layout = QHBoxLayout()
        self.save_every = NoWheelSpinBox()
        self.save_every.setRange(1, 10000)
        self.save_every.setValue(250)
        save_layout.addWidget(QLabel("Save Every (Steps):"))
        save_layout.addWidget(self.save_every)
        output_layout.addLayout(save_layout)

        output_group.setLayout(output_layout)
        layout.addWidget(output_group)

        # --- Sample Prompts & Preview ---
        sample_group = QGroupBox("Sample Prompts & Preview")
        sample_layout = QVBoxLayout()

        # Enable sampling checkbox
        self.enable_sampling = QCheckBox("Enable preview generation during training")
        self.enable_sampling.setChecked(True)
        self.enable_sampling.stateChanged.connect(self.on_sampling_toggled)
        sample_layout.addWidget(self.enable_sampling)

        # Sample Every
        sample_every_layout = QHBoxLayout()
        self.sample_every = NoWheelSpinBox()
        self.sample_every.setRange(1, 10000)
        self.sample_every.setValue(250)
        sample_every_layout.addWidget(QLabel("Sample Every (Steps):"))
        sample_every_layout.addWidget(self.sample_every)
        sample_layout.addLayout(sample_every_layout)

        # Sample dimensions and guidance
        sample_params_layout = QHBoxLayout()

        sample_params_layout.addWidget(QLabel("Width:"))
        self.sample_width = NoWheelSpinBox()
        self.sample_width.setRange(256, 2048)
        self.sample_width.setSingleStep(64)
        self.sample_width.setValue(1024)
        sample_params_layout.addWidget(self.sample_width)

        sample_params_layout.addWidget(QLabel("Height:"))
        self.sample_height = NoWheelSpinBox()
        self.sample_height.setRange(256, 2048)
        self.sample_height.setSingleStep(64)
        self.sample_height.setValue(1024)
        sample_params_layout.addWidget(self.sample_height)

        sample_params_layout.addWidget(QLabel("Guidance:"))
        self.guidance_scale = NoWheelSpinBox()
        self.guidance_scale.setRange(1, 20)
        self.guidance_scale.setValue(4)
        sample_params_layout.addWidget(self.guidance_scale)

        sample_params_layout.addWidget(QLabel("Steps:"))
        self.sample_steps = NoWheelSpinBox()
        self.sample_steps.setRange(1, 100)
        self.sample_steps.setValue(20)
        sample_params_layout.addWidget(self.sample_steps)

        sample_params_layout.addStretch()
        sample_layout.addLayout(sample_params_layout)

        # Prompt input with auto-fill button
        prompt_header = QHBoxLayout()
        prompt_label = QLabel("Sample Prompts:")
        self.auto_fill_btn = QPushButton("Auto-fill from dataset")
        self.auto_fill_btn.clicked.connect(self.auto_fill_prompts)
        self.auto_fill_btn.setStyleSheet("background-color: #2196F3; color: white; padding: 5px;")
        prompt_header.addWidget(prompt_label)
        prompt_header.addWidget(self.auto_fill_btn)
        prompt_header.addStretch()
        sample_layout.addLayout(prompt_header)

        self.sample_prompts = QTextEdit()
        self.sample_prompts.setPlaceholderText("Enter prompts for sampling (one per line) or click Auto-fill")
        self.sample_prompts.setMinimumHeight(80)
        sample_layout.addWidget(self.sample_prompts)

        sample_group.setLayout(sample_layout)
        layout.addWidget(sample_group)

        # --- Advanced Options Toggle ---
        self.advanced_toggle = QCheckBox("Show Advanced Options")
        self.advanced_toggle.setStyleSheet("font-weight: bold; color: #007acc; margin-top: 10px;")
        layout.addWidget(self.advanced_toggle)

        # --- Advanced Options Container ---
        self.advanced_container = QWidget()
        advanced_layout = QVBoxLayout(self.advanced_container)
        advanced_layout.setContentsMargins(0, 0, 0, 0)

        # Trigger Word
        trigger_layout = QHBoxLayout()
        trigger_layout.addWidget(QLabel("Trigger Word:"))
        self.trigger_word = QLineEdit()
        self.trigger_word.setPlaceholderText("Optional trigger word (e.g., p3rs0n)")
        trigger_layout.addWidget(self.trigger_word)
        advanced_layout.addLayout(trigger_layout)

        # Caption Dropout
        dropout_layout = QHBoxLayout()
        dropout_layout.addWidget(QLabel("Caption Dropout Rate:"))
        self.caption_dropout = NoWheelDoubleSpinBox()
        self.caption_dropout.setRange(0.0, 1.0)
        self.caption_dropout.setDecimals(2)
        self.caption_dropout.setSingleStep(0.01)
        self.caption_dropout.setValue(0.05)
        dropout_layout.addWidget(self.caption_dropout)
        dropout_layout.addStretch()
        advanced_layout.addLayout(dropout_layout)

        # Resume Training
        resume_group = QGroupBox("Resume Training")
        resume_layout = QVBoxLayout()

        self.resume_checkbox = QCheckBox("Resume from checkpoint")
        resume_layout.addWidget(self.resume_checkbox)

        resume_path_layout = QHBoxLayout()
        self.resume_path = QLineEdit()
        self.resume_path.setEnabled(False)
        self.resume_path.setPlaceholderText("Path to LoRA checkpoint (.safetensors)")
        select_resume = QPushButton("Browse")
        select_resume.setEnabled(False)
        select_resume.clicked.connect(lambda: self.select_path(self.resume_path, is_file=True))
        resume_path_layout.addWidget(self.resume_path)
        resume_path_layout.addWidget(select_resume)
        resume_layout.addLayout(resume_path_layout)

        self.resume_checkbox.stateChanged.connect(lambda state: [
            self.resume_path.setEnabled(state == Qt.CheckState.Checked.value),
            select_resume.setEnabled(state == Qt.CheckState.Checked.value)
        ])

        resume_group.setLayout(resume_layout)
        advanced_layout.addWidget(resume_group)

        layout.addWidget(self.advanced_container)
        self.advanced_container.setVisible(False)
        self.advanced_toggle.toggled.connect(self.advanced_container.setVisible)

        # --- Buttons ---
        buttons_layout = QHBoxLayout()

        # Reload Config Button
        self.reload_btn = QPushButton("Reload Config")
        self.reload_btn.setToolTip("Reload settings from flux_config.json")
        self.reload_btn.clicked.connect(self.load_saved_config)
        buttons_layout.addWidget(self.reload_btn)

        # Start Button
        self.train_button = QPushButton("Start Flux Training")
        self.train_button.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; padding: 10px;")
        buttons_layout.addWidget(self.train_button)

        layout.addLayout(buttons_layout)
        layout.addStretch()

        scroll.setWidget(content_widget)
        main_layout.addWidget(scroll)
        self.setLayout(main_layout)

    def select_path(self, line_edit, is_file=False):
        if is_file:
            path, _ = QFileDialog.getOpenFileName(self, "Select File", filter="Model files (*.safetensors *.pt)")
        else:
            path = QFileDialog.getExistingDirectory(self, "Select Directory")
        if path:
            line_edit.setText(path)
            self.save_current_config()

    def on_sampling_toggled(self, state):
        """Enable/disable sample prompts based on checkbox"""
        is_checked = (state == Qt.CheckState.Checked.value)
        self.sample_prompts.setEnabled(is_checked)
        self.auto_fill_btn.setEnabled(is_checked)
        self.sample_every.setEnabled(is_checked)
        self.sample_width.setEnabled(is_checked)
        self.sample_height.setEnabled(is_checked)
        self.guidance_scale.setEnabled(is_checked)
        self.sample_steps.setEnabled(is_checked)
        self.save_current_config()

    def auto_fill_prompts(self):
        """Auto-fill prompts with the first caption from the dataset"""
        try:
            if not self.parent or not hasattr(self.parent, 'parent'):
                from PyQt6.QtWidgets import QMessageBox
                QMessageBox.warning(self, "Error", "Unable to access dataset path.")
                return

            main_window = self.parent.parent
            if not hasattr(main_window, 'get_effective_dataset_path'):
                from PyQt6.QtWidgets import QMessageBox
                QMessageBox.warning(self, "Error", "Unable to access dataset path.")
                return

            dataset_path = main_window.get_effective_dataset_path()

            if not dataset_path:
                from PyQt6.QtWidgets import QMessageBox
                QMessageBox.warning(self, "Dataset Not Set", "Please select a dataset folder first in the main window.")
                return

            if not dataset_path.exists():
                from PyQt6.QtWidgets import QMessageBox
                QMessageBox.warning(self, "Dataset Not Found", "Selected dataset path does not exist.")
                return

            # Look for caption files
            caption_extensions = ["*.txt", "*.caption"]
            caption_files = []

            for ext in caption_extensions:
                caption_files.extend(dataset_path.rglob(ext))

            if not caption_files:
                from PyQt6.QtWidgets import QMessageBox
                QMessageBox.warning(self, "No Captions Found", f"No caption files found in dataset directory:\n{dataset_path}")
                return

            # Sort and get first caption
            caption_files = sorted(caption_files)
            first_caption_file = caption_files[0]

            # Read the caption
            with open(first_caption_file, "r", encoding="utf-8") as f:
                caption = f.read().strip()

            if caption:
                self.sample_prompts.setPlainText(caption)
                from PyQt6.QtWidgets import QMessageBox
                preview = caption[:100] + "..." if len(caption) > 100 else caption
                msg = f"Loaded from {first_caption_file.name}\n\n{preview}"
                QMessageBox.information(self, "Caption Loaded", msg)
            else:
                from PyQt6.QtWidgets import QMessageBox
                QMessageBox.warning(self, "Empty Caption", f"File {first_caption_file.name} is empty.")

        except Exception as e:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.critical(self, "Error", f"Failed to load caption: {str(e)}")

    def get_resolution_list(self):
        """Returns list of selected resolutions"""
        resolutions = []
        if self.res_512.isChecked():
            resolutions.append(512)
        if self.res_768.isChecked():
            resolutions.append(768)
        if self.res_1024.isChecked():
            resolutions.append(1024)
        return resolutions if resolutions else [1024]  # Default to 1024 if none selected

    def get_config(self):
        """Returns the current configuration as a dictionary"""
        return {
            "model_name_or_path": self.model_path.text(),
            "output_dir": self.output_dir.text(),
            "output_name": self.output_name.text(),
            "network_dim": self.network_dim.value(),
            "network_alpha": self.network_alpha.value(),
            "learning_rate": self.learning_rate.text(),
            "steps": self.steps.value(),
            "batch_size": self.batch_size.value(),
            "resolution": self.get_resolution_list(),
            "mixed_precision": self.mixed_precision.currentText(),
            "save_precision": self.save_precision.currentText(),
            "save_every": self.save_every.value(),
            "quantize": self.quantize.isChecked(),
            "low_vram": self.low_vram.isChecked(),
            "use_ema": self.use_ema.isChecked(),
            "ema_decay": self.ema_decay.value(),
            "optimizer": self.optimizer.currentText(),
            "cache_latents_to_disk": self.cache_latents.isChecked(),
            "caption_dropout_rate": self.caption_dropout.value(),
            "trigger_word": self.trigger_word.text(),
            "enable_sampling": self.enable_sampling.isChecked(),
            "sample_every": self.sample_every.value(),
            "sample_width": self.sample_width.value(),
            "sample_height": self.sample_height.value(),
            "sample_prompts": self.sample_prompts.toPlainText(),
            "guidance_scale": self.guidance_scale.value(),
            "sample_steps": self.sample_steps.value(),
            "seed": self.seed.value(),
            "resume_training": self.resume_checkbox.isChecked(),
            "resume_path": self.resume_path.text()
        }

    def save_current_config(self):
        """Salva a configuração atual no arquivo JSON"""
        save_config(self.get_config())

    def load_saved_config(self):
        config = load_config()
        if not config:
            return

        if "model_name_or_path" in config: self.model_path.setText(config["model_name_or_path"])
        if "output_dir" in config: self.output_dir.setText(config["output_dir"])
        if "output_name" in config: self.output_name.setText(config["output_name"])
        if "network_dim" in config: self.network_dim.setValue(config["network_dim"])
        if "network_alpha" in config: self.network_alpha.setValue(config["network_alpha"])
        if "learning_rate" in config: self.learning_rate.setText(config["learning_rate"])
        if "steps" in config: self.steps.setValue(config["steps"])
        if "batch_size" in config: self.batch_size.setValue(config["batch_size"])

        # Resolution checkboxes
        if "resolution" in config:
            res_list = config["resolution"]
            self.res_512.setChecked(512 in res_list)
            self.res_768.setChecked(768 in res_list)
            self.res_1024.setChecked(1024 in res_list)

        if "mixed_precision" in config: self.mixed_precision.setCurrentText(config["mixed_precision"])
        if "save_precision" in config: self.save_precision.setCurrentText(config["save_precision"])
        if "save_every" in config: self.save_every.setValue(config["save_every"])
        if "quantize" in config: self.quantize.setChecked(config["quantize"])
        if "low_vram" in config: self.low_vram.setChecked(config["low_vram"])
        if "use_ema" in config: self.use_ema.setChecked(config["use_ema"])
        if "ema_decay" in config: self.ema_decay.setValue(config["ema_decay"])
        if "optimizer" in config: self.optimizer.setCurrentText(config["optimizer"])
        if "cache_latents_to_disk" in config: self.cache_latents.setChecked(config["cache_latents_to_disk"])
        if "caption_dropout_rate" in config: self.caption_dropout.setValue(config["caption_dropout_rate"])
        if "trigger_word" in config: self.trigger_word.setText(config["trigger_word"])
        if "enable_sampling" in config: self.enable_sampling.setChecked(config["enable_sampling"])
        if "sample_every" in config: self.sample_every.setValue(config["sample_every"])
        if "sample_width" in config: self.sample_width.setValue(config["sample_width"])
        if "sample_height" in config: self.sample_height.setValue(config["sample_height"])
        if "sample_prompts" in config: self.sample_prompts.setPlainText(config["sample_prompts"])
        if "guidance_scale" in config: self.guidance_scale.setValue(config["guidance_scale"])
        if "sample_steps" in config: self.sample_steps.setValue(config["sample_steps"])
        if "seed" in config: self.seed.setValue(config["seed"])
        if "resume_training" in config: self.resume_checkbox.setChecked(config["resume_training"])
        if "resume_path" in config: self.resume_path.setText(config["resume_path"])
