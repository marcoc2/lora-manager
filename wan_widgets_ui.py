from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                            QLineEdit, QPushButton, QSpinBox, QDoubleSpinBox,
                            QCheckBox, QGroupBox, QFileDialog, QTextEdit,
                            QScrollArea, QFrame, QRadioButton, QButtonGroup,
                            QFormLayout, QMessageBox)
from PyQt6.QtCore import Qt
import json
from pathlib import Path

CONFIG_FILE = "wan_config.json"

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
    """SpinBox that ignores wheel events to prevent accidental changes"""
    def wheelEvent(self, event):
        event.ignore()

class NoWheelDoubleSpinBox(QDoubleSpinBox):
    """DoubleSpinBox that ignores wheel events to prevent accidental changes"""
    def wheelEvent(self, event):
        event.ignore()

class WanTrainingWidgets(QWidget):
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
        layout.setSpacing(10)

        # --- Model Version Selection ---
        version_group = QGroupBox("Wan Model Version")
        version_layout = QVBoxLayout()

        self.version_button_group = QButtonGroup(self)

        # Wan 2.1 Options
        wan21_frame = QFrame()
        wan21_layout = QHBoxLayout(wan21_frame)
        wan21_layout.setContentsMargins(0, 0, 0, 0)
        self.wan21_14b_radio = QRadioButton("Wan 2.1 14B (24GB+ VRAM)")
        self.wan21_1b_radio = QRadioButton("Wan 2.1 1.3B (12GB+ VRAM)")
        self.version_button_group.addButton(self.wan21_14b_radio, 0)
        self.version_button_group.addButton(self.wan21_1b_radio, 1)
        wan21_layout.addWidget(self.wan21_14b_radio)
        wan21_layout.addWidget(self.wan21_1b_radio)
        wan21_layout.addStretch()

        # Wan 2.2 Options
        wan22_frame = QFrame()
        wan22_layout = QHBoxLayout(wan22_frame)
        wan22_layout.setContentsMargins(0, 0, 0, 0)
        self.wan22_14b_radio = QRadioButton("Wan 2.2 14B MOE (24GB+ VRAM)")
        self.version_button_group.addButton(self.wan22_14b_radio, 2)
        wan22_layout.addWidget(self.wan22_14b_radio)
        wan22_layout.addStretch()

        # Default selection
        self.wan21_14b_radio.setChecked(True)

        # Connect version change
        self.version_button_group.buttonClicked.connect(self.on_version_changed)

        version_layout.addWidget(QLabel("Wan 2.1 (Sigmoid timestep, unload_text_encoder):"))
        version_layout.addWidget(wan21_frame)
        version_layout.addWidget(QLabel("Wan 2.2 (Linear timestep, cache_text_embeddings, MOE):"))
        version_layout.addWidget(wan22_frame)

        version_group.setLayout(version_layout)
        layout.addWidget(version_group)

        # --- Training Mode ---
        mode_group = QGroupBox("Training Mode")
        mode_layout = QHBoxLayout()

        self.mode_button_group = QButtonGroup(self)
        self.image_mode_radio = QRadioButton("Image Training (num_frames=1)")
        self.video_mode_radio = QRadioButton("Video Training (multi-frame)")
        self.mode_button_group.addButton(self.image_mode_radio, 0)
        self.mode_button_group.addButton(self.video_mode_radio, 1)
        self.image_mode_radio.setChecked(True)

        self.mode_button_group.buttonClicked.connect(self.on_mode_changed)

        mode_layout.addWidget(self.image_mode_radio)
        mode_layout.addWidget(self.video_mode_radio)
        mode_layout.addStretch()

        mode_group.setLayout(mode_layout)
        layout.addWidget(mode_group)

        # --- Output Configuration ---
        output_group = QGroupBox("Output Configuration")
        output_layout = QVBoxLayout()

        # Output Directory
        output_dir_layout = QHBoxLayout()
        self.output_dir = QLineEdit()
        self.output_dir.setPlaceholderText("Output directory (default: output)")
        self.output_dir.setText("output")
        btn_output_dir = QPushButton("Browse")
        btn_output_dir.clicked.connect(lambda: self.select_path(self.output_dir, is_file=False))
        output_dir_layout.addWidget(QLabel("Output Directory:"))
        output_dir_layout.addWidget(self.output_dir)
        output_dir_layout.addWidget(btn_output_dir)
        output_layout.addLayout(output_dir_layout)

        # Output Name
        name_layout = QHBoxLayout()
        self.output_name = QLineEdit()
        self.output_name.setPlaceholderText("my_wan_lora")
        self.output_name.setText("wan_lora")
        name_layout.addWidget(QLabel("Output Name:"))
        name_layout.addWidget(self.output_name)
        output_layout.addLayout(name_layout)

        # Trigger Word
        trigger_layout = QHBoxLayout()
        self.trigger_word = QLineEdit()
        self.trigger_word.setPlaceholderText("Optional trigger word (e.g., p3r5on)")
        trigger_layout.addWidget(QLabel("Trigger Word:"))
        trigger_layout.addWidget(self.trigger_word)
        output_layout.addLayout(trigger_layout)

        output_group.setLayout(output_layout)
        layout.addWidget(output_group)

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
        self.batch_size.setRange(1, 16)
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

        # Save Every
        save_layout = QVBoxLayout()
        self.save_every = NoWheelSpinBox()
        self.save_every.setRange(50, 10000)
        self.save_every.setValue(250)
        save_layout.addWidget(QLabel("Save Every (Steps):"))
        save_layout.addWidget(self.save_every)
        grid_layout.addLayout(save_layout)

        params_layout.addLayout(grid_layout)
        params_group.setLayout(params_layout)
        layout.addWidget(params_group)

        # --- LoRA Configuration ---
        lora_group = QGroupBox("LoRA Configuration")
        lora_layout = QHBoxLayout()

        self.lora_rank = NoWheelSpinBox()
        self.lora_rank.setRange(1, 256)
        self.lora_rank.setValue(32)
        lora_layout.addWidget(QLabel("LoRA Rank (linear):"))
        lora_layout.addWidget(self.lora_rank)

        self.lora_alpha = NoWheelSpinBox()
        self.lora_alpha.setRange(1, 256)
        self.lora_alpha.setValue(32)
        lora_layout.addWidget(QLabel("LoRA Alpha:"))
        lora_layout.addWidget(self.lora_alpha)

        lora_group.setLayout(lora_layout)
        layout.addWidget(lora_group)

        # --- Video Parameters (hidden by default) ---
        self.video_params_group = QGroupBox("Video Parameters")
        video_layout = QHBoxLayout()

        self.num_frames = NoWheelSpinBox()
        self.num_frames.setRange(1, 200)
        self.num_frames.setValue(40)
        video_layout.addWidget(QLabel("Number of Frames:"))
        video_layout.addWidget(self.num_frames)

        self.fps = NoWheelSpinBox()
        self.fps.setRange(1, 60)
        self.fps.setValue(16)
        video_layout.addWidget(QLabel("FPS:"))
        video_layout.addWidget(self.fps)

        self.video_params_group.setLayout(video_layout)
        self.video_params_group.setVisible(False)
        layout.addWidget(self.video_params_group)

        # --- Resolution ---
        resolution_group = QGroupBox("Resolution")
        resolution_layout = QVBoxLayout()

        res_inputs = QHBoxLayout()
        self.resolution_width = NoWheelSpinBox()
        self.resolution_width.setRange(256, 2048)
        self.resolution_width.setSingleStep(64)
        self.resolution_width.setValue(832)
        res_inputs.addWidget(QLabel("Width:"))
        res_inputs.addWidget(self.resolution_width)

        self.resolution_height = NoWheelSpinBox()
        self.resolution_height.setRange(256, 2048)
        self.resolution_height.setSingleStep(64)
        self.resolution_height.setValue(480)
        res_inputs.addWidget(QLabel("Height:"))
        res_inputs.addWidget(self.resolution_height)
        resolution_layout.addLayout(res_inputs)

        # Resolution presets
        preset_layout = QHBoxLayout()
        preset_480p = QPushButton("480p (832x480)")
        preset_480p.clicked.connect(lambda: self.set_resolution(832, 480))
        preset_720p = QPushButton("720p (1280x720)")
        preset_720p.clicked.connect(lambda: self.set_resolution(1280, 720))
        preset_1024 = QPushButton("1024x1024")
        preset_1024.clicked.connect(lambda: self.set_resolution(1024, 1024))
        preset_layout.addWidget(preset_480p)
        preset_layout.addWidget(preset_720p)
        preset_layout.addWidget(preset_1024)
        resolution_layout.addLayout(preset_layout)

        resolution_group.setLayout(resolution_layout)
        layout.addWidget(resolution_group)

        # --- Low VRAM Preset Button ---
        low_vram_layout = QHBoxLayout()
        self.low_vram_btn = QPushButton("Apply Low VRAM Preset (24GB)")
        self.low_vram_btn.setStyleSheet("background-color: #FF9800; color: white; font-weight: bold; padding: 10px;")
        self.low_vram_btn.clicked.connect(self.apply_low_vram_preset)
        low_vram_layout.addWidget(self.low_vram_btn)
        layout.addLayout(low_vram_layout)

        # --- Memory Optimization ---
        memory_group = QGroupBox("Memory Optimization")
        memory_layout = QVBoxLayout()

        self.quantize = QCheckBox("Quantize Model (4-bit)")
        self.quantize.setChecked(True)
        memory_layout.addWidget(self.quantize)

        self.quantize_te = QCheckBox("Quantize Text Encoder")
        self.quantize_te.setChecked(True)
        memory_layout.addWidget(self.quantize_te)

        self.low_vram = QCheckBox("Low VRAM Mode")
        self.low_vram.setChecked(True)
        memory_layout.addWidget(self.low_vram)

        self.gradient_checkpointing = QCheckBox("Gradient Checkpointing")
        self.gradient_checkpointing.setChecked(True)
        memory_layout.addWidget(self.gradient_checkpointing)

        memory_group.setLayout(memory_layout)
        layout.addWidget(memory_group)

        # --- Advanced Settings Toggle ---
        self.advanced_toggle = QCheckBox("Show Advanced Options")
        self.advanced_toggle.setStyleSheet("font-weight: bold; color: #007acc; margin-top: 10px;")
        layout.addWidget(self.advanced_toggle)

        # --- Advanced Settings Container ---
        self.advanced_container = QWidget()
        advanced_layout = QVBoxLayout(self.advanced_container)
        advanced_layout.setContentsMargins(0, 0, 0, 0)

        # EMA Settings
        ema_group = QGroupBox("EMA Settings")
        ema_layout = QFormLayout()

        self.use_ema = QCheckBox("Use EMA")
        self.use_ema.setChecked(True)
        ema_layout.addRow("", self.use_ema)

        self.ema_decay = NoWheelDoubleSpinBox()
        self.ema_decay.setRange(0.9, 0.9999)
        self.ema_decay.setDecimals(4)
        self.ema_decay.setSingleStep(0.001)
        self.ema_decay.setValue(0.99)
        ema_layout.addRow("EMA Decay:", self.ema_decay)

        ema_group.setLayout(ema_layout)
        advanced_layout.addWidget(ema_group)

        # Wan 2.1 Specific Settings
        self.wan21_group = QGroupBox("Wan 2.1 Specific")
        wan21_specific_layout = QFormLayout()

        self.unload_text_encoder = QCheckBox("Unload Text Encoder (use trigger word only)")
        self.unload_text_encoder.setChecked(True)
        wan21_specific_layout.addRow("", self.unload_text_encoder)

        self.wan21_group.setLayout(wan21_specific_layout)
        advanced_layout.addWidget(self.wan21_group)

        # Wan 2.2 Specific Settings
        self.wan22_group = QGroupBox("Wan 2.2 Specific (MOE)")
        wan22_layout = QFormLayout()

        self.switch_boundary_every = NoWheelSpinBox()
        self.switch_boundary_every.setRange(1, 100)
        self.switch_boundary_every.setValue(10)
        wan22_layout.addRow("Switch Boundary Every:", self.switch_boundary_every)

        self.train_high_noise = QCheckBox("Train High Noise Stage")
        self.train_high_noise.setChecked(True)
        wan22_layout.addRow("", self.train_high_noise)

        self.train_low_noise = QCheckBox("Train Low Noise Stage")
        self.train_low_noise.setChecked(True)
        wan22_layout.addRow("", self.train_low_noise)

        self.cache_text_embeddings = QCheckBox("Cache Text Embeddings (Wan 2.2)")
        self.cache_text_embeddings.setChecked(True)
        wan22_layout.addRow("", self.cache_text_embeddings)

        self.wan22_group.setLayout(wan22_layout)
        self.wan22_group.setVisible(False)  # Hidden by default (Wan 2.1)
        advanced_layout.addWidget(self.wan22_group)

        # Advanced container visibility
        layout.addWidget(self.advanced_container)
        self.advanced_container.setVisible(False)
        self.advanced_toggle.toggled.connect(self.advanced_container.setVisible)

        # --- Sample Configuration ---
        sample_group = QGroupBox("Sample Generation")
        sample_layout = QVBoxLayout()

        self.enable_sampling = QCheckBox("Enable preview generation during training")
        self.enable_sampling.setChecked(True)
        sample_layout.addWidget(self.enable_sampling)

        sample_params_layout = QHBoxLayout()

        self.sample_every = NoWheelSpinBox()
        self.sample_every.setRange(50, 10000)
        self.sample_every.setValue(250)
        sample_params_layout.addWidget(QLabel("Sample Every:"))
        sample_params_layout.addWidget(self.sample_every)

        self.guidance_scale = NoWheelDoubleSpinBox()
        self.guidance_scale.setRange(1.0, 20.0)
        self.guidance_scale.setValue(5.0)
        sample_params_layout.addWidget(QLabel("Guidance Scale:"))
        sample_params_layout.addWidget(self.guidance_scale)

        self.sample_steps = NoWheelSpinBox()
        self.sample_steps.setRange(10, 100)
        self.sample_steps.setValue(30)
        sample_params_layout.addWidget(QLabel("Sample Steps:"))
        sample_params_layout.addWidget(self.sample_steps)

        sample_layout.addLayout(sample_params_layout)

        # Prompt input
        prompt_header = QHBoxLayout()
        prompt_label = QLabel("Sample Prompts (one per line):")
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

        # --- Buttons ---
        buttons_layout = QHBoxLayout()

        # Reload Config Button
        self.reload_btn = QPushButton("Reload Config")
        self.reload_btn.setToolTip("Reload settings from wan_config.json")
        self.reload_btn.clicked.connect(self.load_saved_config)
        buttons_layout.addWidget(self.reload_btn)

        # Training Button
        self.train_button = QPushButton("Start Wan Training")
        self.train_button.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; padding: 10px;")
        buttons_layout.addWidget(self.train_button)

        layout.addLayout(buttons_layout)
        layout.addStretch()

        scroll.setWidget(content_widget)
        main_layout.addWidget(scroll)
        self.setLayout(main_layout)

    def select_path(self, line_edit, is_file=False):
        if is_file:
            path, _ = QFileDialog.getOpenFileName(self, "Select File")
        else:
            path = QFileDialog.getExistingDirectory(self, "Select Directory")
        if path:
            line_edit.setText(path)
            self.save_current_config()

    def set_resolution(self, width, height):
        self.resolution_width.setValue(width)
        self.resolution_height.setValue(height)
        self.save_current_config()

    def on_version_changed(self, button):
        """Handle model version changes"""
        version_id = self.version_button_group.id(button)

        # Show/hide version-specific settings
        is_wan22 = (version_id == 2)
        self.wan22_group.setVisible(is_wan22)
        self.wan21_group.setVisible(not is_wan22)

        # Update defaults based on version
        if version_id == 0:  # Wan 2.1 14B
            self.resolution_width.setValue(832)
            self.resolution_height.setValue(480)
            self.quantize.setChecked(True)
        elif version_id == 1:  # Wan 2.1 1.3B
            self.resolution_width.setValue(832)
            self.resolution_height.setValue(480)
            self.quantize.setChecked(False)  # 1.3B doesn't need quantization
        elif version_id == 2:  # Wan 2.2 14B
            self.resolution_width.setValue(1024)
            self.resolution_height.setValue(1024)
            self.quantize.setChecked(True)

        self.save_current_config()

    def on_mode_changed(self, button):
        """Handle training mode changes"""
        is_video = self.mode_button_group.id(button) == 1
        self.video_params_group.setVisible(is_video)

        if not is_video:
            self.num_frames.setValue(1)
        else:
            self.num_frames.setValue(40)

        self.save_current_config()

    def apply_low_vram_preset(self):
        """Apply all low VRAM optimizations"""
        self.quantize.setChecked(True)
        self.quantize_te.setChecked(True)
        self.low_vram.setChecked(True)
        self.gradient_checkpointing.setChecked(True)
        self.batch_size.setValue(1)

        # For Wan 2.1, also enable unload_text_encoder
        if not self.wan22_14b_radio.isChecked():
            self.unload_text_encoder.setChecked(True)
        else:
            self.cache_text_embeddings.setChecked(True)

        self.save_current_config()

        QMessageBox.information(self, "Low VRAM Preset Applied",
            "Applied settings for 24GB VRAM cards:\n"
            "- Quantize Model: Enabled\n"
            "- Quantize Text Encoder: Enabled\n"
            "- Low VRAM Mode: Enabled\n"
            "- Gradient Checkpointing: Enabled\n"
            "- Batch Size: 1")

    def auto_fill_prompts(self):
        """Auto-fill prompts with the first caption from the dataset"""
        try:
            # Get dataset path from parent (main window)
            if not self.parent or not hasattr(self.parent, 'parent'):
                QMessageBox.warning(self, "Error", "Unable to access dataset path.")
                return

            main_window = self.parent.parent
            if not hasattr(main_window, 'get_effective_dataset_path'):
                QMessageBox.warning(self, "Error", "Unable to access dataset path.")
                return

            dataset_path = main_window.get_effective_dataset_path()

            if not dataset_path:
                QMessageBox.warning(self, "Dataset Not Set", "Please select a dataset folder first in the main window.")
                return

            if not dataset_path.exists():
                QMessageBox.warning(self, "Dataset Not Found", "Selected dataset path does not exist.")
                return

            # Look for caption files
            caption_extensions = ["*.txt", "*.caption"]
            caption_files = []

            for ext in caption_extensions:
                caption_files.extend(dataset_path.rglob(ext))

            if not caption_files:
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
                preview = caption[:100] + "..." if len(caption) > 100 else caption
                msg = f"Loaded from {first_caption_file.name}\n\n{preview}"
                QMessageBox.information(self, "Caption Loaded", msg)
            else:
                QMessageBox.warning(self, "Empty Caption", f"File {first_caption_file.name} is empty.")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load caption: {str(e)}")

    def get_model_version(self):
        """Get the selected model version"""
        if self.wan21_14b_radio.isChecked():
            return "wan21_14b"
        elif self.wan21_1b_radio.isChecked():
            return "wan21_1b"
        elif self.wan22_14b_radio.isChecked():
            return "wan22_14b"
        return "wan21_14b"

    def get_model_path(self):
        """Get the HuggingFace model path for selected version"""
        version = self.get_model_version()
        paths = {
            "wan21_14b": "Wan-AI/Wan2.1-T2V-14B-Diffusers",
            "wan21_1b": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
            "wan22_14b": "ai-toolkit/Wan2.2-T2V-A14B-Diffusers-bf16"
        }
        return paths.get(version, paths["wan21_14b"])

    def get_arch(self):
        """Get the architecture string for selected version"""
        version = self.get_model_version()
        archs = {
            "wan21_14b": "wan21",
            "wan21_1b": "wan21",
            "wan22_14b": "wan22_14b"
        }
        return archs.get(version, "wan21")

    def get_config(self):
        """Returns the current configuration as a dictionary"""
        return {
            "model_version": self.get_model_version(),
            "name_or_path": self.get_model_path(),
            "arch": self.get_arch(),
            "output_dir": self.output_dir.text(),
            "output_name": self.output_name.text(),
            "trigger_word": self.trigger_word.text(),
            "training_mode": "video" if self.video_mode_radio.isChecked() else "image",
            "num_frames": self.num_frames.value(),
            "fps": self.fps.value(),
            "resolution_width": self.resolution_width.value(),
            "resolution_height": self.resolution_height.value(),
            "steps": self.steps.value(),
            "batch_size": self.batch_size.value(),
            "learning_rate": self.learning_rate.text(),
            "lora_rank": self.lora_rank.value(),
            "lora_alpha": self.lora_alpha.value(),
            "save_every": self.save_every.value(),
            "sample_every": self.sample_every.value(),
            "quantize": self.quantize.isChecked(),
            "quantize_te": self.quantize_te.isChecked(),
            "low_vram": self.low_vram.isChecked(),
            "gradient_checkpointing": self.gradient_checkpointing.isChecked(),
            "use_ema": self.use_ema.isChecked(),
            "ema_decay": self.ema_decay.value(),
            "enable_sampling": self.enable_sampling.isChecked(),
            "sample_prompts": self.sample_prompts.toPlainText(),
            "guidance_scale": self.guidance_scale.value(),
            "sample_steps": self.sample_steps.value(),
            # Wan 2.1 specific
            "unload_text_encoder": self.unload_text_encoder.isChecked(),
            # Wan 2.2 specific
            "switch_boundary_every": self.switch_boundary_every.value(),
            "train_high_noise": self.train_high_noise.isChecked(),
            "train_low_noise": self.train_low_noise.isChecked(),
            "cache_text_embeddings": self.cache_text_embeddings.isChecked()
        }

    def save_current_config(self):
        """Save current configuration to JSON"""
        save_config(self.get_config())

    def load_saved_config(self):
        """Load saved configuration from JSON"""
        config = load_config()
        if not config:
            return

        # Model version
        version = config.get("model_version", "wan21_14b")
        if version == "wan21_14b":
            self.wan21_14b_radio.setChecked(True)
        elif version == "wan21_1b":
            self.wan21_1b_radio.setChecked(True)
        elif version == "wan22_14b":
            self.wan22_14b_radio.setChecked(True)

        # Training mode
        if config.get("training_mode") == "video":
            self.video_mode_radio.setChecked(True)
            self.video_params_group.setVisible(True)
        else:
            self.image_mode_radio.setChecked(True)

        # Load all other values
        if "output_dir" in config: self.output_dir.setText(config["output_dir"])
        if "output_name" in config: self.output_name.setText(config["output_name"])
        if "trigger_word" in config: self.trigger_word.setText(config["trigger_word"])
        if "num_frames" in config: self.num_frames.setValue(config["num_frames"])
        if "fps" in config: self.fps.setValue(config["fps"])
        if "resolution_width" in config: self.resolution_width.setValue(config["resolution_width"])
        if "resolution_height" in config: self.resolution_height.setValue(config["resolution_height"])
        if "steps" in config: self.steps.setValue(config["steps"])
        if "batch_size" in config: self.batch_size.setValue(config["batch_size"])
        if "learning_rate" in config: self.learning_rate.setText(config["learning_rate"])
        if "lora_rank" in config: self.lora_rank.setValue(config["lora_rank"])
        if "lora_alpha" in config: self.lora_alpha.setValue(config["lora_alpha"])
        if "save_every" in config: self.save_every.setValue(config["save_every"])
        if "sample_every" in config: self.sample_every.setValue(config["sample_every"])
        if "quantize" in config: self.quantize.setChecked(config["quantize"])
        if "quantize_te" in config: self.quantize_te.setChecked(config["quantize_te"])
        if "low_vram" in config: self.low_vram.setChecked(config["low_vram"])
        if "gradient_checkpointing" in config: self.gradient_checkpointing.setChecked(config["gradient_checkpointing"])
        if "use_ema" in config: self.use_ema.setChecked(config["use_ema"])
        if "ema_decay" in config: self.ema_decay.setValue(config["ema_decay"])
        if "enable_sampling" in config: self.enable_sampling.setChecked(config["enable_sampling"])
        if "sample_prompts" in config: self.sample_prompts.setPlainText(config["sample_prompts"])
        if "guidance_scale" in config: self.guidance_scale.setValue(config["guidance_scale"])
        if "sample_steps" in config: self.sample_steps.setValue(config["sample_steps"])
        # Wan 2.1
        if "unload_text_encoder" in config: self.unload_text_encoder.setChecked(config["unload_text_encoder"])
        # Wan 2.2
        if "switch_boundary_every" in config: self.switch_boundary_every.setValue(config["switch_boundary_every"])
        if "train_high_noise" in config: self.train_high_noise.setChecked(config["train_high_noise"])
        if "train_low_noise" in config: self.train_low_noise.setChecked(config["train_low_noise"])
        if "cache_text_embeddings" in config: self.cache_text_embeddings.setChecked(config["cache_text_embeddings"])

        # Update visibility based on version
        is_wan22 = version == "wan22_14b"
        self.wan22_group.setVisible(is_wan22)
        self.wan21_group.setVisible(not is_wan22)
