from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                            QLineEdit, QPushButton, QSpinBox, QDoubleSpinBox, 
                            QCheckBox, QComboBox, QGroupBox, QFileDialog, QTextEdit,
                            QScrollArea, QFrame)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QPixmap
import json
from pathlib import Path

def save_config(config, filename="zimage_training_config.json"):
    try:
        with open(filename, 'w') as f:
            json.dump(config, f, indent=4)
    except Exception as e:
        print(f"Error saving config: {e}")

def load_config(filename="zimage_training_config.json"):
    try:
        if Path(filename).exists():
            with open(filename, 'r') as f:
                return json.load(f)
    except Exception as e:
        print(f"Error loading config: {e}")
    return {}

class ZImageTrainingWidgets(QWidget):
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
        
        # Model Path
        model_path_layout = QHBoxLayout()
        self.model_path = QLineEdit()
        self.model_path.setPlaceholderText("Path to Z-Image Model (HuggingFace ID or Local Path)")
        self.model_path.setText("Tongyi-MAI/Z-Image-Turbo") # Official repo default
        btn_model = QPushButton("Select Model")
        btn_model.clicked.connect(lambda: self.select_path(self.model_path, is_file=True))
        model_path_layout.addWidget(QLabel("Model Path:"))
        model_path_layout.addWidget(self.model_path)
        model_path_layout.addWidget(btn_model)
        model_layout.addLayout(model_path_layout)

        # Adapter Path
        adapter_path_layout = QHBoxLayout()
        self.adapter_path = QLineEdit()
        self.adapter_path.setPlaceholderText("Path to Adapter (.safetensors)")
        self.adapter_path.setText("./models/zimage_turbo_training_adapter_v2.safetensors")
        btn_adapter = QPushButton("Select Adapter")
        btn_adapter.clicked.connect(lambda: self.select_path(self.adapter_path, is_file=True))
        adapter_path_layout.addWidget(QLabel("Adapter Path:"))
        adapter_path_layout.addWidget(self.adapter_path)
        adapter_path_layout.addWidget(btn_adapter)
        model_layout.addLayout(adapter_path_layout)

        # Text Encoder Path (Optional)
        te_path_layout = QHBoxLayout()
        self.te_path = QLineEdit()
        self.te_path.setPlaceholderText("Path to Text Encoder (Optional, for single file models)")
        btn_te = QPushButton("Select TE")
        btn_te.clicked.connect(lambda: self.select_path(self.te_path, is_file=False)) # Can be file or folder
        te_path_layout.addWidget(QLabel("Text Encoder:"))
        te_path_layout.addWidget(self.te_path)
        te_path_layout.addWidget(btn_te)
        model_layout.addLayout(te_path_layout)

        # Tokenizer Path (Optional)
        tokenizer_path_layout = QHBoxLayout()
        self.tokenizer_path = QLineEdit()
        self.tokenizer_path.setPlaceholderText("Path to Tokenizer (Optional, for single file models)")
        btn_tokenizer = QPushButton("Select Tokenizer")
        btn_tokenizer.clicked.connect(lambda: self.select_path(self.tokenizer_path, is_file=False)) # Can be file or folder
        tokenizer_path_layout.addWidget(QLabel("Tokenizer:"))
        tokenizer_path_layout.addWidget(self.tokenizer_path)
        tokenizer_path_layout.addWidget(btn_tokenizer)
        model_layout.addLayout(tokenizer_path_layout)

        model_group.setLayout(model_layout)
        layout.addWidget(model_group)

        # --- Training Parameters ---
        params_group = QGroupBox("Training Parameters")
        params_layout = QVBoxLayout()
        
        # Grid for basic params
        grid_layout = QHBoxLayout()
        
        # Steps
        steps_layout = QVBoxLayout()
        self.steps = QSpinBox()
        self.steps.setRange(100, 100000)
        self.steps.setValue(2000)
        steps_layout.addWidget(QLabel("Training Steps:"))
        steps_layout.addWidget(self.steps)
        grid_layout.addLayout(steps_layout)

        # Batch Size
        batch_layout = QVBoxLayout()
        self.batch_size = QSpinBox()
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

        # Resolution
        res_layout = QVBoxLayout()
        self.resolution = QSpinBox()
        self.resolution.setRange(256, 2048)
        self.resolution.setSingleStep(64)
        self.resolution.setValue(1024)
        res_layout.addWidget(QLabel("Resolution:"))
        res_layout.addWidget(self.resolution)
        grid_layout.addLayout(res_layout)

        params_layout.addLayout(grid_layout)

        # LoRA Config
        lora_layout = QHBoxLayout()
        
        self.rank = QSpinBox()
        self.rank.setRange(1, 1024)
        self.rank.setValue(16)
        lora_layout.addWidget(QLabel("LoRA Rank:"))
        lora_layout.addWidget(self.rank)

        self.alpha = QSpinBox()
        self.alpha.setRange(1, 1024)
        self.alpha.setValue(16)
        lora_layout.addWidget(QLabel("LoRA Alpha:"))
        lora_layout.addWidget(self.alpha)

        params_layout.addLayout(lora_layout)
        params_group.setLayout(params_layout)
        layout.addWidget(params_group)

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
        output_dir_layout.addWidget(self.output_dir)
        output_dir_layout.addWidget(btn_output_dir)

        output_layout.addWidget(QLabel("Output Directory:"))
        output_layout.addLayout(output_dir_layout)

        # Output Name
        name_layout = QHBoxLayout()
        self.output_name = QLineEdit()
        self.output_name.setPlaceholderText("my_zimage_lora")
        name_layout.addWidget(QLabel("Output Name:"))
        name_layout.addWidget(self.output_name)
        output_layout.addLayout(name_layout)

        # Save Every
        save_layout = QHBoxLayout()
        self.save_every = QSpinBox()
        self.save_every.setRange(1, 10000)
        self.save_every.setValue(250)
        save_layout.addWidget(QLabel("Save Every (Steps):"))
        save_layout.addWidget(self.save_every)
        output_layout.addLayout(save_layout)

        # Sample Every
        sample_every_layout = QHBoxLayout()
        self.sample_every = QSpinBox()
        self.sample_every.setRange(1, 10000)
        self.sample_every.setValue(250)
        sample_every_layout.addWidget(QLabel("Sample Every (Steps):"))
        sample_every_layout.addWidget(self.sample_every)
        output_layout.addLayout(sample_every_layout)

        output_group.setLayout(output_layout)
        layout.addWidget(output_group)

        # --- Sample Prompts ---
        sample_group = QGroupBox("Sample Prompts & Preview")
        sample_layout = QVBoxLayout()

        # Enable sampling checkbox
        self.enable_sampling = QCheckBox("Enable preview generation during training")
        self.enable_sampling.setChecked(True)  # Default enabled
        self.enable_sampling.stateChanged.connect(self.on_sampling_toggled)
        sample_layout.addWidget(self.enable_sampling)

        # Sampler Selection
        sampler_layout = QHBoxLayout()
        self.sampler = QComboBox()
        self.sampler.addItems(["flowmatch", "euler", "euler_a", "dpm_2", "dpm_2_a", "ddim"])
        self.sampler.setCurrentText("flowmatch")
        sampler_layout.addWidget(QLabel("Preview Sampler:"))
        sampler_layout.addWidget(self.sampler)
        sample_layout.addLayout(sampler_layout)

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
        self.sample_prompts.setMinimumHeight(100)
        sample_layout.addWidget(self.sample_prompts)



        sample_group.setLayout(sample_layout)
        layout.addWidget(sample_group)



        # Buttons Layout
        buttons_layout = QHBoxLayout()
        
        # Reload Config Button
        self.reload_btn = QPushButton("Reload Config")
        self.reload_btn.setToolTip("Reload settings from zimage_training_config.json")
        self.reload_btn.clicked.connect(self.load_saved_config)
        buttons_layout.addWidget(self.reload_btn)

        # Start Button
        self.train_button = QPushButton("Start Z-Image Training")
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
        self.auto_fill_btn.setEnabled(is_checked)
        self.save_current_config()

    def on_sampling_toggled(self, state):
        """Enable/disable sample prompts based on checkbox"""
        from PyQt6.QtCore import Qt
        is_checked = (state == Qt.CheckState.Checked.value)
        self.sample_prompts.setEnabled(is_checked)
        self.auto_fill_btn.setEnabled(is_checked)
        self.save_current_config()



    def auto_fill_prompts(self):
        """Auto-fill prompts with the first caption from the dataset"""
        try:
            # Get dataset path from parent (main window)
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
                QMessageBox.warning(self, "No Captions Found", f"No caption files found in dataset directory (recursively):\n{dataset_path}")
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

    def get_config(self):
        """Returns the current configuration as a dictionary"""
        return {
            "model_path": self.model_path.text(),
            "adapter_path": self.adapter_path.text(),
            "text_encoder_path": self.te_path.text(),
            "tokenizer_path": self.tokenizer_path.text(),
            "steps": self.steps.value(),
            "batch_size": self.batch_size.value(),
            "learning_rate": self.learning_rate.text(),
            "resolution": self.resolution.value(),
            "rank": self.rank.value(),
            "alpha": self.alpha.value(),
            "output_dir": self.output_dir.text(),
            "output_name": self.output_name.text(),
            "save_every": self.save_every.value(),
            "sample_every": self.sample_every.value(),
            "sample_prompts": self.sample_prompts.toPlainText(),
            "enable_sampling": self.enable_sampling.isChecked(),
            "sampler": self.sampler.currentText()
        }

    def save_current_config(self):
        """Salva a configuração atual no arquivo JSON"""
        save_config(self.get_config())

    def load_saved_config(self):
        config = load_config()
        if not config:
            return

        if "model_path" in config: self.model_path.setText(config["model_path"])
        if "adapter_path" in config: self.adapter_path.setText(config["adapter_path"])
        if "text_encoder_path" in config: self.te_path.setText(config["text_encoder_path"])
        if "tokenizer_path" in config: self.tokenizer_path.setText(config["tokenizer_path"])
        if "steps" in config: self.steps.setValue(config["steps"])
        if "batch_size" in config: self.batch_size.setValue(config["batch_size"])
        if "learning_rate" in config: self.learning_rate.setText(config["learning_rate"])
        if "resolution" in config: self.resolution.setValue(config["resolution"])
        if "rank" in config: self.rank.setValue(config["rank"])
        if "alpha" in config: self.alpha.setValue(config["alpha"])
        if "output_dir" in config: self.output_dir.setText(config["output_dir"])
        if "output_name" in config: self.output_name.setText(config["output_name"])
        if "save_every" in config: self.save_every.setValue(config["save_every"])
        if "sample_every" in config: self.sample_every.setValue(config["sample_every"])
        if "sample_prompts" in config: self.sample_prompts.setText(config["sample_prompts"])
        if "enable_sampling" in config: self.enable_sampling.setChecked(config["enable_sampling"])
        if "sampler" in config: self.sampler.setCurrentText(config["sampler"])
