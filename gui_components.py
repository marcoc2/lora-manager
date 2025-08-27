from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, 
                           QPushButton, QLineEdit, QSpinBox, QFormLayout,
                           QComboBox, QTextEdit, QCheckBox)

class SuffixInputDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Enter Suffix")
        self.setModal(True)

        layout = QVBoxLayout()
        self.suffix_input = QLineEdit()
        self.suffix_input.setPlaceholderText("Enter suffix (e.g., _XXX)")
        layout.addWidget(QLabel("Suffix for renaming:"))
        layout.addWidget(self.suffix_input)

        buttons = QHBoxLayout()
        ok_button = QPushButton("OK")
        cancel_button = QPushButton("Cancel")
        ok_button.clicked.connect(self.accept)
        cancel_button.clicked.connect(self.reject)
        buttons.addWidget(ok_button)
        buttons.addWidget(cancel_button)

        layout.addLayout(buttons)
        self.setLayout(layout)

    def get_suffix(self):
        return self.suffix_input.text().strip()

class CaptionConfigDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Caption Configuration")
        self.setModal(True)
        
        layout = QFormLayout()
        
        # Method selection
        self.method_combo = QComboBox()
        self.method_combo.addItems(["Florence-2", "Danbooru", "Janus-7B"])
        layout.addRow("Captioning Method:", self.method_combo)
        
        # Prefix field
        self.prefix = QLineEdit()
        layout.addRow("Caption Prefix:", self.prefix)
        
        # Danbooru model selection
        self.model_combo = QComboBox()
        self.model_combo.addItems(["vit", "swinv2", "convnext"])
        self.model_combo.setVisible(False)
        layout.addRow("Danbooru Model:", self.model_combo)
        
        # Janus options
        self.janus_context = QTextEdit()
        self.janus_context.setPlaceholderText("Enter additional context for Janus prompt (optional)")
        self.janus_context.setMaximumHeight(100)
        self.janus_context.setVisible(False)
        layout.addRow("Janus Context:", self.janus_context)
        
        self.replace_prompt = QCheckBox("Replace Default Prompt")
        self.replace_prompt.setVisible(False)
        layout.addRow("", self.replace_prompt)
        
        self.method_combo.currentTextChanged.connect(self.on_method_changed)
        
        # Buttons
        buttons = QHBoxLayout()
        ok_button = QPushButton("OK")
        cancel_button = QPushButton("Cancel")
        
        ok_button.clicked.connect(self.accept)
        cancel_button.clicked.connect(self.reject)
        
        buttons.addWidget(ok_button)
        buttons.addWidget(cancel_button)
        
        final_layout = QVBoxLayout()
        final_layout.addLayout(layout)
        final_layout.addLayout(buttons)
        
        self.setLayout(final_layout)
    
    def on_method_changed(self, text):
        self.model_combo.setVisible(text == "Danbooru")
        self.janus_context.setVisible(text == "Janus-7B")
        self.replace_prompt.setVisible(text == "Janus-7B")
        
    def get_values(self):
        return {
            'method': self.method_combo.currentText(),
            'prefix': self.prefix.text(),
            'model_type': self.model_combo.currentText() if self.method_combo.currentText() == "Danbooru" else None,
            'janus_context': self.janus_context.toPlainText() if self.method_combo.currentText() == "Janus-7B" else None,
            'replace_prompt': self.replace_prompt.isChecked() if self.method_combo.currentText() == "Janus-7B" else False
        }

class TomlConfigDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Dataset Configuration")
        self.setModal(True)
        
        layout = QFormLayout()
        
        self.class_tokens = QLineEdit()
        self.num_repeats = QSpinBox()
        self.num_repeats.setRange(1, 100)
        self.num_repeats.setValue(1)
        
        self.resolution = QSpinBox()
        self.resolution.setRange(64, 2048)
        self.resolution.setValue(512)
        self.resolution.setSingleStep(64)
        
        layout.addRow("Class Tokens:", self.class_tokens)
        layout.addRow("Number of Repeats:", self.num_repeats)
        layout.addRow("Resolution:", self.resolution)
        
        buttons = QHBoxLayout()
        ok_button = QPushButton("OK")
        cancel_button = QPushButton("Cancel")
        
        ok_button.clicked.connect(self.accept)
        cancel_button.clicked.connect(self.reject)
        
        buttons.addWidget(ok_button)
        buttons.addWidget(cancel_button)
        
        final_layout = QVBoxLayout()
        final_layout.addLayout(layout)
        final_layout.addLayout(buttons)
        
        self.setLayout(final_layout)
    
    def get_values(self):
        return {
            'class_tokens': self.class_tokens.text(),
            'num_repeats': self.num_repeats.value(),
            'resolution': self.resolution.value()
        }

class QwenTomlConfigDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Qwen-Image Dataset Configuration")
        self.setModal(True)
        
        layout = QFormLayout()
        
        # Qwen-specific configuration
        self.caption_extension = QLineEdit()
        self.caption_extension.setText(".txt")
        
        self.batch_size = QSpinBox()
        self.batch_size.setRange(1, 16)
        self.batch_size.setValue(1)
        
        self.enable_bucket = QCheckBox("Enable Bucket")
        self.enable_bucket.setChecked(True)
        
        self.bucket_no_upscale = QCheckBox("Bucket No Upscale")
        self.bucket_no_upscale.setChecked(False)
        
        # Resolution with width/height
        self.resolution_width = QSpinBox()
        self.resolution_width.setRange(256, 2048)
        self.resolution_width.setValue(768)
        self.resolution_width.setSingleStep(64)
        
        self.resolution_height = QSpinBox()
        self.resolution_height.setRange(256, 2048)
        self.resolution_height.setValue(768)
        self.resolution_height.setSingleStep(64)
        
        self.num_repeats = QSpinBox()
        self.num_repeats.setRange(1, 100)
        self.num_repeats.setValue(1)
        
        layout.addRow("Caption Extension:", self.caption_extension)
        layout.addRow("Batch Size:", self.batch_size)
        layout.addRow("", self.enable_bucket)
        layout.addRow("", self.bucket_no_upscale)
        
        # Resolution layout
        res_layout = QHBoxLayout()
        res_layout.addWidget(self.resolution_width)
        res_layout.addWidget(QLabel("x"))
        res_layout.addWidget(self.resolution_height)
        layout.addRow("Resolution (W x H):", res_layout)
        
        layout.addRow("Number of Repeats:", self.num_repeats)
        
        buttons = QHBoxLayout()
        ok_button = QPushButton("OK")
        cancel_button = QPushButton("Cancel")
        
        ok_button.clicked.connect(self.accept)
        cancel_button.clicked.connect(self.reject)
        
        buttons.addWidget(ok_button)
        buttons.addWidget(cancel_button)
        
        final_layout = QVBoxLayout()
        final_layout.addLayout(layout)
        final_layout.addLayout(buttons)
        
        self.setLayout(final_layout)
    
    def get_values(self):
        return {
            'caption_extension': self.caption_extension.text(),
            'batch_size': self.batch_size.value(),
            'enable_bucket': self.enable_bucket.isChecked(),
            'bucket_no_upscale': self.bucket_no_upscale.isChecked(),
            'resolution': [self.resolution_width.value(), self.resolution_height.value()],
            'num_repeats': self.num_repeats.value()
        }