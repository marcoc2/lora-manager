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