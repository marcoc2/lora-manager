from PyQt6.QtWidgets import (QDialog, QFormLayout, QLineEdit, QComboBox, QTextEdit, QCheckBox, QHBoxLayout, QPushButton, QVBoxLayout)

class CaptionConfigDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Caption Configuration")
        self.setModal(True)
        
        layout = QFormLayout()
        
        # Adiciona seleção do método
        self.method_combo = QComboBox()
        self.method_combo.addItems(["Florence-2", "Danbooru", "Janus-7B"])
        layout.addRow("Captioning Method:", self.method_combo)
        
        # Campo para prefixo
        self.prefix = QLineEdit()
        layout.addRow("Caption Prefix:", self.prefix)
        
        # Para método Danbooru, adiciona seleção do modelo
        self.model_combo = QComboBox()
        self.model_combo.addItems(["vit", "swinv2", "convnext"])
        self.model_combo.setVisible(False)
        layout.addRow("Danbooru Model:", self.model_combo)
        
        # Para Janus-7B, adiciona campo de contexto e checkbox
        self.janus_context = QTextEdit()
        self.janus_context.setPlaceholderText("Enter additional context for Janus prompt (optional)")
        self.janus_context.setMaximumHeight(100)
        self.janus_context.setVisible(False)
        layout.addRow("Janus Context:", self.janus_context)
        
        self.replace_prompt = QCheckBox("Replace Default Prompt")
        self.replace_prompt.setVisible(False)
        layout.addRow("", self.replace_prompt)
        
        # Conecta evento de mudança do método
        self.method_combo.currentTextChanged.connect(self.on_method_changed)
        
        # Botões
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
        """Mostra/esconde opções específicas de cada método"""
        # Danbooru options
        self.model_combo.setVisible(text == "Danbooru")
        
        # Janus options
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
