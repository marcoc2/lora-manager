from PyQt6.QtWidgets import (QDialog, QFormLayout, QLineEdit, QSpinBox, QHBoxLayout, QPushButton, QVBoxLayout)

class TomlConfigDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Dataset Configuration")
        self.setModal(True)
        
        # Layout
        layout = QFormLayout()
        
        # Campos
        self.class_tokens = QLineEdit()
        self.num_repeats = QSpinBox()
        self.num_repeats.setRange(1, 100)
        self.num_repeats.setValue(1)
        
        self.resolution = QSpinBox()
        self.resolution.setRange(64, 2048)
        self.resolution.setValue(512)
        self.resolution.setSingleStep(64)
        
        # Adiciona campos ao layout
        layout.addRow("Class Tokens:", self.class_tokens)
        layout.addRow("Number of Repeats:", self.num_repeats)
        layout.addRow("Resolution:", self.resolution)
        
        # Botões
        buttons = QHBoxLayout()
        ok_button = QPushButton("OK")
        cancel_button = QPushButton("Cancel")
        
        ok_button.clicked.connect(self.accept)
        cancel_button.clicked.connect(self.reject)
        
        buttons.addWidget(ok_button)
        buttons.addWidget(cancel_button)
        
        # Layout final
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
