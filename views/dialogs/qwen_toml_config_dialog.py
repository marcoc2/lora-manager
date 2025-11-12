from PyQt6.QtWidgets import (QDialog, QFormLayout, QLineEdit, QSpinBox, QHBoxLayout, QPushButton, QVBoxLayout, QCheckBox, QLabel)

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
