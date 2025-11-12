from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QLineEdit, QHBoxLayout, QPushButton, QLabel)

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
