from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QScrollArea
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QPixmap

class PreviewWindow(QWidget):
    finished = pyqtSignal()

    def __init__(self, image_path, parent=None):
        super().__init__()
        self.setWindowTitle("Training Preview")
        self.resize(800, 800)
        
        layout = QVBoxLayout()
        self.label = QLabel()
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        scroll = QScrollArea()
        scroll.setWidget(self.label)
        scroll.setWidgetResizable(True)
        layout.addWidget(scroll)
        
        self.setLayout(layout)
        
        if image_path:
            self.update_image(image_path)

    def update_image(self, image_path):
        pixmap = QPixmap(str(image_path))
        if not pixmap.isNull():
            self.label.setPixmap(pixmap)
            self.label.adjustSize()

    def closeEvent(self, event):
        self.finished.emit()
        super().closeEvent(event)
