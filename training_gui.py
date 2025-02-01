# training_gui.py
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, 
                           QPushButton, QMessageBox, QDialog)
from training_tabs import TrainingTabs
from training_widgets import CommandOutputDialog

class TrainingPanel(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.dataset_path = None
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        
        # Training tabs
        self.training_tabs = TrainingTabs(self)
        layout.addWidget(self.training_tabs)
        
        # Training control buttons
        buttons_layout = QHBoxLayout()
        
        train_button = QPushButton("Start Training")
        train_button.clicked.connect(self.start_training)
        buttons_layout.addWidget(train_button)
        
        layout.addLayout(buttons_layout)
        self.setLayout(layout)

    def on_dataset_changed(self, dataset_path):
        self.dataset_path = dataset_path

    def start_training(self):
        if not self.dataset_path:
            QMessageBox.warning(self, "Warning", "Please select a dataset folder first!")
            return
        
        try:
            self.training_tabs.save_config()
            
            command = self.training_tabs.get_command(self.dataset_path)
            if command is None:
                return
            
            msg = QMessageBox()
            msg.setWindowTitle("Training Command")
            msg.setText("The following command will be executed:")
            msg.setDetailedText(command)
            msg.setStandardButtons(QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel)
            
            if msg.exec() == QMessageBox.StandardButton.Ok:
                output_dialog = CommandOutputDialog(command, self)
                output_dialog.exec()
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error starting training: {str(e)}")