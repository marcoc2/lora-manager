from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QTabWidget,
                            QMessageBox)
from training_widgets import TrainingWidgets
from flux_widgets_ui import FluxTrainingWidgets
from qwen_widgets_ui import QwenTrainingWidgets

class TrainingTabs(QWidget):
    def __init__(self, parent=None, queue_manager=None):
        super().__init__(parent)
        self.parent = parent
        self.queue_manager = queue_manager
        self.init_ui()

    def init_ui(self):
        # Main layout
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(10, 10, 10, 10)
        
        # Create tab widget for training options
        self.tabs = QTabWidget()
        
        # Create training widgets
        self.training_widget = TrainingWidgets(self)
        self.flux_widget = FluxTrainingWidgets(self)
        self.qwen_widget = QwenTrainingWidgets(self)
        
        # Add widgets to tabs
        self.tabs.addTab(self.training_widget, "LoRA Training")
        self.tabs.addTab(self.flux_widget, "Flux Training")
        self.tabs.addTab(self.qwen_widget, "Qwen-Image Training")
        
        # Connect training buttons to queue
        self.training_widget.train_button.clicked.connect(self.queue_training_task)
        self.flux_widget.train_button.clicked.connect(self.queue_flux_training_task)
        self.qwen_widget.train_button.clicked.connect(self.queue_qwen_training_task)
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QTabWidget,
                            QMessageBox)
from training_widgets import TrainingWidgets
from flux_widgets_ui import FluxTrainingWidgets
from qwen_widgets_ui import QwenTrainingWidgets

class TrainingTabs(QWidget):
    def __init__(self, parent=None, queue_manager=None):
        super().__init__(parent)
        self.parent = parent
        self.queue_manager = queue_manager
        self.init_ui()

    def init_ui(self):
        # Main layout
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(10, 10, 10, 10)
        
        # Create tab widget for training options
        self.tabs = QTabWidget()
        
        # Create training widgets
        self.training_widget = TrainingWidgets(self)
        self.flux_widget = FluxTrainingWidgets(self)
        self.qwen_widget = QwenTrainingWidgets(self)
        
        # Add widgets to tabs
        self.tabs.addTab(self.training_widget, "LoRA Training")
        self.tabs.addTab(self.flux_widget, "Flux Training")
        self.tabs.addTab(self.qwen_widget, "Qwen-Image Training")
        
        # Connect training buttons to queue
        self.training_widget.train_button.clicked.connect(self.queue_training_task)
        self.flux_widget.train_button.clicked.connect(self.queue_flux_training_task)
        self.qwen_widget.train_button.clicked.connect(self.queue_qwen_training_task)
        
        # Add tabs to main layout
        main_layout.addWidget(self.tabs)
        
        self.setLayout(main_layout)

    def queue_training_task(self):
        """Add a LoRA training task to the queue"""
        dataset_path = self.parent.get_effective_dataset_path()
        if not dataset_path:
            QMessageBox.warning(self, "Warning", "Please select a dataset folder first!")
            return
            
        try:
            # Save current config
            self.training_widget.save_current_config()
            
            # Get command
            command = self.training_widget.get_command(dataset_path)
            if command is None:
                return
                
            # Add to queue
            output_name = self.training_widget.output_name.text() or "lora_training"
            print(f"Queueing task: {output_name}")  # Debug print
            self.queue_manager.add_task(command, dataset_path, output_name)
            QMessageBox.information(self, "Success", f"Training task '{output_name}' added to queue!")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error queuing training task: {str(e)}")

    def queue_flux_training_task(self):
        """Add a Flux training task to the queue"""
        dataset_path = self.parent.get_effective_dataset_path()
        if not dataset_path:
            QMessageBox.warning(self, "Warning", "Please select a dataset folder first!")
            return
            
        try:
            # Save current config
            self.flux_widget.save_current_config()
            
            # Get command
            command = self.flux_widget.get_command(dataset_path)
            if command is None:
                return
                
            # Add to queue
            output_name = self.flux_widget.output_name.text() or "flux_training"
            print(f"Queueing task: {output_name}")  # Debug print
            self.queue_manager.add_task(command, dataset_path, output_name)
            QMessageBox.information(self, "Success", f"Training task '{output_name}' added to queue!")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error queuing training task: {str(e)}")

    def queue_qwen_training_task(self):
        """Add a Qwen-Image training task to the queue"""
        dataset_path = self.parent.get_effective_dataset_path()
        if not dataset_path:
            QMessageBox.warning(self, "Warning", "Please select a dataset folder first!")
            return
            
        try:
            # Save current config
            self.qwen_widget.save_current_config()
            
            # Get command
            command = self.qwen_widget.get_command(dataset_path)
            if command is None:
                return
                
            # Add to queue
            output_name = self.qwen_widget.output_name.text() or "qwen_training"
            print(f"Queueing task: {output_name}")  # Debug print
            self.queue_manager.add_task(command, dataset_path, output_name)
            QMessageBox.information(self, "Success", f"Training task '{output_name}' added to queue!")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error queuing training task: {str(e)}")

    def save_config(self):
        """Save configurations for all widgets"""
        self.training_widget.save_current_config()
        self.flux_widget.save_current_config()
        self.qwen_widget.save_current_config()