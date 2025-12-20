from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QTabWidget,
                            QMessageBox)
from training_widgets import TrainingWidgets
from flux_widgets_ui import FluxTrainingWidgets
from qwen_widgets_ui import QwenTrainingWidgets
from zimage_widgets_ui import ZImageTrainingWidgets
from wan_widgets_ui import WanTrainingWidgets
from controllers.training_controller import TrainingController

class TrainingTabs(QWidget):
    def __init__(self, parent=None, queue_manager=None):
        super().__init__(parent)
        self.parent = parent
        self.queue_manager = queue_manager
        self.controller = TrainingController(self)

        # Pass script manager if available from parent
        if hasattr(self.parent, 'script_manager'):
            self.controller.set_script_manager(self.parent.script_manager)

        self.init_ui()

    @property
    def project_manager(self):
        """Get project manager from parent window"""
        if hasattr(self.parent, 'project_manager'):
            return self.parent.project_manager
        return None

    def _ensure_project_exists(self, dataset_path):
        """Ensure a project exists for the dataset, create if needed"""
        pm = self.project_manager
        if pm is None:
            return

        if not pm.has_project:
            # Create project automatically when training starts
            pm.create_project(dataset_path)
            print(f"[TrainingTabs] Auto-created project for training")

    def _record_training_run(self, model_type: str, config: dict, output_name: str):
        """Record a training run in the project history"""
        pm = self.project_manager
        if pm is None or not pm.has_project:
            return

        steps = config.get("steps", 0)
        output_path = config.get("output_dir", "")

        pm.add_training_run(
            model_type=model_type,
            steps=steps,
            output_name=output_name,
            output_path=output_path,
            config_snapshot=config
        )

        # Also save the model config to project
        pm.set_model_config(model_type, config)

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
        self.zimage_widget = ZImageTrainingWidgets(self)
        self.wan_widget = WanTrainingWidgets(self)

        # Add widgets to tabs
        self.tabs.addTab(self.training_widget, "LoRA Training")
        self.tabs.addTab(self.flux_widget, "Flux Training")
        self.tabs.addTab(self.qwen_widget, "Qwen-Image Training")
        self.tabs.addTab(self.zimage_widget, "Z-Image Training")
        self.tabs.addTab(self.wan_widget, "Wan 2.1/2.2 Training")
        
        # Connect training buttons to queue
        self.training_widget.train_button.clicked.connect(self.queue_training_task)
        self.flux_widget.train_button.clicked.connect(self.queue_flux_training_task)
        self.qwen_widget.train_button.clicked.connect(self.queue_qwen_training_task)
        self.zimage_widget.train_button.clicked.connect(self.queue_zimage_training_task)
        self.wan_widget.train_button.clicked.connect(self.queue_wan_training_task)

        # Add tabs to main layout
        main_layout.addWidget(self.tabs)
        
        self.setLayout(main_layout)

    def queue_training_task(self):
        """Add a LoRA training task to the queue"""
        # Legacy SD1.5/XL training - keeping as is for now or should refactor too?
        # The user asked for Flux and Qwen refactoring specifically.
        # I'll keep this one as is to minimize scope creep unless requested.
        dataset_path = self.parent.get_effective_dataset_path()
        if not dataset_path:
            QMessageBox.warning(self, "Warning", "Please select a dataset folder first!")
            return
            
        try:
            self.training_widget.save_current_config()
            command = self.training_widget.get_command(dataset_path)
            if command is None:
                return
            output_name = self.training_widget.output_name.text() or "lora_training"
            self.queue_manager.add_task(command, dataset_path, output_name, metadata={"model_type": "sd"})
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
            # Ensure project exists (auto-create if needed)
            self._ensure_project_exists(self.parent.dataset_path)

            # Save config
            self.flux_widget.save_current_config()

            # Get config from view
            config = self.flux_widget.get_config()

            # Get command from controller
            command, error = self.controller.get_flux_command(config, dataset_path)

            if error:
                QMessageBox.warning(self, "Validation Error", error)
                return

            if command is None:
                return

            # Add to queue
            output_name = config.get("output_name") or "flux_training"
            print(f"Queueing task: {output_name}")
            self.queue_manager.add_task(command, dataset_path, output_name, metadata={
                "model_type": "flux",
                "config_snapshot": config
            })

            # Record in project history
            self._record_training_run("flux", config, output_name)

            QMessageBox.information(self, "Success", f"Training task '{output_name}' added to queue!")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error queuing training task: {str(e)}")

    def queue_qwen_training_task(self):
        """Add a Qwen-Image training task to the queue (ai-toolkit)"""
        dataset_path = self.parent.get_effective_dataset_path()
        if not dataset_path:
            QMessageBox.warning(self, "Warning", "Please select a dataset folder first!")
            return

        try:
            # Ensure project exists (auto-create if needed)
            self._ensure_project_exists(self.parent.dataset_path)

            # Save config
            self.qwen_widget.save_current_config()

            # Get config from view
            config = self.qwen_widget.get_config()

            # Get command from controller
            command, error = self.controller.get_qwen_command(config, dataset_path)

            if error:
                QMessageBox.warning(self, "Validation Error", error)
                return

            if command is None:
                return

            # Add to queue
            output_name = config.get("output_name") or "qwen_training"
            print(f"Queueing task: {output_name}")
            self.queue_manager.add_task(command, dataset_path, output_name, metadata={
                "model_type": "qwen",
                "config_snapshot": config
            })

            # Record in project history
            self._record_training_run("qwen", config, output_name)

            QMessageBox.information(self, "Success", f"Training task '{output_name}' added to queue!")

        except Exception as e:
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "Error", f"Error queuing training task: {str(e)}")

    def queue_zimage_training_task(self):
        """Add a Z-Image training task to the queue"""
        dataset_path = self.parent.get_effective_dataset_path()
        if not dataset_path:
            QMessageBox.warning(self, "Warning", "Please select a dataset folder first!")
            return

        try:
            # Ensure project exists (auto-create if needed)
            self._ensure_project_exists(self.parent.dataset_path)

            self.zimage_widget.save_current_config()
            config = self.zimage_widget.get_config()

            command, error = self.controller.get_zimage_command(config, dataset_path)

            if error:
                QMessageBox.warning(self, "Validation Error", error)
                return

            if command is None:
                return

            output_name = config.get("output_name") or "zimage_training"
            self.queue_manager.add_task(command, dataset_path, output_name, metadata={
                "model_type": "zimage_turbo",
                "config_snapshot": config
            })

            # Record in project history
            self._record_training_run("zimage", config, output_name)

            QMessageBox.information(self, "Success", f"Training task '{output_name}' added to queue!")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error queuing training task: {str(e)}")

    def queue_wan_training_task(self):
        """Add a Wan 2.1/2.2 training task to the queue"""
        dataset_path = self.parent.get_effective_dataset_path()
        if not dataset_path:
            QMessageBox.warning(self, "Warning", "Please select a dataset folder first!")
            return

        try:
            # Ensure project exists (auto-create if needed)
            self._ensure_project_exists(self.parent.dataset_path)

            self.wan_widget.save_current_config()
            config = self.wan_widget.get_config()

            command, error = self.controller.get_wan_command(config, dataset_path)

            if error:
                QMessageBox.warning(self, "Validation Error", error)
                return

            if command is None:
                return

            output_name = config.get("output_name") or "wan_training"
            model_version = config.get("model_version", "wan21_14b")
            self.queue_manager.add_task(command, dataset_path, output_name, metadata={
                "model_type": f"wan_{model_version}",
                "config_snapshot": config
            })

            # Record in project history
            self._record_training_run("wan", config, output_name)

            QMessageBox.information(self, "Success", f"Training task '{output_name}' added to queue!")

        except Exception as e:
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "Error", f"Error queuing training task: {str(e)}")

    def save_config(self):
        """Save configurations for all widgets"""
        self.training_widget.save_current_config()
        self.flux_widget.save_current_config()
        self.qwen_widget.save_current_config()
        self.zimage_widget.save_current_config()
        self.wan_widget.save_current_config()