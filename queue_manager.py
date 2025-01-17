from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                            QPushButton, QListWidget, QListWidgetItem, QGroupBox,
                            QTextEdit, QMessageBox)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, pyqtSlot
import queue
import threading
from pathlib import Path
from datetime import datetime

class TrainingTask:
    def __init__(self, command, dataset_path, output_name):
        self.command = command
        self.dataset_path = Path(dataset_path)
        self.output_name = output_name
        self.status = "Queued"
        self.progress = 0
        self.start_time = None
        self.end_time = None
        
    def get_display_text(self):
        status_emoji = {
            "Queued": "⏳",
            "Running": "▶️",
            "Completed": "✅",
            "Failed": "❌",
        }
        
        elapsed = ""
        if self.start_time:
            if self.end_time:
                elapsed = f" ({(self.end_time - self.start_time).seconds // 60}m)"
            else:
                elapsed = f" ({(datetime.now() - self.start_time).seconds // 60}m)"
                
        return f"{status_emoji[self.status]} {self.output_name} - {self.status}{elapsed}"

class TrainingWorker(QThread):
    task_progress = pyqtSignal(str)
    task_completed = pyqtSignal(bool)
    
    def __init__(self, task):
        super().__init__()
        self.task = task
        
    def run(self):
        try:
            import subprocess
            self.task.start_time = datetime.now()
            print(f"Executing command: {self.task.command}")
            
            process = subprocess.Popen(
                self.task.command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                shell=True,
                universal_newlines=False
            )
            
            while True:
                output = process.stdout.readline()
                if output == b'' and process.poll() is not None:
                    break
                if output:
                    try:
                        line = output.decode('utf-8', errors='replace').strip()
                        print(f"Training output: {line}")
                        self.task_progress.emit(line)
                    except Exception as e:
                        print(f"Error processing output: {e}")
            
            self.task.end_time = datetime.now()
            success = process.returncode == 0
            self.task_completed.emit(success)
            
        except Exception as e:
            print(f"Error in training worker: {e}")
            self.task.end_time = datetime.now()
            self.task_progress.emit(f"Error: {str(e)}")
            self.task_completed.emit(False)

class QueueManager(QWidget):
    signal_add_task = pyqtSignal(object)
    signal_update_task = pyqtSignal(object)
    signal_append_log = pyqtSignal(str)
    signal_clear_log = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.task_queue = queue.Queue()
        self.current_task = None
        self.workers = []
        
        # Conecta sinais aos slots
        self.signal_add_task.connect(self._add_task_to_list)
        self.signal_update_task.connect(self._update_task_in_list)
        self.signal_append_log.connect(self._append_to_log)
        self.signal_clear_log.connect(self._clear_log)
        
        self.init_ui()
        
        # Start the queue processing
        self.queue_processor = threading.Thread(target=self.process_queue, daemon=True)
        self.queue_processor.start()
        
    def init_ui(self):
        layout = QVBoxLayout()
        
        # Queue display group
        queue_group = QGroupBox("Training Queue")
        queue_layout = QVBoxLayout()
        
        # Queue list
        self.queue_list = QListWidget()
        self.queue_list.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.queue_list.setMaximumHeight(200)
        queue_layout.addWidget(self.queue_list)
        
        # Control buttons
        button_layout = QHBoxLayout()
        
        self.clear_completed_btn = QPushButton("Clear Completed")
        self.clear_completed_btn.clicked.connect(self.clear_completed_tasks)
        
        self.clear_all_btn = QPushButton("Clear All")
        self.clear_all_btn.clicked.connect(self.clear_all_tasks)
        
        button_layout.addWidget(self.clear_completed_btn)
        button_layout.addWidget(self.clear_all_btn)
        
        queue_layout.addLayout(button_layout)
        queue_group.setLayout(queue_layout)
        layout.addWidget(queue_group)
        
        # Log output group
        log_group = QGroupBox("Training Output")
        log_layout = QVBoxLayout()
        
        # Log text area
        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.log_output.setMinimumHeight(200)
        log_layout.addWidget(self.log_output)
        
        log_group.setLayout(log_layout)
        layout.addWidget(log_group)
        
        self.setLayout(layout)
    
    @pyqtSlot(object)
    def _add_task_to_list(self, task):
        item = QListWidgetItem(task.get_display_text())
        item.setData(Qt.ItemDataRole.UserRole, task)
        self.queue_list.addItem(item)
    
    @pyqtSlot(object)
    def _update_task_in_list(self, task):
        for i in range(self.queue_list.count()):
            item = self.queue_list.item(i)
            if item.data(Qt.ItemDataRole.UserRole) == task:
                item.setText(task.get_display_text())
                break
    
    @pyqtSlot(str)
    def _append_to_log(self, message):
        self.log_output.append(message)
        cursor = self.log_output.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        self.log_output.setTextCursor(cursor)
    
    @pyqtSlot()
    def _clear_log(self):
        self.log_output.clear()
    
    def add_task(self, command, dataset_path, output_name):
        """Add a new training task to the queue"""
        task = TrainingTask(command, dataset_path, output_name)
        self.task_queue.put(task)
        self.signal_add_task.emit(task)
    
    def process_queue(self):
        """Process tasks in the queue"""
        while True:
            if self.current_task is None:
                try:
                    task = self.task_queue.get(timeout=1)
                    self.current_task = task
                    self.execute_task(task)
                except queue.Empty:
                    continue
            threading.Event().wait(1)
    
    def execute_task(self, task):
        """Execute a single training task"""
        print(f"Starting task: {task.output_name}")
        task.status = "Running"
        self.signal_update_task.emit(task)
        
        self.signal_clear_log.emit()
        self.signal_append_log.emit(f"Starting training for: {task.output_name}\n")
        self.signal_append_log.emit(f"Command: {task.command}\n")
        self.signal_append_log.emit("="*50 + "\n")
        
        worker = TrainingWorker(task)
        worker.task_progress.connect(lambda msg: self.signal_append_log.emit(msg + "\n"))
        worker.task_completed.connect(lambda success: self.task_finished(task, success))
        worker.start()
        self.workers.append(worker)
    
    def task_finished(self, task, success):
        """Handle task completion"""
        task.status = "Completed" if success else "Failed"
        self.signal_update_task.emit(task)
        self.current_task = None
        
        status_msg = "Training completed successfully!" if success else "Training failed!"
        self.signal_append_log.emit(f"\n{status_msg}\n{'='*50}\n")
        
        # Clean up finished worker
        for worker in self.workers[:]:
            if worker.isFinished():
                self.workers.remove(worker)
    
    def clear_completed_tasks(self):
        """Remove completed tasks from the display"""
        for i in range(self.queue_list.count() - 1, -1, -1):
            item = self.queue_list.item(i)
            task = item.data(Qt.ItemDataRole.UserRole)
            if task.status in ["Completed", "Failed"]:
                self.queue_list.takeItem(i)
    
    def clear_all_tasks(self):
        """Clear all tasks from the queue"""
        # Only clear tasks that aren't currently running
        for i in range(self.queue_list.count() - 1, -1, -1):
            item = self.queue_list.item(i)
            task = item.data(Qt.ItemDataRole.UserRole)
            if task.status != "Running":
                self.queue_list.takeItem(i)
                
        # Clear the queue except for the currently running task
        while True:
            try:
                self.task_queue.get_nowait()
            except queue.Empty:
                break