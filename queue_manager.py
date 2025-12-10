import sys
import os
import queue
import subprocess
import threading
import time
import re
import yaml
import video_utils
from pathlib import Path
from datetime import datetime

from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QListWidget, 
                            QListWidgetItem, QPushButton, QLabel, QGroupBox, 
                            QTextEdit, QProgressBar, QMessageBox)
from PyQt6.QtCore import Qt, pyqtSignal, pyqtSlot, QThread, QTimer, QUrl
from PyQt6.QtGui import QPixmap, QDesktopServices

from preview_window import PreviewWindow

class ClickableLabel(QLabel):
    clicked = pyqtSignal()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()

class TrainingTask:
    def __init__(self, command, dataset_path, output_name, metadata=None):
        self.command = command
        self.dataset_path = Path(dataset_path)
        self.output_name = output_name
        self.metadata = metadata or {}
        self.status = "Pending"  # Pending, Running, Completed, Failed
        self.created_at = datetime.now()
        self.start_time = None
        self.end_time = None

    def get_display_text(self):
        return f"[{self.status}] {self.output_name} ({self.created_at.strftime('%H:%M')})"

class TrainingWorker(QThread):
    task_completed = pyqtSignal(bool)
    task_progress = pyqtSignal(str)
    
    def __init__(self, task):
        super().__init__()
        self.task = task
        self.process = None
        self.is_running = True

    def run(self):
        try:
            print(f"[WORKER] Starting worker for task: {self.task.output_name}")
            # Configurar environment para UTF-8
            env = os.environ.copy()
            env['PYTHONIOENCODING'] = 'utf-8'
            env['PYTHONLEGACYWINDOWSSTDIO'] = '0'

            # Criar processo
            print(f"[WORKER] Creating subprocess...")
            self.process = subprocess.Popen(
                self.task.command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                shell=True,
                universal_newlines=True,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
                bufsize=1,
                encoding='utf-8',
                errors='replace',
                env=env
            )
            print(f"[WORKER] Subprocess started with PID: {self.process.pid}")

            # Ler saída em tempo real
            for line in self.process.stdout:
                if not self.is_running:
                    print(f"[WORKER] is_running=False, breaking loop")
                    break
                line = line.strip()
                if line:
                    self.task_progress.emit(line)

            print(f"[WORKER] Output loop finished, waiting for process...")
            self.process.wait()
            success = self.process.returncode == 0
            print(f"[WORKER] Process finished with returncode={self.process.returncode}, success={success}")
            print(f"[WORKER] Emitting task_completed signal...")
            self.task_completed.emit(success)
            print(f"[WORKER] task_completed signal emitted")

        except Exception as e:
            print(f"[WORKER] Exception: {str(e)}")
            self.task_progress.emit(f"Error running task: {str(e)}")
            self.task_completed.emit(False)

    def stop(self):
        self.is_running = False
        if self.process:
            try:
                self.process.terminate()
                self.process.kill()
            except:
                pass

class PostProcessingWorker(QThread):
    finished = pyqtSignal(bool)
    progress = pyqtSignal(str)
    
    def __init__(self, command, name="Post-Processing"):
        super().__init__()
        self.command = command
        self.name = name
        self.process = None
        self.is_running = True

    def run(self):
        try:
            # Configurar environment
            env = os.environ.copy()
            env['PYTHONIOENCODING'] = 'utf-8'
            env['PYTHONLEGACYWINDOWSSTDIO'] = '0'
            
            # Criar processo
            self.process = subprocess.Popen(
                self.command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                shell=False, # False because we pass list of args
                universal_newlines=True,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
                bufsize=1,
                encoding='utf-8',
                errors='replace',
                env=env
            )
            
            for line in self.process.stdout:
                if not self.is_running:
                    break
                self.progress.emit(f"[{self.name}] {line.strip()}")
            
            self.process.wait()
            self.finished.emit(self.process.returncode == 0)
            
        except Exception as e:
            self.progress.emit(f"Error in {self.name}: {str(e)}")
            self.finished.emit(False)

    def stop(self):
        self.is_running = False
        if self.process:
            try:
                self.process.terminate()
            except:
                pass

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
        self.is_processing = False

        # Preview monitoring
        self.preview_enabled = False
        self.current_samples_dir = None
        self.last_preview_file = None
        self.active_preview_window = None

        # Conecta sinais aos slots
        self.signal_add_task.connect(self._add_task_to_list)
        self.signal_update_task.connect(self._update_task_in_list)
        self.signal_append_log.connect(self._append_to_log)
        self.signal_clear_log.connect(self._clear_log)

        self.init_ui()

        # Start the queue processing
        self.queue_processor = threading.Thread(target=self.process_queue, daemon=True)
        self.queue_processor.start()

        # Start preview monitoring timer
        self.preview_timer = QTimer()
        self.preview_timer.timeout.connect(self.check_for_new_samples)
        self.preview_timer.start(5000)  # Check every 5 seconds
        
    def init_ui(self):
        layout = QVBoxLayout()

        # Upper section: Queue list and Preview side-by-side
        upper_layout = QHBoxLayout()

        # Queue display group (left side)
        queue_group = QGroupBox("Training Queue")
        queue_layout = QVBoxLayout()

        # Queue list
        self.queue_list = QListWidget()
        self.queue_list.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.queue_list.setMaximumHeight(200)
        queue_layout.addWidget(self.queue_list)

        # Control buttons
        button_layout = QHBoxLayout()

        self.skip_current_btn = QPushButton("Skip Current")
        self.skip_current_btn.clicked.connect(self.skip_current_task)
        self.skip_current_btn.setStyleSheet("QPushButton { background-color: #FF9800; color: white; }")
        self.skip_current_btn.setToolTip("Mata o processo atual e passa para o próximo da fila")

        self.clear_completed_btn = QPushButton("Clear Completed")
        self.clear_completed_btn.clicked.connect(self.clear_completed_tasks)

        self.clear_all_btn = QPushButton("Clear All")
        self.clear_all_btn.clicked.connect(self.clear_all_tasks)

        self.reset_queue_btn = QPushButton("Reset Queue")
        self.reset_queue_btn.clicked.connect(self.reset_queue)
        self.reset_queue_btn.setStyleSheet("QPushButton { background-color: #ff6b6b; color: white; }")

        button_layout.addWidget(self.skip_current_btn)
        button_layout.addWidget(self.clear_completed_btn)
        button_layout.addWidget(self.clear_all_btn)
        button_layout.addWidget(self.reset_queue_btn)

        queue_layout.addLayout(button_layout)
        queue_group.setLayout(queue_layout)
        upper_layout.addWidget(queue_group)

        # Preview group (right side)
        preview_group = QGroupBox("Training Preview")
        preview_layout = QVBoxLayout()

        self.preview_label = ClickableLabel("No samples yet")
        self.preview_label.setCursor(Qt.CursorShape.PointingHandCursor)
        self.preview_label.clicked.connect(self.open_preview_window)
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setMinimumSize(400, 200)
        self.preview_label.setMaximumHeight(200)
        self.preview_label.setStyleSheet("border: 1px dashed #666; background-color: #222;")
        self.preview_label.setScaledContents(False)  # Keep aspect ratio
        preview_layout.addWidget(self.preview_label)

        self.preview_status = QLabel("Waiting for training to start...")
        self.preview_status.setStyleSheet("color: #999; font-size: 10px;")
        self.preview_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        preview_layout.addWidget(self.preview_status)

        preview_group.setLayout(preview_layout)
        upper_layout.addWidget(preview_group)

        layout.addLayout(upper_layout)

        # Log output group (bottom)
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
    
    def add_task(self, command, dataset_path, output_name, metadata=None):
        """Add a new training task to the queue"""
        task = TrainingTask(command, dataset_path, output_name, metadata)
        self.task_queue.put(task)
        self.signal_add_task.emit(task)
    
    def process_queue(self):
        """Process tasks in the queue"""
        print("[QUEUE] Queue processor thread started")
        while True:
            print(f"[QUEUE] Loop - is_processing={self.is_processing}, current_task={self.current_task is not None}, queue_size={self.task_queue.qsize()}")
            if not self.is_processing and self.current_task is None:
                print("[QUEUE] Ready to process next task...")
                try:
                    task = self.task_queue.get(timeout=1)
                    print(f"[QUEUE] Got task from queue: {task.output_name}")
                    self.current_task = task
                    self.is_processing = True
                    print(f"[QUEUE] Executing task: {task.output_name}")
                    self.execute_task(task)
                except queue.Empty:
                    pass
            else:
                if self.is_processing:
                    print(f"[QUEUE] Waiting - task in progress")
                if self.current_task:
                    print(f"[QUEUE] Waiting - current_task: {self.current_task.output_name}, status: {self.current_task.status}")
            threading.Event().wait(2)  # Aumentado para 2s para reduzir spam de log
    
    def execute_task(self, task):
        """Execute a single training task (com cache automático se necessário)"""
        try:
            cmd = task.command
            if "dataset_config" in cmd:
                cmd = cmd.replace("cropped_images\\cropped_images", "cropped_images")

            # Try to detect and configure preview monitoring
            self._setup_preview_for_task(cmd)

            self.signal_clear_log.emit()
            self.signal_append_log.emit(f"Starting training for: {task.output_name}\n")
            self.signal_append_log.emit(f"Command: {cmd}\n")
            self.signal_append_log.emit("="*50 + "\n")

            # Atualizar status para "Running" ANTES de iniciar o worker
            task.status = "Running"
            task.start_time = datetime.now().timestamp()
            self.signal_update_task.emit(task)  # Atualiza UI imediatamente

            worker = TrainingWorker(task)
            worker.task_progress.connect(self._handle_task_progress)
            # Usar conexão direta em vez de lambda para evitar problemas de closure em threads
            worker.task_completed.connect(self._on_worker_completed)
            worker.task.command = cmd
            worker.start()
            self.workers.append(worker)

        except Exception as e:
            error_msg = f"Error starting task: {str(e)}"
            self.signal_append_log.emit(f"\nError starting task: {error_msg}\n")
            self.signal_append_log.emit(f"Stack trace: {str(e)}\n")
            self.task_finished(task, False)
            # Log do erro mas não re-raise para manter a fila funcionando
    
    def execute_cache_and_training(self, task):
        """Executa caches sequencialmente e depois o treinamento"""
        try:
            cache_info = task.command
            cache_commands = cache_info["cache_commands"]
            training_command = cache_info["training_command"]
            
            task.status = "Running"
            task.start_time = datetime.now().timestamp()
            self.signal_update_task.emit(task)
            self.signal_clear_log.emit()
            
            # Executar caches sequencialmente
            for i, cache_cmd in enumerate(cache_commands):
                cache_type = "Latent Cache" if "cache_latents" in str(cache_cmd) else "Text Encoder Cache"
                self.signal_append_log.emit(f"Step {i+1}/{len(cache_commands)}: Running {cache_type}...\n")
                self.signal_append_log.emit(f"Command: {' '.join(cache_cmd)}\n")
                self.signal_append_log.emit("="*50 + "\n")
                
                # Executar comando de cache
                import subprocess
                import os
                
                # Configurar environment para UTF-8
                env = os.environ.copy()
                env['PYTHONIOENCODING'] = 'utf-8'
                env['PYTHONLEGACYWINDOWSSTDIO'] = '0'
                
                process = subprocess.Popen(
                    cache_cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    shell=True,
                    universal_newlines=True,
                    creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
                    bufsize=1,
                    encoding='utf-8',
                    errors='replace',
                    env=env
                )
                
                # Ler saída em tempo real
                for line in process.stdout:
                    line = line.strip()
                    if line:
                        self.signal_append_log.emit(line + "\n")
                
                process.wait()
                
                if process.returncode != 0:
                    self.signal_append_log.emit(f"\nError: {cache_type} failed with return code {process.returncode}\n")
                    self.task_finished(task, False)
                    return
                else:
                    self.signal_append_log.emit(f"\n{cache_type} completed successfully!\n")
            
            # Agora executar treinamento
            self.signal_append_log.emit(f"\nStep {len(cache_commands)+1}/{len(cache_commands)+1}: Starting Training...\n")
            self.signal_append_log.emit("="*50 + "\n")
            
            # Atualizar comando da task para o treinamento
            task.command = training_command
            
            # Executar treinamento normalmente
            worker = TrainingWorker(task)
            worker.task_progress.connect(self._handle_task_progress)
            worker.task_completed.connect(self._on_worker_completed)
            worker.start()
            self.workers.append(worker)
            
        except Exception as e:
            error_msg = f"Error in cache and training: {str(e)}"
            self.signal_append_log.emit(f"\nError: {error_msg}\n")
            self.task_finished(task, False)
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

    def _setup_preview_for_task(self, command):
        """Parse command to extract YAML config and set up preview monitoring"""
        try:
            # Try to find YAML config file in command
            # Pattern: python.exe run.py path/to/config.yaml
            yaml_match = re.search(r'[\w/\\:.-]+\.yaml', command)
            if not yaml_match:
                self.set_preview_disabled("No config file detected")
                return

            yaml_path = Path(yaml_match.group(0))
            if not yaml_path.exists():
                self.set_preview_disabled("Config file not found")
                return

            # Parse YAML config
            with open(yaml_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)

            # Check if sampling is disabled
            # Look in config.process[0].sample for disable_sampling
            try:
                process_config = config.get('config', {}).get('process', [{}])[0]
                sample_config = process_config.get('sample', {})
                disable_sampling = sample_config.get('disable_sampling', False)

                if disable_sampling:
                    self.set_preview_disabled("Sampling disabled in config")
                    return
            except (KeyError, IndexError, TypeError):
                # If we can't find the config, assume sampling is enabled
                pass

            # Get output directory (training_folder from config.process[0])
            try:
                training_folder = process_config.get('training_folder', '')
                if not training_folder:
                    self.set_preview_disabled("No training folder in config")
                    return

                # Build samples directory path
                samples_dir = Path(training_folder) / 'samples'
                self.enable_preview(str(samples_dir))

            except Exception as e:
                self.set_preview_disabled(f"Error parsing config: {str(e)}")

        except Exception as e:
            # If anything fails, disable preview
            self.set_preview_disabled(f"Config error: {str(e)}")

    def check_for_new_samples(self):
        """Automatically check for new sample images from ai-toolkit"""
        if not self.preview_enabled or not self.current_samples_dir:
            return

        try:
            samples_path = Path(self.current_samples_dir)
            if not samples_path.exists():
                return

            # Find all image files in samples directory
            image_files = []
            for ext in ['*.png', '*.jpg', '*.jpeg']:
                image_files.extend(samples_path.glob(ext))

            if not image_files:
                return

            # Filter images created after task start
            if self.current_task and self.current_task.start_time:
                image_files = [p for p in image_files if p.stat().st_mtime > self.current_task.start_time]

            if not image_files:
                return

            # Get the most recent image
            latest_image = max(image_files, key=lambda p: p.stat().st_mtime)

            # Only update if it's a new image
            if str(latest_image) != self.last_preview_file:
                self.last_preview_file = str(latest_image)
                self.update_preview_image(latest_image)
                
                # Copy to destination immediately
                if self.current_task:
                    self._copy_sample_to_destination(latest_image, self.current_task)

        except Exception as e:
            # Silently ignore errors (directory might not exist yet)
            pass

    def _copy_sample_to_destination(self, image_path, task):
        """Copy a sample image to the destination folder immediately"""
        try:
            model_type = task.metadata.get("model_type")
            if not model_type:
                return

            dataset_path = task.dataset_path
            dest_folder = dataset_path / model_type
            dest_folder.mkdir(parents=True, exist_ok=True)
            
            import shutil
            shutil.copy2(image_path, dest_folder / image_path.name)
            
        except Exception as e:
            print(f"Error copying sample: {e}")

    def update_preview_image(self, image_path):
        """Update the preview label with the new image"""
        try:
            pixmap = QPixmap(str(image_path))
            if not pixmap.isNull():
                # Scale image to fit while maintaining aspect ratio
                scaled_pixmap = pixmap.scaled(
                    self.preview_label.width() - 10,
                    self.preview_label.height() - 10,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation
                )
                self.preview_label.setPixmap(scaled_pixmap)

                # Update status
                filename = Path(image_path).name
                file_time = datetime.fromtimestamp(Path(image_path).stat().st_mtime)
                time_str = file_time.strftime("%H:%M:%S")
                self.preview_status.setText(f"Latest sample: {filename} ({time_str})")
                self.preview_status.setStyleSheet("color: #4CAF50; font-size: 10px;")

                # Update external window if open
                if self.active_preview_window and self.active_preview_window.isVisible():
                    self.active_preview_window.update_image(image_path)
        except Exception as e:
            self.preview_status.setText(f"Error loading image: {str(e)}")
            self.preview_status.setStyleSheet("color: #ff6b6b; font-size: 10px;")

    def set_preview_disabled(self, reason="Sampling disabled in config"):
        """Disable preview widget with a reason"""
        self.preview_enabled = False
        self.preview_label.setText(reason)
        self.preview_label.setStyleSheet("border: 1px dashed #666; background-color: #1a1a1a; color: #666;")
        self.preview_status.setText("Preview is disabled")
        self.preview_status.setStyleSheet("color: #666; font-size: 10px;")

    def enable_preview(self, samples_dir):
        """Enable preview monitoring for a samples directory"""
        self.preview_enabled = True
        self.current_samples_dir = samples_dir
        self.last_preview_file = None
        self.preview_label.setText("Waiting for samples...")
        self.preview_label.setStyleSheet("border: 1px dashed #666; background-color: #222; color: #999;")
        self.preview_status.setText("Monitoring for new samples...")
        self.preview_status.setStyleSheet("color: #2196F3; font-size: 10px;")

    def skip_current_task(self):
        """Skip the current running task and move to the next one in queue"""
        if not self.current_task or not self.is_processing:
            self.signal_append_log.emit("No task currently running to skip.\n")
            return

        reply = QMessageBox.question(
            self, "Skip Current Task",
            f"Matar o processo '{self.current_task.output_name}' e passar para o próximo?\n\n"
            "Os pesos intermediários salvos serão preservados.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )

        if reply == QMessageBox.StandardButton.Yes:
            task = self.current_task
            self.signal_append_log.emit(f"\n>>> Skipping task: {task.output_name}\n")

            # Kill all workers associated with current task
            for worker in self.workers[:]:
                if worker.task == task or (hasattr(worker, 'task') and worker.task is None):
                    try:
                        if hasattr(worker, 'process') and worker.process:
                            self.signal_append_log.emit("Terminating process...\n")
                            worker.process.terminate()
                            try:
                                worker.process.wait(timeout=3)
                            except:
                                worker.process.kill()
                        worker.is_running = False
                        worker.quit()
                        worker.wait(2000)
                    except Exception as e:
                        self.signal_append_log.emit(f"Warning during termination: {e}\n")
                    try:
                        self.workers.remove(worker)
                    except:
                        pass

            # Update task status
            task.status = "Skipped"
            task.end_time = datetime.now().timestamp()
            self.signal_update_task.emit(task)

            self.signal_append_log.emit(f"Task skipped: {task.output_name}\n")

            # Cleanup and proceed to next
            self._cleanup_memory()
            self.signal_append_log.emit("Aguardando 3 segundos antes do próximo...\n")
            self._delayed_finish_task(3)

    def reset_queue(self):
        """Reset the entire queue system (emergency reset)"""
        from PyQt6.QtWidgets import QMessageBox
        
        reply = QMessageBox.question(self, "Reset Queue", 
                                   "This will force-stop all running tasks and reset the queue. Continue?",
                                   QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        
        if reply == QMessageBox.StandardButton.Yes:
            # Force-stop all workers
            for worker in self.workers[:]:
                try:
                    if worker.process:
                        worker.process.terminate()
                        worker.process.kill()
                    worker.quit()
                    worker.wait(3000)  # Wait up to 3 seconds
                except:
                    pass
                self.workers.remove(worker)
            
            # Clear queue completely
            while True:
                try:
                    self.task_queue.get_nowait()
                except queue.Empty:
                    break
            
            # Reset states
            self.current_task = None
            self.is_processing = False
            
            # Clear UI
            self.queue_list.clear()
            self.log_output.clear()
            
            self.signal_append_log.emit("Queue has been reset. You can now add new tasks.\n")

    def open_preview_window(self):
        """Open the current preview image in a separate window"""
        if not self.last_preview_file or not Path(self.last_preview_file).exists():
            return

        if self.active_preview_window is None:
            self.active_preview_window = PreviewWindow(self.last_preview_file, self)
            self.active_preview_window.finished.connect(self._on_preview_closed)
            self.active_preview_window.show()
        else:
            self.active_preview_window.update_image(self.last_preview_file)
            self.active_preview_window.show()
            self.active_preview_window.raise_()
            self.active_preview_window.activateWindow()

    def _on_preview_closed(self):
        self.active_preview_window = None

    def _handle_task_progress(self, line):
        """Handle progress updates from the worker"""
        self.signal_append_log.emit(line)

    def _on_worker_completed(self, success):
        """Slot chamado quando um worker emite task_completed"""
        print(f"[QUEUE] _on_worker_completed called with success={success}")
        # Encontrar o worker que emitiu o sinal
        sender_worker = self.sender()
        if sender_worker and hasattr(sender_worker, 'task') and sender_worker.task:
            task = sender_worker.task
            print(f"[QUEUE] Found task from sender: {task.output_name}")
            self.task_finished(task, success)
        elif self.current_task:
            # Fallback: usar current_task se não conseguir identificar o sender
            print(f"[QUEUE] Using current_task as fallback: {self.current_task.output_name}")
            self.task_finished(self.current_task, success)
        else:
            print(f"[QUEUE] ERROR: Could not identify task for completed worker!")

    def _cleanup_memory(self):
        """Limpa memória GPU e RAM entre tarefas"""
        import gc

        # Força garbage collection
        gc.collect()

        # Limpa cache CUDA se disponível
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                self.signal_append_log.emit("CUDA cache cleared.\n")
        except ImportError:
            pass
        except Exception as e:
            self.signal_append_log.emit(f"Warning: Could not clear CUDA cache: {e}\n")

    def _cleanup_worker(self, task):
        """Remove worker da lista e limpa referências"""
        for worker in self.workers[:]:  # Itera sobre cópia da lista
            if worker.task == task:
                # Quebra referência circular
                worker.task = None
                try:
                    worker.quit()
                    worker.wait(2000)  # Espera até 2 segundos
                except:
                    pass
                self.workers.remove(worker)
                break

    def _finish_task_processing(self):
        """Finaliza processamento após cooldown - libera fila para próxima tarefa"""
        print(f"[QUEUE] _finish_task_processing called - BEFORE: is_processing={self.is_processing}, current_task={self.current_task is not None}")
        self.current_task = None
        self.is_processing = False
        print(f"[QUEUE] _finish_task_processing - AFTER: is_processing={self.is_processing}, current_task={self.current_task is not None}")
        self.signal_append_log.emit("="*50 + "\n")
        self.signal_append_log.emit("Ready for next task in queue.\n")

    def _delayed_finish_task(self, delay_seconds=5):
        """Executa cooldown em thread separada e libera a fila"""
        print(f"[QUEUE] _delayed_finish_task called with delay={delay_seconds}s")
        def do_finish():
            print(f"[QUEUE] Cooldown thread started, sleeping {delay_seconds}s...")
            time.sleep(delay_seconds)
            print(f"[QUEUE] Cooldown finished, calling _finish_task_processing...")
            self._finish_task_processing()

        finish_thread = threading.Thread(target=do_finish, daemon=True)
        finish_thread.start()
        print(f"[QUEUE] Cooldown thread started")

    def task_finished(self, task, success):
        """Handle task completion with proper cleanup and cooldown"""
        print(f"[QUEUE] task_finished called - task={task.output_name}, success={success}")

        # 1. Atualizar status
        task.status = "Completed" if success else "Failed"
        task.end_time = datetime.now().timestamp()
        self.signal_update_task.emit(task)
        print(f"[QUEUE] Task status updated to: {task.status}")

        # 2. Log
        status_msg = "completed successfully" if success else "failed"
        self.signal_append_log.emit(f"\nTask {status_msg}: {task.output_name}\n")

        # 3. Remover worker corretamente
        print(f"[QUEUE] Cleaning up worker...")
        self._cleanup_worker(task)

        # 4. Pós-processamento (video) - async
        if success:
            print(f"[QUEUE] Starting post-processing...")
            self._handle_post_processing(task)

        # 5. Limpar memória GPU/RAM
        print(f"[QUEUE] Cleaning up memory...")
        self._cleanup_memory()

        # 6. Cooldown de 5 segundos antes de liberar próxima tarefa
        # Usa threading.Thread em vez de QTimer porque process_queue() roda em thread regular
        self.signal_append_log.emit("Aguardando 5 segundos para limpeza de memória...\n")
        print(f"[QUEUE] Starting cooldown...")
        self._delayed_finish_task(5)

    def _handle_post_processing(self, task):
        """Handle post-processing steps like video generation"""
        try:
            model_type = task.metadata.get("model_type")
            if not model_type:
                self.signal_append_log.emit("Skipping post-processing: No model_type found in metadata.\n")
                return

            # Images are in dataset_path/cropped_images/model_type
            samples_dir = task.dataset_path / "cropped_images" / model_type

            # Fallback: check if images are directly in dataset_path/model_type (for non-standard setups)
            if not samples_dir.exists():
                samples_dir = task.dataset_path / model_type

            self.signal_append_log.emit(f"Post-processing: Checking for images in {samples_dir}\n")

            if samples_dir.exists():
                # Count images
                image_count = len(list(samples_dir.glob("*.png"))) + len(list(samples_dir.glob("*.jpg")))
                if image_count == 0:
                    self.signal_append_log.emit(f"No images found in {samples_dir}\n")
                    return

                self.signal_append_log.emit(f"Found {image_count} images\n")

                # Primary: Use RIFE interpolation for smooth video
                video_path = samples_dir / "training_preview.mp4"
                rife_script = Path(__file__).parent / "interpolate_rife_torch.py"

                if rife_script.exists():
                    self.signal_append_log.emit(f"Starting RIFE frame interpolation...\n")
                    self.signal_append_log.emit(f"Input: {samples_dir}\n")
                    self.signal_append_log.emit(f"Output: {video_path}\n")

                    cmd = [
                        sys.executable,
                        str(rife_script),
                        "--input", str(samples_dir),
                        "--output", str(video_path),
                        "--multiplier", "16",
                        "--fps", "16"
                    ]

                    worker = PostProcessingWorker(cmd, "RIFE Interpolation")
                    worker.progress.connect(self._handle_task_progress)
                    worker.finished.connect(lambda success: self._on_rife_video_finished(success, video_path, samples_dir))
                    worker.start()
                    self.workers.append(worker)
                else:
                    # Fallback: simple video without interpolation
                    self.signal_append_log.emit(f"RIFE script not found, creating simple video...\n")
                    self._create_simple_video(samples_dir, video_path)

            else:
                self.signal_append_log.emit(f"Samples directory not found: {samples_dir}\n")

        except Exception as e:
            self.signal_append_log.emit(f"Error in post-processing: {str(e)}\n")
            import traceback
            self.signal_append_log.emit(traceback.format_exc() + "\n")

    def _on_rife_video_finished(self, success, video_path, samples_dir):
        """Handle completion of RIFE interpolation"""
        if success:
            self.signal_append_log.emit(f"\nRIFE video created successfully: {video_path}\n")
        else:
            self.signal_append_log.emit(f"\nRIFE interpolation failed. Creating simple video as fallback...\n")
            self._create_simple_video(samples_dir, video_path)
        self.signal_append_log.emit("="*50 + "\n")

    def _create_simple_video(self, samples_dir, video_path):
        """Create a simple video without interpolation as fallback"""
        try:
            video_gen_script = Path(__file__).parent / "video_generator.py"
            if video_gen_script.exists():
                cmd = [
                    sys.executable,
                    str(video_gen_script),
                    "--input", str(samples_dir),
                    "--output", str(video_path),
                    "--fps", "8"
                ]
                worker = PostProcessingWorker(cmd, "Simple Video")
                worker.progress.connect(self._handle_task_progress)
                worker.finished.connect(lambda s: self._on_simple_video_done(s, video_path))
                worker.start()
                self.workers.append(worker)
            else:
                # Direct call to video_utils
                video_utils.create_video_from_folder(str(samples_dir), str(video_path), fps=8)
                self.signal_append_log.emit(f"Simple video created: {video_path}\n")
        except Exception as e:
            self.signal_append_log.emit(f"Error creating simple video: {str(e)}\n")

    def _on_simple_video_done(self, success, video_path):
        """Handle completion of simple video fallback"""
        if success:
            self.signal_append_log.emit(f"\nSimple video created: {video_path}\n")
        else:
            self.signal_append_log.emit(f"\nFailed to create video.\n")
        self.signal_append_log.emit("="*50 + "\n")

    def clear_completed_tasks(self):
        """Clear completed and failed tasks from the list"""
        for i in range(self.queue_list.count() - 1, -1, -1):
            item = self.queue_list.item(i)
            task = item.data(Qt.ItemDataRole.UserRole)
            if task.status in ["Completed", "Failed"]:
                self.queue_list.takeItem(i)

    def clear_all_tasks(self):
        """Clear all tasks except running ones"""
        for i in range(self.queue_list.count() - 1, -1, -1):
            item = self.queue_list.item(i)
            task = item.data(Qt.ItemDataRole.UserRole)
            if task.status != "Running":
                self.queue_list.takeItem(i)