from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                            QPushButton, QListWidget, QListWidgetItem, QGroupBox,
                            QTextEdit, QMessageBox)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, pyqtSlot, QTimer
import queue
import threading
import platform
import subprocess
import time
import select
import io
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
    
    def __init__(self, task, timeout=36000):  # Timeout padrão de 1 hora
        super().__init__()
        self.task = task
        self.process = None
        self.timeout = timeout
        self.is_running = True
        self._last_messages = []  # Armazena as últimas mensagens para verificar indicadores de sucesso
        
    def run(self):
        try:
            self.task.start_time = datetime.now()
            
            # Verificar o sistema operacional
            is_windows = platform.system() == 'Windows'
            
            # Configurar os argumentos do Popen de acordo com o sistema operacional
            popen_kwargs = {
                'stdout': subprocess.PIPE,
                'stderr': subprocess.STDOUT,
                'shell': True,
                'universal_newlines': True,
                'bufsize': 1,
                'encoding': 'utf-8',
                'errors': 'replace'
            }
            
            # Adicionar flags específicas do Windows se estivermos no Windows
            if is_windows:
                popen_kwargs['creationflags'] = subprocess.CREATE_NEW_PROCESS_GROUP
            
            # Cria o processo com as configurações apropriadas para o sistema operacional
            self.process = subprocess.Popen(
                self.task.command,
                **popen_kwargs
            )
            
            # Usar leitura não-bloqueante com timeout para evitar travamento
            start_time = time.time()
            output_buffer = ""
            
            # Definir um timer para verificar periodicamente o processo
            check_interval = 0.1  # 100ms
            
            while self.is_running:
                # Verificar timeout
                if time.time() - start_time > self.timeout:
                    self.task_progress.emit("Processo excedeu o tempo limite e será encerrado.")
                    self.terminate_process()
                    self.task_completed.emit(False)
                    return
                
                # Verifica se o processo ainda está executando
                if self.process.poll() is not None:
                    # Processo terminou, ler o restante da saída
                    remaining_output = self.process.stdout.read()
                    if remaining_output:
                        for line in remaining_output.splitlines():
                            if line.strip():
                                self.task_progress.emit(line.strip())
                    
                    returncode = self.process.returncode
                    
                    # Verificar se há mensagens de sucesso no buffer de saída
                    success_indicators = ["model saved", "saving checkpoint", "100%"]
                    success_found = False
                    
                    # Verificar buffer de mensagens anteriores para indicadores de sucesso
                    for indicator in success_indicators:
                        if hasattr(self, '_last_messages'):
                            for msg in self._last_messages:
                                if indicator in msg:
                                    success_found = True
                                    self.task_progress.emit(f"Indicador de sucesso detectado: '{indicator}'")
                                    break
                            if success_found:
                                break
                    
                    if returncode == 3221225477:  # 0xC0000005 (específico do Windows)
                        error_msg = ("Memory access error (0xC0000005). This usually means:\n"
                                   "1. Not enough RAM/VRAM for the current settings\n"
                                   "2. Try reducing batch size or model dimensions\n"
                                   "3. Check if other programs are using GPU memory\n"
                                   "4. Try restarting your computer if problem persists")
                        self.task_progress.emit(error_msg)
                        self.task_completed.emit(False)
                    elif success_found:
                        # Se encontramos indicadores de sucesso, consideramos como sucesso mesmo se o código de retorno não for 0
                        self.task_progress.emit("Treinamento completado com sucesso baseado em indicadores de progresso.")
                        self.task_completed.emit(True)
                    else:
                        # Verificar se o código de retorno foi 0 ou diferente de 0
                        is_success = returncode == 0
                        self.task_completed.emit(is_success)
                    
                    return
                
                # Lê a saída disponível sem bloquear
                if is_windows:
                    # No Windows, não podemos usar select com pipes
                    # Usamos uma abordagem alternativa com leitura de linha
                    output_line = self.process.stdout.readline()
                    if output_line:
                        line = output_line.strip()
                        if line:
                            self.task_progress.emit(line)
                else:
                    # No Linux/Mac, podemos usar select para leitura não-bloqueante
                    rlist, _, _ = select.select([self.process.stdout], [], [], check_interval)
                    if rlist:
                        output_line = self.process.stdout.readline()
                        if output_line:
                            line = output_line.strip()
                            if line:
                                self.task_progress.emit(line)
                                
                                # Detecta mensagens específicas de sucesso
                                if "model saved" in line or "saving checkpoint" in line:
                                    self.task_progress.emit("Detectado sinal de sucesso no treinamento.")
                
                # Pequena pausa para não sobrecarregar a CPU
                QThread.msleep(10)
                
        except Exception as e:
            self.task.end_time = datetime.now()
            error_msg = f"Error in training process: {str(e)}"
            print(error_msg)
            self.task_progress.emit(error_msg)
            self.task_completed.emit(False)
            
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Limpa recursos e fecha streams do processo"""
        if self.process:
            try:
                if self.process.stdout:
                    self.process.stdout.close()
                
                # Verifica se o processo ainda está em execução
                if self.process.poll() is None:
                    self.terminate_process()
                    
            except Exception as e:
                print(f"Error during cleanup: {str(e)}")
    
    def terminate_process(self):
        """Força o encerramento do processo se ainda estiver em execução"""
        try:
            if self.process and self.process.poll() is None:
                if platform.system() == 'Windows':
                    # No Windows, usa taskkill para encerrar o processo e seus filhos
                    subprocess.run(f"taskkill /F /T /PID {self.process.pid}", shell=True)
                else:
                    # No Linux/Mac, tenta com SIGTERM primeiro, depois SIGKILL
                    self.process.terminate()
                    # Espera um pouco para ver se o processo termina
                    time.sleep(0.5)
                    if self.process.poll() is None:
                        self.process.kill()
        except Exception as e:
            print(f"Error terminating process: {str(e)}")
    
    def stop(self):
        """Método para interromper o worker de fora"""
        self.is_running = False
        self.terminate_process()

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
        
        # Conecta sinais aos slots
        self.signal_add_task.connect(self._add_task_to_list)
        self.signal_update_task.connect(self._update_task_in_list)
        self.signal_append_log.connect(self._append_to_log)
        self.signal_clear_log.connect(self._clear_log)
        
        # Timer para verificar periodicamente o estado dos workers
        self.check_timer = QTimer(self)
        self.check_timer.timeout.connect(self.check_workers_status)
        self.check_timer.start(1000)  # Verifica a cada 1 segundo
        
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
        
        self.stop_current_btn = QPushButton("Stop Current Task")
        self.stop_current_btn.clicked.connect(self.stop_current_task)
        
        button_layout.addWidget(self.clear_completed_btn)
        button_layout.addWidget(self.clear_all_btn)
        button_layout.addWidget(self.stop_current_btn)
        
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
    
    def check_workers_status(self):
        """Verifica periodicamente o status dos workers e remove os finalizados"""
        workers_to_remove = []
        
        for worker in self.workers:
            if not worker.isRunning():
                workers_to_remove.append(worker)
        
        for worker in workers_to_remove:
            self.workers.remove(worker)
            worker.deleteLater()  # Importante para liberar recursos Qt
            
        # Se não houver workers ativos mas o current_task ainda estiver definido,
        # isso pode indicar que o worker terminou mas o task_finished não foi chamado
        if not self.workers and self.current_task and self.is_processing:
            self.signal_append_log.emit("Detectado processo que terminou sem notificação. Liberando fila...")
            self.task_finished(self.current_task, False)
    
    def stop_current_task(self):
        """Para a tarefa atual em execução"""
        if self.current_task and self.current_task.status == "Running":
            for worker in self.workers:
                worker.stop()
            
            self.signal_append_log.emit("Tarefa interrompida pelo usuário.")
            
            # Não chamamos task_finished aqui, deixamos o worker sinalizar o término
    
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
            if not self.is_processing and self.current_task is None:
                try:
                    task = self.task_queue.get(timeout=1)
                    self.current_task = task
                    self.is_processing = True
                    self.execute_task(task)
                except queue.Empty:
                    pass
            threading.Event().wait(1)  # Pequena pausa para não sobrecarregar a CPU
    
    def execute_task(self, task):
        """Execute a single training task"""
        try:
            # Verifica se o dataset.toml existe
            dataset_toml = task.dataset_path / "cropped_images" / "dataset.toml"
            if not dataset_toml.exists():
                self.signal_append_log.emit(f"Erro: dataset.toml não encontrado em {dataset_toml}")
                self.task_finished(task, False)
                return
            
            task.status = "Running"
            self.signal_update_task.emit(task)
            
            # Verifica e corrige o caminho do dataset.toml
            cmd = task.command
            
            # Verificação de caminho no Windows vs Linux
            if platform.system() == 'Windows':
                if "cropped_images\\cropped_images" in cmd:
                    cmd = cmd.replace("cropped_images\\cropped_images", "cropped_images")
            else:
                # Verifica e corrige caminhos no formato Linux
                if "cropped_images/cropped_images" in cmd:
                    cmd = cmd.replace("cropped_images/cropped_images", "cropped_images")
            
            self.signal_clear_log.emit()
            self.signal_append_log.emit(f"Starting training for: {task.output_name}\n")
            self.signal_append_log.emit(f"Command: {cmd}\n")
            self.signal_append_log.emit("="*50 + "\n")
            
            worker = TrainingWorker(task)
            worker.task_progress.connect(self._handle_task_progress)
            worker.task_completed.connect(lambda success: self.task_finished(task, success))
            worker.task.command = cmd
            worker.start()
            self.workers.append(worker)
            
        except Exception as e:
            error_msg = f"Error starting task: {str(e)}"
            self.signal_append_log.emit(f"\n{error_msg}\n")
            self.task_finished(task, False)
    
    def _handle_task_progress(self, message):
        """Processa as mensagens de progresso do treinamento"""
        self.signal_append_log.emit(message + "\n")
    
    def task_finished(self, task, success):
        """Handle task completion"""
        try:
            # Evitar chamar task_finished várias vezes para a mesma tarefa
            if task.status in ["Completed", "Failed"]:
                return
                
            task.status = "Completed" if success else "Failed"
            task.end_time = datetime.now()
            self.signal_update_task.emit(task)
            
            status_msg = "Training completed successfully!" if success else "Training failed!"
            self.signal_append_log.emit(f"\n{status_msg}\n{'='*50}\n")
            
        finally:
            # Garante que os estados são resetados mesmo se houver erro
            self.current_task = None
            self.is_processing = False
    
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
    
    def closeEvent(self, event):
        """Garante a limpeza adequada ao fechar o widget"""
        # Para todos os workers ativos
        for worker in self.workers:
            worker.stop()
            worker.wait(1000)  # Espera até 1 segundo
        
        super().closeEvent(event)