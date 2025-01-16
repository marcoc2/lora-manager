from PyQt6.QtWidgets import QDialog, QVBoxLayout, QPushButton, QTextEdit
from PyQt6.QtCore import QProcess
from pathlib import Path
import tempfile
import shutil
import os
import sys
import codecs

class CommandOutputDialog(QDialog):
    def __init__(self, command, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Training Output")
        self.setMinimumSize(600, 400)
        
        self.command = command
        self.process = QProcess(self)

        # Layout
        layout = QVBoxLayout()
        self.text_output = QTextEdit()
        self.text_output.setReadOnly(True)
        layout.addWidget(self.text_output)
        
        self.close_button = QPushButton("Close")
        self.close_button.setEnabled(False)
        self.close_button.clicked.connect(self.close)
        layout.addWidget(self.close_button)
        
        self.setLayout(layout)
        
        self.start_process()

    def start_process(self):
        """Inicia o subprocesso para rodar o comando"""
        self.process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self.process.readyReadStandardOutput.connect(self.read_output)
        self.process.readyReadStandardError.connect(self.read_output)
        self.process.finished.connect(self.process_finished)
        
        # Configura ambiente para UTF-8
        env = self.process.processEnvironment()
        env.insert("PYTHONIOENCODING", "utf-8")
        env.insert("PYTHONUTF8", "1")
        self.process.setProcessEnvironment(env)
        
        # Divide o comando em partes para compatibilidade com QProcess
        command_parts = self.command.split()
        self.process.start(command_parts[0], command_parts[1:])

    def read_output(self):
        """Lê a saída do subprocesso e exibe no QTextEdit"""
        output = self.process.readAllStandardOutput().data().decode()
        self.text_output.append(output)
        
        error = self.process.readAllStandardError().data().decode()
        if error:
            self.text_output.append(f"ERROR: {error}")

    def process_finished(self):
        """Habilita o botão de fechamento quando o processo termina e limpa arquivos temporários"""
        self.text_output.append("\nTraining finished.")
        self.close_button.setEnabled(True)
        
        # Limpar arquivos temporários se o parent implementa cleanup_temp_files
        if hasattr(self.parent(), 'cleanup_temp_files'):
            self.parent().cleanup_temp_files()

class ScriptManager:
    def __init__(self):
        self.temp_script_path = None

    def create_temp_script(self, original_script_path):
        """
        Cria uma versão temporária do script sem caracteres não-ASCII
        """
        # Criar diretório temporário se não existir
        temp_dir = Path(tempfile.gettempdir()) / "temp_training_scripts"
        temp_dir.mkdir(exist_ok=True)
        
        # Criar nome para arquivo temporário
        temp_script = temp_dir / f"temp_{os.path.basename(original_script_path)}"
        
        try:
            # Ler arquivo original e converter caracteres
            with open(original_script_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Substituir caracteres não-ASCII por seus equivalentes mais próximos
            ascii_content = content.encode('ascii', 'ignore').decode('ascii')
            
            # Adicionar configuração de encoding no início do arquivo
            final_content = f'''import sys
import os
if sys.platform.startswith('win'):
    import locale
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
    os.environ["PYTHONIOENCODING"] = "utf-8"

{ascii_content}
'''
            # Escrever versão limpa no arquivo temporário
            with open(temp_script, 'w', encoding='utf-8', newline='') as f:
                f.write(final_content)
            
            self.temp_script_path = str(temp_script)
            return self.temp_script_path
            
        except Exception as e:
            print(f"Erro ao criar script temporário: {e}")
            return original_script_path

    def cleanup_temp_files(self):
        """Limpa arquivos temporários"""
        temp_dir = Path(tempfile.gettempdir()) / "temp_training_scripts"
        if temp_dir.exists():
            try:
                shutil.rmtree(temp_dir)
                self.temp_script_path = None
            except Exception as e:
                print(f"Erro ao limpar arquivos temporários: {e}")

def format_command_args(args_text):
    """
    Formata argumentos de linha de comando de forma segura
    """
    if not args_text.strip():
        return []
        
    parts = []
    for arg in args_text.split():
        # Limpa aspas e adiciona à lista
        cleaned_arg = arg.replace('"', '').replace("'", '')
        parts.append(cleaned_arg)
    
    if parts:
        return [" ".join(parts)]
    return []