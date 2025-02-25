import sys
import os
import importlib.util
import traceback
import ast
import json

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QListWidget, QFileDialog, QLabel, QTextEdit, QLineEdit,
    QFormLayout, QMessageBox
)

CONFIG_FILE = "config.json"

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Processador de Textos com Plugins")
        self.plugins_dir = ""
        self.input_dir = ""
        self.plugins = {}      # dicionário: nome do plugin -> módulo carregado
        self.param_widgets = {}  # dicionário: nome do parâmetro -> (widget, tipo)
        self.config = {}

        # Carrega as configurações persistidas, se existirem
        self.load_config()

        # Configuração da interface
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # Seleção do diretório de textos
        input_layout = QHBoxLayout()
        self.input_dir_button = QPushButton("Selecionar diretório de textos")
        self.input_dir_button.clicked.connect(self.select_input_dir)
        self.input_dir_label = QLabel("Nenhum diretório selecionado")
        input_layout.addWidget(self.input_dir_button)
        input_layout.addWidget(self.input_dir_label)
        main_layout.addLayout(input_layout)

        # Seleção do diretório dos scripts
        plugins_layout = QHBoxLayout()
        self.plugins_dir_button = QPushButton("Selecionar diretório dos scripts")
        self.plugins_dir_button.clicked.connect(self.select_plugins_dir)
        self.plugins_dir_label = QLabel("Nenhum diretório selecionado")
        plugins_layout.addWidget(self.plugins_dir_button)
        plugins_layout.addWidget(self.plugins_dir_label)
        main_layout.addLayout(plugins_layout)

        # Lista de plugins disponíveis
        main_layout.addWidget(QLabel("Scripts disponíveis:"))
        self.plugin_list = QListWidget()
        self.plugin_list.itemSelectionChanged.connect(self.plugin_selected)
        main_layout.addWidget(self.plugin_list)

        # Área para parâmetros adicionais (criada dinamicamente)
        main_layout.addWidget(QLabel("Parâmetros adicionais:"))
        self.param_widget = QWidget()
        self.param_form_layout = QFormLayout(self.param_widget)
        main_layout.addWidget(self.param_widget)

        # Botão para executar o plugin selecionado
        self.execute_button = QPushButton("Executar script selecionado")
        self.execute_button.clicked.connect(self.execute_plugin)
        main_layout.addWidget(self.execute_button)

        # Área de log para mensagens
        main_layout.addWidget(QLabel("Log:"))
        self.log_text_edit = QTextEdit()
        self.log_text_edit.setReadOnly(True)
        main_layout.addWidget(self.log_text_edit)

        # Atualiza os labels com os diretórios persistidos (se houver)
        if self.input_dir:
            self.input_dir_label.setText(self.input_dir)
        if self.plugins_dir:
            self.plugins_dir_label.setText(self.plugins_dir)
            self.load_plugins()

    def log(self, message):
        self.log_text_edit.append(message)

    def load_config(self):
        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                    self.config = json.load(f)
            except Exception as e:
                self.log(f"Erro ao carregar config: {str(e)}")
                self.config = {}
        # Atualiza os diretórios com os valores salvos, se existirem
        self.input_dir = self.config.get("input_dir", "")
        self.plugins_dir = self.config.get("plugins_dir", "")

    def save_config(self):
        self.config["input_dir"] = self.input_dir
        self.config["plugins_dir"] = self.plugins_dir
        try:
            with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, ensure_ascii=False, indent=4)
        except Exception as e:
            self.log(f"Erro ao salvar config: {str(e)}")

    def select_input_dir(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Selecione o diretório de textos")
        if dir_path:
            self.input_dir = dir_path
            self.input_dir_label.setText(dir_path)
            self.log(f"Diretório de textos selecionado: {dir_path}")
            self.save_config()

    def select_plugins_dir(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Selecione o diretório dos scripts")
        if dir_path:
            self.plugins_dir = dir_path
            self.plugins_dir_label.setText(dir_path)
            self.log(f"Diretório dos scripts selecionado: {dir_path}")
            self.save_config()
            self.load_plugins()

    def load_plugins(self):
        self.plugin_list.clear()
        self.plugins = {}
        # Varre o diretório dos scripts procurando arquivos .py
        for filename in os.listdir(self.plugins_dir):
            if filename.endswith(".py"):
                plugin_path = os.path.join(self.plugins_dir, filename)
                plugin_name = os.path.splitext(filename)[0]
                try:
                    spec = importlib.util.spec_from_file_location(plugin_name, plugin_path)
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    # Verifica se o módulo tem a função process_files
                    if hasattr(module, "process_files"):
                        self.plugins[plugin_name] = module
                        self.plugin_list.addItem(plugin_name)
                        self.log(f"Plugin carregado: {plugin_name}")
                    else:
                        self.log(f"Ignorado (não possui process_files): {plugin_name}")
                except Exception as e:
                    self.log(f"Erro ao carregar plugin {plugin_name}: {str(e)}")
                    traceback.print_exc()

    def plugin_selected(self):
        # Limpa os widgets de parâmetros anteriores
        for i in reversed(range(self.param_form_layout.count())):
            widget = self.param_form_layout.itemAt(i).widget()
            if widget is not None:
                widget.setParent(None)
        self.param_widgets = {}

        selected_items = self.plugin_list.selectedItems()
        if not selected_items:
            return

        plugin_name = selected_items[0].text()
        module = self.plugins.get(plugin_name)
        if module and hasattr(module, "get_parameters"):
            try:
                params = module.get_parameters()
                for param in params:
                    # Cada parâmetro é um dicionário com chaves: name, label, type e default
                    name = param.get("name")
                    label_text = param.get("label", name)
                    default = param.get("default", "")
                    widget = QLineEdit()
                    widget.setText(str(default))
                    self.param_form_layout.addRow(label_text, widget)
                    self.param_widgets[name] = (widget, param.get("type", "str"))
            except Exception as e:
                self.log(f"Erro ao obter parâmetros do plugin {plugin_name}: {str(e)}")
                traceback.print_exc()

    def execute_plugin(self):
        selected_items = self.plugin_list.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "Aviso", "Nenhum script selecionado")
            return
        if not self.input_dir:
            QMessageBox.warning(self, "Aviso", "Nenhum diretório de textos selecionado")
            return

        plugin_name = selected_items[0].text()
        module = self.plugins.get(plugin_name)
        if not module:
            self.log(f"Plugin {plugin_name} não encontrado.")
            return

        # Coleta os parâmetros adicionais
        kwargs = {}
        for name, (widget, ptype) in self.param_widgets.items():
            value = widget.text()
            if ptype == "dict":
                try:
                    value = ast.literal_eval(value)
                except Exception as e:
                    QMessageBox.warning(self, "Aviso", f"Erro ao interpretar parâmetro '{name}' como dict: {str(e)}")
                    return
            elif ptype == "int":
                try:
                    value = int(value)
                except Exception as e:
                    QMessageBox.warning(self, "Aviso", f"Erro ao interpretar parâmetro '{name}' como int: {str(e)}")
                    return
            # Se for 'str' ou outro, mantém como string
            kwargs[name] = value

        self.log(f"Executando plugin '{plugin_name}' com input_dir: {self.input_dir} e parâmetros: {kwargs}")
        try:
            # Executa a função process_files do plugin
            module.process_files(self.input_dir, **kwargs)
            self.log(f"Plugin '{plugin_name}' executado com sucesso.")
        except Exception as e:
            self.log(f"Erro na execução do plugin '{plugin_name}': {str(e)}")
            traceback.print_exc()

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
