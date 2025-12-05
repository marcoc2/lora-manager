"""
Enhanced Caption Configuration Dialog
"""
from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel,
                           QPushButton, QLineEdit, QFormLayout,
                           QComboBox, QTextEdit, QCheckBox, QGroupBox)
from PyQt6.QtCore import Qt
import json
from pathlib import Path


# Prompt templates for different caption styles
PROMPT_TEMPLATES = {
    "Custom": "",
    "Detailed": "Describe this image in rich detail, including the subject, setting, colors, mood, and any notable elements.",
    "Concise": "Provide a brief, clear description of this image.",
    "Booru-style": "Generate comma-separated tags describing this image, focusing on visual elements, style, and composition.",
    "Character Focus": "Describe the character in this image, including their appearance, clothing, pose, and expression.",
    "Scene Description": "Describe the scene and environment in this image, including the setting, atmosphere, and background elements."
}


class CaptionConfigDialog(QDialog):
    """Enhanced configuration dialog for caption generation"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Caption Configuration")
        self.setModal(True)
        self.setMinimumWidth(500)

        main_layout = QVBoxLayout()

        # Basic Configuration Group
        basic_group = QGroupBox("Configuração Básica")
        basic_layout = QFormLayout()

        # Method selection
        self.method_combo = QComboBox()
        self.method_combo.addItems(["Florence-2", "Danbooru", "Janus-7B", "Qwen3-VL"])
        self.method_combo.setToolTip("Selecione o modelo de anotação de imagens")
        basic_layout.addRow("Método de Caption:", self.method_combo)

        # Trigger word - NEW
        self.trigger_word = QLineEdit()
        self.trigger_word.setPlaceholderText("Ex: character_name, style_tag, etc.")
        self.trigger_word.setToolTip("Esta palavra será adicionada no início de todas as captions geradas")
        basic_layout.addRow("Trigger Word:", self.trigger_word)

        # Prefix field (legacy - kept for compatibility)
        self.prefix = QLineEdit()
        self.prefix.setPlaceholderText("Prefixo adicional (opcional)")
        self.prefix.setToolTip("Texto adicional para adicionar após a trigger word")
        basic_layout.addRow("Prefixo Adicional:", self.prefix)

        basic_group.setLayout(basic_layout)
        main_layout.addWidget(basic_group)

        # Custom Prompt Group - NEW
        prompt_group = QGroupBox("Prompt Customizado")
        prompt_layout = QVBoxLayout()

        # Template selection
        template_layout = QHBoxLayout()
        template_layout.addWidget(QLabel("Template:"))
        self.prompt_template = QComboBox()
        self.prompt_template.addItems(list(PROMPT_TEMPLATES.keys()))
        self.prompt_template.setToolTip("Selecione um template predefinido ou escolha 'Custom' para criar seu próprio")
        self.prompt_template.currentTextChanged.connect(self.on_template_changed)
        template_layout.addWidget(self.prompt_template)
        template_layout.addStretch()
        prompt_layout.addLayout(template_layout)

        # Custom prompt text
        self.custom_prompt = QTextEdit()
        self.custom_prompt.setPlaceholderText("Digite seu prompt customizado aqui...\n\nEste prompt substituirá ou complementará o prompt padrão do modelo selecionado.")
        self.custom_prompt.setMaximumHeight(100)
        self.custom_prompt.setToolTip("Prompt customizado para o modelo de anotação")
        prompt_layout.addWidget(self.custom_prompt)

        prompt_group.setLayout(prompt_layout)
        main_layout.addWidget(prompt_group)

        # Model-specific options
        self.options_group = QGroupBox("Opções Específicas do Modelo")
        options_layout = QFormLayout()

        # Danbooru model selection
        self.model_combo = QComboBox()
        self.model_combo.addItems(["vit", "swinv2", "convnext"])
        self.model_combo.setVisible(False)
        self.model_label = QLabel("Modelo Danbooru:")
        self.model_label.setVisible(False)
        options_layout.addRow(self.model_label, self.model_combo)

        # Janus options
        self.janus_context = QTextEdit()
        self.janus_context.setPlaceholderText("Contexto adicional para o Janus (opcional)")
        self.janus_context.setMaximumHeight(80)
        self.janus_context.setVisible(False)
        self.janus_label = QLabel("Contexto Janus:")
        self.janus_label.setVisible(False)
        options_layout.addRow(self.janus_label, self.janus_context)

        self.replace_prompt = QCheckBox("Substituir Prompt Padrão")
        self.replace_prompt.setVisible(False)
        self.replace_prompt.setToolTip("Se marcado, o prompt customizado substituirá completamente o prompt padrão do Janus")
        options_layout.addRow("", self.replace_prompt)

        self.options_group.setLayout(options_layout)
        main_layout.addWidget(self.options_group)

        # Connect method change
        self.method_combo.currentTextChanged.connect(self.on_method_changed)

        # Buttons
        buttons = QHBoxLayout()
        ok_button = QPushButton("Gerar Captions")
        cancel_button = QPushButton("Cancelar")

        ok_button.setDefault(True)
        ok_button.clicked.connect(self.accept)
        cancel_button.clicked.connect(self.reject)

        buttons.addStretch()
        buttons.addWidget(ok_button)
        buttons.addWidget(cancel_button)

        main_layout.addLayout(buttons)
        self.setLayout(main_layout)

        # Load saved configuration
        self.load_config()

        # Update visibility for initial method
        self.on_method_changed(self.method_combo.currentText())

    def on_method_changed(self, method: str):
        """Update visible options based on selected method"""
        # Danbooru options
        is_danbooru = method == "Danbooru"
        self.model_combo.setVisible(is_danbooru)
        self.model_label.setVisible(is_danbooru)

        # Janus options
        is_janus = method == "Janus-7B"
        self.janus_context.setVisible(is_janus)
        self.janus_label.setVisible(is_janus)
        self.replace_prompt.setVisible(is_janus)

        # Update prompt group visibility
        # Custom prompt is available for Florence-2, Janus-7B, and Qwen3-VL
        # Danbooru uses tags, not prompts
        prompt_enabled = method in ["Florence-2", "Janus-7B", "Qwen3-VL"]
        self.custom_prompt.setEnabled(prompt_enabled)
        self.prompt_template.setEnabled(prompt_enabled)

        if not prompt_enabled:
            self.custom_prompt.setPlaceholderText(
                "Prompt customizado não disponível para o método selecionado"
            )
        else:
            self.custom_prompt.setPlaceholderText(
                "Digite seu prompt customizado aqui...\n\n"
                "Este prompt substituirá ou complementará o prompt padrão do modelo selecionado."
            )

    def on_template_changed(self, template_name: str):
        """Update custom prompt when template is changed"""
        if template_name in PROMPT_TEMPLATES:
            template_text = PROMPT_TEMPLATES[template_name]
            if template_text:  # Don't overwrite if "Custom" is selected
                self.custom_prompt.setPlainText(template_text)

    def get_values(self) -> dict:
        """Get all configuration values"""
        return {
            'method': self.method_combo.currentText(),
            'trigger_word': self.trigger_word.text().strip(),
            'prefix': self.prefix.text().strip(),
            'custom_prompt': self.custom_prompt.toPlainText().strip(),
            'prompt_template': self.prompt_template.currentText(),
            'model_type': self.model_combo.currentText() if self.method_combo.currentText() == "Danbooru" else None,
            'janus_context': self.janus_context.toPlainText() if self.method_combo.currentText() == "Janus-7B" else None,
            'replace_prompt': self.replace_prompt.isChecked() if self.method_combo.currentText() == "Janus-7B" else False
        }

    def load_config(self):
        """Load saved configuration from file"""
        config_file = Path("caption_config.json")
        if config_file.exists():
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)

                # Apply saved values
                if 'method' in config:
                    index = self.method_combo.findText(config['method'])
                    if index >= 0:
                        self.method_combo.setCurrentIndex(index)

                if 'trigger_word' in config:
                    self.trigger_word.setText(config['trigger_word'])

                if 'prefix' in config:
                    self.prefix.setText(config['prefix'])

                if 'custom_prompt' in config:
                    self.custom_prompt.setPlainText(config['custom_prompt'])

                if 'prompt_template' in config:
                    index = self.prompt_template.findText(config['prompt_template'])
                    if index >= 0:
                        self.prompt_template.setCurrentIndex(index)

                if 'model_type' in config and config['model_type']:
                    index = self.model_combo.findText(config['model_type'])
                    if index >= 0:
                        self.model_combo.setCurrentIndex(index)

            except Exception as e:
                print(f"Erro ao carregar configuração: {e}")

    def save_config(self):
        """Save configuration to file"""
        config = self.get_values()
        config_file = Path("caption_config.json")

        try:
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Erro ao salvar configuração: {e}")

    def accept(self):
        """Override accept to save config before closing"""
        self.save_config()
        super().accept()
