"""
Caption Processing View - Comprehensive caption management interface
"""
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QPushButton,
                           QMessageBox, QDialog, QLabel, QLineEdit, QListWidget,
                           QListWidgetItem, QTextEdit, QSplitter, QComboBox,
                           QFormLayout, QProgressDialog, QCheckBox, QFrame,
                           QScrollArea, QSizePolicy)
from PyQt6.QtCore import pyqtSignal, Qt, QTimer
from PyQt6.QtGui import QFont
from pathlib import Path
import json


# Prompt templates for caption styles
PROMPT_TEMPLATES = {
    "Custom": "",
    "Detailed": "Describe this image in rich detail, including the subject, setting, colors, mood, and any notable elements.",
    "Concise": "Provide a brief, clear description of this image.",
    "Booru-style": "Generate comma-separated tags describing this image, focusing on visual elements, style, and composition.",
    "Character Focus": "Describe the character in this image, including their appearance, clothing, pose, and expression.",
    "Scene Description": "Describe the scene and environment in this image, including the setting, atmosphere, and background elements."
}


class CaptionProcessingView(QWidget):
    """
    Comprehensive view for caption generation and management.
    """

    # Signals
    generate_clicked = pyqtSignal(dict)
    analyze_clicked = pyqtSignal()

    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.dataset_path = None
        self.captions_dir = None
        self.current_caption_file = None
        self.caption_files = []
        self.init_ui()

        # Auto-refresh timer
        self.refresh_timer = QTimer()
        self.refresh_timer.timeout.connect(self.check_for_updates)

    def init_ui(self):
        """Initialize the UI components"""
        main_layout = QVBoxLayout()
        main_layout.setSpacing(8)
        main_layout.setContentsMargins(8, 8, 8, 8)

        # Top section: Generation controls
        generation_group = self.create_generation_group()
        main_layout.addWidget(generation_group)

        # Middle section: Keyword/Prefix tools
        keyword_group = self.create_keyword_group()
        main_layout.addWidget(keyword_group)

        # Bottom section: Caption browser (splitter for resizable)
        browser_group = self.create_caption_browser()
        main_layout.addWidget(browser_group, 1)  # stretch factor 1

        self.setLayout(main_layout)

    def create_generation_group(self):
        """Create caption generation controls"""
        group = QGroupBox("Geração de Captions")
        layout = QVBoxLayout()
        layout.setSpacing(8)

        # Row 1: Method selection and trigger word
        row1 = QHBoxLayout()

        # Method
        row1.addWidget(QLabel("Método:"))
        self.method_combo = QComboBox()
        self.method_combo.addItems(["Florence-2", "Danbooru", "Janus-7B", "Qwen3-VL"])
        self.method_combo.setMinimumWidth(120)
        self.method_combo.currentTextChanged.connect(self.on_method_changed)
        row1.addWidget(self.method_combo)

        row1.addSpacing(20)

        # Trigger word
        row1.addWidget(QLabel("Trigger Word:"))
        self.trigger_word = QLineEdit()
        self.trigger_word.setPlaceholderText("Ex: p3rs0n, style_name")
        self.trigger_word.setMinimumWidth(150)
        row1.addWidget(self.trigger_word)

        row1.addStretch()
        layout.addLayout(row1)

        # Row 2: Danbooru model (conditional)
        self.danbooru_row = QHBoxLayout()
        self.danbooru_label = QLabel("Modelo Danbooru:")
        self.danbooru_combo = QComboBox()
        self.danbooru_combo.addItems(["vit", "swinv2", "convnext"])
        self.danbooru_row.addWidget(self.danbooru_label)
        self.danbooru_row.addWidget(self.danbooru_combo)
        self.danbooru_row.addStretch()

        self.danbooru_widget = QWidget()
        self.danbooru_widget.setLayout(self.danbooru_row)
        self.danbooru_widget.setVisible(False)
        layout.addWidget(self.danbooru_widget)

        # Row 3: Custom prompt section
        prompt_layout = QHBoxLayout()
        prompt_layout.addWidget(QLabel("Prompt Template:"))
        self.prompt_template = QComboBox()
        self.prompt_template.addItems(list(PROMPT_TEMPLATES.keys()))
        self.prompt_template.currentTextChanged.connect(self.on_template_changed)
        prompt_layout.addWidget(self.prompt_template)
        prompt_layout.addStretch()
        layout.addLayout(prompt_layout)

        self.custom_prompt = QTextEdit()
        self.custom_prompt.setPlaceholderText("Prompt customizado (opcional)...")
        self.custom_prompt.setMaximumHeight(60)
        layout.addWidget(self.custom_prompt)

        # Row 4: Generate button
        btn_row = QHBoxLayout()
        self.generate_btn = QPushButton("Gerar Captions")
        self.generate_btn.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; padding: 8px;")
        self.generate_btn.clicked.connect(self.on_generate_clicked)
        btn_row.addWidget(self.generate_btn)

        self.refresh_btn = QPushButton("Atualizar Lista")
        self.refresh_btn.clicked.connect(self.refresh_captions_list)
        btn_row.addWidget(self.refresh_btn)

        btn_row.addStretch()
        layout.addLayout(btn_row)

        group.setLayout(layout)
        return group

    def create_keyword_group(self):
        """Create keyword/prefix management tools"""
        group = QGroupBox("Adicionar Palavra-Chave às Captions")
        layout = QVBoxLayout()
        layout.setSpacing(8)

        # Info label
        info_label = QLabel("Adiciona uma palavra-chave no início de todas as captions existentes:")
        info_label.setStyleSheet("color: #666;")
        layout.addWidget(info_label)

        # Input row
        input_row = QHBoxLayout()

        input_row.addWidget(QLabel("Palavra-chave:"))
        self.keyword_input = QLineEdit()
        self.keyword_input.setPlaceholderText("Ex: trigger_word, ")
        self.keyword_input.setMinimumWidth(200)
        input_row.addWidget(self.keyword_input)

        self.add_comma_check = QCheckBox("Adicionar vírgula após")
        self.add_comma_check.setChecked(True)
        input_row.addWidget(self.add_comma_check)

        input_row.addStretch()
        layout.addLayout(input_row)

        # Buttons row
        btn_row = QHBoxLayout()

        self.prepend_btn = QPushButton("Adicionar no Início de Todas")
        self.prepend_btn.setStyleSheet("background-color: #FF9800; color: white; padding: 6px;")
        self.prepend_btn.clicked.connect(self.prepend_keyword_to_all)
        btn_row.addWidget(self.prepend_btn)

        self.remove_keyword_btn = QPushButton("Remover de Todas")
        self.remove_keyword_btn.clicked.connect(self.remove_keyword_from_all)
        btn_row.addWidget(self.remove_keyword_btn)

        self.remove_newlines_btn = QPushButton("Remover Quebras de Linha")
        self.remove_newlines_btn.setToolTip("Remove todas as quebras de linha de todas as captions")
        self.remove_newlines_btn.clicked.connect(self.remove_newlines_from_all)
        btn_row.addWidget(self.remove_newlines_btn)

        btn_row.addStretch()

        # Stats label
        self.stats_label = QLabel("")
        self.stats_label.setStyleSheet("color: #007acc; font-weight: bold;")
        btn_row.addWidget(self.stats_label)

        layout.addLayout(btn_row)

        group.setLayout(layout)
        return group

    def create_caption_browser(self):
        """Create caption file browser with preview"""
        group = QGroupBox("Navegador de Captions")
        layout = QVBoxLayout()

        # Splitter for resizable panels
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left panel: File list
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)

        list_header = QHBoxLayout()
        list_header.addWidget(QLabel("Arquivos de Caption:"))
        self.file_count_label = QLabel("(0 arquivos)")
        self.file_count_label.setStyleSheet("color: #666;")
        list_header.addWidget(self.file_count_label)
        list_header.addStretch()
        left_layout.addLayout(list_header)

        self.caption_list = QListWidget()
        self.caption_list.setMinimumWidth(200)
        self.caption_list.currentItemChanged.connect(self.on_caption_selected)
        left_layout.addWidget(self.caption_list)

        splitter.addWidget(left_panel)

        # Right panel: Caption preview/edit
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)

        preview_header = QHBoxLayout()
        self.preview_filename = QLabel("Selecione um arquivo")
        self.preview_filename.setStyleSheet("font-weight: bold;")
        preview_header.addWidget(self.preview_filename)
        preview_header.addStretch()
        right_layout.addLayout(preview_header)

        self.caption_preview = QTextEdit()
        self.caption_preview.setPlaceholderText("Selecione um arquivo da lista para ver/editar a caption...")
        right_layout.addWidget(self.caption_preview)

        # Edit buttons
        edit_btn_row = QHBoxLayout()
        self.save_caption_btn = QPushButton("Salvar Alterações")
        self.save_caption_btn.clicked.connect(self.save_current_caption)
        self.save_caption_btn.setEnabled(False)
        edit_btn_row.addWidget(self.save_caption_btn)

        self.delete_caption_btn = QPushButton("Excluir Caption")
        self.delete_caption_btn.clicked.connect(self.delete_current_caption)
        self.delete_caption_btn.setEnabled(False)
        self.delete_caption_btn.setStyleSheet("color: #d32f2f;")
        edit_btn_row.addWidget(self.delete_caption_btn)

        edit_btn_row.addStretch()
        right_layout.addLayout(edit_btn_row)

        splitter.addWidget(right_panel)

        # Set initial sizes (40% list, 60% preview)
        splitter.setSizes([300, 450])

        layout.addWidget(splitter)
        group.setLayout(layout)
        return group

    def on_method_changed(self, method: str):
        """Handle caption method change"""
        is_danbooru = method == "Danbooru"
        self.danbooru_widget.setVisible(is_danbooru)

        # Enable/disable prompt for methods that support it
        prompt_enabled = method in ["Florence-2", "Janus-7B", "Qwen3-VL"]
        self.custom_prompt.setEnabled(prompt_enabled)
        self.prompt_template.setEnabled(prompt_enabled)

    def on_template_changed(self, template_name: str):
        """Update custom prompt when template changes"""
        if template_name in PROMPT_TEMPLATES:
            template_text = PROMPT_TEMPLATES[template_name]
            if template_text:
                self.custom_prompt.setPlainText(template_text)

    def on_generate_clicked(self):
        """Handle generate captions button click"""
        # Validation is handled by MainController via PathResolver
        # Just build config and emit signal - controller will validate

        # Build config and emit signal
        config = {
            'method': self.method_combo.currentText(),
            'trigger_word': self.trigger_word.text().strip(),
            'prefix': self.trigger_word.text().strip(),  # For compatibility
            'custom_prompt': self.custom_prompt.toPlainText().strip(),
            'prompt_template': self.prompt_template.currentText(),
            'model_type': self.danbooru_combo.currentText() if self.method_combo.currentText() == "Danbooru" else None,
            'janus_context': self.custom_prompt.toPlainText().strip() if self.method_combo.currentText() == "Janus-7B" else None,
            'replace_prompt': False
        }

        self.generate_clicked.emit(config)

    def on_dataset_changed(self, dataset_path):
        """Update when dataset changes"""
        self.dataset_path = dataset_path
        self.refresh_captions_list()

    def refresh_captions_list(self):
        """Refresh the list of caption files"""
        self.caption_list.clear()
        self.caption_files = []
        self.current_caption_file = None
        self.caption_preview.clear()
        self.preview_filename.setText("Selecione um arquivo")
        self.save_caption_btn.setEnabled(False)
        self.delete_caption_btn.setEnabled(False)

        # Use PathResolver if available
        if hasattr(self.main_window, 'path_resolver') and self.main_window.path_resolver:
            resolver = self.main_window.path_resolver
            self.captions_dir = resolver.get_captions_directory()
            self.dataset_path = str(resolver.get_images_directory()) if resolver.get_images_directory() else None
        else:
            # Fallback to old logic if resolver not available
            if not self.dataset_path:
                if hasattr(self.main_window, 'get_effective_dataset_path'):
                    self.dataset_path = self.main_window.get_effective_dataset_path()

            if not self.dataset_path:
                self.file_count_label.setText("(0 arquivos)")
                self.stats_label.setText("")
                return

            dataset_path = Path(self.dataset_path)

            # Find captions directory - check multiple possible locations
            possible_dirs = [
                dataset_path / "captions",  # Direct captions in selected folder
            ]

            # Add cropped_images variants (cropped_images, cropped_images_512x512, etc.)
            for variant in dataset_path.glob("cropped_images*"):
                if variant.is_dir():
                    possible_dirs.append(variant / "captions")

            # Also check if dataset_path itself has captions alongside images
            possible_dirs.append(dataset_path)

            self.captions_dir = None
            for d in possible_dirs:
                if d.exists() and any(d.glob("*.txt")):
                    self.captions_dir = d
                    break

        if not self.captions_dir or not self.captions_dir.exists():
            self.file_count_label.setText("(0 arquivos)")
            self.stats_label.setText("Nenhuma caption encontrada")
            return

        # Check if there are txt files
        if not any(self.captions_dir.glob("*.txt")):
            self.file_count_label.setText("(0 arquivos)")
            self.stats_label.setText("Nenhuma caption encontrada")
            return

        # Load caption files
        self.caption_files = sorted(self.captions_dir.glob("*.txt"), key=lambda x: x.name)

        for caption_file in self.caption_files:
            item = QListWidgetItem(caption_file.name)
            item.setData(Qt.ItemDataRole.UserRole, str(caption_file))
            self.caption_list.addItem(item)

        self.file_count_label.setText(f"({len(self.caption_files)} arquivos)")

        # Update stats
        self.update_stats()

    def update_stats(self):
        """Update statistics label"""
        if not self.caption_files:
            self.stats_label.setText("")
            return

        # Count images using PathResolver if available
        image_count = 0
        if hasattr(self.main_window, 'path_resolver') and self.main_window.path_resolver:
            resolver = self.main_window.path_resolver
            images_dir = resolver.get_images_directory()
            if images_dir:
                image_count = resolver.count_images(images_dir)
        elif self.dataset_path:
            # Fallback to manual counting
            dataset_path = Path(self.dataset_path)
            seen = set()
            for ext in ["*.jpg", "*.jpeg", "*.png", "*.webp"]:
                for f in dataset_path.glob(ext):
                    if f.name.lower() not in seen:
                        seen.add(f.name.lower())
                        image_count += 1

        self.stats_label.setText(f"Imagens: {image_count} | Captions: {len(self.caption_files)}")

    def on_caption_selected(self, current, previous):
        """Handle caption file selection"""
        if not current:
            return

        caption_path = current.data(Qt.ItemDataRole.UserRole)
        if not caption_path:
            return

        self.current_caption_file = Path(caption_path)
        self.preview_filename.setText(self.current_caption_file.name)

        try:
            content = self.current_caption_file.read_text(encoding='utf-8')
            self.caption_preview.setPlainText(content)
            self.save_caption_btn.setEnabled(True)
            self.delete_caption_btn.setEnabled(True)
        except Exception as e:
            self.caption_preview.setPlainText(f"Erro ao ler arquivo: {e}")
            self.save_caption_btn.setEnabled(False)

    def save_current_caption(self):
        """Save changes to the current caption file"""
        if not self.current_caption_file:
            return

        try:
            content = self.caption_preview.toPlainText()
            self.current_caption_file.write_text(content, encoding='utf-8')
            QMessageBox.information(self, "Sucesso", f"Caption salva: {self.current_caption_file.name}")
        except Exception as e:
            QMessageBox.critical(self, "Erro", f"Erro ao salvar: {e}")

    def delete_current_caption(self):
        """Delete the current caption file"""
        if not self.current_caption_file:
            return

        reply = QMessageBox.question(
            self, "Confirmar",
            f"Excluir {self.current_caption_file.name}?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )

        if reply == QMessageBox.StandardButton.Yes:
            try:
                self.current_caption_file.unlink()
                self.refresh_captions_list()
                QMessageBox.information(self, "Sucesso", "Caption excluída!")
            except Exception as e:
                QMessageBox.critical(self, "Erro", f"Erro ao excluir: {e}")

    def prepend_keyword_to_all(self):
        """Prepend keyword to all captions"""
        keyword = self.keyword_input.text().strip()
        if not keyword:
            QMessageBox.warning(self, "Aviso", "Digite uma palavra-chave!")
            return

        if not self.caption_files:
            QMessageBox.warning(self, "Aviso", "Nenhuma caption encontrada!")
            return

        # Add comma if requested
        if self.add_comma_check.isChecked() and not keyword.endswith(','):
            keyword = keyword + ", "
        elif not keyword.endswith(' '):
            keyword = keyword + " "

        reply = QMessageBox.question(
            self, "Confirmar",
            f"Adicionar '{keyword.strip()}' no início de {len(self.caption_files)} captions?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )

        if reply != QMessageBox.StandardButton.Yes:
            return

        # Process all files
        success = 0
        errors = []

        for caption_file in self.caption_files:
            try:
                content = caption_file.read_text(encoding='utf-8').strip()

                # Check if keyword already exists at start
                if content.lower().startswith(keyword.lower().strip()):
                    continue  # Skip if already has keyword

                new_content = keyword + content
                caption_file.write_text(new_content, encoding='utf-8')
                success += 1
            except Exception as e:
                errors.append(f"{caption_file.name}: {e}")

        # Show result
        msg = f"Modificadas: {success} captions"
        if errors:
            msg += f"\nErros: {len(errors)}"

        QMessageBox.information(self, "Concluído", msg)

        # Refresh view
        self.refresh_captions_list()

        # Reload current caption if any
        if self.current_caption_file:
            try:
                content = self.current_caption_file.read_text(encoding='utf-8')
                self.caption_preview.setPlainText(content)
            except:
                pass

    def remove_keyword_from_all(self):
        """Remove keyword from all captions"""
        keyword = self.keyword_input.text().strip()
        if not keyword:
            QMessageBox.warning(self, "Aviso", "Digite a palavra-chave para remover!")
            return

        if not self.caption_files:
            QMessageBox.warning(self, "Aviso", "Nenhuma caption encontrada!")
            return

        reply = QMessageBox.question(
            self, "Confirmar",
            f"Remover '{keyword}' do início de todas as captions?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )

        if reply != QMessageBox.StandardButton.Yes:
            return

        # Variations to check
        variations = [
            keyword + ", ",
            keyword + " ",
            keyword + ",",
            keyword
        ]

        success = 0
        for caption_file in self.caption_files:
            try:
                content = caption_file.read_text(encoding='utf-8')

                modified = False
                for var in variations:
                    if content.startswith(var):
                        content = content[len(var):].lstrip()
                        modified = True
                        break

                if modified:
                    caption_file.write_text(content, encoding='utf-8')
                    success += 1
            except:
                pass

        QMessageBox.information(self, "Concluído", f"Removido de {success} captions")
        self.refresh_captions_list()

        # Reload current
        if self.current_caption_file and self.current_caption_file.exists():
            try:
                content = self.current_caption_file.read_text(encoding='utf-8')
                self.caption_preview.setPlainText(content)
            except:
                pass

    def remove_newlines_from_all(self):
        """Remove all newlines from all caption files"""
        if not self.caption_files:
            QMessageBox.warning(self, "Aviso", "Nenhuma caption encontrada!")
            return

        reply = QMessageBox.question(
            self, "Confirmar",
            f"Remover todas as quebras de linha de {len(self.caption_files)} captions?\n\n"
            "Isso vai transformar captions multi-linha em uma única linha.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )

        if reply != QMessageBox.StandardButton.Yes:
            return

        success = 0
        for caption_file in self.caption_files:
            try:
                content = caption_file.read_text(encoding='utf-8')

                # Replace newlines with space, then clean up multiple spaces
                new_content = content.replace('\r\n', ' ').replace('\n', ' ').replace('\r', ' ')
                # Remove multiple spaces
                while '  ' in new_content:
                    new_content = new_content.replace('  ', ' ')
                new_content = new_content.strip()

                if new_content != content:
                    caption_file.write_text(new_content, encoding='utf-8')
                    success += 1
            except:
                pass

        QMessageBox.information(self, "Concluído", f"Modificadas: {success} captions")
        self.refresh_captions_list()

        # Reload current
        if self.current_caption_file and self.current_caption_file.exists():
            try:
                content = self.current_caption_file.read_text(encoding='utf-8')
                self.caption_preview.setPlainText(content)
            except:
                pass

    def check_for_updates(self):
        """Check if captions folder has changed"""
        if not self.captions_dir or not self.captions_dir.exists():
            return

        current_count = len(list(self.captions_dir.glob("*.txt")))
        if current_count != len(self.caption_files):
            self.refresh_captions_list()

    def add_caption_to_list(self, image_path: str, caption: str, thumbnail_path: str):
        """
        Add a single caption to the list as it's generated.
        Called incrementally during caption generation for real-time updates.
        """
        try:
            # Determine caption file path
            img_path = Path(image_path)

            # Update captions_dir if not set - use PathResolver
            if not self.captions_dir:
                if hasattr(self.main_window, 'path_resolver') and self.main_window.path_resolver:
                    self.captions_dir = self.main_window.path_resolver.get_captions_directory()
                else:
                    # Fallback: captions are in a "captions" subfolder of where the image is
                    image_parent = img_path.parent
                    potential_captions_dir = image_parent / "captions"
                    if potential_captions_dir.exists():
                        self.captions_dir = potential_captions_dir

            if not self.captions_dir:
                return

            caption_file = self.captions_dir / f"{img_path.stem}.txt"

            # Check if already in list
            for i in range(self.caption_list.count()):
                item = self.caption_list.item(i)
                if item and item.text() == caption_file.name:
                    return  # Already exists

            # Add to list
            if caption_file.exists():
                item = QListWidgetItem(caption_file.name)
                item.setData(Qt.ItemDataRole.UserRole, str(caption_file))
                self.caption_list.addItem(item)
                self.caption_files.append(caption_file)

                # Update count
                self.file_count_label.setText(f"({len(self.caption_files)} arquivos)")

                # Scroll to show new item
                self.caption_list.scrollToBottom()

        except Exception as e:
            print(f"Error adding caption to list: {e}")

    def show_success_message(self, processed: int, failed: int):
        """Show success message after caption generation"""
        msg = f"Geração de captions concluída!\n\nProcessadas com sucesso: {processed}\nFalhas: {failed}"

        if failed > 0:
            QMessageBox.warning(self, "Concluído com Erros", msg)
        else:
            QMessageBox.information(self, "Sucesso", msg)

        # Refresh caption list
        self.refresh_captions_list()

        # Refresh image grid
        if hasattr(self.main_window, 'populate_image_grid') and self.dataset_path:
            self.main_window.populate_image_grid(Path(self.dataset_path))

    def showEvent(self, event):
        """Called when widget becomes visible"""
        super().showEvent(event)
        # Refresh captions when tab is shown
        QTimer.singleShot(100, self.refresh_captions_list)

    def load_config(self):
        """Load saved configuration"""
        config_file = Path("caption_config.json")
        if config_file.exists():
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)

                if 'method' in config:
                    idx = self.method_combo.findText(config['method'])
                    if idx >= 0:
                        self.method_combo.setCurrentIndex(idx)

                if 'trigger_word' in config:
                    self.trigger_word.setText(config['trigger_word'])

                if 'custom_prompt' in config:
                    self.custom_prompt.setPlainText(config['custom_prompt'])

            except Exception as e:
                print(f"Error loading caption config: {e}")

    def save_config(self):
        """Save current configuration"""
        config = {
            'method': self.method_combo.currentText(),
            'trigger_word': self.trigger_word.text(),
            'custom_prompt': self.custom_prompt.toPlainText(),
            'prompt_template': self.prompt_template.currentText(),
        }

        try:
            with open("caption_config.json", 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error saving caption config: {e}")
