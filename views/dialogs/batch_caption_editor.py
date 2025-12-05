"""
Batch Caption Editor - Edit multiple captions at once
"""
from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel,
                           QPushButton, QLineEdit, QTableWidget,
                           QTableWidgetItem, QGroupBox, QMessageBox,
                           QHeaderView, QAbstractItemView)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap
from pathlib import Path


class BatchCaptionEditor(QDialog):
    """
    Dialog for batch editing captions with find/replace functionality.
    """

    def __init__(self, captions_data: list, parent=None):
        """
        Initialize the batch caption editor.

        Args:
            captions_data: List of (image_path, caption_text) tuples
            parent: Parent widget
        """
        super().__init__(parent)
        self.captions_data = captions_data
        self.setWindowTitle("Editor de Captions em Lote")
        self.setModal(True)
        self.setMinimumWidth(900)
        self.setMinimumHeight(600)

        self.init_ui()
        self.load_data()

    def init_ui(self):
        """Initialize the UI components"""
        main_layout = QVBoxLayout()

        # Find/Replace section
        find_replace_group = QGroupBox("Buscar e Substituir")
        find_replace_layout = QHBoxLayout()

        # Find
        find_replace_layout.addWidget(QLabel("Buscar:"))
        self.find_input = QLineEdit()
        self.find_input.setPlaceholderText("Texto para buscar...")
        find_replace_layout.addWidget(self.find_input)

        # Replace
        find_replace_layout.addWidget(QLabel("Substituir por:"))
        self.replace_input = QLineEdit()
        self.replace_input.setPlaceholderText("Novo texto...")
        find_replace_layout.addWidget(self.replace_input)

        # Buttons
        self.find_button = QPushButton("Buscar")
        self.find_button.clicked.connect(self.find_text)
        find_replace_layout.addWidget(self.find_button)

        self.replace_all_button = QPushButton("Substituir Tudo")
        self.replace_all_button.clicked.connect(self.find_replace_all)
        find_replace_layout.addWidget(self.replace_all_button)

        find_replace_group.setLayout(find_replace_layout)
        main_layout.addWidget(find_replace_group)

        # Table for captions
        self.table = QTableWidget()
        self.table.setColumnCount(3)
        self.table.setHorizontalHeaderLabels(["Thumbnail", "Nome do Arquivo", "Caption"])

        # Configure table
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Fixed)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)

        self.table.setColumnWidth(0, 100)  # Thumbnail column
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.DoubleClicked |
                                  QAbstractItemView.EditTrigger.SelectedClicked)

        main_layout.addWidget(self.table)

        # Info label
        self.info_label = QLabel(f"Total: {len(self.captions_data)} captions")
        main_layout.addWidget(self.info_label)

        # Bottom buttons
        buttons_layout = QHBoxLayout()

        self.save_button = QPushButton("Salvar Todas")
        self.save_button.clicked.connect(self.save_all)
        self.save_button.setDefault(True)

        self.cancel_button = QPushButton("Cancelar")
        self.cancel_button.clicked.connect(self.reject)

        buttons_layout.addStretch()
        buttons_layout.addWidget(self.save_button)
        buttons_layout.addWidget(self.cancel_button)

        main_layout.addLayout(buttons_layout)

        self.setLayout(main_layout)

    def load_data(self):
        """Populate table with caption data"""
        self.table.setRowCount(len(self.captions_data))

        for row, (img_path, caption) in enumerate(self.captions_data):
            # Column 0: Thumbnail
            thumbnail_label = QLabel()
            thumbnail_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

            try:
                pixmap = QPixmap(img_path)
                if not pixmap.isNull():
                    scaled_pixmap = pixmap.scaled(
                        80, 80,
                        Qt.AspectRatioMode.KeepAspectRatio,
                        Qt.TransformationMode.SmoothTransformation
                    )
                    thumbnail_label.setPixmap(scaled_pixmap)
                else:
                    thumbnail_label.setText("N/A")
            except Exception:
                thumbnail_label.setText("N/A")

            self.table.setCellWidget(row, 0, thumbnail_label)
            self.table.setRowHeight(row, 90)

            # Column 1: Filename
            filename_item = QTableWidgetItem(Path(img_path).name)
            filename_item.setFlags(filename_item.flags() & ~Qt.ItemFlag.ItemIsEditable)  # Read-only
            self.table.setItem(row, 1, filename_item)

            # Column 2: Caption (editable)
            caption_item = QTableWidgetItem(caption)
            self.table.setItem(row, 2, caption_item)

    def find_text(self):
        """Highlight rows containing the search text"""
        find_text = self.find_input.text()
        if not find_text:
            QMessageBox.warning(self, "Aviso", "Digite um texto para buscar.")
            return

        # Clear previous selection
        self.table.clearSelection()

        # Find and select matching rows
        matches = 0
        for row in range(self.table.rowCount()):
            caption_item = self.table.item(row, 2)
            if caption_item and find_text.lower() in caption_item.text().lower():
                self.table.selectRow(row)
                matches += 1

        if matches == 0:
            QMessageBox.information(self, "Resultado", "Nenhuma caption encontrada com esse texto.")
        else:
            QMessageBox.information(self, "Resultado", f"Encontradas {matches} captions com '{find_text}'.")

    def find_replace_all(self):
        """Replace text in all captions"""
        find_text = self.find_input.text()
        replace_text = self.replace_input.text()

        if not find_text:
            QMessageBox.warning(self, "Aviso", "Digite um texto para buscar.")
            return

        # Confirm replacement
        reply = QMessageBox.question(
            self,
            "Confirmar Substituição",
            f"Substituir todas as ocorrências de '{find_text}' por '{replace_text}'?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )

        if reply != QMessageBox.StandardButton.Yes:
            return

        # Perform replacement
        replaced_count = 0
        for row in range(self.table.rowCount()):
            caption_item = self.table.item(row, 2)
            if caption_item and find_text in caption_item.text():
                new_text = caption_item.text().replace(find_text, replace_text)
                caption_item.setText(new_text)
                replaced_count += 1

        QMessageBox.information(
            self,
            "Concluído",
            f"Substituído em {replaced_count} captions."
        )

    def save_all(self):
        """Save all captions back to files"""
        try:
            saved_count = 0
            errors = []

            for row in range(self.table.rowCount()):
                # Get original image path
                img_path = Path(self.captions_data[row][0])

                # Get edited caption text
                caption_item = self.table.item(row, 2)
                if not caption_item:
                    continue

                caption_text = caption_item.text()

                # Determine caption file path
                # Check if caption is in a subdirectory
                if (img_path.parent / "captions").exists():
                    caption_dir = img_path.parent / "captions"
                else:
                    # Try parent's captions directory
                    caption_dir = img_path.parent.parent / "captions"
                    if not caption_dir.exists():
                        # Use same directory as image
                        caption_dir = img_path.parent

                caption_file = caption_dir / f"{img_path.stem}.txt"

                # Save caption
                try:
                    caption_file.write_text(caption_text, encoding='utf-8')
                    saved_count += 1
                except Exception as e:
                    errors.append(f"{img_path.name}: {str(e)}")

            # Show result
            if errors:
                error_msg = "\n".join(errors[:10])  # Show first 10 errors
                if len(errors) > 10:
                    error_msg += f"\n... e mais {len(errors) - 10} erros"

                QMessageBox.warning(
                    self,
                    "Salvo com Erros",
                    f"Salvos: {saved_count}\nErros: {len(errors)}\n\n{error_msg}"
                )
            else:
                QMessageBox.information(
                    self,
                    "Sucesso",
                    f"Todas as {saved_count} captions foram salvas com sucesso!"
                )
                self.accept()

        except Exception as e:
            QMessageBox.critical(
                self,
                "Erro",
                f"Erro ao salvar captions: {str(e)}"
            )

    def get_edited_data(self):
        """
        Get all edited caption data.

        Returns:
            List of (image_path, caption_text) tuples
        """
        edited_data = []
        for row in range(self.table.rowCount()):
            img_path = self.captions_data[row][0]
            caption_item = self.table.item(row, 2)
            if caption_item:
                caption_text = caption_item.text()
                edited_data.append((img_path, caption_text))

        return edited_data
