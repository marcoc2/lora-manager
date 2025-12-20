"""
Training History Dialog

Janela para visualizar o histórico completo de treinamentos,
com tabela ordenável, filtros, e painel de detalhes.
"""
import json
from pathlib import Path
from datetime import datetime
from typing import Optional, List

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QTableWidget, QTableWidgetItem,
    QPushButton, QLabel, QComboBox, QTextEdit, QSplitter, QGroupBox,
    QHeaderView, QWidget, QMessageBox
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap

from views.dataset_view import StarRatingWidget


class TrainingHistoryDialog(QDialog):
    """Janela de histórico de treinamentos"""

    def __init__(self, project_manager, parent=None):
        super().__init__(parent)
        self.project_manager = project_manager
        self.setWindowTitle("Training History")
        self.setMinimumSize(900, 600)
        self.selected_index = -1

        self.init_ui()
        self.load_history()

    def init_ui(self):
        layout = QVBoxLayout()

        # Filtros
        filter_layout = QHBoxLayout()

        filter_layout.addWidget(QLabel("Filter by Model:"))
        self.model_filter = QComboBox()
        self.model_filter.addItem("All Models", None)
        self.model_filter.currentIndexChanged.connect(self.apply_filters)
        filter_layout.addWidget(self.model_filter)

        filter_layout.addWidget(QLabel("Min Rating:"))
        self.rating_filter = QComboBox()
        self.rating_filter.addItem("Any", 0)
        for i in range(1, 6):
            self.rating_filter.addItem("★" * i, i)
        self.rating_filter.currentIndexChanged.connect(self.apply_filters)
        filter_layout.addWidget(self.rating_filter)

        filter_layout.addStretch()
        layout.addLayout(filter_layout)

        # Splitter para tabela e detalhes
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Tabela de histórico
        self.table = QTableWidget()
        self.table.setColumnCount(7)
        self.table.setHorizontalHeaderLabels([
            "Date", "Name", "Model", "Steps", "LR", "Loss", "Rating"
        ])
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.table.setSortingEnabled(True)
        self.table.setAlternatingRowColors(True)

        # Ajusta colunas
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(5, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(6, QHeaderView.ResizeMode.ResizeToContents)

        self.table.itemSelectionChanged.connect(self.on_selection_changed)
        splitter.addWidget(self.table)

        # Painel de detalhes
        details_widget = QWidget()
        details_layout = QVBoxLayout()
        details_layout.setContentsMargins(10, 0, 0, 0)

        # Rating do item selecionado
        rating_group = QGroupBox("Rate This Training")
        rating_layout = QVBoxLayout()
        self.detail_rating = StarRatingWidget()
        self.detail_rating.rating_changed.connect(self.on_rating_changed)
        rating_layout.addWidget(self.detail_rating)
        rating_group.setLayout(rating_layout)
        details_layout.addWidget(rating_group)

        # Notas
        notes_group = QGroupBox("Notes")
        notes_layout = QVBoxLayout()
        self.notes_edit = QTextEdit()
        self.notes_edit.setPlaceholderText("Add notes about this training run...")
        self.notes_edit.setMaximumHeight(100)
        self.notes_edit.textChanged.connect(self.on_notes_changed)
        notes_layout.addWidget(self.notes_edit)
        notes_group.setLayout(notes_layout)
        details_layout.addWidget(notes_group)

        # Config
        config_group = QGroupBox("Training Config")
        config_layout = QVBoxLayout()
        self.config_display = QTextEdit()
        self.config_display.setReadOnly(True)
        self.config_display.setStyleSheet("font-family: monospace; font-size: 11px;")
        config_layout.addWidget(self.config_display)
        config_group.setLayout(config_layout)
        details_layout.addWidget(config_group)

        # Preview (se existir)
        self.preview_group = QGroupBox("Preview")
        preview_layout = QVBoxLayout()
        self.preview_label = QLabel("No preview available")
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setMinimumHeight(150)
        self.preview_label.setStyleSheet("background-color: #1a1a1a;")
        preview_layout.addWidget(self.preview_label)

        open_folder_btn = QPushButton("Open Output Folder")
        open_folder_btn.clicked.connect(self.open_output_folder)
        preview_layout.addWidget(open_folder_btn)

        self.preview_group.setLayout(preview_layout)
        details_layout.addWidget(self.preview_group)

        details_widget.setLayout(details_layout)
        splitter.addWidget(details_widget)

        # Proporções do splitter
        splitter.setSizes([500, 400])
        layout.addWidget(splitter)

        # Botão fechar
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        layout.addWidget(close_btn)

        self.setLayout(layout)

    def load_history(self):
        """Carrega o histórico do projeto"""
        history = self.project_manager.get_training_history()

        # Popula o filtro de modelos
        model_types = set()
        for run in history:
            model_types.add(run.model_type)

        self.model_filter.blockSignals(True)
        for model in sorted(model_types):
            self.model_filter.addItem(model, model)
        self.model_filter.blockSignals(False)

        self.populate_table(history)

    def populate_table(self, history: list):
        """Popula a tabela com os treinamentos"""
        self.table.setRowCount(0)

        for idx, run in enumerate(history):
            row = self.table.rowCount()
            self.table.insertRow(row)

            # Data
            date_str = run.timestamp.strftime("%m/%d %H:%M")
            date_item = QTableWidgetItem(date_str)
            date_item.setData(Qt.ItemDataRole.UserRole, idx)  # Guarda índice original
            self.table.setItem(row, 0, date_item)

            # Nome
            self.table.setItem(row, 1, QTableWidgetItem(run.output_name))

            # Modelo
            self.table.setItem(row, 2, QTableWidgetItem(run.model_type))

            # Steps
            steps_item = QTableWidgetItem()
            steps_item.setData(Qt.ItemDataRole.DisplayRole, run.steps)
            self.table.setItem(row, 3, steps_item)

            # Learning Rate
            lr = run.config_snapshot.get("learning_rate", 0)
            if lr:
                try:
                    lr_float = float(lr)
                    lr_str = f"{lr_float:.0e}" if lr_float < 0.01 else f"{lr_float}"
                except (ValueError, TypeError):
                    lr_str = str(lr)
            else:
                lr_str = "-"
            self.table.setItem(row, 4, QTableWidgetItem(lr_str))

            # Loss
            if run.final_loss:
                loss_item = QTableWidgetItem(f"{run.final_loss:.4f}")
            else:
                loss_item = QTableWidgetItem("-")
            self.table.setItem(row, 5, loss_item)

            # Rating
            if run.rating:
                rating_str = "★" * run.rating + "☆" * (5 - run.rating)
            else:
                rating_str = "☆☆☆☆☆"
            rating_item = QTableWidgetItem(rating_str)
            rating_item.setData(Qt.ItemDataRole.UserRole + 1, run.rating or 0)
            self.table.setItem(row, 6, rating_item)

    def apply_filters(self):
        """Aplica os filtros selecionados"""
        history = self.project_manager.get_training_history()

        # Filtro por modelo
        model_filter = self.model_filter.currentData()
        if model_filter:
            history = [r for r in history if r.model_type == model_filter]

        # Filtro por rating mínimo
        min_rating = self.rating_filter.currentData()
        if min_rating:
            history = [r for r in history if (r.rating or 0) >= min_rating]

        self.populate_table(history)

    def on_selection_changed(self):
        """Chamado quando a seleção na tabela muda"""
        selected = self.table.selectedItems()
        if not selected:
            self.selected_index = -1
            self.clear_details()
            return

        # Pega o índice original do primeiro item da linha
        row = selected[0].row()
        idx_item = self.table.item(row, 0)
        self.selected_index = idx_item.data(Qt.ItemDataRole.UserRole)

        self.show_details(self.selected_index)

    def show_details(self, index: int):
        """Mostra detalhes do treinamento selecionado"""
        history = self.project_manager.get_training_history()
        if index < 0 or index >= len(history):
            self.clear_details()
            return

        run = history[index]

        # Rating
        self.detail_rating.blockSignals(True)
        self.detail_rating.set_rating(run.rating or 0)
        self.detail_rating.blockSignals(False)

        # Notas
        self.notes_edit.blockSignals(True)
        self.notes_edit.setText(run.notes or "")
        self.notes_edit.blockSignals(False)

        # Config
        config_json = json.dumps(run.config_snapshot, indent=2, default=str)
        self.config_display.setText(config_json)

        # Preview
        self.load_preview(run.output_path)

    def clear_details(self):
        """Limpa o painel de detalhes"""
        self.detail_rating.set_rating(0)
        self.notes_edit.clear()
        self.config_display.clear()
        self.preview_label.setText("Select a training to view details")
        self.preview_label.setPixmap(QPixmap())

    def load_preview(self, output_path: Optional[str]):
        """Carrega preview do treinamento"""
        if not output_path:
            self.preview_label.setText("No output folder")
            return

        path = Path(output_path)
        if not path.exists():
            self.preview_label.setText("Folder not found")
            return

        # Procura por imagens de preview
        preview_files = list(path.glob("*.png")) + list(path.glob("*.jpg"))
        if preview_files:
            # Pega a mais recente
            latest = max(preview_files, key=lambda p: p.stat().st_mtime)
            pixmap = QPixmap(str(latest))
            if not pixmap.isNull():
                scaled = pixmap.scaled(
                    self.preview_label.width() - 10,
                    self.preview_label.height() - 10,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation
                )
                self.preview_label.setPixmap(scaled)
                return

        self.preview_label.setText("No preview images found")

    def on_rating_changed(self, rating: int):
        """Atualiza o rating do treinamento selecionado"""
        if self.selected_index < 0:
            return

        self.project_manager.update_training_rating(self.selected_index, rating)

        # Atualiza a tabela
        for row in range(self.table.rowCount()):
            idx_item = self.table.item(row, 0)
            if idx_item and idx_item.data(Qt.ItemDataRole.UserRole) == self.selected_index:
                rating_str = "★" * rating + "☆" * (5 - rating)
                self.table.item(row, 6).setText(rating_str)
                break

    def on_notes_changed(self):
        """Atualiza as notas do treinamento selecionado"""
        if self.selected_index < 0:
            return

        notes = self.notes_edit.toPlainText()
        self.project_manager.update_training_notes(self.selected_index, notes)

    def open_output_folder(self):
        """Abre a pasta de saída no explorador"""
        if self.selected_index < 0:
            return

        history = self.project_manager.get_training_history()
        if self.selected_index >= len(history):
            return

        run = history[self.selected_index]
        if not run.output_path:
            QMessageBox.information(self, "No Path", "No output path recorded for this training.")
            return

        path = Path(run.output_path)
        if not path.exists():
            QMessageBox.warning(self, "Not Found", f"Folder not found:\n{path}")
            return

        import os
        os.startfile(str(path))
