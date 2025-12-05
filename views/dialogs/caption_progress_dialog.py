"""
Caption Progress Dialog - Rich progress UI with live preview
"""
from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel,
                           QPushButton, QProgressBar, QTextEdit,
                           QListWidget, QGroupBox, QSizePolicy)
from PyQt6.QtCore import pyqtSignal, Qt
from PyQt6.QtGui import QPixmap
from pathlib import Path


class CaptionProgressDialog(QDialog):
    """
    Rich progress dialog for caption generation.
    Shows live preview of images, captions, and statistics.
    """

    # Signals
    cancel_requested = pyqtSignal()
    batch_edit_requested = pyqtSignal(list)  # list of (path, caption) tuples

    MAX_RECENT_ITEMS = 20  # Limit recent captions list

    def __init__(self, parent=None):
        super().__init__(parent)
        self.captions_data = []  # Store all generated captions
        self.setWindowTitle("Gerando Captions")
        self.setModal(True)
        self.setMinimumWidth(800)
        self.setMinimumHeight(600)

        self.init_ui()

    def init_ui(self):
        """Initialize the UI components"""
        main_layout = QVBoxLayout()

        # Progress bar at top
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        main_layout.addWidget(self.progress_bar)

        # Status label
        self.status_label = QLabel("Iniciando...")
        self.status_label.setWordWrap(True)
        main_layout.addWidget(self.status_label)

        # Statistics row
        stats_layout = QHBoxLayout()

        self.speed_label = QLabel("Velocidade: -")
        self.eta_label = QLabel("ETA: -")
        self.count_label = QLabel("Processadas: 0/0")

        stats_layout.addWidget(self.speed_label)
        stats_layout.addWidget(self.eta_label)
        stats_layout.addWidget(self.count_label)
        stats_layout.addStretch()

        main_layout.addLayout(stats_layout)

        # Preview section (image + caption side by side)
        preview_layout = QHBoxLayout()

        # Image preview group
        image_group = QGroupBox("Imagem Atual")
        image_layout = QVBoxLayout()

        self.thumbnail_label = QLabel()
        self.thumbnail_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.thumbnail_label.setMinimumSize(300, 300)
        self.thumbnail_label.setMaximumSize(300, 300)
        self.thumbnail_label.setScaledContents(False)
        self.thumbnail_label.setText("Aguardando...")
        self.thumbnail_label.setStyleSheet("QLabel { border: 1px solid #555; background-color: #2b2b2b; }")

        self.image_name_label = QLabel("")
        self.image_name_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_name_label.setWordWrap(True)

        image_layout.addWidget(self.thumbnail_label)
        image_layout.addWidget(self.image_name_label)
        image_group.setLayout(image_layout)
        preview_layout.addWidget(image_group)

        # Caption preview group
        caption_group = QGroupBox("Caption Gerada")
        caption_layout = QVBoxLayout()

        self.caption_text = QTextEdit()
        self.caption_text.setReadOnly(True)
        self.caption_text.setPlaceholderText("A caption aparecerá aqui...")
        self.caption_text.setMinimumHeight(250)

        caption_layout.addWidget(self.caption_text)
        caption_group.setLayout(caption_layout)
        preview_layout.addWidget(caption_group)

        main_layout.addLayout(preview_layout)

        # Recent captions list
        recent_group = QGroupBox("Captions Recentes")
        recent_layout = QVBoxLayout()

        self.recent_list = QListWidget()
        self.recent_list.setMinimumHeight(150)
        recent_layout.addWidget(self.recent_list)

        recent_group.setLayout(recent_layout)
        main_layout.addWidget(recent_group)

        # Buttons
        buttons_layout = QHBoxLayout()

        self.cancel_button = QPushButton("Cancelar")
        self.cancel_button.clicked.connect(self.on_cancel_clicked)

        self.batch_edit_button = QPushButton("Editar em Lote")
        self.batch_edit_button.clicked.connect(self.on_batch_edit_clicked)
        self.batch_edit_button.setEnabled(False)  # Enabled when processing completes

        self.close_button = QPushButton("Fechar")
        self.close_button.clicked.connect(self.accept)
        self.close_button.setEnabled(False)  # Enabled when processing completes

        buttons_layout.addWidget(self.batch_edit_button)
        buttons_layout.addStretch()
        buttons_layout.addWidget(self.cancel_button)
        buttons_layout.addWidget(self.close_button)

        main_layout.addLayout(buttons_layout)

        self.setLayout(main_layout)

    def on_caption_ready(self, image_path: str, caption: str, thumbnail_path: str):
        """
        Slot: Update UI when caption is generated.

        Args:
            image_path: Path to the image file
            caption: Generated caption text
            thumbnail_path: Path to thumbnail (usually same as image_path)
        """
        # Update thumbnail
        try:
            pixmap = QPixmap(thumbnail_path)
            if not pixmap.isNull():
                scaled_pixmap = pixmap.scaled(
                    300, 300,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation
                )
                self.thumbnail_label.setPixmap(scaled_pixmap)
            else:
                self.thumbnail_label.setText("Erro ao carregar imagem")
        except Exception as e:
            self.thumbnail_label.setText(f"Erro: {e}")

        # Update image name
        self.image_name_label.setText(Path(image_path).name)

        # Update caption display
        self.caption_text.setPlainText(caption)

        # Add to recent list (with success icon)
        filename = Path(image_path).name
        caption_preview = caption[:50] + "..." if len(caption) > 50 else caption
        self.recent_list.insertItem(0, f"✓ {filename}: {caption_preview}")

        # Limit list size to prevent memory issues
        while self.recent_list.count() > self.MAX_RECENT_ITEMS:
            # Remove the last (oldest) item
            self.recent_list.takeItem(self.recent_list.count() - 1)

        # Store for batch editor
        self.captions_data.append((image_path, caption))

    def on_progress_updated(self, current: int, total: int, message: str):
        """
        Slot: Update progress bar and statistics.

        Args:
            current: Number of images processed so far
            total: Total number of images
            message: Detailed status message
        """
        # Update progress bar
        if total > 0:
            percentage = int((current / total) * 100)
            self.progress_bar.setValue(percentage)
            self.progress_bar.setFormat(f"{current}/{total} ({percentage}%)")

        # Update status message
        self.status_label.setText(message)

        # Parse statistics from message if available
        # Message format: "Processando file.jpg (10/100) | Velocidade: 5.2 img/min | ETA: 17.3 min"
        parts = message.split("|")
        if len(parts) >= 3:
            # Update count
            count_part = parts[0].strip()
            self.count_label.setText(f"Processadas: {current}/{total}")

            # Update speed
            speed_part = parts[1].strip()
            self.speed_label.setText(speed_part)

            # Update ETA
            eta_part = parts[2].strip()
            self.eta_label.setText(eta_part)
        else:
            # Fallback
            self.count_label.setText(f"Processadas: {current}/{total}")

    def on_error(self, image_path: str, error_msg: str):
        """
        Slot: Display error in recent list.

        Args:
            image_path: Path to the image that failed
            error_msg: Error message
        """
        if image_path:
            filename = Path(image_path).name
            error_text = f"✗ {filename}: ERRO - {error_msg[:100]}"
        else:
            # General error
            error_text = f"✗ ERRO: {error_msg[:100]}"

        self.recent_list.insertItem(0, error_text)

        # Limit list size
        while self.recent_list.count() > self.MAX_RECENT_ITEMS:
            # Remove the last (oldest) item
            self.recent_list.takeItem(self.recent_list.count() - 1)

    def on_complete(self, processed: int, failed: int):
        """
        Slot: Handle completion of caption generation.

        Args:
            processed: Number of successfully processed images
            failed: Number of failed images
        """
        # Update status
        total = processed + failed
        self.status_label.setText(
            f"Concluído! {processed} captions geradas com sucesso, {failed} falhas."
        )

        # Set progress to 100%
        self.progress_bar.setValue(100)
        self.progress_bar.setFormat(f"Concluído ({total} imagens)")

        # Enable/disable buttons
        self.cancel_button.setEnabled(False)
        self.close_button.setEnabled(True)

        if processed > 0:
            self.batch_edit_button.setEnabled(True)

    def on_cancel_clicked(self):
        """Emit cancel signal to controller"""
        self.cancel_requested.emit()
        self.status_label.setText("Cancelando...")
        self.cancel_button.setEnabled(False)

    def on_batch_edit_clicked(self):
        """Launch batch editor with current captions"""
        if self.captions_data:
            self.batch_edit_requested.emit(self.captions_data)

    def closeEvent(self, event):
        """Override close event to handle cancellation"""
        if self.cancel_button.isEnabled():
            # Still processing - ask to cancel
            self.cancel_requested.emit()
        event.accept()
