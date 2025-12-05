"""
Caption Processing View - MVC-compliant view for caption generation
"""
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QGroupBox, QPushButton,
                           QMessageBox, QDialog)
from PyQt6.QtCore import pyqtSignal
from pathlib import Path


class CaptionProcessingView(QWidget):
    """
    View component for caption processing tab.
    Emits signals instead of processing directly (follows MVC pattern).
    """

    # Signals
    generate_clicked = pyqtSignal(dict)  # config
    analyze_clicked = pyqtSignal()
    generate_toml_clicked = pyqtSignal()

    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.dataset_path = None
        self.init_ui()

    def init_ui(self):
        """Initialize the UI components"""
        layout = QVBoxLayout()
        layout.setSpacing(10)

        # Caption Generation Group
        caption_group = self.create_caption_generation_group()
        layout.addWidget(caption_group)

        self.setLayout(layout)

    def create_caption_generation_group(self):
        """Create the caption generation UI group"""
        group = QGroupBox("1. Geração de Captions")
        group.setMinimumHeight(100)
        layout = QVBoxLayout()
        layout.setContentsMargins(10, 15, 10, 15)

        # Generate Captions button
        generate_captions_btn = QPushButton("Gerar Captions")
        generate_captions_btn.setToolTip(
            "Gerar captions para as imagens usando modelos de IA\n"
            "Disponível: Florence-2, Danbooru, Janus-7B, Qwen3-VL"
        )
        generate_captions_btn.clicked.connect(self.on_generate_clicked)
        layout.addWidget(generate_captions_btn)

        # Analyze Dataset button (optional - could be moved to separate group)
        analyze_btn = QPushButton("Analisar Dataset")
        analyze_btn.setToolTip("Ver estatísticas sobre imagens e captions no dataset")
        analyze_btn.clicked.connect(self.on_analyze_clicked)
        layout.addWidget(analyze_btn)

        group.setLayout(layout)
        return group

    def on_generate_clicked(self):
        """Handle generate captions button click"""
        # Update dataset path from main window
        if hasattr(self.main_window, 'get_effective_dataset_path'):
            self.dataset_path = self.main_window.get_effective_dataset_path()

        # Validate dataset path
        if not self.dataset_path:
            QMessageBox.warning(
                self,
                "Aviso",
                "Por favor, selecione uma pasta de dataset primeiro!"
            )
            return

        # Check if dataset has images
        try:
            dataset_path = Path(self.dataset_path)

            # Check if dataset_path itself has images (e.g. artifact folder)
            has_images = any(dataset_path.glob("*.[jp][pn][g]"))

            if not has_images:
                # Check for cropped_images subdirectory
                cropped_dir = dataset_path / "cropped_images"
                if cropped_dir.exists():
                    has_images = any(cropped_dir.glob("*.[jp][pn][g]"))

            if not has_images:
                QMessageBox.warning(
                    self,
                    "Aviso",
                    "Nenhuma imagem encontrada! Por favor, selecione uma pasta com imagens ou processe as imagens primeiro."
                )
                return

        except Exception as e:
            QMessageBox.critical(
                self,
                "Erro",
                f"Erro ao verificar dataset: {str(e)}"
            )
            return

        # Show configuration dialog
        from views.dialogs.caption_config_dialog import CaptionConfigDialog

        config_dialog = CaptionConfigDialog(self)
        if config_dialog.exec() == QDialog.DialogCode.Accepted:
            config = config_dialog.get_values()
            # Emit signal with configuration
            self.generate_clicked.emit(config)

    def on_analyze_clicked(self):
        """Handle analyze dataset button click"""
        # Update dataset path from main window
        if hasattr(self.main_window, 'get_effective_dataset_path'):
            self.dataset_path = self.main_window.get_effective_dataset_path()

        if not self.dataset_path:
            QMessageBox.warning(
                self,
                "Aviso",
                "Por favor, selecione uma pasta de dataset primeiro!"
            )
            return

        try:
            dataset_path = Path(self.dataset_path)
            stats = {
                "total_images": 0,
                "total_captions": 0,
                "missing_captions": []
            }

            # Check if dataset_path has images directly
            has_images = any(dataset_path.glob("*.[jp][pn][g]"))

            if has_images:
                images_dir = dataset_path
            else:
                images_dir = dataset_path / "cropped_images"

            if images_dir.exists():
                image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]
                for ext in image_extensions:
                    stats["total_images"] += len(list(images_dir.glob(ext)))

            captions_dir = images_dir / "captions"
            if captions_dir.exists():
                stats["total_captions"] = len(list(captions_dir.glob("*.txt")))

                # Find missing captions
                for ext in image_extensions:
                    for img_path in images_dir.glob(ext):
                        caption_path = captions_dir / f"{img_path.stem}.txt"
                        if not caption_path.exists():
                            stats["missing_captions"].append(img_path.name)

            # Display statistics
            msg = f"""Análise do Dataset:

Total de Imagens: {stats['total_images']}
Total de Captions: {stats['total_captions']}
Captions Faltando: {len(stats['missing_captions'])}"""

            if stats["missing_captions"]:
                msg += "\n\nArquivos sem captions:"
                for file in stats["missing_captions"][:10]:
                    msg += f"\n- {file}"
                if len(stats["missing_captions"]) > 10:
                    msg += f"\n... e mais {len(stats['missing_captions']) - 10}"

            QMessageBox.information(self, "Análise do Dataset", msg)

        except Exception as e:
            QMessageBox.critical(
                self,
                "Erro",
                f"Erro ao analisar dataset: {str(e)}"
            )

    def on_dataset_changed(self, dataset_path):
        """Update dataset path when it changes"""
        self.dataset_path = dataset_path

    def show_success_message(self, processed: int, failed: int):
        """Show success message after caption generation"""
        msg = f"Geração de captions concluída!\n\nProcessadas com sucesso: {processed}\nFalhas: {failed}"

        if failed > 0:
            QMessageBox.warning(self, "Concluído com Erros", msg)
        else:
            QMessageBox.information(self, "Sucesso", msg)

        # Refresh image grid
        if hasattr(self.main_window, 'populate_image_grid') and self.dataset_path:
            self.main_window.populate_image_grid(Path(self.dataset_path))
