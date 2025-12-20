from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QListWidget, QListWidgetItem, QGroupBox,
    QPushButton, QLabel, QSpinBox, QSplitter, QComboBox, QTextEdit, QLineEdit,
    QFrame
)
from PyQt6.QtGui import QIcon, QStandardItemModel, QStandardItem, QCursor
from PyQt6.QtCore import Qt, QSize, pyqtSignal


class StarRatingWidget(QWidget):
    """Widget clicável de 5 estrelas para rating"""
    rating_changed = pyqtSignal(int)  # Emite 1-5

    def __init__(self, parent=None):
        super().__init__(parent)
        self._rating = 0  # 0 = não avaliado
        self._hover_rating = 0
        self.init_ui()

    def init_ui(self):
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        self.stars = []
        for i in range(5):
            star = QLabel("☆")
            star.setStyleSheet("font-size: 18px; color: #FFD700;")
            star.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
            star.mousePressEvent = lambda e, idx=i: self._on_star_clicked(idx + 1)
            star.enterEvent = lambda e, idx=i: self._on_star_hover(idx + 1)
            star.leaveEvent = lambda e: self._on_star_leave()
            self.stars.append(star)
            layout.addWidget(star)

        layout.addStretch()
        self.setLayout(layout)

    def _on_star_clicked(self, rating: int):
        self._rating = rating
        self._update_display()
        self.rating_changed.emit(rating)

    def _on_star_hover(self, rating: int):
        self._hover_rating = rating
        self._update_display()

    def _on_star_leave(self):
        self._hover_rating = 0
        self._update_display()

    def _update_display(self):
        display_rating = self._hover_rating if self._hover_rating > 0 else self._rating
        for i, star in enumerate(self.stars):
            if i < display_rating:
                star.setText("★")
            else:
                star.setText("☆")

    def set_rating(self, rating: int):
        """Define o rating programaticamente"""
        self._rating = max(0, min(5, rating)) if rating else 0
        self._update_display()

    def get_rating(self) -> int:
        return self._rating

class DatasetView(QWidget):
    # Signals for project changes
    project_name_changed = pyqtSignal(str)
    trigger_word_changed = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_window = parent  # Reference to main window for actions
        self._project = None  # Reference to current project
        self.init_ui()
        
    def init_ui(self):
        main_layout = QHBoxLayout()
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(15)
        
        # Splitter to allow resizing between tree and tools
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Left side: Tree View
        left_widget = self.create_tree_panel()
        splitter.addWidget(left_widget)
        
        # Right side: Tools
        right_widget = self.create_tools_panel()
        splitter.addWidget(right_widget)
        
        # Set initial sizes (40% tree, 60% tools)
        splitter.setSizes([400, 600])
        
        main_layout.addWidget(splitter)
        self.setLayout(main_layout)
        
    def create_tree_panel(self):
        panel = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Header with Select Button
        header_layout = QHBoxLayout()
        select_button = QPushButton("Select Dataset Folder")
        select_button.setObjectName("primaryButton") # For styling
        select_button.setMinimumHeight(40)
        select_button.clicked.connect(self.parent_window.select_dataset_folder)
        header_layout.addWidget(select_button)
        layout.addLayout(header_layout)
        
        # Image Grid View
        self.image_list = QListWidget()
        self.image_list.setViewMode(QListWidget.ViewMode.IconMode)
        self.image_list.setIconSize(QSize(128, 128))
        self.image_list.setResizeMode(QListWidget.ResizeMode.Adjust)
        self.image_list.setSpacing(10)
        self.image_list.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        layout.addWidget(self.image_list)
        
        panel.setLayout(layout)
        return panel
        
    def create_tools_panel(self):
        panel = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(15)
        
        # 0. Project Info
        project_group = QGroupBox("Project")
        project_layout = QVBoxLayout()

        # Project name
        name_layout = QHBoxLayout()
        name_layout.addWidget(QLabel("Name:"))
        self.project_name_edit = QLineEdit()
        self.project_name_edit.setPlaceholderText("No project loaded")
        self.project_name_edit.textChanged.connect(self._on_project_name_changed)
        name_layout.addWidget(self.project_name_edit)
        project_layout.addLayout(name_layout)

        # Trigger word (global)
        trigger_layout = QHBoxLayout()
        trigger_layout.addWidget(QLabel("Trigger:"))
        self.project_trigger_edit = QLineEdit()
        self.project_trigger_edit.setPlaceholderText("ex: p3rs0n")
        self.project_trigger_edit.textChanged.connect(self._on_trigger_word_changed)
        trigger_layout.addWidget(self.project_trigger_edit)
        project_layout.addLayout(trigger_layout)

        # Save status indicator
        self.save_status_label = QLabel("")
        self.save_status_label.setStyleSheet("color: #888; font-size: 10px;")
        project_layout.addWidget(self.save_status_label)

        project_group.setLayout(project_layout)
        layout.addWidget(project_group)

        # 1. Target Image Folder
        target_group = QGroupBox("Target Image Folder")
        target_layout = QVBoxLayout()

        self.artifact_combo = QComboBox()
        self.artifact_combo.setPlaceholderText("Select target folder...")
        self.artifact_combo.setToolTip(
            "Selecione a pasta de imagens para processamento/treino.\n"
            "• original_dataset: imagens originais (ideal para ai-toolkit)\n"
            "• cropped_images_*: imagens pré-processadas (necessário para kohya)"
        )
        self.artifact_combo.currentIndexChanged.connect(self.parent_window.on_artifact_selected)
        target_layout.addWidget(self.artifact_combo)

        self.toml_info = QTextEdit()
        self.toml_info.setReadOnly(True)
        self.toml_info.setMaximumHeight(80)
        self.toml_info.setPlaceholderText("Select a folder to view info...")
        target_layout.addWidget(self.toml_info)

        target_group.setLayout(target_layout)
        layout.addWidget(target_group)
        
        # 2. Image Processing
        img_group = QGroupBox("2. Image Processing")
        img_layout = QVBoxLayout()
        
        size_layout = QHBoxLayout()
        size_layout.addWidget(QLabel("Target Size:"))
        self.crop_width = QSpinBox()
        self.crop_width.setRange(64, 2048)
        self.crop_width.setValue(512)
        self.crop_height = QSpinBox()
        self.crop_height.setRange(64, 2048)
        self.crop_height.setValue(512)
        size_layout.addWidget(self.crop_width)
        size_layout.addWidget(QLabel("x"))
        size_layout.addWidget(self.crop_height)
        img_layout.addLayout(size_layout)
        
        self.face_detection = QPushButton("Face Detection: ON")
        self.face_detection.setCheckable(True)
        self.face_detection.setChecked(True)
        self.face_detection.clicked.connect(self.parent_window.toggle_face_detection)
        img_layout.addWidget(self.face_detection)
        
        process_btn = QPushButton("Process Images")
        process_btn.setObjectName("actionButton")
        process_btn.clicked.connect(self.parent_window.process_images)
        img_layout.addWidget(process_btn)
        
        img_group.setLayout(img_layout)
        layout.addWidget(img_group)
        
        # 3. Last Training - Resumo do último treino
        self.last_training_group = QGroupBox("Last Training")
        last_training_layout = QVBoxLayout()

        # Nome do treino
        self.last_train_name = QLabel("No training yet")
        self.last_train_name.setStyleSheet("font-weight: bold;")
        last_training_layout.addWidget(self.last_train_name)

        # Data e modelo
        self.last_train_info = QLabel("")
        self.last_train_info.setStyleSheet("color: #888; font-size: 11px;")
        last_training_layout.addWidget(self.last_train_info)

        # Steps e Loss
        self.last_train_stats = QLabel("")
        self.last_train_stats.setStyleSheet("font-size: 12px;")
        last_training_layout.addWidget(self.last_train_stats)

        # Rating com estrelas
        rating_layout = QHBoxLayout()
        rating_layout.addWidget(QLabel("Rating:"))
        self.last_train_rating = StarRatingWidget()
        self.last_train_rating.rating_changed.connect(self._on_last_training_rated)
        rating_layout.addWidget(self.last_train_rating)
        rating_layout.addStretch()
        last_training_layout.addLayout(rating_layout)

        # Botão para ver histórico completo
        history_btn = QPushButton("View Full History")
        history_btn.setObjectName("actionButton")
        history_btn.clicked.connect(self._open_training_history)
        last_training_layout.addWidget(history_btn)

        self.last_training_group.setLayout(last_training_layout)
        layout.addWidget(self.last_training_group)
        

        
        layout.addStretch() # Push everything up
        panel.setLayout(layout)
        return panel

    def populate_image_grid(self, path):
        self.image_list.clear()
        
        if not path.exists():
            return

        image_extensions = {'.png', '.jpg', '.jpeg', '.webp', '.bmp'}
        
        # If path is the root dataset folder, check if we should look into "cropped_images"
        # But since we have the artifact selector, we might be passed the specific folder directly.
        # Let's assume 'path' is the directory we want to show images from.
        
        try:
            for item_path in sorted(path.iterdir()):
                if item_path.is_file() and item_path.suffix.lower() in image_extensions:
                    icon = QIcon(str(item_path))
                    item = QListWidgetItem(icon, item_path.name)
                    self.image_list.addItem(item)
        except Exception as e:
            print(f"Error loading images from {path}: {e}")

    # -------------------------------------------------------------------------
    # Project Management Methods
    # -------------------------------------------------------------------------

    def update_project_info(self, project):
        """Update UI with project information"""
        self._project = project

        # Block signals to avoid triggering auto-save during population
        self.project_name_edit.blockSignals(True)
        self.project_trigger_edit.blockSignals(True)

        if project is None:
            self.project_name_edit.setText("")
            self.project_name_edit.setPlaceholderText("No project loaded")
            self.project_name_edit.setEnabled(False)
            self.project_trigger_edit.setText("")
            self.project_trigger_edit.setEnabled(False)
            self.save_status_label.setText("")
        else:
            self.project_name_edit.setText(project.project_name)
            self.project_name_edit.setEnabled(True)
            self.project_trigger_edit.setText(project.trigger_word or "")
            self.project_trigger_edit.setEnabled(True)
            self.save_status_label.setText("Project loaded")
            self.save_status_label.setStyleSheet("color: #4CAF50; font-size: 10px;")

        self.project_name_edit.blockSignals(False)
        self.project_trigger_edit.blockSignals(False)

        # Atualiza painel de último treinamento
        self.update_last_training()

    def update_save_status(self, status: str):
        """Update save status indicator"""
        if status == "saving":
            self.save_status_label.setText("Saving...")
            self.save_status_label.setStyleSheet("color: #FF9800; font-size: 10px;")
        elif status == "saved":
            self.save_status_label.setText("Saved")
            self.save_status_label.setStyleSheet("color: #4CAF50; font-size: 10px;")
        elif status == "error":
            self.save_status_label.setText("Save error!")
            self.save_status_label.setStyleSheet("color: #f44336; font-size: 10px;")

    def _on_project_name_changed(self, text: str):
        """Handle project name changes - trigger auto-save"""
        if self._project and hasattr(self.parent_window, 'project_manager'):
            self.parent_window.project_manager.update_project_info(name=text)

    def _on_trigger_word_changed(self, text: str):
        """Handle trigger word changes - trigger auto-save"""
        if self._project and hasattr(self.parent_window, 'project_manager'):
            self.parent_window.project_manager.set_trigger_word(text)

    # -------------------------------------------------------------------------
    # Last Training Panel Methods
    # -------------------------------------------------------------------------

    def update_last_training(self):
        """Atualiza o painel com informações do último treinamento"""
        if not hasattr(self.parent_window, 'project_manager'):
            return

        pm = self.parent_window.project_manager
        last_run = pm.get_last_training_run()

        if last_run is None:
            self.last_train_name.setText("No training yet")
            self.last_train_info.setText("")
            self.last_train_stats.setText("")
            self.last_train_rating.set_rating(0)
            self.last_train_rating.setEnabled(False)
            return

        self.last_train_rating.setEnabled(True)

        # Nome do treino
        self.last_train_name.setText(last_run.output_name)

        # Data e modelo
        date_str = last_run.timestamp.strftime("%Y-%m-%d %H:%M")
        self.last_train_info.setText(f"{date_str} | {last_run.model_type}")

        # Steps e Loss
        loss_str = f"{last_run.final_loss:.4f}" if last_run.final_loss else "N/A"
        self.last_train_stats.setText(f"Steps: {last_run.steps} | Loss: {loss_str}")

        # Rating
        self.last_train_rating.set_rating(last_run.rating or 0)

    def _on_last_training_rated(self, rating: int):
        """Chamado quando o usuário clica nas estrelas do último treino"""
        if not hasattr(self.parent_window, 'project_manager'):
            return

        pm = self.parent_window.project_manager
        history = pm.get_training_history()

        if history:
            # Atualiza o rating do último treino (último da lista)
            pm.update_training_rating(len(history) - 1, rating)

    def _open_training_history(self):
        """Abre a janela de histórico de treinamentos"""
        if not hasattr(self.parent_window, 'project_manager'):
            return

        pm = self.parent_window.project_manager
        if not pm.has_project:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.information(self, "No Project", "Please load a project first.")
            return

        from views.training_history_dialog import TrainingHistoryDialog
        dialog = TrainingHistoryDialog(pm, self)
        dialog.exec()

        # Atualiza o painel após fechar (pode ter mudado ratings)
        self.update_last_training()
