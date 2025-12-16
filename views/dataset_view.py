from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QListWidget, QListWidgetItem, QGroupBox,
    QPushButton, QLabel, QSpinBox, QSplitter, QComboBox, QTextEdit, QLineEdit
)
from PyQt6.QtGui import QIcon, QStandardItemModel, QStandardItem
from PyQt6.QtCore import Qt, QSize, pyqtSignal

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
        
        # 3. Utilities
        util_group = QGroupBox("3. Utilities")
        util_layout = QVBoxLayout()
        
        rename_btn = QPushButton("Rename and Convert Images")
        rename_btn.clicked.connect(self.parent_window.rename_and_convert_images)
        util_layout.addWidget(rename_btn)
        
        analyze_btn = QPushButton("Analyze Dataset")
        analyze_btn.clicked.connect(self.parent_window.analyze_dataset)
        util_layout.addWidget(analyze_btn)
        
        util_group.setLayout(util_layout)
        layout.addWidget(util_group)
        

        
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
