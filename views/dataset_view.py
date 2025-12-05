from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QListWidget, QListWidgetItem, QGroupBox, 
    QPushButton, QLabel, QSpinBox, QSplitter, QComboBox, QTextEdit
)
from PyQt6.QtGui import QIcon, QStandardItemModel, QStandardItem
from PyQt6.QtCore import Qt, QSize

class DatasetView(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_window = parent  # Reference to main window for actions
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
        
        # 0. Project Artifacts (New)
        artifact_group = QGroupBox("Project Artifacts")
        artifact_layout = QVBoxLayout()
        
        self.artifact_combo = QComboBox()
        self.artifact_combo.setPlaceholderText("Select an artifact folder...")
        self.artifact_combo.currentIndexChanged.connect(self.parent_window.on_artifact_selected)
        artifact_layout.addWidget(self.artifact_combo)
        
        self.toml_info = QTextEdit()
        self.toml_info.setReadOnly(True)
        self.toml_info.setMaximumHeight(100)
        self.toml_info.setPlaceholderText("Select an artifact to view dataset.toml info...")
        artifact_layout.addWidget(self.toml_info)
        
        artifact_group.setLayout(artifact_layout)
        layout.addWidget(artifact_group)
        
        # 1. Image Processing
        img_group = QGroupBox("1. Image Processing")
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
        
        # 2. Utilities
        util_group = QGroupBox("2. Utilities")
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
