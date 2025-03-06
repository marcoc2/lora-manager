# Lora Training Manager

![Logo](icon.svg)

A graphical tool for managing datasets and training LoRA models for Stable Diffusion, with integration to Kohya scripts.

## Features

- Cross-platform graphical interface using PyQt6
- Tree-view visualization and management of dataset structure
- Automated image processing:
  - Facial detection and intelligent cropping
  - Resizing with aspect ratio preservation
  - Upscaling when necessary
- Automatic caption generation using BLIP
- Automatic dataset.toml generation for training
- Integration with Kohya scripts

## Prerequisites

### Python
- Python 3.10 or higher
- pip (Python package manager)

### System Dependencies (Linux)
```bash
sudo apt-get install python3-pyqt6 libxcb-cursor0 xcb python3-pyqt6.qtsvg libxcb-xinerama0 libxcb-randr0 libxcb-xtest0 libxcb-shape0 libxcb-xkb1
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/your-username/lora-training-manager.git
cd lora-training-manager
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
# or
venv\Scripts\activate  # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the program:
```bash
python main.py
```

2. Select your dataset folder using the "Select Dataset Folder" button

3. Use the context menu (right click) in the TreeView to access operations:
   - Process Images: Process dataset images
   - Generate Captions: Generate captions using BLIP
   - Generate dataset.toml: Create configuration file for training
   - Analyze Dataset: Show statistics for the current dataset

## Project Structure

```
lora-training-manager/
├── main.py                  # Main graphical interface
├── dataset_manager.py       # Dataset management module
├── image_processor.py       # Image processing
├── caption_generator.py     # Caption generation with BLIP
├── requirements.txt         # Project dependencies
└── README.md                # This file
```

## Contributing

Contributions are welcome! Please feel free to submit pull requests.

1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.

Copyright (C) 2024 

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
