import sys
from PyQt6.QtWidgets import QApplication
from gui import DatasetManagerGUI

def main():
    app = QApplication(sys.argv)
    window = DatasetManagerGUI()
    window.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()
