import sys
from PyQt6.QtWidgets import QApplication
from PyQt6.QtGui import QIcon
from gui import DatasetManagerGUI

def main():
    app = QApplication(sys.argv)
    
    # Definir o ícone do aplicativo
    app.setWindowIcon(QIcon('icon.svg'))
    
    window = DatasetManagerGUI()
    window.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()