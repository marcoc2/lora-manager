import sys
from PyQt6.QtWidgets import QApplication

from views.main_window import DatasetManagerGUI
from controllers.main_controller import MainController

def main():
    app = QApplication(sys.argv)
    view = DatasetManagerGUI()
    controller = MainController(view)
    view.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()