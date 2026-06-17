"""
main.py
───────
Entry point for the Road Network Visualizer PySide6 application.

Usage
─────
python main.py
"""

import sys

from PySide6.QtWidgets import QApplication

from assets.main_window import MainWindow


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("DSF - GUI")
    app.setOrganizationName("Grufoony")
    app.setStyle("Fusion")

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
