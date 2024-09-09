
# main_window.py
# The main window for the QVOX Editor user interface, which includes menus, toolbars, and the voxel editor workspace.

import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QAction, QFileDialog, QMessageBox
from PyQt5.QtGui import QIcon

class MainWindow(QMainWindow):
    """Main window for the QVOX Editor."""

    def __init__(self):
        super().__init__()

        # Initialize window
        self.setWindowTitle('QVOX Editor')
        self.setGeometry(100, 100, 1200, 800)
        self._create_menu()
        self._create_toolbar()
        self._create_statusbar()

    def _create_menu(self):
        """Creates the main menu for the application."""
        menu_bar = self.menuBar()

        # File menu
        file_menu = menu_bar.addMenu('File')

        open_action = QAction(QIcon('open.png'), 'Open', self)
        open_action.setShortcut('Ctrl+O')
        open_action.triggered.connect(self.open_file)
        file_menu.addAction(open_action)

        save_action = QAction(QIcon('save.png'), 'Save', self)
        save_action.setShortcut('Ctrl+S')
        save_action.triggered.connect(self.save_file)
        file_menu.addAction(save_action)

        exit_action = QAction(QIcon('exit.png'), 'Exit', self)
        exit_action.setShortcut('Ctrl+Q')
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

    def _create_toolbar(self):
        """Creates a toolbar for quick access to actions."""
        toolbar = self.addToolBar('Main Toolbar')

        open_action = QAction(QIcon('open.png'), 'Open', self)
        open_action.triggered.connect(self.open_file)
        toolbar.addAction(open_action)

        save_action = QAction(QIcon('save.png'), 'Save', self)
        save_action.triggered.connect(self.save_file)
        toolbar.addAction(save_action)

        exit_action = QAction(QIcon('exit.png'), 'Exit', self)
        exit_action.triggered.connect(self.close)
        toolbar.addAction(exit_action)

    def _create_statusbar(self):
        """Creates the status bar at the bottom of the window."""
        self.statusBar().showMessage('Ready')

    def open_file(self):
        """Opens a file dialog to select and open a QVOX file."""
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, 'Open QVOX File', '', 'QVOX Files (*.qvox);;All Files (*)', options=options)
        if file_name:
            self.statusBar().showMessage(f'Opened file: {file_name}')

    def save_file(self):
        """Opens a file dialog to save the current voxel data to a QVOX file."""
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getSaveFileName(self, 'Save QVOX File', '', 'QVOX Files (*.qvox);;All Files (*)', options=options)
        if file_name:
            self.statusBar().showMessage(f'Saved file: {file_name}')

    def closeEvent(self, event):
        """Confirms if the user really wants to close the application."""
        reply = QMessageBox.question(self, 'Message', 'Are you sure you want to quit?', QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
