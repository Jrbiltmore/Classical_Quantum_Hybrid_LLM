import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QAction, QToolBar, QVBoxLayout, QWidget, QLabel
from PyQt5.QtCore import Qt
from hexal_viewer import HexalViewer
from quantum_viewer import QuantumViewer
from attribute_viewer import AttributeViewer

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Hexal Illustrator")
        self.setGeometry(100, 100, 1200, 800)

        # Initialize main components
        self.hexal_viewer = HexalViewer(self)
        self.quantum_viewer = QuantumViewer(self)
        self.attribute_viewer = AttributeViewer(self)

        # Create the central widget
        central_widget = QWidget(self)
        layout = QVBoxLayout(central_widget)
        layout.addWidget(self.hexal_viewer)
        layout.addWidget(self.quantum_viewer)
        layout.addWidget(self.attribute_viewer)
        self.setCentralWidget(central_widget)

        # Create the menu bar
        self.menu_bar = self.menuBar()
        self.create_menus()

        # Create toolbars
        self.toolbar = QToolBar("Main Toolbar", self)
        self.addToolBar(self.toolbar)
        self.create_toolbar()

    def create_menus(self):
        """
        Creates the menu bar with file, view, and settings options.
        """
        # File Menu
        file_menu = self.menu_bar.addMenu("&File")

        open_action = QAction("&Open", self)
        open_action.triggered.connect(self.open_file)
        file_menu.addAction(open_action)

        save_action = QAction("&Save", self)
        save_action.triggered.connect(self.save_file)
        file_menu.addAction(save_action)

        exit_action = QAction("E&xit", self)
        exit_action.triggered.connect(self.close_application)
        file_menu.addAction(exit_action)

        # View Menu
        view_menu = self.menu_bar.addMenu("&View")

        zoom_in_action = QAction("Zoom &In", self)
        zoom_in_action.triggered.connect(self.zoom_in)
        view_menu.addAction(zoom_in_action)

        zoom_out_action = QAction("Zoom &Out", self)
        zoom_out_action.triggered.connect(self.zoom_out)
        view_menu.addAction(zoom_out_action)

    def create_toolbar(self):
        """
        Creates a toolbar with common actions.
        """
        open_action = QAction("Open", self)
        open_action.triggered.connect(self.open_file)
        self.toolbar.addAction(open_action)

        save_action = QAction("Save", self)
        save_action.triggered.connect(self.save_file)
        self.toolbar.addAction(save_action)

        self.toolbar.addSeparator()

        zoom_in_action = QAction("Zoom In", self)
        zoom_in_action.triggered.connect(self.zoom_in)
        self.toolbar.addAction(zoom_in_action)

        zoom_out_action = QAction("Zoom Out", self)
        zoom_out_action.triggered.connect(self.zoom_out)
        self.toolbar.addAction(zoom_out_action)

    def open_file(self):
        """
        Opens a file dialog for loading hexal grid data.
        """
        # Placeholder for file loading logic
        print("Opening file...")

    def save_file(self):
        """
        Opens a file dialog for saving hexal grid data.
        """
        # Placeholder for file saving logic
        print("Saving file...")

    def zoom_in(self):
        """
        Increases the zoom level in the hexal viewer.
        """
        self.hexal_viewer.zoom_in()

    def zoom_out(self):
        """
        Decreases the zoom level in the hexal viewer.
        """
        self.hexal_viewer.zoom_out()

    def close_application(self):
        """
        Closes the application.
        """
        sys.exit()

    def render_viewport(self):
        """
        Renders the central viewport where the hexal grid, quantum states, and attributes are visualized.
        """
        self.hexal_viewer.render_grid()
        self.quantum_viewer.render_quantum_states()
        self.attribute_viewer.render_attributes()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())
