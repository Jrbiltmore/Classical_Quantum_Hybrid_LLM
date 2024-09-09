import numpy as np
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QWidget

class HexalViewer(QWidget):
    def __init__(self, parent=None, grid_size=(100, 100)):
        super(HexalViewer, self).__init__(parent)
        self.grid_size = grid_size
        self.hexal_grid = self.initialize_grid()
        self.zoom_level = 1.0

    def initialize_grid(self):
        """
        Initializes the hexal grid with default values.
        """
        return np.zeros(self.grid_size, dtype=np.float32)

    def render_grid(self):
        """
        Renders the hexal grid using matplotlib, applying the current zoom level.
        """
        plt.figure(figsize=(6, 6))
        plt.imshow(self.hexal_grid, cmap='viridis', extent=(0, self.grid_size[0], 0, self.grid_size[1]))
        plt.colorbar(label="Hexal Value")
        plt.title(f"Hexal Grid Visualization (Zoom: {self.zoom_level:.1f}x)")
        plt.show()

    def zoom_in(self):
        """
        Increases the zoom level of the grid.
        """
        self.zoom_level = min(self.zoom_level * 1.1, 10.0)  # Max zoom is 10x
        self.render_grid()

    def zoom_out(self):
        """
        Decreases the zoom level of the grid.
        """
        self.zoom_level = max(self.zoom_level / 1.1, 0.1)  # Min zoom is 0.1x
        self.render_grid()

    def update_grid(self, updates):
        """
        Updates the hexal grid with new values.
        - updates: Dictionary where keys are (x, y) coordinates and values are new hexal values.
        """
        for coords, value in updates.items():
            x, y = coords
            self.hexal_grid[x, y] = value
        self.render_grid()

    def highlight_hexal(self, coords, color='red'):
        """
        Highlights a specific hexal in the grid.
        - coords: Tuple (x, y) of the hexal to highlight.
        - color: The color used to highlight the hexal.
        """
        x, y = coords
        plt.imshow(self.hexal_grid, cmap='viridis')
        plt.scatter([x], [y], color=color, edgecolors='black', s=100, label=f'Highlighted Hexal: {coords}')
        plt.legend()
        plt.show()

    def overlay_hexals(self, overlay_grid):
        """
        Overlays another hexal grid on top of the current one, useful for comparing datasets.
        - overlay_grid: 2D array of hexal values to overlay.
        """
        combined_grid = np.maximum(self.hexal_grid, overlay_grid)
        plt.imshow(combined_grid, cmap='plasma')
        plt.colorbar(label="Overlayed Hexal Value")
        plt.title("Hexal Grid with Overlay")
        plt.show()

    def reset_grid(self):
        """
        Resets the hexal grid to its default state (all zeros).
        """
        self.hexal_grid = self.initialize_grid()
        self.render_grid()
