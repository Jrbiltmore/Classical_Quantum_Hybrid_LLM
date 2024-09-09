import numpy as np
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QWidget

class ObserverViewer(QWidget):
    def __init__(self, parent=None, grid_size=(100, 100)):
        super(ObserverViewer, self).__init__(parent)
        self.grid_size = grid_size
        self.observer_position = np.array([50, 50])  # Default observer position
        self.observer_angle = 0  # Default observer angle in degrees

    def update_observer_position(self, new_position):
        """
        Updates the observer's position on the hexal grid.
        - new_position: Tuple representing the new position (x, y) of the observer.
        """
        self.observer_position = np.array(new_position)
        self.render_observer_view()

    def update_observer_angle(self, new_angle):
        """
        Updates the observer's viewing angle.
        - new_angle: The new angle of the observer in degrees.
        """
        self.observer_angle = new_angle % 360  # Ensure angle stays within 0-360 degrees
        self.render_observer_view()

    def show_observer_effects(self, grid):
        """
        Shows the observer's effect on the grid, visualizing perspective shifts or changes in the grid based on the observer's position and angle.
        - grid: The hexal grid to which observer effects will be applied.
        """
        shifted_grid = np.roll(grid, self.observer_position.astype(int), axis=(0, 1))
        rotated_grid = np.rot90(shifted_grid, k=int(self.observer_angle // 90))

        plt.imshow(rotated_grid, cmap='plasma')
        plt.title(f'Observer View (Position: {self.observer_position}, Angle: {self.observer_angle}°)')
        plt.show()

    def render_observer_view(self):
        """
        Renders a visual representation of the observer's current perspective on the grid.
        """
        x, y = self.observer_position
        plt.scatter([x], [y], color='blue', edgecolors='black', s=100, label=f'Observer Position: {self.observer_position}')
        plt.legend()
        plt.show()

    def reset_observer_view(self):
        """
        Resets the observer's position and angle to the default values.
        """
        self.observer_position = np.array([50, 50])
        self.observer_angle = 0
        self.render_observer_view()

    def apply_relativity_effects(self, grid, speed):
        """
        Applies relativistic effects to the observer's view, such as time dilation or length contraction, based on the observer's speed.
        - grid: The hexal grid affected by relativity.
        - speed: The velocity of the observer relative to the speed of light.
        """
        time_dilation_factor = np.sqrt(1 - speed ** 2)
        contracted_grid = np.clip(grid * time_dilation_factor, 0, np.max(grid))

        plt.imshow(contracted_grid, cmap='inferno')
        plt.title(f'Relativistic View (Speed: {speed}c)')
        plt.show()

    def update_based_on_environment(self, environmental_factors):
        """
        Adjusts the observer's perspective based on external environmental factors such as gravity or temperature.
        - environmental_factors: A dictionary containing the environmental conditions affecting the observer.
        """
        gravity = environmental_factors.get('gravity', 9.8)
        temperature = environmental_factors.get('temperature', 300)
        electromagnetic_field = environmental_factors.get('electromagnetic_field', np.array([0, 0, 0]))

        # Update observer's position and angle based on environmental conditions
        self.observer_position += electromagnetic_field[:2]
        self.observer_angle += temperature / 100  # Simulating how temperature might affect angle over time
        self.render_observer_view()
