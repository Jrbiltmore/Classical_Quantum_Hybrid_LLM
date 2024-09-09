import numpy as np
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QWidget

class AttributeViewer(QWidget):
    def __init__(self, parent=None, grid_size=(100, 100)):
        super(AttributeViewer, self).__init__(parent)
        self.grid_size = grid_size
        self.attributes = self.initialize_attributes()

    def initialize_attributes(self):
        """
        Initializes multidimensional attributes such as spin, entropy, and entanglement for the hexal grid.
        """
        attributes = {
            "spin": np.random.choice([-0.5, 0.5], size=self.grid_size),
            "entropy": np.random.rand(self.grid_size[0], self.grid_size[1]),
            "entanglement": np.random.rand(self.grid_size[0], self.grid_size[1])
        }
        return attributes

    def render_attributes(self):
        """
        Renders the multidimensional attributes of the hexal grid.
        """
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        
        axs[0].imshow(self.attributes["spin"], cmap='coolwarm')
        axs[0].set_title("Spin")
        
        axs[1].imshow(self.attributes["entropy"], cmap='viridis')
        axs[1].set_title("Entropy")
        
        axs[2].imshow(self.attributes["entanglement"], cmap='plasma')
        axs[2].set_title("Entanglement")
        
        plt.show()

    def update_attributes(self, attribute_updates):
        """
        Updates the multidimensional attributes of hexal cells.
        - attribute_updates: Dictionary with attribute names as keys and updated values as items.
        """
        for attribute, updates in attribute_updates.items():
            for coords, value in updates.items():
                x, y = coords
                self.attributes[attribute][x, y] = value
        self.render_attributes()

    def visualize_spin(self):
        """
        Visualizes the spin attribute of the hexal grid.
        """
        plt.imshow(self.attributes["spin"], cmap='coolwarm')
        plt.colorbar(label="Spin Value")
        plt.title("Spin Visualization")
        plt.show()

    def visualize_entropy(self):
        """
        Visualizes the entropy attribute of the hexal grid.
        """
        plt.imshow(self.attributes["entropy"], cmap='viridis')
        plt.colorbar(label="Entropy Value")
        plt.title("Entropy Visualization")
        plt.show()

    def visualize_entanglement(self):
        """
        Visualizes the entanglement attribute of the hexal grid.
        """
        plt.imshow(self.attributes["entanglement"], cmap='plasma')
        plt.colorbar(label="Entanglement Level")
        plt.title("Entanglement Visualization")
        plt.show()

    def reset_attributes(self):
        """
        Resets the multidimensional attributes of the hexal grid to their initial random states.
        """
        self.attributes = self.initialize_attributes()
        self.render_attributes()
