# wavefunction_renderer.py
# Quantum_Game_Development_Engine/Quantum_Visuals/wavefunction_renderer.py

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

class WavefunctionRenderer:
    def __init__(self, resolution=(512, 512), wavefunction=None):
        """
        Initializes the WavefunctionRenderer with the specified resolution and wavefunction.
        :param resolution: Tuple specifying the resolution of the visualizer (width, height)
        :param wavefunction: Function representing the quantum wavefunction
        """
        self.resolution = resolution
        self.wavefunction = wavefunction if wavefunction is not None else self.default_wavefunction

    def default_wavefunction(self, x, y):
        """
        Default wavefunction for demonstration purposes.
        :param x: X-coordinate
        :param y: Y-coordinate
        :return: Value of the wavefunction at (x, y)
        """
        return np.sin(x) * np.cos(y)

    def generate_wavefunction(self):
        """
        Generates a visual representation of the quantum wavefunction.
        """
        x = np.linspace(-np.pi, np.pi, self.resolution[0])
        y = np.linspace(-np.pi, np.pi, self.resolution[1])
        X, Y = np.meshgrid(x, y)
        Z = self.wavefunction(X, Y)
        
        # Normalize Z to [0, 1]
        Z_normalized = (Z - Z.min()) / (Z.max() - Z.min())

        # Set canvas colors
        self.canvas = np.zeros((self.resolution[0], self.resolution[1], 3))
        self.canvas[..., 0] = Z_normalized  # Red channel
        self.canvas[..., 1] = 1.0 - Z_normalized  # Green channel
        self.canvas[..., 2] = (Z_normalized + 1.0) / 2.0  # Blue channel

    def display_wavefunction(self):
        """
        Displays the generated wavefunction visualization using matplotlib.
        """
        plt.imshow(self.canvas)
        plt.title('Wavefunction Visualization')
        plt.axis('off')
        plt.colorbar(label='Wavefunction Amplitude')
        plt.show()

# Example usage
if __name__ == "__main__":
    # Create and display a wavefunction visualization using the default wavefunction
    renderer = WavefunctionRenderer()
    renderer.generate_wavefunction()
    renderer.display_wavefunction()

    # Create and display a wavefunction visualization using a custom wavefunction
    def custom_wavefunction(x, y):
        return np.sin(x) * np.exp(-y**2)

    renderer = WavefunctionRenderer(wavefunction=custom_wavefunction)
    renderer.generate_wavefunction()
    renderer.display_wavefunction()
