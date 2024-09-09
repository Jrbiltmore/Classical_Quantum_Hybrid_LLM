import numpy as np
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QWidget

class QuantumViewer(QWidget):
    def __init__(self, parent=None, grid_size=(100, 100)):
        super(QuantumViewer, self).__init__(parent)
        self.grid_size = grid_size
        self.quantum_grid = self.initialize_quantum_grid()

    def initialize_quantum_grid(self):
        """
        Initializes the quantum grid with random complex quantum states.
        """
        return np.random.rand(self.grid_size[0], self.grid_size[1]) + 1j * np.random.rand(self.grid_size[0], self.grid_size[1])

    def render_quantum_states(self):
        """
        Renders the quantum grid, visualizing both the real and imaginary components.
        """
        real_part = np.real(self.quantum_grid)
        imag_part = np.imag(self.quantum_grid)
        
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        axs[0].imshow(real_part, cmap='coolwarm')
        axs[0].set_title("Real Part of Quantum State")
        axs[1].imshow(imag_part, cmap='coolwarm')
        axs[1].set_title("Imaginary Part of Quantum State")
        plt.show()

    def update_quantum_states(self, updates):
        """
        Updates the quantum grid with new complex states.
        - updates: Dictionary where keys are (x, y) coordinates and values are new quantum states (complex numbers).
        """
        for coords, state in updates.items():
            x, y = coords
            self.quantum_grid[x, y] = state
        self.render_quantum_states()

    def visualize_entanglement(self, coords1, coords2):
        """
        Visualizes the quantum entanglement between two hexals.
        - coords1, coords2: Coordinates of the two hexals to visualize entanglement.
        """
        state1 = self.quantum_grid[coords1]
        state2 = self.quantum_grid[coords2]
        entanglement = np.abs(np.dot(np.conj(state1), state2))  # Dot product of conjugate of state1 and state2
        
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.text(0.5, 0.5, f'Entanglement: {entanglement:.4f}', fontsize=15, ha='center')
        ax.set_title(f'Entanglement between {coords1} and {coords2}')
        plt.show()

    def collapse_quantum_state(self, coords):
        """
        Collapses the quantum state at the specified coordinates to a real number based on the probability amplitude.
        """
        real_state = np.abs(self.quantum_grid[coords]) ** 2
        self.quantum_grid[coords] = real_state + 0j  # Collapse to a definite real state
        self.render_quantum_states()

    def simulate_wavefunction_evolution(self, steps=100):
        """
        Simulates the time evolution of the quantum wavefunction over the grid.
        Uses a simplified Schrödinger equation approximation.
        """
        for _ in range(steps):
            self.quantum_grid += 0.01j * np.gradient(self.quantum_grid)  # Simplified Schrödinger-like evolution
        self.render_quantum_states()

    def reset_quantum_grid(self):
        """
        Resets the quantum grid to its initial random state.
        """
        self.quantum_grid = self.initialize_quantum_grid()
        self.render_quantum_states()
