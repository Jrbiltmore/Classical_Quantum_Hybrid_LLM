# entanglement_visualizer.py
# Quantum_Game_Development_Engine/Quantum_Visuals/entanglement_visualizer.py

import matplotlib.pyplot as plt
import numpy as np

class EntanglementVisualizer:
    def __init__(self, state_matrix):
        """
        Initializes the EntanglementVisualizer with a quantum state matrix.
        :param state_matrix: A 2D numpy array representing the quantum state matrix
        """
        self.state_matrix = state_matrix
        self.validate_state_matrix()

    def validate_state_matrix(self):
        """
        Validates that the state matrix is a proper quantum state matrix.
        """
        if not self.is_square_matrix(self.state_matrix):
            raise ValueError("State matrix must be square.")
        if not self.is_unitary(self.state_matrix):
            raise ValueError("State matrix must be unitary.")

    def is_square_matrix(self, matrix):
        """
        Checks if the matrix is square.
        :param matrix: A numpy array representing the matrix
        :return: Boolean indicating if the matrix is square
        """
        return matrix.shape[0] == matrix.shape[1]

    def is_unitary(self, matrix):
        """
        Checks if the matrix is unitary.
        :param matrix: A numpy array representing the matrix
        :return: Boolean indicating if the matrix is unitary
        """
        identity = np.eye(matrix.shape[0])
        return np.allclose(np.dot(matrix.conj().T, matrix), identity)

    def visualize_entanglement(self):
        """
        Visualizes the quantum entanglement by plotting the state matrix.
        """
        plt.imshow(np.abs(self.state_matrix)**2, cmap='viridis', interpolation='nearest')
        plt.colorbar(label='Probability Amplitude Magnitude Squared')
        plt.title('Quantum Entanglement Visualization')
        plt.xlabel('Basis States')
        plt.ylabel('Basis States')
        plt.show()

# Example usage
if __name__ == "__main__":
    # Example state matrix representing an entangled quantum state
    state_matrix = np.array([[0.5, 0.5], [0.5, -0.5]])

    # Create an EntanglementVisualizer instance
    visualizer = EntanglementVisualizer(state_matrix=state_matrix)

    # Visualize the entanglement
    visualizer.visualize_entanglement()
