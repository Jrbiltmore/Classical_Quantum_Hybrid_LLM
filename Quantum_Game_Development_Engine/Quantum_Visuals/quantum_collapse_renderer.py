# quantum_collapse_renderer.py
# Quantum_Game_Development_Engine/Quantum_Visuals/quantum_collapse_renderer.py

import matplotlib.pyplot as plt
import numpy as np

class QuantumCollapseRenderer:
    def __init__(self, initial_state, collapsed_state):
        """
        Initializes the QuantumCollapseRenderer with the initial and collapsed states.
        :param initial_state: A 1D numpy array representing the initial quantum state vector
        :param collapsed_state: A 1D numpy array representing the collapsed quantum state vector
        """
        self.initial_state = initial_state
        self.collapsed_state = collapsed_state
        self.validate_states()

    def validate_states(self):
        """
        Validates that the initial and collapsed states are valid quantum states.
        """
        if not self.is_valid_state(self.initial_state):
            raise ValueError("Initial state is not a valid quantum state.")
        if not self.is_valid_state(self.collapsed_state):
            raise ValueError("Collapsed state is not a valid quantum state.")

    def is_valid_state(self, state):
        """
        Checks if the state vector is a valid quantum state (i.e., normalized).
        :param state: A numpy array representing the state vector
        :return: Boolean indicating if the state is valid
        """
        norm = np.linalg.norm(state)
        return np.allclose(norm, 1)

    def render_collapse(self):
        """
        Renders the quantum collapse by plotting the initial and collapsed state vectors.
        """
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))

        # Plot the initial state
        ax[0].bar(range(len(self.initial_state)), np.abs(self.initial_state)**2, color='blue', alpha=0.7)
        ax[0].set_title('Initial Quantum State')
        ax[0].set_xlabel('Basis States')
        ax[0].set_ylabel('Probability Amplitude Magnitude Squared')

        # Plot the collapsed state
        ax[1].bar(range(len(self.collapsed_state)), np.abs(self.collapsed_state)**2, color='red', alpha=0.7)
        ax[1].set_title('Collapsed Quantum State')
        ax[1].set_xlabel('Basis States')
        ax[1].set_ylabel('Probability Amplitude Magnitude Squared')

        plt.suptitle('Quantum Wavefunction Collapse')
        plt.tight_layout()
        plt.show()

# Example usage
if __name__ == "__main__":
    # Example state vectors for demonstration purposes
    initial_state = np.array([0.6, 0.8])
    collapsed_state = np.array([0.9, 0.1])

    # Create a QuantumCollapseRenderer instance
    renderer = QuantumCollapseRenderer(initial_state=initial_state, collapsed_state=collapsed_state)

    # Render the quantum collapse
    renderer.render_collapse()
