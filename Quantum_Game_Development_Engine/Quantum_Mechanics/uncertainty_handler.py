# uncertainty_handler.py
# Quantum_Game_Development_Engine/Quantum_Mechanics/uncertainty_handler.py

import numpy as np

class UncertaintyHandler:
    def __init__(self, state_vector):
        """
        Initializes the UncertaintyHandler with a quantum state vector.
        :param state_vector: A numpy array representing the quantum state vector
        """
        self.state_vector = state_vector
        self.validate_state_vector()

    def validate_state_vector(self):
        """
        Validates that the quantum state vector is a proper quantum state.
        """
        norm = np.linalg.norm(self.state_vector)
        if not np.isclose(norm, 1.0):
            raise ValueError("Quantum state vector must be normalized.")

    def calculate_variance(self, observable_matrix):
        """
        Calculates the variance of an observable with respect to the current quantum state.
        :param observable_matrix: A numpy array representing the observable's matrix
        :return: Variance of the observable
        """
        if observable_matrix.shape[0] != self.state_vector.shape[0]:
            raise ValueError("Observable matrix must have the same dimension as the state vector.")

        # Calculate expectation value
        expectation_value = np.real(np.vdot(self.state_vector, np.dot(observable_matrix, self.state_vector)))

        # Calculate expectation value of the observable squared
        observable_squared = np.dot(observable_matrix, observable_matrix)
        expectation_value_squared = np.real(np.vdot(self.state_vector, np.dot(observable_squared, self.state_vector)))

        # Variance
        variance = expectation_value_squared - expectation_value**2
        return variance

    def calculate_uncertainty(self, observable_matrix):
        """
        Calculates the uncertainty of an observable with respect to the current quantum state.
        :param observable_matrix: A numpy array representing the observable's matrix
        :return: Uncertainty (standard deviation) of the observable
        """
        variance = self.calculate_variance(observable_matrix)
        uncertainty = np.sqrt(variance)
        return uncertainty

# Example usage
if __name__ == "__main__":
    # Initialize a state vector for a 2-dimensional system
    state_vector = np.array([1/np.sqrt(2), 1/np.sqrt(2)])  # |0> + |1> / sqrt(2)

    # Define an observable matrix (Pauli-X matrix for example)
    observable_matrix = np.array([[0, 1], [1, 0]])  # Pauli-X matrix

    # Create an UncertaintyHandler instance
    handler = UncertaintyHandler(state_vector=state_vector)

    # Calculate variance and uncertainty
    variance = handler.calculate_variance(observable_matrix)
    uncertainty = handler.calculate_uncertainty(observable_matrix)

    print(f"Variance of the observable: {variance}")
    print(f"Uncertainty (standard deviation) of the observable: {uncertainty}")
