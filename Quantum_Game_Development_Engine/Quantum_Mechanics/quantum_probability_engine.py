# quantum_probability_engine.py
# Quantum_Game_Development_Engine/Quantum_Mechanics/quantum_probability_engine.py

import numpy as np

class QuantumProbabilityEngine:
    def __init__(self, quantum_state_vector):
        """
        Initializes the QuantumProbabilityEngine with a quantum state vector.
        :param quantum_state_vector: A numpy array representing the quantum state vector
        """
        self.quantum_state_vector = quantum_state_vector

    def compute_probabilities(self):
        """
        Computes the probabilities of measurement outcomes based on the quantum state vector.
        :return: A dictionary with measurement outcomes and their associated probabilities
        """
        # Ensure the state vector is normalized
        norm = np.linalg.norm(self.quantum_state_vector)
        if norm != 1.0:
            raise ValueError("Quantum state vector must be normalized.")

        probabilities = np.abs(self.quantum_state_vector)**2
        return {f"outcome_{i}": prob for i, prob in enumerate(probabilities)}

    def update_quantum_state(self, new_quantum_state_vector):
        """
        Updates the quantum state vector.
        :param new_quantum_state_vector: A numpy array representing the new quantum state vector
        """
        self.quantum_state_vector = new_quantum_state_vector

    def get_quantum_state(self):
        """
        Retrieves the current quantum state vector.
        :return: The current quantum state vector
        """
        return self.quantum_state_vector

# Example usage
if __name__ == "__main__":
    # Initialize a quantum state vector for a 2-qubit system
    state_vector = np.array([1/np.sqrt(2), 1/np.sqrt(2)])  # |0> + |1> / sqrt(2)
    
    # Create a QuantumProbabilityEngine instance
    probability_engine = QuantumProbabilityEngine(quantum_state_vector=state_vector)
    
    # Compute probabilities of measurement outcomes
    probabilities = probability_engine.compute_probabilities()
    print("Measurement probabilities:", probabilities)
    
    # Update quantum state vector
    new_state_vector = np.array([1, 0])  # |0>
    probability_engine.update_quantum_state(new_quantum_state_vector=new_state_vector)
    
    # Retrieve updated quantum state
    updated_state = probability_engine.get_quantum_state()
    print("Updated quantum state:", updated_state)
