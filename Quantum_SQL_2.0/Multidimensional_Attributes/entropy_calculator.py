
# entropy_calculator.py
# This file calculates the entropy (disorder) and other thermodynamic properties for quantum systems.

import numpy as np

class EntropyCalculator:
    def calculate_entropy(self, probabilities):
        """Calculates the Shannon entropy based on given probabilities."""
        probabilities = np.array(probabilities)
        entropy = -np.sum(probabilities * np.log2(probabilities + 1e-9))  # Adding epsilon to avoid log(0)
        return entropy

# Example usage:
# calculator = EntropyCalculator()
# entropy = calculator.calculate_entropy([0.5, 0.5])
