
# quantum_state_loader.py
# Loads quantum state data from QVOX files for editing.

import json
import numpy as np
from typing import Dict, Tuple

class QuantumStateLoader:
    """Class responsible for loading quantum state data from QVOX files."""

    def load_states(self, filename: str) -> Dict[Tuple[int, int, int], np.ndarray]:
        """Loads quantum states from a QVOX file and returns a dictionary of voxel states."""
        with open(filename, 'r') as file:
            data = json.load(file)

        quantum_states = {
            tuple(map(int, key.split(','))): np.array(value)
            for key, value in data["quantum_states"].items()
        }

        return quantum_states
