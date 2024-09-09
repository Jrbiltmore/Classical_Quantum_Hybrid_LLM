
# quantum_state_saver.py
# Saves modified quantum state data back into the QVOX file format.

import json
import numpy as np
from typing import Dict, Tuple

class QuantumStateSaver:
    """Class responsible for saving quantum state data to QVOX files."""

    def save_states(self, filename: str, quantum_states: Dict[Tuple[int, int, int], np.ndarray]):
        """Saves quantum states to a QVOX file."""
        data = {
            "quantum_states": {
                ','.join(map(str, key)): value.tolist()
                for key, value in quantum_states.items()
            }
        }

        with open(filename, 'w') as file:
            json.dump(data, file, indent=4)
