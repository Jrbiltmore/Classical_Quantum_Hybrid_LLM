
# file_saver.py
# Handles saving edited voxel data, quantum states, and multidimensional attributes back into QVOX format.

import json
import numpy as np
from typing import Dict, Tuple

class FileSaver:
    """Class responsible for saving voxel grid and quantum states to a QVOX file."""

    def save_qvox_file(self, filename: str, voxel_grid: np.ndarray, quantum_states: Dict[Tuple[int, int, int], np.ndarray]):
        """Saves the current voxel grid and quantum states to a QVOX file."""
        data = {
            "voxel_grid": voxel_grid.tolist(),
            "quantum_states": {
                ','.join(map(str, key)): value.tolist()
                for key, value in quantum_states.items()
            }
        }

        with open(filename, "w") as file:
            json.dump(data, file, indent=4)

    def save_voxel_grid(self, filename: str, voxel_grid: np.ndarray):
        """Saves just the voxel grid to a QVOX file."""
        data = {"voxel_grid": voxel_grid.tolist()}

        with open(filename, "w") as file:
            json.dump(data, file, indent=4)

    def save_quantum_states(self, filename: str, quantum_states: Dict[Tuple[int, int, int], np.ndarray]):
        """Saves just the quantum states to a QVOX file."""
        data = {
            "quantum_states": {
                ','.join(map(str, key)): value.tolist()
                for key, value in quantum_states.items()
            }
        }

        with open(filename, "w") as file:
            json.dump(data, file, indent=4)
