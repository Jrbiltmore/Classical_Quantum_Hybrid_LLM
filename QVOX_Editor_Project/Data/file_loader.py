
# file_loader.py
# Responsible for loading QVOX files into the editor and reading their contents.

import json
import numpy as np
from typing import Dict, Tuple

class FileLoader:
    """Class responsible for loading QVOX files and reading their voxel and quantum state data."""

    def load_qvox_file(self, filename: str) -> Dict[str, np.ndarray]:
        """Loads a QVOX file and returns the voxel grid and quantum states as a dictionary."""
        with open(filename, "r") as file:
            data = json.load(file)

        voxel_grid = np.array(data["voxel_grid"])
        quantum_states = {
            tuple(map(int, key.split(','))): np.array(value)
            for key, value in data["quantum_states"].items()
        }

        return {"voxel_grid": voxel_grid, "quantum_states": quantum_states}

    def load_voxel_grid(self, data: Dict[str, np.ndarray]) -> np.ndarray:
        """Returns the voxel grid from the loaded QVOX data."""
        return data["voxel_grid"]

    def load_quantum_states(self, data: Dict[str, np.ndarray]) -> Dict[Tuple[int, int, int], np.ndarray]:
        """Returns the quantum states from the loaded QVOX data."""
        return data["quantum_states"]
