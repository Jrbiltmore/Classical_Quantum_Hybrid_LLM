
# exporter.py
# Exports QVOX files to other formats (e.g., for visualization in external applications).

import json
import numpy as np
from typing import Dict, Tuple

class Exporter:
    """Class responsible for exporting voxel data and quantum states to other formats."""

    def export_to_json(self, filename: str, voxel_grid: np.ndarray, quantum_states: Dict[Tuple[int, int, int], np.ndarray]):
        """Exports voxel data and quantum states to a JSON file."""
        data = {
            "voxel_grid": voxel_grid.tolist(),
            "quantum_states": {
                ','.join(map(str, key)): value.tolist()
                for key, value in quantum_states.items()
            }
        }
        with open(filename, 'w') as file:
            json.dump(data, file, indent=4)
        print(f"Data exported to {filename}")

    def export_to_csv(self, filename: str, voxel_grid: np.ndarray):
        """Exports voxel grid data to a CSV file."""
        np.savetxt(filename, voxel_grid.reshape(-1, voxel_grid.shape[-1]), delimiter=",", fmt="%d")
        print(f"Voxel grid exported to {filename}")
