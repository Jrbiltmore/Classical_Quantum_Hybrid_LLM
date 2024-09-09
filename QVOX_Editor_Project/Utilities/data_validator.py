
# data_validator.py
# Validates QVOX files to ensure that the voxel data, quantum states, and attributes are correctly formatted.

import json
import numpy as np

class DataValidator:
    """Class responsible for validating QVOX files and ensuring data integrity."""

    def validate_qvox_file(self, filename: str) -> bool:
        """Validates the contents of a QVOX file."""
        try:
            with open(filename, 'r') as file:
                data = json.load(file)

            voxel_grid = np.array(data.get("voxel_grid", []))
            quantum_states = data.get("quantum_states", {})

            if voxel_grid.size == 0:
                raise ValueError("Voxel grid is empty or invalid.")
            if not all(isinstance(state, list) for state in quantum_states.values()):
                raise ValueError("Invalid format for quantum states.")
            
            print(f"Validation passed for {filename}")
            return True
        except Exception as e:
            print(f"Validation failed for {filename}: {e}")
            return False
