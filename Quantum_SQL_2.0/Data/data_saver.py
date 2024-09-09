
# data_saver.py
# This file saves new quantum state or voxel data into the appropriate formats (QVOX, QDAT, etc.).

import json
import numpy as np

class DataSaver:
    def save_voxel_data(self, filepath, voxel_data):
        """Saves voxel data to a specified file path."""
        try:
            with open(filepath, 'w') as file:
                json.dump(voxel_data.tolist(), file)
        except Exception as e:
            raise Exception(f"Error saving voxel data: {e}")

    def save_quantum_state(self, filepath, state_data):
        """Saves quantum state data to a specified file path."""
        try:
            with open(filepath, 'w') as file:
                json.dump(state_data.tolist(), file)
        except Exception as e:
            raise Exception(f"Error saving quantum state data: {e}")

# Example usage:
# saver = DataSaver()
# saver.save_voxel_data("path_to_voxel_file.qvox", voxel_data)
# saver.save_quantum_state("path_to_state_file.qdat", quantum_state_data)
