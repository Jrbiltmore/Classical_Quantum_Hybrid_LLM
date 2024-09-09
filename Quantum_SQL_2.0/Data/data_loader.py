
# data_loader.py
# This file handles the loading and retrieval of data from voxel storage and quantum state files.

import json
import numpy as np

class DataLoader:
    def load_voxel_data(self, filepath):
        """Loads voxel data from a specified file path."""
        try:
            with open(filepath, 'r') as file:
                data = json.load(file)
            return np.array(data)
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {filepath}")
        except Exception as e:
            raise Exception(f"Error loading voxel data: {e}")

    def load_quantum_state(self, filepath):
        """Loads quantum state data from a specified file path."""
        try:
            with open(filepath, 'r') as file:
                data = json.load(file)
            return np.array(data)
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {filepath}")
        except Exception as e:
            raise Exception(f"Error loading quantum state data: {e}")

# Example usage:
# loader = DataLoader()
# voxel_data = loader.load_voxel_data("path_to_voxel_file.qvox")
# quantum_state_data = loader.load_quantum_state("path_to_state_file.qdat")
