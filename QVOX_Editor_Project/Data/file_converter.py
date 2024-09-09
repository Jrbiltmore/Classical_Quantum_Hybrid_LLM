
# file_converter.py
# Converts between different voxel formats (e.g., QVOX, HDF5, etc.).

import json
import h5py
import numpy as np
from typing import Dict, Tuple

class FileConverter:
    """Class responsible for converting voxel data between different formats."""

    def qvox_to_hdf5(self, qvox_filename: str, hdf5_filename: str):
        """Converts a QVOX file to HDF5 format."""
        with open(qvox_filename, "r") as file:
            qvox_data = json.load(file)

        with h5py.File(hdf5_filename, "w") as hdf5_file:
            voxel_grid = np.array(qvox_data["voxel_grid"])
            hdf5_file.create_dataset("voxel_grid", data=voxel_grid)

            quantum_states_grp = hdf5_file.create_group("quantum_states")
            for key, value in qvox_data["quantum_states"].items():
                quantum_states_grp.create_dataset(key, data=np.array(value))

    def hdf5_to_qvox(self, hdf5_filename: str, qvox_filename: str):
        """Converts an HDF5 file to QVOX format."""
        with h5py.File(hdf5_filename, "r") as hdf5_file:
            voxel_grid = hdf5_file["voxel_grid"][:]

            quantum_states = {}
            quantum_states_grp = hdf5_file["quantum_states"]
            for key in quantum_states_grp:
                quantum_states[key] = quantum_states_grp[key][:].tolist()

        qvox_data = {
            "voxel_grid": voxel_grid.tolist(),
            "quantum_states": quantum_states
        }

        with open(qvox_filename, "w") as file:
            json.dump(qvox_data, file, indent=4)
