
# Transition Saver for Voxel-Hexel Visualization Engine
# Supports saving transition data between voxel and hexal grids for analysis and replay

import json
import h5py
import numpy as np

class TransitionSaver:
    """
    Handles saving of transition data between voxel and hexal grids.
    Supports JSON, HDF5, and custom transition formats.
    """
    
    @staticmethod
    def save_transition_to_json(voxel_grid, hexal_grid, file_path):
        """
        Save transition data between voxel and hexal grids to a JSON file.
        :param voxel_grid: The voxel grid state.
        :param hexal_grid: The hexal grid state.
        :param file_path: The path to the JSON file.
        """
        transition_data = {
            "voxel_grid": voxel_grid.tolist(),
            "hexal_grid": hexal_grid.tolist()
        }
        with open(file_path, 'w') as json_file:
            json.dump(transition_data, json_file, indent=4)
        print(f"Transition data saved to JSON at {file_path}")

    @staticmethod
    def save_transition_to_hdf5(voxel_grid, hexal_grid, file_path, compression="gzip"):
        """
        Save transition data between voxel and hexal grids to an HDF5 file with optional compression.
        :param voxel_grid: The voxel grid state.
        :param hexal_grid: The hexal grid state.
        :param file_path: The path to the HDF5 file.
        :param compression: Compression type to use (e.g., "gzip", None).
        """
        with h5py.File(file_path, 'w') as hdf5_file:
            hdf5_file.create_dataset('voxel_grid', data=voxel_grid, compression=compression)
            hdf5_file.create_dataset('hexal_grid', data=hexal_grid, compression=compression)
        print(f"Transition data saved to HDF5 at {file_path} with compression {compression}")

    @staticmethod
    def save_transition_to_custom(voxel_grid, hexal_grid, file_path):
        """
        Save transition data to a custom format file (CSV or plain text).
        :param voxel_grid: The voxel grid state.
        :param hexal_grid: The hexal grid state.
        :param file_path: The path to the custom file.
        """
        np.savetxt(file_path + "_voxel.csv", voxel_grid, delimiter=',')
        np.savetxt(file_path + "_hexal.csv", hexal_grid, delimiter=',')
        print(f"Transition data saved in custom format at {file_path}")

    @staticmethod
    def save(voxel_grid, hexal_grid, file_path, file_type="json", compression=None):
        """
        Generic save function that saves transition data based on the file type.
        :param voxel_grid: The voxel grid state.
        :param hexal_grid: The hexal grid state.
        :param file_path: Path to the file.
        :param file_type: Type of the file (json, hdf5, custom).
        :param compression: Compression type (only used for HDF5).
        """
        if file_type == "json":
            TransitionSaver.save_transition_to_json(voxel_grid, hexal_grid, file_path)
        elif file_type == "hdf5":
            TransitionSaver.save_transition_to_hdf5(voxel_grid, hexal_grid, file_path, compression=compression)
        elif file_type == "custom":
            TransitionSaver.save_transition_to_custom(voxel_grid, hexal_grid, file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
