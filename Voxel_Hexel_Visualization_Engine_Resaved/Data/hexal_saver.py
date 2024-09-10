# Hexal Saver for Voxel-Hexel Visualization Engine
# Supports saving to multiple formats: JSON, HDF5, QHEX

import json
import h5py
import numpy as np

class HexalSaver:
    """
    Handles saving of hexal grid data to different formats.
    Supports JSON, HDF5, and QHEX formats with optional compression.
    """
    
    @staticmethod
    def save_to_json(grid, file_path):
        """
        Save hexal grid data to a JSON file.
        :param grid: The hexal grid data to save.
        :param file_path: The path to the JSON file.
        """
        data = {"hexal_grid": grid.tolist()}
        with open(file_path, 'w') as json_file:
            json.dump(data, json_file, indent=4)
        print(f"Hexal grid saved to JSON at {file_path}")
    
    @staticmethod
    def save_to_hdf5(grid, file_path, compression="gzip"):
        """
        Save hexal grid data to an HDF5 file with optional compression.
        :param grid: The hexal grid data to save.
        :param file_path: The path to the HDF5 file.
        :param compression: Compression type to use (e.g., "gzip", None).
        """
        with h5py.File(file_path, 'w') as hdf5_file:
            hdf5_file.create_dataset('hexal_grid', data=grid, compression=compression)
        print(f"Hexal grid saved to HDF5 at {file_path} with compression {compression}")
    
    @staticmethod
    def save_to_qhex(grid, file_path):
        """
        Save hexal grid data to a QHEX (Quantum Hexal) file.
        :param grid: The hexal grid data to save.
        :param file_path: The path to the QHEX file.
        """
        np.savetxt(file_path, grid, delimiter=',')
        print(f"Hexal grid saved to QHEX at {file_path}")
    
    @staticmethod
    def save(grid, file_path, file_type="json", compression=None):
        """
        Generic save function that saves a hexal grid based on the file type.
        :param grid: The hexal grid data to save.
        :param file_path: Path to the file.
        :param file_type: Type of the file (json, hdf5, qhex).
        :param compression: Compression type (only used for HDF5).
        """
        if file_type == "json":
            HexalSaver.save_to_json(grid, file_path)
        elif file_type == "hdf5":
            HexalSaver.save_to_hdf5(grid, file_path, compression=compression)
        elif file_type == "qhex":
            HexalSaver.save_to_qhex(grid, file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
