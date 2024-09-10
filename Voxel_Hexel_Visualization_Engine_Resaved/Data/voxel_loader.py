
# Voxel Loader for Voxel-Hexel Visualization Engine
# Supports loading from multiple formats: JSON, HDF5, QVOX

import json
import h5py
import numpy as np

class VoxelLoader:
    """
    Handles loading of voxel grid data from different formats.
    Supports JSON, HDF5, and QVOX formats.
    """
    
    @staticmethod
    def load_from_json(file_path):
        """
        Load voxel grid data from a JSON file.
        :param file_path: The path to the JSON file.
        :return: Voxel grid data as a NumPy array.
        """
        try:
            with open(file_path, 'r') as json_file:
                data = json.load(json_file)
            return np.array(data["voxel_grid"])
        except FileNotFoundError:
            raise Exception(f"File not found: {file_path}")
        except json.JSONDecodeError:
            raise Exception(f"Failed to decode JSON file: {file_path}")
    
    @staticmethod
    def load_from_hdf5(file_path):
        """
        Load voxel grid data from an HDF5 file.
        :param file_path: The path to the HDF5 file.
        :return: Voxel grid data as a NumPy array.
        """
        try:
            with h5py.File(file_path, 'r') as hdf5_file:
                data = hdf5_file['voxel_grid'][:]
            return data
        except FileNotFoundError:
            raise Exception(f"File not found: {file_path}")
        except OSError:
            raise Exception(f"Error opening HDF5 file: {file_path}")
    
    @staticmethod
    def load_from_qvox(file_path):
        """
        Load voxel grid data from a QVOX (Quantum Voxel) file.
        :param file_path: The path to the QVOX file.
        :return: Voxel grid data as a NumPy array.
        """
        try:
            with open(file_path, 'r') as qvox_file:
                # Placeholder logic for QVOX format, assuming it's a custom format
                data = np.genfromtxt(qvox_file, delimiter=',')
            return data
        except FileNotFoundError:
            raise Exception(f"File not found: {file_path}")
        except ValueError:
            raise Exception(f"Failed to parse QVOX file: {file_path}")
    
    @staticmethod
    def load(file_path, file_type="json"):
        """
        Generic load function that loads a voxel grid based on the file type.
        :param file_path: Path to the file.
        :param file_type: Type of the file (json, hdf5, qvox).
        :return: Voxel grid data as a NumPy array.
        """
        if file_type == "json":
            return VoxelLoader.load_from_json(file_path)
        elif file_type == "hdf5":
            return VoxelLoader.load_from_hdf5(file_path)
        elif file_type == "qvox":
            return VoxelLoader.load_from_qvox(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
