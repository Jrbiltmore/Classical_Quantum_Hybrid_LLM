
# voxel_manager.py
# This file handles the creation, retrieval, update, and deletion of voxel data.

import numpy as np

class VoxelManager:
    def __init__(self):
        self.voxels = {}

    def create_voxel(self, voxel_id, dimensions, data=None):
        """Creates a new voxel with the given ID and dimensions."""
        if data is None:
            data = np.zeros(dimensions)
        self.voxels[voxel_id] = data

    def retrieve_voxel(self, voxel_id):
        """Retrieves a voxel's data by its ID."""
        return self.voxels.get(voxel_id, None)

    def update_voxel(self, voxel_id, new_data):
        """Updates the data of an existing voxel."""
        if voxel_id in self.voxels:
            self.voxels[voxel_id] = new_data
        else:
            raise KeyError(f"Voxel with ID {voxel_id} does not exist.")

    def delete_voxel(self, voxel_id):
        """Deletes a voxel by its ID."""
        if voxel_id in self.voxels:
            del self.voxels[voxel_id]
        else:
            raise KeyError(f"Voxel with ID {voxel_id} does not exist.")

    def list_voxels(self):
        """Lists all voxel IDs managed by this manager."""
        return list(self.voxels.keys())

# Example usage:
# voxel_manager = VoxelManager()
# voxel_manager.create_voxel("v1", (10, 10, 10))
# print(voxel_manager.retrieve_voxel("v1"))
