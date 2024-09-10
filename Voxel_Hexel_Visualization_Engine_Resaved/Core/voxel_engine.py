# Voxel Engine for Voxel-Hexel Visualization Engine

import numpy as np

class VoxelGrid:
    def __init__(self, dimensions=(100, 100, 100), default_value=0):
        # Create a 3D grid of voxels initialized to the default value
        self.grid = np.full(dimensions, default_value)
        self.dimensions = dimensions
        self.default_value = default_value

    def set_voxel(self, x, y, z, value):
        # Set the value of a voxel at a specific position
        if self.is_within_bounds(x, y, z):
            self.grid[x, y, z] = value
        else:
            raise IndexError("Voxel coordinates are out of bounds.")

    def get_voxel(self, x, y, z):
        # Get the value of a voxel at a specific position
        if self.is_within_bounds(x, y, z):
            return self.grid[x, y, z]
        else:
            raise IndexError("Voxel coordinates are out of bounds.")

    def is_within_bounds(self, x, y, z):
        # Check if the voxel coordinates are within grid bounds
        return 0 <= x < self.dimensions[0] and 0 <= y < self.dimensions[1] and 0 <= z < self.dimensions[2]

    def clear_grid(self):
        # Reset the grid to the default value
        self.grid.fill(self.default_value)

    def fill_region(self, start, end, value):
        # Fill a cuboidal region of the grid with a specific value
        x1, y1, z1 = start
        x2, y2, z2 = end
        if self.is_within_bounds(x1, y1, z1) and self.is_within_bounds(x2-1, y2-1, z2-1):
            self.grid[x1:x2, y1:y2, z1:z2] = value
        else:
            raise IndexError("Voxel coordinates for region are out of bounds.")

    def save_grid_to_file(self, file_path):
        # Save the voxel grid to a file
        np.save(file_path, self.grid)

    def load_grid_from_file(self, file_path):
        # Load a voxel grid from a file
        self.grid = np.load(file_path)
        self.dimensions = self.grid.shape
