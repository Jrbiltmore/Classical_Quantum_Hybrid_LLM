
# Voxel Grid for Voxel-Hexel Visualization Engine
# Manages voxel grid creation, manipulation, and scaling

import numpy as np

class VoxelGrid:
    """
    Represents a voxel grid structure. Provides functionality for creating, modifying,
    and scaling voxel units. Supports efficient data storage and retrieval for large grids.
    """
    
    def __init__(self, dimensions=(100, 100, 100), default_value=0):
        self.grid = np.full(dimensions, default_value)
        self.dimensions = dimensions
        self.default_value = default_value

    def set_voxel(self, x, y, z, value):
        """
        Set the value of a voxel at the specified (x, y, z) coordinates.
        :param x: x-coordinate of the voxel.
        :param y: y-coordinate of the voxel.
        :param z: z-coordinate of the voxel.
        :param value: Value to set in the voxel.
        """
        if self.is_within_bounds(x, y, z):
            self.grid[x, y, z] = value
        else:
            raise IndexError("Voxel coordinates are out of bounds.")

    def get_voxel(self, x, y, z):
        """
        Retrieve the value of a voxel at the specified (x, y, z) coordinates.
        :param x: x-coordinate of the voxel.
        :param y: y-coordinate of the voxel.
        :param z: z-coordinate of the voxel.
        :return: Value at the voxel position.
        """
        if self.is_within_bounds(x, y, z):
            return self.grid[x, y, z]
        else:
            raise IndexError("Voxel coordinates are out of bounds.")

    def is_within_bounds(self, x, y, z):
        """
        Check if the given voxel coordinates are within the bounds of the voxel grid.
        :param x: x-coordinate.
        :param y: y-coordinate.
        :param z: z-coordinate.
        :return: True if within bounds, False otherwise.
        """
        return 0 <= x < self.dimensions[0] and 0 <= y < self.dimensions[1] and 0 <= z < self.dimensions[2]

    def clear_grid(self):
        """
        Reset the voxel grid to the default value.
        """
        self.grid.fill(self.default_value)

    def fill_region(self, start, end, value):
        """
        Fill a cuboidal region of the voxel grid with a specific value.
        :param start: Starting coordinates (x1, y1, z1) of the region.
        :param end: Ending coordinates (x2, y2, z2) of the region.
        :param value: Value to fill the region with.
        """
        x1, y1, z1 = start
        x2, y2, z2 = end
        if self.is_within_bounds(x1, y1, z1) and self.is_within_bounds(x2 - 1, y2 - 1, z2 - 1):
            self.grid[x1:x2, y1:y2, z1:z2] = value
        else:
            raise IndexError("Voxel coordinates for the region are out of bounds.")

    def scale_grid(self, scale_factor):
        """
        Scale the voxel grid by a uniform scale factor.
        :param scale_factor: The factor by which to scale the grid.
        """
        new_dimensions = tuple(int(dim * scale_factor) for dim in self.dimensions)
        self.grid = self._resize_voxel_grid(self.grid, new_dimensions)
        self.dimensions = new_dimensions

    def _resize_voxel_grid(self, grid, new_dimensions):
        """
        Internal helper function to resize a voxel grid.
        :param grid: The current voxel grid.
        :param new_dimensions: The target dimensions for the resized grid.
        :return: The resized voxel grid.
        """
        new_grid = np.zeros(new_dimensions, dtype=grid.dtype)
        min_dims = tuple(min(od, nd) for od, nd in zip(grid.shape, new_dimensions))
        new_grid[:min_dims[0], :min_dims[1], :min_dims[2]] = grid[:min_dims[0], :min_dims[1], :min_dims[2]]
        return new_grid
