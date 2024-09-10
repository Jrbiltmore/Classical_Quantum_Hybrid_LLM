
# Scaling Manager for Voxel-Hexel Visualization Engine
# Handles scaling operations for both voxel and hexal grids

class ScalingManager:
    """
    Manages the scaling of voxel and hexal grids. Supports uniform and non-uniform scaling
    to adjust spatial resolution and optimize grid structures for specific applications.
    """
    
    def __init__(self, voxel_grid=None, hexal_grid=None):
        self.voxel_grid = voxel_grid
        self.hexal_grid = hexal_grid

    def scale_voxel_grid(self, scale_factor):
        """
        Scale the voxel grid by a uniform scale factor.
        :param scale_factor: The factor by which to scale the grid.
        """
        if not self.voxel_grid:
            raise ValueError("Voxel grid not provided.")
        
        new_dimensions = tuple(int(dim * scale_factor) for dim in self.voxel_grid.shape)
        self.voxel_grid = self._resize_voxel_grid(self.voxel_grid, new_dimensions)
        print(f"Voxel grid scaled by a factor of {scale_factor}")

    def scale_hexal_grid(self, scale_factor):
        """
        Scale the hexal grid by a uniform scale factor.
        :param scale_factor: The factor by which to scale the grid.
        """
        if not self.hexal_grid:
            raise ValueError("Hexal grid not provided.")
        
        new_radius = int(self.hexal_grid.radius * scale_factor)
        self.hexal_grid.scale_grid(new_radius)
        print(f"Hexal grid scaled by a factor of {scale_factor}")

    def scale_voxel_grid_non_uniform(self, scale_factors):
        """
        Perform non-uniform scaling on the voxel grid.
        :param scale_factors: Tuple of scale factors (sx, sy, sz) for each dimension.
        """
        if not self.voxel_grid:
            raise ValueError("Voxel grid not provided.")
        
        new_dimensions = tuple(int(dim * sf) for dim, sf in zip(self.voxel_grid.shape, scale_factors))
        self.voxel_grid = self._resize_voxel_grid(self.voxel_grid, new_dimensions)
        print(f"Voxel grid scaled non-uniformly by factors {scale_factors}")

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

