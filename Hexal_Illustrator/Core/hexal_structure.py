import numpy as np

class HexalStructure:
    def __init__(self, grid_size=(100, 100), default_value=0):
        self.grid_size = grid_size
        self.default_value = default_value
        self.hexal_grid = self.create_hexal_grid()

    def create_hexal_grid(self):
        """
        Initializes the hexagonal grid, with each hexal containing default values.
        """
        return np.full(self.grid_size, self.default_value, dtype=np.float32)

    def modify_hexal(self, coords, new_value):
        """
        Modifies the value of a specific hexal at the given coordinates.
        - coords: Tuple (x, y) representing the coordinates of the hexal.
        - new_value: The new value to assign to the hexal.
        """
        x, y = coords
        self.hexal_grid[x, y] = new_value

    def add_hexal_row(self, position, values):
        """
        Adds a row of hexals to the grid at a specified position. Used for dynamic resizing of the grid.
        - position: The row index where the new row should be added.
        - values: The values to fill in the new row.
        """
        self.hexal_grid = np.insert(self.hexal_grid, position, values, axis=0)

    def add_hexal_column(self, position, values):
        """
        Adds a column of hexals to the grid at a specified position. Used for dynamic resizing of the grid.
        - position: The column index where the new column should be added.
        - values: The values to fill in the new column.
        """
        self.hexal_grid = np.insert(self.hexal_grid, position, values, axis=1)

    def remove_hexal_row(self, position):
        """
        Removes a row of hexals from the grid at the specified position.
        - position: The row index to remove.
        """
        self.hexal_grid = np.delete(self.hexal_grid, position, axis=0)

    def remove_hexal_column(self, position):
        """
        Removes a column of hexals from the grid at the specified position.
        - position: The column index to remove.
        """
        self.hexal_grid = np.delete(self.hexal_grid, position, axis=1)

    def get_hexal_value(self, coords):
        """
        Retrieves the value of a specific hexal in the grid.
        - coords: Tuple (x, y) representing the coordinates of the hexal.
        """
        x, y = coords
        return self.hexal_grid[x, y]

    def rotate_hexal_grid(self, degrees=60):
        """
        Rotates the entire hexal grid by a specified number of degrees.
        - degrees: The number of degrees to rotate the grid (60, 120, 180, etc. for hexagonal symmetry).
        """
        if degrees % 60 != 0:
            raise ValueError("Rotation degrees must be a multiple of 60 for hexagonal grids.")
        return np.rot90(self.hexal_grid, k=degrees // 60)

    def scale_hexal_grid(self, scale_factor):
        """
        Scales the entire hexal grid by a specified factor.
        - scale_factor: The factor by which to scale the grid.
        """
        new_size = (int(self.grid_size[0] * scale_factor), int(self.grid_size[1] * scale_factor))
        return np.resize(self.hexal_grid, new_size)

    def merge_hexal_grids(self, other_grid, position=(0, 0)):
        """
        Merges another hexal grid into this one at a specified position.
        - other_grid: The hexal grid to be merged.
        - position: Tuple (x, y) representing where to insert the other grid.
        """
        x, y = position
        new_grid = self.hexal_grid.copy()
        new_grid[x:x+other_grid.shape[0], y:y+other_grid.shape[1]] = other_grid
        return new_grid

    def export_hexal_grid(self, filename):
        """
        Exports the current hexal grid to a file for analysis or further use.
        """
        np.savetxt(filename, self.hexal_grid, delimiter=',', fmt='%.4f')

    def import_hexal_grid(self, filename):
        """
        Imports a hexal grid from a file.
        """
        self.hexal_grid = np.loadtxt(filename, delimiter=',')

    def apply_random_noise(self, noise_level=0.1):
        """
        Applies random noise to the hexal grid, modifying values within a certain noise level.
        - noise_level: The maximum change applied to each hexal (as a fraction of its value).
        """
        noise = np.random.rand(self.grid_size[0], self.grid_size[1]) * noise_level
        self.hexal_grid += noise
