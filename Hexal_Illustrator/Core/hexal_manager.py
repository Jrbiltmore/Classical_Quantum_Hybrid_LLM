import numpy as np
from hexal_structure import HexalStructure

class HexalManager:
    def __init__(self, grid_size=(100, 100), default_value=0):
        self.structure = HexalStructure(grid_size, default_value)
        self.voxel_data = self.initialize_voxel_data()
        self.grid_history = []

    def initialize_voxel_data(self):
        """
        Initializes the voxel data that will be integrated with the hexal grid.
        """
        return np.zeros(self.structure.grid_size, dtype=np.float32)

    def integrate_voxel_data(self, voxel_updates):
        """
        Integrates voxel data into the hexal grid.
        - voxel_updates: A dictionary where keys are hexal coordinates and values are the voxel data to update.
        """
        for coords, value in voxel_updates.items():
            self.voxel_data[coords] = value
        self.update_hexal_with_voxels()

    def update_hexal_with_voxels(self):
        """
        Updates the hexal grid based on the current voxel data.
        """
        for x in range(self.structure.grid_size[0]):
            for y in range(self.structure.grid_size[1]):
                self.structure.modify_hexal((x, y), self.voxel_data[x, y])
        self.grid_history.append(self.structure.hexal_grid.copy())

    def update_real_time(self, rule):
        """
        Updates the hexal grid in real-time, applying a user-defined rule.
        - rule: A function that defines how each hexal should be updated based on its current state.
        """
        new_grid = np.copy(self.structure.hexal_grid)
        for x in range(self.structure.grid_size[0]):
            for y in range(self.structure.grid_size[1]):
                voxel_value = self.voxel_data[x, y]
                new_grid[x, y] = rule(voxel_value)
        self.structure.hexal_grid = new_grid
        self.grid_history.append(self.structure.hexal_grid.copy())

    def revert_to_previous_state(self):
        """
        Reverts the hexal grid to its previous state.
        """
        if len(self.grid_history) > 1:
            self.grid_history.pop()
            self.structure.hexal_grid = self.grid_history[-1]

    def add_layer_to_voxel(self, layer_values):
        """
        Adds a layer of voxel data to the hexal grid, useful for multi-layered simulations or visualization.
        - layer_values: The voxel data to add as a new layer.
        """
        self.voxel_data = np.dstack([self.voxel_data, layer_values])

    def calculate_total_energy(self):
        """
        Calculates the total energy of the hexal grid, based on the sum of voxel data values.
        """
        return np.sum(self.voxel_data)

    def apply_spatial_transformation(self, transformation_matrix):
        """
        Applies a spatial transformation (rotation, scaling, etc.) to the hexal grid using a transformation matrix.
        - transformation_matrix: A 2D array representing the transformation to apply.
        """
        transformed_grid = np.dot(self.structure.hexal_grid, transformation_matrix)
        self.structure.hexal_grid = transformed_grid
        self.grid_history.append(self.structure.hexal_grid.copy())

    def load_voxel_data_from_file(self, filename):
        """
        Loads voxel data from a file.
        """
        self.voxel_data = np.loadtxt(filename, delimiter=',')

    def save_voxel_data_to_file(self, filename):
        """
        Saves the current voxel data to a file.
        """
        np.savetxt(filename, self.voxel_data, delimiter=',')

    def apply_noise_to_voxel_data(self, noise_level=0.05):
        """
        Applies random noise to the voxel data, useful for simulating natural variations or errors.
        - noise_level: The maximum change applied to each voxel as a fraction of its value.
        """
        noise = np.random.rand(self.structure.grid_size[0], self.structure.grid_size[1]) * noise_level
        self.voxel_data += noise
        self.update_hexal_with_voxels()
