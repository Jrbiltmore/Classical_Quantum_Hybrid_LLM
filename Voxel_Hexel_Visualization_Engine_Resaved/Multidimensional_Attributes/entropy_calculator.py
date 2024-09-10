
# Entropy Calculator for Voxel-Hexel Visualization Engine
# Calculates the entropy or disorder across voxel and hexal grids

import numpy as np

class EntropyCalculator:
    """
    Handles entropy calculation for both voxel and hexal grids, measuring the degree of disorder.
    This is useful for analyzing physical properties and quantum data distributions.
    """
    
    @staticmethod
    def calculate_entropy(grid):
        """
        Calculate the entropy (disorder) for a given grid.
        :param grid: The voxel or hexal grid data.
        :return: The entropy value of the grid.
        """
        values, counts = np.unique(grid, return_counts=True)
        probabilities = counts / len(grid.flatten())
        entropy = -np.sum(probabilities * np.log2(probabilities))
        return entropy

    @staticmethod
    def calculate_entropy_change(initial_grid, final_grid):
        """
        Calculate the change in entropy between two grid states.
        :param initial_grid: The initial grid state.
        :param final_grid: The final grid state.
        :return: The change in entropy between the two grids.
        """
        initial_entropy = EntropyCalculator.calculate_entropy(initial_grid)
        final_entropy = EntropyCalculator.calculate_entropy(final_grid)
        return final_entropy - initial_entropy
