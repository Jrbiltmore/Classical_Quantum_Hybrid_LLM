
import numpy as np
from typing import Tuple

class HexalGrid:
    def __init__(self):
        self.grid = None

    def generate_grid(self, dimensions: Tuple[int, int], grid_size: int, orientation: str = 'flat-topped') -> np.ndarray:
        '''
        Generates a hexagonal grid based on the specified dimensions, grid size, and orientation.
        
        :param dimensions: A tuple specifying the width (x) and height (y) of the grid.
        :param grid_size: The size of each hexagonal tile.
        :param orientation: The orientation of the hexagons ('flat-topped' or 'pointy-topped').
        :return: A NumPy array representing the hexagonal grid.
        '''
        width, height = dimensions
        self.grid = np.zeros((width, height), dtype=object)

        for x in range(width):
            for y in range(height):
                self.grid[x, y] = self._create_hexagon(x, y, grid_size, orientation)

        return self.grid

    def _create_hexagon(self, x: int, y: int, grid_size: int, orientation: str) -> dict:
        '''
        Creates an individual hexagon with the specified properties.
        
        :param x: The x-coordinate of the hexagon.
        :param y: The y-coordinate of the hexagon.
        :param grid_size: The size of the hexagon.
        :param orientation: The orientation of the hexagon ('flat-topped' or 'pointy-topped').
        :return: A dictionary representing the properties of the hexagon.
        '''
        hexagon = {
            'x': x,
            'y': y,
            'size': grid_size,
            'orientation': orientation
        }
        return hexagon

    def get_hexagon(self, x: int, y: int) -> dict:
        '''
        Retrieves the properties of a specific hexagon.
        
        :param x: The x-coordinate of the hexagon.
        :param y: The y-coordinate of the hexagon.
        :return: A dictionary representing the properties of the hexagon.
        '''
        if self.grid is None:
            raise ValueError("Grid has not been generated.")
        return self.grid[x, y]

    def get_neighbors(self, x: int, y: int) -> list:
        '''
        Retrieves the neighboring hexagons for a specific hexagon.
        
        :param x: The x-coordinate of the hexagon.
        :param y: The y-coordinate of the hexagon.
        :return: A list of neighboring hexagons.
        '''
        neighbors = []
        directions = [(1, 0), (0, 1), (-1, 0), (0, -1), (1, -1), (-1, 1)]
        
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.grid.shape[0] and 0 <= ny < self.grid.shape[1]:
                neighbors.append(self.grid[nx, ny])

        return neighbors
