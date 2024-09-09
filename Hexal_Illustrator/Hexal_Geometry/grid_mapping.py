
import numpy as np
from typing import Tuple, Dict

class GridMapping:
    def __init__(self):
        pass

    def axial_to_cartesian(self, q: int, r: int, size: int) -> Tuple[float, float]:
        '''
        Converts axial coordinates (q, r) to Cartesian coordinates (x, y).
        
        :param q: The axial coordinate q (column).
        :param r: The axial coordinate r (row).
        :param size: The size of the hexagonal grid cell.
        :return: A tuple representing the Cartesian coordinates (x, y).
        '''
        x = size * (3/2 * q)
        y = size * (np.sqrt(3) * (r + q / 2))
        return (x, y)

    def cartesian_to_axial(self, x: float, y: float, size: int) -> Tuple[int, int]:
        '''
        Converts Cartesian coordinates (x, y) to axial coordinates (q, r).
        
        :param x: The Cartesian x-coordinate.
        :param y: The Cartesian y-coordinate.
        :param size: The size of the hexagonal grid cell.
        :return: A tuple representing the axial coordinates (q, r).
        '''
        q = (2/3 * x) / size
        r = (-x / 3 + np.sqrt(3) / 3 * y) / size
        return (int(round(q)), int(round(r)))

    def cube_to_axial(self, x: int, y: int, z: int) -> Tuple[int, int]:
        '''
        Converts cube coordinates (x, y, z) to axial coordinates (q, r).
        
        :param x: The cube x-coordinate.
        :param y: The cube y-coordinate.
        :param z: The cube z-coordinate.
        :return: A tuple representing the axial coordinates (q, r).
        '''
        q = x
        r = z
        return (q, r)

    def axial_to_cube(self, q: int, r: int) -> Tuple[int, int, int]:
        '''
        Converts axial coordinates (q, r) to cube coordinates (x, y, z).
        
        :param q: The axial coordinate q.
        :param r: The axial coordinate r.
        :return: A tuple representing the cube coordinates (x, y, z).
        '''
        x = q
        z = r
        y = -x - z
        return (x, y, z)

    def generate_axial_map(self, dimensions: Tuple[int, int], size: int) -> Dict[Tuple[int, int], Tuple[float, float]]:
        '''
        Generates a mapping between axial coordinates (q, r) and Cartesian coordinates (x, y).
        
        :param dimensions: A tuple specifying the width and height of the grid in hexagons.
        :param size: The size of each hexagonal grid cell.
        :return: A dictionary mapping axial coordinates to Cartesian coordinates.
        '''
        mapping = {}
        width, height = dimensions
        for q in range(-width, width):
            for r in range(-height, height):
                if -q - r >= -width and -q - r <= height:
                    cartesian_coords = self.axial_to_cartesian(q, r, size)
                    mapping[(q, r)] = cartesian_coords
        return mapping
