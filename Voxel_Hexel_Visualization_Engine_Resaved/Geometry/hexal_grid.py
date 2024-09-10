
# Hexal Grid for Voxel-Hexel Visualization Engine
# Manages hexagonal grids and related geometric transformations

class HexalGrid:
    """
    Represents a hexagonal grid structure. Provides functionality for creating,
    modifying, and scaling hexal cells. Supports efficient storage and retrieval
    of hexal data.
    """
    
    def __init__(self, radius=50, default_value=0):
        self.radius = radius
        self.default_value = default_value
        self.grid = self._initialize_grid()

    def _initialize_grid(self):
        """
        Initialize a hexagonal grid using axial coordinates (q, r).
        :return: Dictionary representing the hexagonal grid.
        """
        grid = {}
        for q in range(-self.radius, self.radius + 1):
            for r in range(max(-self.radius, -q - self.radius), min(self.radius, -q + self.radius) + 1):
                grid[(q, r)] = self.default_value
        return grid

    def set_hexal(self, q, r, value):
        """
        Set the value of a hexal at the given q, r coordinates.
        :param q: Axial q-coordinate.
        :param r: Axial r-coordinate.
        :param value: Value to set in the hexal.
        """
        if (q, r) in self.grid:
            self.grid[(q, r)] = value
        else:
            raise IndexError("Hexal coordinates are out of bounds.")

    def get_hexal(self, q, r):
        """
        Get the value of a hexal at the given q, r coordinates.
        :param q: Axial q-coordinate.
        :param r: Axial r-coordinate.
        :return: Value at the hexal position.
        """
        return self.grid.get((q, r), None)

    def is_within_bounds(self, q, r):
        """
        Check if the given hexal coordinates are within the bounds of the grid.
        :param q: Axial q-coordinate.
        :param r: Axial r-coordinate.
        :return: True if within bounds, False otherwise.
        """
        return (q, r) in self.grid

    def clear_grid(self):
        """
        Reset the hexal grid to the default value.
        """
        for key in self.grid:
            self.grid[key] = self.default_value

    def scale_grid(self, scale_factor):
        """
        Scale the hexal grid by a given scale factor.
        :param scale_factor: The factor by which to scale the grid.
        """
        new_radius = int(self.radius * scale_factor)
        self.radius = max(1, new_radius)
        self.grid = self._initialize_grid()

    def get_neighbors(self, q, r):
        """
        Get the neighboring hexals of a given hexal using axial coordinates.
        :param q: Axial q-coordinate.
        :param r: Axial r-coordinate.
        :return: List of neighboring hexal coordinates.
        """
        directions = [(+1, 0), (-1, 0), (0, +1), (0, -1), (+1, -1), (-1, +1)]
        neighbors = [(q + dq, r + dr) for dq, dr in directions if self.is_within_bounds(q + dq, r + dr)]
        return neighbors
