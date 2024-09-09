import numpy as np

class InteractionManager:
    def __init__(self, hexal_grid, zoom_level=1.0, pan_offset=(0, 0)):
        self.hexal_grid = hexal_grid
        self.zoom_level = zoom_level
        self.pan_offset = np.array(pan_offset)

    def handle_zoom(self, zoom_factor):
        """
        Handles zooming interaction for the hexal grid.
        - zoom_factor: A multiplier that increases or decreases the zoom level.
        """
        self.zoom_level *= zoom_factor
        self.zoom_level = np.clip(self.zoom_level, 0.1, 10.0)  # Limit zoom levels between 0.1x and 10x

    def apply_zoom(self):
        """
        Applies the current zoom level to the hexal grid, adjusting the scale of the visualization.
        """
        zoomed_grid_size = np.array(self.hexal_grid.shape) * self.zoom_level
        zoomed_grid = np.resize(self.hexal_grid, zoomed_grid_size.astype(int))
        return zoomed_grid

    def handle_panning(self, pan_vector):
        """
        Handles panning interaction for the hexal grid, adjusting the grid's position based on user input.
        - pan_vector: A 2D vector representing the direction and magnitude of panning.
        """
        self.pan_offset += np.array(pan_vector)

    def apply_panning(self):
        """
        Applies the current panning offset to the hexal grid, adjusting its position on the screen.
        """
        shifted_grid = np.roll(self.hexal_grid, self.pan_offset.astype(int), axis=(0, 1))
        return shifted_grid

    def handle_selection(self, selection_coords):
        """
        Handles the selection of a specific hexal cell based on user input.
        - selection_coords: The coordinates of the hexal cell being selected.
        Returns the value of the selected hexal cell.
        """
        x, y = selection_coords
        return self.hexal_grid[x, y]

    def modify_selected_hexal(self, selection_coords, new_value):
        """
        Modifies the value of a selected hexal cell.
        - selection_coords: The coordinates of the hexal cell to be modified.
        - new_value: The new value to assign to the selected hexal.
        """
        x, y = selection_coords
        self.hexal_grid[x, y] = new_value

    def draw_on_grid(self, start_coords, end_coords, value):
        """
        Draws a line of hexal cells on the grid between the start and end coordinates, assigning them a specific value.
        - start_coords: Starting coordinates for the line.
        - end_coords: Ending coordinates for the line.
        - value: The value to assign to the drawn cells.
        """
        start = np.array(start_coords)
        end = np.array(end_coords)
        num_points = max(np.abs(end - start)) + 1
        points = np.linspace(start, end, num_points).astype(int)
        for point in points:
            self.hexal_grid[point[0], point[1]] = value

    def erase_on_grid(self, selection_coords):
        """
        Erases a specific hexal cell, returning its value to the default state.
        - selection_coords: The coordinates of the hexal cell to erase.
        """
        self.modify_selected_hexal(selection_coords, 0)

    def rotate_selected_hexals(self, selection_coords, degrees=60):
        """
        Rotates the selected hexal cells by a specified number of degrees.
        - selection_coords: Coordinates of the selected hexal cells.
        - degrees: The number of degrees to rotate (60, 120, 180, etc.).
        """
        if degrees % 60 != 0:
            raise ValueError("Rotation degrees must be a multiple of 60 for hexagonal grids.")
        
        # For simplicity, rotating hexagonal grid cells involves rotating the grid view, not individual cells
        rotated_grid = np.rot90(self.hexal_grid, k=degrees // 60)
        for (x, y) in selection_coords:
            self.hexal_grid[x, y] = rotated_grid[x, y]

    def undo_last_action(self, grid_history):
        """
        Reverts the grid to its previous state by undoing the last action.
        - grid_history: A list of previous grid states.
        """
        if len(grid_history) > 1:
            grid_history.pop()  # Remove the current state
            self.hexal_grid = grid_history[-1]  # Revert to the previous state

    def apply_brush_stroke(self, brush_pattern, start_coords):
        """
        Applies a brush stroke on the hexal grid using a defined pattern.
        - brush_pattern: A 2D array representing the pattern of the brush.
        - start_coords: Starting coordinates where the brush stroke begins.
        """
        pattern_height, pattern_width = brush_pattern.shape
        start_x, start_y = start_coords
        for i in range(pattern_height):
            for j in range(pattern_width):
                self.hexal_grid[start_x + i, start_y + j] = brush_pattern[i, j]

    def rotate_brush(self, brush_pattern, degrees=60):
        """
        Rotates the brush pattern by a specified number of degrees.
        - brush_pattern: The pattern to rotate.
        - degrees: The number of degrees to rotate the brush pattern (multiple of 60).
        Returns the rotated brush pattern.
        """
        if degrees % 60 != 0:
            raise ValueError("Rotation degrees must be a multiple of 60 for hexagonal symmetry.")
        
        rotated_brush = np.rot90(brush_pattern, k=degrees // 60)
        return rotated_brush
