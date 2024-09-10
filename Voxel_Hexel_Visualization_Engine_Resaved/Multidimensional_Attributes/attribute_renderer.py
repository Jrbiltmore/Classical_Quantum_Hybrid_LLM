
# Attribute Renderer for Voxel-Hexel Visualization Engine
# Renders multidimensional attributes across both voxel and hexal grids

import matplotlib.pyplot as plt

class AttributeRenderer:
    """
    Handles rendering of multidimensional attributes (e.g., quantum spin, entropy, custom attributes)
    across both voxel and hexal grids, providing visual representation of data properties.
    """
    
    def __init__(self, attribute_manager):
        self.attribute_manager = attribute_manager

    def render_attributes(self, grid, is_voxel_grid=True):
        """
        Renders attributes as visual overlays on a 2D representation of the voxel or hexal grid.
        :param grid: The voxel or hexal grid data.
        :param is_voxel_grid: Boolean indicating whether it's a voxel (True) or hexal grid (False).
        """
        if is_voxel_grid:
            self._render_voxel_attributes(grid)
        else:
            self._render_hexal_attributes(grid)

    def _render_voxel_attributes(self, grid):
        """
        Renders attributes over a voxel grid, visualizing data properties for each voxel.
        :param grid: The voxel grid data to render.
        """
        fig, ax = plt.subplots()
        for position, attributes in self.attribute_manager.attributes.items():
            x, y, z = position
            attribute_summary = self._summarize_attributes(attributes)
            ax.text(x, y, attribute_summary, fontsize=8, color="blue")
        plt.title("Voxel Grid Attributes")
        plt.show()

    def _render_hexal_attributes(self, grid):
        """
        Renders attributes over a hexal grid, visualizing data properties for each hexal.
        :param grid: The hexal grid data to render.
        """
        fig, ax = plt.subplots()
        for position, attributes in self.attribute_manager.attributes.items():
            q, r = position
            attribute_summary = self._summarize_attributes(attributes)
            ax.text(q, r, attribute_summary, fontsize=8, color="green")
        plt.title("Hexal Grid Attributes")
        plt.show()

    def _summarize_attributes(self, attributes):
        """
        Creates a summary string of attributes to display on the grid.
        :param attributes: Dictionary of attribute names and values.
        :return: A formatted string summarizing the attributes.
        """
        return ', '.join([f"{k}: {v}" for k, v in attributes.items()])
