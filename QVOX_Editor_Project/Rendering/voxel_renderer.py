
# voxel_renderer.py
# Core engine for rendering voxel grids and voxel-level interactions.

from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
import numpy as np

class VoxelRenderer:
    """Class responsible for rendering voxels in a 3D grid."""

    def __init__(self):
        self.voxel_color_map = self._generate_voxel_color_map()

    def _generate_voxel_color_map(self):
        """Generates a color map for rendering voxels based on their properties."""
        return {
            "default": (0.0, 0.5, 1.0),
            "highlight": (1.0, 0.0, 0.0),
            "selected": (0.0, 1.0, 0.0)
        }

    def render_voxel(self, x: int, y: int, z: int, color_type: str = "default"):
        """Renders a single voxel at the specified coordinates with a given color."""
        color = self.voxel_color_map.get(color_type, self.voxel_color_map["default"])
        glPushMatrix()
        glTranslatef(x, y, z)
        glColor3f(*color)
        glutSolidCube(1.0)
        glPopMatrix()

    def render_voxel_grid(self, voxel_grid: np.ndarray):
        """Renders the entire voxel grid in 3D space."""
        grid_size = voxel_grid.shape
        for x in range(grid_size[0]):
            for y in range(grid_size[1]):
                for z in range(grid_size[2]):
                    if voxel_grid[x, y, z] is not None:
                        self.render_voxel(x, y, z)

    def highlight_voxel(self, x: int, y: int, z: int):
        """Highlights a voxel by rendering it with a different color."""
        self.render_voxel(x, y, z, color_type="highlight")

    def select_voxel(self, x: int, y: int, z: int):
        """Selects a voxel by rendering it with a selection color."""
        self.render_voxel(x, y, z, color_type="selected")
