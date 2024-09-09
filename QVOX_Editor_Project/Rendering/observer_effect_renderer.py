
# observer_effect_renderer.py
# Handles rendering based on observer-dependent effects, changing the view as the observer interacts with the voxel space.

from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
import numpy as np

class ObserverEffectRenderer:
    """Class responsible for rendering observer-dependent effects on the voxel grid."""

    def __init__(self):
        self.observer_effect_color_map = self._generate_observer_effect_color_map()

    def _generate_observer_effect_color_map(self):
        """Generates a color map for visualizing observer effects on the voxel grid."""
        return {
            "high_effect": (1.0, 0.0, 0.0),  # Red for high observer effect (collapse)
            "medium_effect": (1.0, 1.0, 0.0),  # Yellow for medium observer effect
            "low_effect": (0.0, 1.0, 0.0)  # Green for low observer effect
        }

    def render_observer_effect(self, x: int, y: int, z: int, observer_effect: float):
        """Renders a single voxel with color corresponding to the observer effect."""
        if observer_effect > 0.8:
            color = self.observer_effect_color_map["high_effect"]
        elif observer_effect > 0.4:
            color = self.observer_effect_color_map["medium_effect"]
        else:
            color = self.observer_effect_color_map["low_effect"]

        glPushMatrix()
        glTranslatef(x, y, z)
        glColor3f(*color)
        glutSolidCube(1.0)
        glPopMatrix()

    def render_observer_effect_grid(self, voxel_grid: np.ndarray, observer_effects: Dict[Tuple[int, int, int], float]):
        """Renders the entire voxel grid with visualizations for observer effects on each voxel."""
        grid_size = voxel_grid.shape
        for x in range(grid_size[0]):
            for y in range(grid_size[1]):
                for z in range(grid_size[2]):
                    if voxel_grid[x, y, z] is not None and (x, y, z) in observer_effects:
                        self.render_observer_effect(x, y, z, observer_effects[(x, y, z)])

