
# spin_visualizer.py
# Visualizes the spin of particles associated with voxels.

from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
import numpy as np

class SpinVisualizer:
    """Class responsible for visualizing the spin of voxels in 3D space."""

    def __init__(self):
        self.spin_color_map = self._generate_spin_color_map()

    def _generate_spin_color_map(self):
        """Generates a color map for spin visualizations."""
        return {
            "spin_up": (0.0, 1.0, 0.0),  # Green for spin-up
            "spin_down": (1.0, 0.0, 0.0)  # Red for spin-down
        }

    def visualize_spin(self, x: int, y: int, z: int, spin: float):
        """Renders the spin attribute as an arrow to indicate direction."""
        glPushMatrix()
        glTranslatef(x, y, z)

        if spin > 0:
            glColor3f(*self.spin_color_map["spin_up"])
        else:
            glColor3f(*self.spin_color_map["spin_down"])

        # Draw arrow to indicate spin direction
        glutSolidCube(0.5)
        glPopMatrix()
