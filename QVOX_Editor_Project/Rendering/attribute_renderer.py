
# attribute_renderer.py
# Renders multidimensional attributes like entropy, spin, and entanglement in visual form.

from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
import numpy as np

class AttributeRenderer:
    """Class responsible for rendering voxel attributes such as entropy, spin, and entanglement in 3D."""

    def __init__(self):
        self.attribute_color_map = self._generate_attribute_color_map()

    def _generate_attribute_color_map(self):
        """Generates a color map based on voxel attributes for rendering purposes."""
        return {
            "entropy": (0.5, 0.5, 0.5),  # Gray for entropy
            "spin_up": (0.0, 1.0, 0.0),  # Green for spin-up
            "spin_down": (1.0, 0.0, 0.0),  # Red for spin-down
            "entanglement": (0.0, 0.0, 1.0)  # Blue for entanglement
        }

    def render_voxel_attributes(self, x: int, y: int, z: int, attributes: dict):
        """Renders the voxel's attributes using colored visualizations."""
        glPushMatrix()
        glTranslatef(x, y, z)

        if "entropy" in attributes:
            self._render_entropy(attributes["entropy"])

        if "spin" in attributes:
            self._render_spin(attributes["spin"])

        if "entanglement" in attributes:
            self._render_entanglement(attributes["entanglement"])

        glPopMatrix()

    def _render_entropy(self, entropy: float):
        """Renders the entropy attribute by adjusting the voxel's transparency."""
        glColor4f(0.5, 0.5, 0.5, entropy)  # Set gray color with varying transparency
        glutSolidCube(1.0)

    def _render_spin(self, spin: float):
        """Renders the spin attribute as a direction indicator."""
        if spin > 0:
            glColor3f(0.0, 1.0, 0.0)  # Green for spin-up
        else:
            glColor3f(1.0, 0.0, 0.0)  # Red for spin-down
        glutSolidCube(1.0)

    def _render_entanglement(self, entanglement: float):
        """Renders the entanglement attribute as a colored cube with blue tones."""
        glColor3f(0.0, 0.0, 1.0)  # Blue for entanglement
        glutWireCube(1.0)  # Render as wireframe to distinguish entanglement
