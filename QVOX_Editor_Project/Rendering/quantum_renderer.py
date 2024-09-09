
# quantum_renderer.py
# Renders quantum state information in 3D, showing wavefunction probability distributions and superpositions.

from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
import numpy as np

class QuantumRenderer:
    """Class responsible for rendering quantum state visualizations in a 3D space."""

    def __init__(self):
        self.state_color_map = self._generate_state_color_map()

    def _generate_state_color_map(self):
        """Generates a color map for rendering quantum states based on probability distributions."""
        return {
            "high_prob": (0.0, 1.0, 0.0),  # Green for high probability
            "medium_prob": (1.0, 1.0, 0.0),  # Yellow for medium probability
            "low_prob": (1.0, 0.0, 0.0)  # Red for low probability
        }

    def render_quantum_state(self, x: int, y: int, z: int, quantum_state: np.ndarray):
        """Renders a voxel's quantum state in 3D based on the probability amplitude."""
        probability_amplitude = np.abs(quantum_state) ** 2
        dominant_state = np.argmax(probability_amplitude)

        # Map probability to color
        if probability_amplitude[dominant_state] > 0.8:
            color = self.state_color_map["high_prob"]
        elif probability_amplitude[dominant_state] > 0.4:
            color = self.state_color_map["medium_prob"]
        else:
            color = self.state_color_map["low_prob"]

        glPushMatrix()
        glTranslatef(x, y, z)
        glColor3f(*color)
        glutSolidCube(1.0)
        glPopMatrix()

    def render_quantum_grid(self, voxel_grid: np.ndarray, quantum_states: Dict[Tuple[int, int, int], np.ndarray]):
        """Renders the entire voxel grid with quantum state visualizations for each voxel."""
        grid_size = voxel_grid.shape
        for x in range(grid_size[0]):
            for y in range(grid_size[1]):
                for z in range(grid_size[2]):
                    if voxel_grid[x, y, z] is not None and (x, y, z) in quantum_states:
                        self.render_quantum_state(x, y, z, quantum_states[(x, y, z)])

