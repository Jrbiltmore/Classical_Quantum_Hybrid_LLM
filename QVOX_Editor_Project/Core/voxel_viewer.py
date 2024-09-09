
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
import numpy as np

class VoxelViewer:
    def __init__(self, voxel_grid, quantum_states):
        self.voxel_grid = voxel_grid
        self.quantum_states = quantum_states
        self.window = None
        self.angle = 0

    def init_window(self):
        glutInit()
        glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH)
        glutInitWindowSize(800, 600)
        self.window = glutCreateWindow(b"QVOX Voxel Viewer")
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_COLOR_MATERIAL)

    def display(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()

        # Set up camera view
        glTranslatef(0.0, 0.0, -30)
        glRotatef(self.angle, 1.0, 1.0, 0.0)

        # Render voxels in the grid
        grid_size = self.voxel_grid.shape
        for x in range(grid_size[0]):
            for y in range(grid_size[1]):
                for z in range(grid_size[2]):
                    if self.voxel_grid[x, y, z] is not None:
                        self.render_voxel(x, y, z)

        glutSwapBuffers()

    def render_voxel(self, x, y, z):
        glPushMatrix()
        glTranslatef(x - 5, y - 5, z - 5)  # Centering the voxel grid
        quantum_state = self.quantum_states.get((x, y, z), None)

        # Color based on quantum state (simple red/blue based on state existence)
        if quantum_state is not None:
            glColor3f(1.0, 0.0, 0.0)  # Red for quantum states
        else:
            glColor3f(0.5, 0.5, 0.5)  # Grey for empty voxels

        glutSolidCube(1)
        glPopMatrix()

    def rotate_view(self):
        self.angle += 1
        if self.angle > 360:
            self.angle = 0
        glutPostRedisplay()

    def start(self):
        self.init_window()
        glutDisplayFunc(self.display)
        glutIdleFunc(self.rotate_view)
        glutMainLoop()
