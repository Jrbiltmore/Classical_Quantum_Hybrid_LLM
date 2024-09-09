
# voxel_viewer.py
# Visualization component for rendering voxels in 3D space, with support for zooming, panning, and rotating the voxel grid.

from PyQt5.QtWidgets import QWidget, QVBoxLayout
from PyQt5.QtOpenGL import QGLWidget
from PyQt5.QtCore import Qt
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
import numpy as np

class VoxelViewer(QGLWidget):
    """Widget for visualizing voxels in a 3D space."""

    def __init__(self, parent=None):
        super(VoxelViewer, self).__init__(parent)
        self.voxel_grid = None  # 3D grid of voxels to render
        self.zoom_level = -10.0  # Initial zoom level
        self.rotation = [0, 0, 0]  # Rotation angles for the grid

    def initializeGL(self):
        """Initializes OpenGL settings."""
        glClearColor(0.0, 0.0, 0.0, 1.0)
        glEnable(GL_DEPTH_TEST)

    def resizeGL(self, w, h):
        """Handles window resizing for OpenGL rendering."""
        glViewport(0, 0, w, h)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45.0, w / h, 0.1, 100.0)
        glMatrixMode(GL_MODELVIEW)

    def paintGL(self):
        """Renders the voxel grid in 3D space."""
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()

        glTranslatef(0.0, 0.0, self.zoom_level)  # Apply zoom
        glRotatef(self.rotation[0], 1.0, 0.0, 0.0)  # Apply rotation
        glRotatef(self.rotation[1], 0.0, 1.0, 0.0)
        glRotatef(self.rotation[2], 0.0, 0.0, 1.0)

        if self.voxel_grid is not None:
            self._draw_voxel_grid(self.voxel_grid)

    def _draw_voxel_grid(self, voxel_grid):
        """Draws the voxel grid by iterating through each voxel and rendering a cube at its position."""
        grid_size = voxel_grid.shape
        for x in range(grid_size[0]):
            for y in range(grid_size[1]):
                for z in range(grid_size[2]):
                    if voxel_grid[x, y, z] is not None:
                        self._draw_voxel(x, y, z)

    def _draw_voxel(self, x, y, z):
        """Renders a single voxel (cube) at the given coordinates."""
        glPushMatrix()
        glTranslatef(x, y, z)
        glColor3f(0.0, 0.5, 1.0)  # Set voxel color
        glutSolidCube(1.0)  # Draw voxel as a cube
        glPopMatrix()

    def load_voxel_grid(self, voxel_grid: np.ndarray):
        """Loads a 3D voxel grid for rendering."""
        self.voxel_grid = voxel_grid
        self.update()

    def mousePressEvent(self, event):
        """Handles mouse press events for rotating the grid."""
        self.last_mouse_pos = event.pos()

    def mouseMoveEvent(self, event):
        """Handles mouse movement for rotating the grid based on mouse drag."""
        dx = event.x() - self.last_mouse_pos.x()
        dy = event.y() - self.last_mouse_pos.y()

        if event.buttons() == Qt.LeftButton:
            self.rotation[0] += dy
            self.rotation[1] += dx

        self.last_mouse_pos = event.pos()
        self.update()

    def wheelEvent(self, event):
        """Handles zooming in and out using the mouse wheel."""
        self.zoom_level += event.angleDelta().y() / 120.0  # Adjust zoom level
        self.update()
