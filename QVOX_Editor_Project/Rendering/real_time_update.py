
# real_time_update.py
# Provides real-time updates to voxel rendering based on user input or state changes.

from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
import numpy as np

class RealTimeUpdate:
    """Class responsible for managing real-time updates to voxel rendering and state transitions."""

    def __init__(self):
        self.update_interval = 100  # Update every 100ms
        self.last_update_time = 0
        self.current_time = 0

    def start_update_loop(self, callback):
        """Starts the real-time update loop with the given callback function for rendering."""
        glutTimerFunc(self.update_interval, self.update_callback, 0)
        self.callback = callback

    def update_callback(self, value):
        """Callback function triggered for each real-time update cycle."""
        self.current_time = glutGet(GLUT_ELAPSED_TIME)
        if self.current_time - self.last_update_time >= self.update_interval:
            self.callback()  # Call the rendering function
            self.last_update_time = self.current_time
        glutTimerFunc(self.update_interval, self.update_callback, 0)

    def stop_update_loop(self):
        """Stops the real-time update loop."""
        self.update_interval = None

