
# observer_manager.py
# Manages the observer’s position, velocity, and view angle within the voxel space.

from typing import Tuple

class ObserverManager:
    """Class responsible for managing the observer's dynamics in the voxel space."""

    def __init__(self):
        self.position = (0.0, 0.0, 0.0)  # Observer's position (x, y, z)
        self.velocity = (0.0, 0.0, 0.0)  # Observer's velocity (dx, dy, dz)
        self.view_angle = 0.0  # Observer's view angle

    def update_position(self, position: Tuple[float, float, float]):
        """Updates the observer's position."""
        self.position = position

    def update_velocity(self, velocity: Tuple[float, float, float]):
        """Updates the observer's velocity."""
        self.velocity = velocity

    def update_view_angle(self, angle: float):
        """Updates the observer's view angle."""
        self.view_angle = angle

    def get_observer_data(self) -> Tuple[Tuple[float, float, float], Tuple[float, float, float], float]:
        """Returns the current observer data: position, velocity, and view angle."""
        return self.position, self.velocity, self.view_angle
