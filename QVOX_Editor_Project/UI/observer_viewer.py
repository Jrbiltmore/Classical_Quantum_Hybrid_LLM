
# observer_viewer.py
# Visual component that shows the observer’s perspective and its effect on the quantum simulation.

from PyQt5.QtWidgets import QWidget, QLabel, QVBoxLayout
from typing import Tuple

class ObserverViewer(QWidget):
    """Widget for displaying the observer's perspective and effects in the voxel space."""

    def __init__(self, parent=None):
        super(ObserverViewer, self).__init__(parent)
        self.layout = QVBoxLayout(self)

        # Observer position and velocity labels
        self.position_label = QLabel("Observer Position: (0.0, 0.0, 0.0)")
        self.velocity_label = QLabel("Observer Velocity: (0.0, 0.0, 0.0)")
        self.effect_label = QLabel("Observer Effect: 0.0")

        # Add labels to layout
        self.layout.addWidget(self.position_label)
        self.layout.addWidget(self.velocity_label)
        self.layout.addWidget(self.effect_label)

    def update_observer_data(self, position: Tuple[float, float, float], velocity: Tuple[float, float, float], effect: float):
        """Updates the observer's position, velocity, and effect in the voxel space."""
        self.position_label.setText(f"Observer Position: {position}")
        self.velocity_label.setText(f"Observer Velocity: {velocity}")
        self.effect_label.setText(f"Observer Effect: {effect}")
