
# quantum_state_viewer.py
# UI component for visualizing and editing quantum states, including wavefunction animations and probability visualizations.

from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel
from PyQt5.QtCore import QTimer
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

class QuantumStateViewer(QWidget):
    """Widget for visualizing and editing quantum states associated with voxels."""

    def __init__(self, parent=None):
        super(QuantumStateViewer, self).__init__(parent)
        self.layout = QVBoxLayout(self)

        # Quantum state data
        self.current_state = None
        self.wavefunction_plot = None

        # Create label and plot area
        self.state_label = QLabel("No Quantum State Loaded")
        self.layout.addWidget(self.state_label)

        self.figure, self.ax = plt.subplots()
        self.canvas = FigureCanvas(self.figure)
        self.layout.addWidget(self.canvas)

        # Timer for animations
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_wavefunction_plot)

    def load_quantum_state(self, quantum_state: np.ndarray):
        """Loads a new quantum state for visualization."""
        self.current_state = quantum_state
        self.state_label.setText(f"Quantum State Loaded: {quantum_state.shape}")
        self.plot_wavefunction()

    def plot_wavefunction(self):
        """Plots the wavefunction of the current quantum state."""
        if self.current_state is None:
            return

        self.ax.clear()
        self.ax.set_title("Quantum State Wavefunction")

        # Plotting the real part of the wavefunction as an example
        x = np.arange(len(self.current_state))
        self.ax.plot(x, np.real(self.current_state), label='Real Part')
        self.ax.plot(x, np.imag(self.current_state), label='Imaginary Part')

        self.ax.legend()
        self.canvas.draw()

    def update_wavefunction_plot(self):
        """Updates the wavefunction plot to simulate animation of time evolution."""
        if self.current_state is None:
            return

        # Simulate evolution by multiplying by a phase factor
        self.current_state *= np.exp(1j * np.pi / 20)
        self.plot_wavefunction()

    def start_wavefunction_animation(self):
        """Starts the wavefunction animation over time."""
        self.timer.start(100)  # Update every 100ms

    def stop_wavefunction_animation(self):
        """Stops the wavefunction animation."""
        self.timer.stop()

