
# wavefunction_simulator.py
# Simulates the time-dependent Schrödinger equation to visualize the evolution of quantum states.

import numpy as np

class WavefunctionSimulator:
    """Class responsible for simulating the time evolution of quantum wavefunctions."""

    def __init__(self):
        self.hbar = 1.0  # Reduced Planck constant (set to 1 for simplicity)

    def simulate_time_evolution(self, initial_state: np.ndarray, potential: np.ndarray, time_steps: int, dt: float) -> np.ndarray:
        """Simulates the time evolution of a quantum state using the Schrödinger equation.

        Args:
            initial_state: The initial wavefunction (as a complex numpy array).
            potential: The potential energy at each point in space (numpy array).
            time_steps: Number of time steps to simulate.
            dt: The time increment for each step.

        Returns:
            A numpy array representing the evolved wavefunction.
        """
        wavefunction = initial_state.copy()
        for _ in range(time_steps):
            wavefunction = self._apply_time_evolution_operator(wavefunction, potential, dt)
        return wavefunction

    def _apply_time_evolution_operator(self, wavefunction: np.ndarray, potential: np.ndarray, dt: float) -> np.ndarray:
        """Applies the time evolution operator to the wavefunction based on the potential and time step."""
        kinetic_operator = -0.5 * np.gradient(np.gradient(wavefunction))
        potential_operator = potential * wavefunction
        hamiltonian = kinetic_operator + potential_operator
        time_evolution_operator = np.exp(-1j * hamiltonian * dt / self.hbar)
        return wavefunction * time_evolution_operator
