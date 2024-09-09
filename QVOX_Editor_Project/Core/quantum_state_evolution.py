
import numpy as np

class QuantumStateEvolution:
    def __init__(self, hbar=1.0):
        self.hbar = hbar  # Reduced Planck constant

    def evolve(self, initial_state, potential, time_steps, dt):
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

    def _apply_time_evolution_operator(self, wavefunction, potential, dt):
        """Applies the time evolution operator based on the potential and time step."""
        kinetic_operator = -0.5 * np.gradient(np.gradient(wavefunction))
        potential_operator = potential * wavefunction
        hamiltonian = kinetic_operator + potential_operator
        time_evolution_operator = np.exp(-1j * hamiltonian * dt / self.hbar)
        return wavefunction * time_evolution_operator

    def simulate_time_series(self, initial_state, potential, total_time, time_interval):
        """Simulates the quantum state evolution and returns snapshots over a time series."""
        time_steps = int(total_time / time_interval)
        state_snapshots = []
        wavefunction = initial_state.copy()

        for step in range(time_steps):
            wavefunction = self._apply_time_evolution_operator(wavefunction, potential, time_interval)
            state_snapshots.append(wavefunction.copy())

        return state_snapshots
