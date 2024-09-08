
from qiskit import QuantumCircuit, Aer, execute
import numpy as np

class StateEvolution:
    def __init__(self, num_qubits, delta_t=0.01):
        self.num_qubits = num_qubits
        self.delta_t = delta_t
        self.backend = Aer.get_backend('statevector_simulator')

    def hamiltonian_evolution(self, quantum_circuit, hamiltonian_matrix, time_steps):
        # Simulate time evolution under a given Hamiltonian
        for t in range(time_steps):
            time = t * self.delta_t
            # Apply unitary time evolution operator: U(t) = exp(-i * H * t)
            quantum_circuit.unitary(hamiltonian_matrix * time, range(self.num_qubits), label='U(t)')
        return quantum_circuit

    def simulate(self, hamiltonian_matrix, time_steps):
        quantum_circuit = QuantumCircuit(self.num_qubits)
        
        # Apply Hamiltonian evolution
        evolved_circuit = self.hamiltonian_evolution(quantum_circuit, hamiltonian_matrix, time_steps)
        
        job = execute(evolved_circuit, self.backend)
        result = job.result()
        statevector = result.get_statevector()
        
        return statevector

if __name__ == "__main__":
    num_qubits = 3
    time_steps = 100

    # Example Hamiltonian matrix (Pauli-Z operator)
    hamiltonian_matrix = np.array([[1, 0], [0, -1]])

    state_evolution = StateEvolution(num_qubits)
    statevector = state_evolution.simulate(hamiltonian_matrix, time_steps)

    print(f"Final Statevector after Evolution: {statevector}")
