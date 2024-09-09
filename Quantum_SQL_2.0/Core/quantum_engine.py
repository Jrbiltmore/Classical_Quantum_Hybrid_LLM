
# quantum_engine.py
# This file contains the core engine for quantum computation, handling quantum state evolution, quantum operations, and execution of quantum algorithms.

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, Aer, execute

class QuantumEngine:
    def __init__(self, num_qubits):
        self.num_qubits = num_qubits
        self.qr = QuantumRegister(num_qubits)
        self.cr = ClassicalRegister(num_qubits)
        self.circuit = QuantumCircuit(self.qr, self.cr)

    def apply_hadamard(self, qubit):
        """Applies a Hadamard gate to a specified qubit."""
        self.circuit.h(self.qr[qubit])

    def apply_cnot(self, control_qubit, target_qubit):
        """Applies a controlled-NOT (CNOT) gate between control and target qubits."""
        self.circuit.cx(self.qr[control_qubit], self.qr[target_qubit])

    def apply_phase_shift(self, qubit, phase):
        """Applies a phase shift to a specified qubit."""
        self.circuit.p(phase, self.qr[qubit])

    def apply_measurement(self):
        """Measures all qubits in the quantum circuit."""
        self.circuit.measure(self.qr, self.cr)

    def execute_circuit(self, shots=1024, backend_name="qasm_simulator"):
        """Executes the quantum circuit using the specified backend and returns the result."""
        backend = Aer.get_backend(backend_name)
        job = execute(self.circuit, backend, shots=shots)
        result = job.result()
        return result.get_counts(self.circuit)

    def reset_circuit(self):
        """Resets the quantum circuit to the initial state."""
        self.circuit.data = []

    def apply_grover_algorithm(self, oracle):
        """Applies Grover's algorithm using a specified oracle function."""
        # Initialize superposition
        for qubit in range(self.num_qubits):
            self.apply_hadamard(qubit)
        
        # Apply oracle
        oracle(self.circuit)

        # Apply diffusion operator (Grover diffusion)
        for qubit in range(self.num_qubits):
            self.apply_hadamard(qubit)
            self.apply_phase_shift(qubit, np.pi)

        self.apply_hadamard(self.num_qubits - 1)
        self.apply_cnot(self.num_qubits - 2, self.num_qubits - 1)
        self.apply_hadamard(self.num_qubits - 1)

    def run_grover(self, oracle):
        """Runs Grover's search algorithm on the quantum engine."""
        self.apply_grover_algorithm(oracle)
        self.apply_measurement()
        return self.execute_circuit()

# Example usage:
# engine = QuantumEngine(2)
# engine.apply_hadamard(0)
# engine.apply_cnot(0, 1)
# engine.apply_measurement()
# print(engine.execute_circuit())
