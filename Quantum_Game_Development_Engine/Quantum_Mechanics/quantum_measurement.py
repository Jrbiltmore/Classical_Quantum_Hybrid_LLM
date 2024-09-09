# quantum_measurement.py
# Quantum_Game_Development_Engine/Quantum_Mechanics/quantum_measurement.py

from qiskit import QuantumCircuit, Aer, execute

class QuantumMeasurement:
    def __init__(self, num_qubits):
        """
        Initializes the QuantumMeasurement with a quantum circuit.
        :param num_qubits: Number of qubits in the quantum circuit
        """
        self.num_qubits = num_qubits
        self.quantum_circuit = QuantumCircuit(num_qubits, num_qubits)
        self.backend = Aer.get_backend('qasm_simulator')

    def apply_quantum_gate(self, gate_type, qubit_index):
        """
        Applies a quantum gate to a qubit in the circuit.
        :param gate_type: Type of quantum gate (e.g., 'h' for Hadamard, 'x' for X-gate)
        :param qubit_index: Index of the qubit to which the gate is applied
        """
        if gate_type == 'h':
            self.quantum_circuit.h(qubit_index)
        elif gate_type == 'x':
            self.quantum_circuit.x(qubit_index)
        elif gate_type == 'z':
            self.quantum_circuit.z(qubit_index)
        elif gate_type == 'cx':
            # This method requires two qubit indices, use 'apply_quantum_gate' for single-qubit gates
            raise ValueError("CNOT gate requires two qubit indices. Use a different method for CNOT.")
        else:
            raise ValueError(f"Unknown gate type: {gate_type}")

    def apply_cnot_gate(self, control_qubit, target_qubit):
        """
        Applies a CNOT (Controlled-NOT) gate between two qubits.
        :param control_qubit: Index of the control qubit
        :param target_qubit: Index of the target qubit
        """
        if control_qubit >= self.num_qubits or target_qubit >= self.num_qubits:
            raise ValueError("Qubit indices must be within the range of the quantum circuit.")
        self.quantum_circuit.cx(control_qubit, target_qubit)

    def measure_qubits(self):
        """
        Measures the qubits in the quantum circuit and retrieves the results.
        :return: A dictionary with the measurement results
        """
        self.quantum_circuit.measure(range(self.num_qubits), range(self.num_qubits))
        job = execute(self.quantum_circuit, backend=self.backend, shots=1)
        result = job.result()
        measurements = result.get_counts()

        return measurements

    def reset_circuit(self):
        """
        Resets the quantum circuit for new operations.
        """
        self.quantum_circuit = QuantumCircuit(self.num_qubits, self.num_qubits)

# Example usage
if __name__ == "__main__":
    # Initialize the QuantumMeasurement with 3 qubits
    qm = QuantumMeasurement(num_qubits=3)
    
    # Apply Hadamard gate to qubit 0
    qm.apply_quantum_gate(gate_type='h', qubit_index=0)
    
    # Apply CNOT gate between qubit 0 (control) and qubit 1 (target)
    qm.apply_cnot_gate(control_qubit=0, target_qubit=1)
    
    # Measure all qubits and print the result
    measurement_result = qm.measure_qubits()
    print(f"Measurement result: {measurement_result}")

    # Reset the circuit
    qm.reset_circuit()
    print("Quantum circuit has been reset.")
