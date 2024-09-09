# quantum_measurement_engine.py
# Quantum_Game_Development_Engine/Quantum_Logic/quantum_measurement_engine.py

import numpy as np
from qiskit import QuantumCircuit, Aer, execute

class QuantumMeasurementEngine:
    def __init__(self, num_qubits):
        self.num_qubits = num_qubits
        self.quantum_circuit = QuantumCircuit(num_qubits, num_qubits)

    def prepare_state(self, initial_state):
        """
        Prepare the quantum circuit with an initial quantum state.
        """
        if len(initial_state) != self.num_qubits:
            raise ValueError("Initial state length must match the number of qubits.")
        self.quantum_circuit.reset(range(self.num_qubits))
        for qubit, value in enumerate(initial_state):
            if value == 1:
                self.quantum_circuit.x(qubit)  # Apply X gate if value is 1
        print("Quantum state prepared.")

    def apply_gate(self, gate, qubit):
        """
        Apply a quantum gate to a specified qubit.
        """
        if gate == 'H':
            self.quantum_circuit.h(qubit)  # Apply Hadamard gate
        elif gate == 'X':
            self.quantum_circuit.x(qubit)  # Apply X gate
        elif gate == 'Z':
            self.quantum_circuit.z(qubit)  # Apply Z gate
        else:
            raise ValueError(f"Unsupported gate: {gate}")
        print(f"{gate} gate applied to qubit {qubit}.")

    def measure(self):
        """
        Measure the quantum state and return the result.
        """
        self.quantum_circuit.measure(range(self.num_qubits), range(self.num_qubits))
        job = execute(self.quantum_circuit, backend=Aer.get_backend('qasm_simulator'), shots=1)
        result = job.result()
        measurements = result.get_counts()
        measurement_result = list(measurements.keys())[0]
        print(f"Measurement result: {measurement_result}")
        return measurement_result

# Example usage
if __name__ == "__main__":
    # Initialize the quantum measurement engine
    qme = QuantumMeasurementEngine(num_qubits=3)
    
    # Prepare an initial quantum state
    initial_state = [1, 0, 1]  # Example state
    qme.prepare_state(initial_state)
    
    # Apply some gates
    qme.apply_gate('H', 0)  # Apply Hadamard gate to qubit 0
    
    # Measure the state
    measurement_result = qme.measure()
    print(f"Final measurement result: {measurement_result}")
