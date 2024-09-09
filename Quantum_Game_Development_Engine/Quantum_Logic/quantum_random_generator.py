# quantum_random_generator.py
# Quantum_Game_Development_Engine/Quantum_Logic/quantum_random_generator.py

import numpy as np
from qiskit import QuantumCircuit, Aer, execute

class QuantumRandomGenerator:
    def __init__(self, num_qubits):
        """
        Initializes the QuantumRandomGenerator with a quantum circuit.
        :param num_qubits: Number of qubits in the quantum circuit
        """
        self.num_qubits = num_qubits
        self.quantum_circuit = QuantumCircuit(num_qubits, num_qubits)

    def generate_random_number(self):
        """
        Generates a random number using quantum mechanics.
        :return: Random number between 0 and 2^num_qubits - 1
        """
        # Apply Hadamard gates to create superposition
        for qubit in range(self.num_qubits):
            self.quantum_circuit.h(qubit)

        # Measure the qubits
        self.quantum_circuit.measure(range(self.num_qubits), range(self.num_qubits))

        # Execute the circuit
        job = execute(self.quantum_circuit, backend=Aer.get_backend('qasm_simulator'), shots=1)
        result = job.result()
        measurements = result.get_counts()
        
        # Convert the measurement result to an integer
        measurement_result = list(measurements.keys())[0]
        random_number = int(measurement_result, 2)

        return random_number

# Example usage
if __name__ == "__main__":
    # Initialize the QuantumRandomGenerator with 3 qubits
    qrg = QuantumRandomGenerator(num_qubits=3)
    
    # Generate a random number
    random_number = qrg.generate_random_number()
    print(f"Generated quantum random number: {random_number}")
