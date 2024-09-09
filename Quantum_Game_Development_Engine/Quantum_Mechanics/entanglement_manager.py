# entanglement_manager.py
# Quantum_Game_Development_Engine/Quantum_Mechanics/entanglement_manager.py

from qiskit import QuantumCircuit, Aer, execute
import numpy as np

class EntanglementManager:
    def __init__(self, num_qubits):
        """
        Initializes the EntanglementManager with a quantum circuit.
        :param num_qubits: Number of qubits in the quantum circuit
        """
        self.num_qubits = num_qubits
        self.quantum_circuit = QuantumCircuit(num_qubits, num_qubits)
        self.backend = Aer.get_backend('qasm_simulator')

    def create_entanglement(self, qubit1, qubit2):
        """
        Creates entanglement between two qubits.
        :param qubit1: Index of the first qubit
        :param qubit2: Index of the second qubit
        """
        if qubit1 >= self.num_qubits or qubit2 >= self.num_qubits:
            raise ValueError("Qubit indices must be within the range of the quantum circuit.")
        
        # Apply Hadamard gate to the first qubit to create superposition
        self.quantum_circuit.h(qubit1)
        
        # Apply CNOT gate to create entanglement between qubit1 and qubit2
        self.quantum_circuit.cx(qubit1, qubit2)
        
        # Measure the qubits
        self.quantum_circuit.measure([qubit1, qubit2], [qubit1, qubit2])
    
    def get_entangled_state(self):
        """
        Retrieves the state of the qubits after entanglement.
        :return: A dictionary with the measurement results
        """
        job = execute(self.quantum_circuit, backend=self.backend, shots=1)
        result = job.result()
        measurements = result.get_counts()
        
        return measurements

    def reset_circuit(self):
        """
        Resets the quantum circuit for a new set of operations.
        """
        self.quantum_circuit = QuantumCircuit(self.num_qubits, self.num_qubits)

# Example usage
if __name__ == "__main__":
    # Initialize the EntanglementManager with 2 qubits
    em = EntanglementManager(num_qubits=2)
    
    # Create entanglement between qubit 0 and qubit 1
    em.create_entanglement(qubit1=0, qubit2=1)
    
    # Retrieve and print the entangled state
    entangled_state = em.get_entangled_state()
    print(f"Entangled state: {entangled_state}")

    # Reset the circuit
    em.reset_circuit()
    print("Quantum circuit has been reset.")
