
from qiskit import Aer, QuantumCircuit, execute
from qiskit.aqua.algorithms import QPCA
from qiskit.aqua import QuantumInstance
import numpy as np

class QuantumPCA:
    def __init__(self, num_qubits):
        self.num_qubits = num_qubits
        self.backend = Aer.get_backend('statevector_simulator')

    def run_pca(self, data):
        # Initialize QPCA
        qpca = QPCA(data, k=self.num_qubits)

        # Execute QPCA on quantum backend
        quantum_instance = QuantumInstance(self.backend)
        result = qpca.run(quantum_instance)

        return result

if __name__ == "__main__":
    # Example data for PCA (covariance matrix)
    data = np.array([[0.9, 0.4], [0.4, 0.5]])

    quantum_pca = QuantumPCA(num_qubits=2)
    result = quantum_pca.run_pca(data)

    print(f"Quantum PCA Result: {result}")
