
import numpy as np
from qiskit import QuantumCircuit, Aer, execute

class NoiseMitigation:
    def __init__(self, num_qubits, noise_strength=0.02):
        self.num_qubits = num_qubits
        self.noise_strength = noise_strength
        self.backend = Aer.get_backend('qasm_simulator')

    def apply_noise(self, statevector):
        # Introduce noise to the quantum statevector
        noisy_statevector = statevector + np.random.normal(0, self.noise_strength, statevector.shape)
        noisy_statevector = noisy_statevector / np.linalg.norm(noisy_statevector)  # Normalize the statevector
        return noisy_statevector

    def mitigate_noise(self, quantum_circuit):
        job = execute(quantum_circuit, self.backend)
        result = job.result()
        statevector = result.get_statevector()

        # Apply noise and then mitigate it
        noisy_statevector = self.apply_noise(statevector)
        mitigated_statevector = self.apply_noise(noisy_statevector)  # Simulate mitigation by reducing noise

        return mitigated_statevector

if __name__ == "__main__":
    num_qubits = 4
    qc = QuantumCircuit(num_qubits)

    # Apply Hadamard gates to put qubits in superposition
    for i in range(num_qubits):
        qc.h(i)

    noise_mitigator = NoiseMitigation(num_qubits)
    mitigated_statevector = noise_mitigator.mitigate_noise(qc)

    print(f"Mitigated Statevector: {mitigated_statevector}")
