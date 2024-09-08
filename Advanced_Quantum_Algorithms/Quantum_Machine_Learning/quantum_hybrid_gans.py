
import torch
import torch.nn as nn
import numpy as np
from qiskit import QuantumCircuit, Aer, execute

class QuantumGenerator(nn.Module):
    def __init__(self, num_qubits):
        super(QuantumGenerator, self).__init__()
        self.num_qubits = num_qubits
        self.fc = nn.Linear(num_qubits, num_qubits)
        self.backend = Aer.get_backend('statevector_simulator')

    def quantum_layer(self, input_tensor):
        qc = QuantumCircuit(self.num_qubits)
        for i in range(self.num_qubits):
            qc.rx(input_tensor[i].item(), i)

        # Simulate the quantum circuit
        job = execute(qc, self.backend)
        result = job.result()
        statevector = result.get_statevector()

        return torch.tensor(np.real(statevector[:self.num_qubits]), dtype=torch.float32)

    def forward(self, x):
        x = self.fc(x)
        quantum_output = self.quantum_layer(x)
        return torch.sigmoid(quantum_output)


class ClassicalDiscriminator(nn.Module):
    def __init__(self, num_qubits):
        super(ClassicalDiscriminator, self).__init__()
        self.fc1 = nn.Linear(num_qubits, num_qubits)
        self.fc2 = nn.Linear(num_qubits, 1)

    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x


if __name__ == "__main__":
    num_qubits = 4
    generator = QuantumGenerator(num_qubits)
    discriminator = ClassicalDiscriminator(num_qubits)

    # Sample input data for the GAN
    sample_input = torch.rand(num_qubits)
    generated_data = generator(sample_input)
    discriminator_output = discriminator(generated_data)

    print(f"Generated Data: {generated_data}")
    print(f"Discriminator Output: {discriminator_output}")
