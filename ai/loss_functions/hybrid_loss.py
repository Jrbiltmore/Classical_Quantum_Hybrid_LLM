
# hybrid_loss.py
# Combines quantum and classical loss functions for hybrid models

import torch
from fidelity_loss import FidelityLoss
from quantum_loss_functions import QuantumCrossEntropyLoss

class HybridLossFunction(torch.nn.Module):
    def __init__(self):
        super(HybridLossFunction, self).__init__()
        self.fidelity_loss = FidelityLoss()
        self.cross_entropy_loss = QuantumCrossEntropyLoss()

    def forward(self, classical_output, quantum_output, classical_target, quantum_target):
        # Classical loss: Mean squared error for classical output
        classical_loss = torch.nn.functional.mse_loss(classical_output, classical_target)

        # Quantum loss: Fidelity loss and cross-entropy loss for quantum output
        quantum_fidelity_loss = self.fidelity_loss(quantum_output, quantum_target)
        quantum_cross_entropy_loss = self.cross_entropy_loss(quantum_output, quantum_target)

        total_loss = classical_loss + quantum_fidelity_loss + quantum_cross_entropy_loss
        return total_loss
