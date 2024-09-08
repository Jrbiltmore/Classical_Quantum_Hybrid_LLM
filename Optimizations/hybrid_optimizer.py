
# hybrid_optimizer.py
# Optimizer for quantum-classical hybrid models, combining classical and quantum optimization methods

import torch.optim as optim
from scipy.optimize import minimize  # For quantum optimization (e.g., COBYLA)

class HybridOptimizer:
    def __init__(self, model):
        self.model = model
        self.classical_optimizer = optim.Adam(self.model.classical_parameters(), lr=0.001)
        self.quantum_optimizer = minimize

    def optimize_classical(self, classical_input):
        # Optimization for classical layers
        self.classical_optimizer.zero_grad()
        output = self.model.forward_classical(classical_input)
        loss = self.model.classical_loss(output)
        loss.backward()
        self.classical_optimizer.step()
        return output

    def optimize_quantum(self, quantum_input):
        # Optimization for quantum layers using COBYLA or other methods
        def quantum_loss(params):
            output = self.model.forward_quantum(quantum_input, params)
            return self.model.quantum_loss(output)

        optimized_params = self.quantum_optimizer(quantum_loss, self.model.initial_quantum_params, method='COBYLA')
        return optimized_params
