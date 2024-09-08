
import torch
import torch.nn as nn

class EntanglementRegularization(nn.Module):
    def __init__(self, regularization_weight=0.1):
        super(EntanglementRegularization, self).__init__()
        self.regularization_weight = regularization_weight

    def forward(self, predicted):
        # Calculate the reduced density matrix for the first half of the qubits
        num_qubits = predicted.size(-1)
        half_qubits = num_qubits // 2
        reduced_density_matrix = torch.matmul(predicted[:, :half_qubits], predicted[:, :half_qubits].T)

        # Calculate the entanglement regularization term (trace of the reduced density matrix squared)
        trace_squared = torch.trace(torch.matmul(reduced_density_matrix, reduced_density_matrix))
        regularization_loss = self.regularization_weight * (1 - trace_squared)

        return regularization_loss

if __name__ == "__main__":
    # Example usage
    predicted = torch.tensor([[0.7, 0.3], [0.5, 0.5]], dtype=torch.float32)

    regularization_fn = EntanglementRegularization(regularization_weight=0.1)
    reg_loss = regularization_fn(predicted)

    print(f"Entanglement Regularization Loss: {reg_loss.item()}")
