
import torch
import torch.nn as nn

class FidelityLoss(nn.Module):
    def __init__(self):
        super(FidelityLoss, self).__init__()

    def forward(self, predicted, target):
        # Calculate fidelity between predicted and target states
        fidelity = torch.sum(torch.sqrt(predicted * target), dim=-1) ** 2
        loss = 1 - fidelity.mean()
        return loss

if __name__ == "__main__":
    # Example usage
    predicted = torch.tensor([[0.7, 0.3], [0.5, 0.5]], dtype=torch.float32)
    target = torch.tensor([[0.6, 0.4], [0.5, 0.5]], dtype=torch.float32)

    loss_fn = FidelityLoss()
    loss = loss_fn(predicted, target)

    print(f"Fidelity Loss: {loss.item()}")
