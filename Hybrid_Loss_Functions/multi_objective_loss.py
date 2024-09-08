
import torch
import torch.nn as nn

class MultiObjectiveLoss(nn.Module):
    def __init__(self, loss_fns, weights=None):
        super(MultiObjectiveLoss, self).__init__()
        self.loss_fns = loss_fns
        if weights is None:
            self.weights = [1.0] * len(loss_fns)
        else:
            self.weights = weights

    def forward(self, predicted, target):
        total_loss = 0.0
        for i, loss_fn in enumerate(self.loss_fns):
            loss = loss_fn(predicted, target)
            total_loss += self.weights[i] * loss
        return total_loss

if __name__ == "__main__":
    # Example usage
    predicted = torch.tensor([[0.7, 0.3], [0.5, 0.5]], dtype=torch.float32)
    target = torch.tensor([[0.6, 0.4], [0.5, 0.5]], dtype=torch.float32)

    # Define individual loss functions
    mse_loss = nn.MSELoss()
    ce_loss = nn.CrossEntropyLoss()

    # Initialize Multi-Objective Loss with weights
    loss_fns = [mse_loss, ce_loss]
    multi_loss_fn = MultiObjectiveLoss(loss_fns, weights=[0.7, 0.3])

    # Calculate the total loss
    total_loss = multi_loss_fn(predicted, target)
    print(f"Multi-Objective Loss: {total_loss.item()}")
