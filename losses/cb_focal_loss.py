import torch
import torch.nn as nn
import torch.nn.functional as F

class CBFocalLoss(nn.Module):
    def __init__(self, class_counts: torch.Tensor, cb_beta: float, focal_gamma: float, num_classes: int):
        super().__init__()
        self.focal_gamma = focal_gamma
        self.num_classes = num_classes

        # Calculate effective number of samples
        effective_num = 1.0 - torch.pow(cb_beta, class_counts)
        weights = (1.0 - cb_beta) / effective_num
        self.register_buffer('weights', weights / weights.sum()) # Normalize weights

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Inputs are logits, targets are class indices

        # Cross-entropy part
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')

        # Focal Loss part (1 - pt)^gamma
        pt = torch.exp(-ce_loss)
        focal_term = (1 - pt)**self.focal_gamma

        # Class-balanced weights
        weights = self.weights[targets]

        # Combined loss
        loss = weights * focal_term * ce_loss

        return loss.mean()
