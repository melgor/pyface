import torch
import torch.nn.functional as F


class InfoNCELoss(torch.nn.Module):
    def __init__(self, temperature: float = 0.07):
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # Normalize features
        features = F.normalize(features, dim=1)

        # Compute cosine similarity matrix
        similarity_matrix = torch.mm(features, features.t()) / self.temperature

        # Create positive mask based on labels
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float()

        # Mask out self-similarity to avoid trivial pairs (diagonal should be zero)
        mask.fill_diagonal_(0)

        # Apply log softmax to similarity matrix
        log_prob = F.log_softmax(similarity_matrix, dim=1)

        # Compute loss using both positive and negative pairs
        pos_loss = -torch.sum(mask * log_prob) / mask.sum()  # Loss for positives
        neg_loss = -torch.sum((1 - mask) * log_prob) / (
            mask.shape[0] * (mask.shape[0] - 1) - mask.sum()
        )  # Loss for negatives

        # Combine positive and negative loss components
        total_loss = pos_loss + neg_loss
        return total_loss
