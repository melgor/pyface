import torch
import torch.nn as nn
import torch.nn.functional as F

"""Code taken from https://github.com/msight-tech/research-xbm/blob/master/ret_benchmark/losses/contrastive_loss.py"""


class ContrastiveLossXBM(nn.Module):
    def __init__(self, margin: float = 0.5):
        super(ContrastiveLossXBM, self).__init__()
        self._margin = margin

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor):
        # Compute similarity matrix
        embeddings = F.normalize(embeddings)
        sim_mat = torch.matmul(embeddings, embeddings.t())
        batch_size = embeddings.size(0)
        epsilon = 1e-5

        # Create a mask for positive pairs (same label) and negative pairs (different label)
        label_eq = labels.unsqueeze(1) == labels.unsqueeze(0)
        pos_mask = label_eq & (sim_mat < 1 - epsilon)
        neg_mask = ~label_eq & (sim_mat > self._margin)

        # Compute positive loss
        pos_pairs = sim_mat[pos_mask]
        pos_loss = torch.sum(1 - pos_pairs)

        # Compute negative loss
        neg_pairs = sim_mat[neg_mask]
        neg_loss = torch.sum(neg_pairs)

        # Calculate final loss
        num_pos_pairs = pos_pairs.size(0)
        num_neg_pairs = neg_pairs.size(0)
        total_pairs = num_pos_pairs + num_neg_pairs

        if total_pairs > 0:
            loss = (pos_loss + neg_loss) / batch_size
        else:
            loss = torch.tensor(0.0, device=embeddings.device)

        return loss


class ContrastivePairMiner:
    """
    Miner for generating positive and negative pairs for Contrastive Loss.
    """

    def __init__(self, positive_threshold: float = 4.0, negative_threshold: float = 4.0):
        self.positive_threshold = positive_threshold
        self.negative_threshold = negative_threshold

    def mine_pairs(self, embeddings: torch.Tensor, labels: torch.Tensor):
        """
        Generate positive and negative pairs based on thresholds.

        Args:
            embeddings (torch.Tensor): The embeddings of shape (batch_size, embedding_dim).
            labels (torch.Tensor): The labels of shape (batch_size).

        Returns:
            pos_pairs (torch.Tensor): Tensor of positive pairs indices (N_pos, 2).
            neg_pairs (torch.Tensor): Tensor of negative pairs indices (N_neg, 2).
        """
        # Compute pairwise distance matrix
        dist_mat = torch.cdist(embeddings, embeddings, p=2)  # Euclidean distance

        # Create mask for positive pairs
        label_equal = labels.unsqueeze(1) == labels.unsqueeze(0)
        pos_mask = label_equal & (dist_mat < self.positive_threshold) & (dist_mat > 0)

        # Create mask for negative pairs
        neg_mask = ~label_equal & (dist_mat < self.negative_threshold)

        # Get indices of positive and negative pairs
        pos_pairs = torch.nonzero(pos_mask, as_tuple=False)
        neg_pairs = torch.nonzero(neg_mask, as_tuple=False)

        return pos_pairs, neg_pairs


class ContrastiveLoss(nn.Module):
    """
    Contrastive Loss with a miner for selecting positive and negative pairs.
    """

    def __init__(self, margin: float = 2.0, positive_threshold: float = 2.0, negative_threshold: float = 2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.miner = ContrastivePairMiner(positive_threshold, negative_threshold)

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor):
        """
        Compute the contrastive loss.

        Args:
            embeddings (torch.Tensor): The embeddings of shape (batch_size, embedding_dim).
            labels (torch.Tensor): The labels of shape (batch_size).

        Returns:
            torch.Tensor: The computed contrastive loss.
        """
        embeddings = torch.nn.functional.normalize(embeddings)
        pos_pairs, neg_pairs = self.miner.mine_pairs(embeddings, labels)

        # Compute the contrastive loss
        loss = torch.tensor(0.0, device=embeddings.device)
        if pos_pairs.size(0) > 0:
            # Extract positive pairs
            pos_distances = torch.norm(embeddings[pos_pairs[:, 0]] - embeddings[pos_pairs[:, 1]], p=2, dim=1)
            pos_loss = torch.mean(pos_distances**2)
            loss += pos_loss

        if neg_pairs.size(0) > 0:
            # Extract negative pairs

            neg_distances = torch.norm(embeddings[neg_pairs[:, 0]] - embeddings[neg_pairs[:, 1]], p=2, dim=1)
            neg_loss = torch.mean(F.relu(self.margin - neg_distances) ** 2)

            loss += neg_loss
        print(
            neg_pairs.size(0),
            neg_loss,
            pos_pairs.size(0),
            pos_loss,
        )
        return loss


# class ContrastiveLoss(nn.Module):
#     def __init__(self, margin: float = 0.5):
#         super(ContrastiveLoss, self).__init__()
#         self._margin = margin
#
#     def forward(self, embeddings: torch.Tensor, labels: torch.Tensor):
#         batch_size = embeddings.size(0)
#         # Compute similarity matrix
#         sim_mat = torch.matmul(embeddings, embeddings.t())
#         epsilon = 1e-5
#
#         loss = list()
#         neg_count = list()
#         for i in range(batch_size):
#             pos_pair_ = torch.masked_select(sim_mat[i], labels[i] == labels)
#             pos_pair_ = torch.masked_select(pos_pair_, pos_pair_ < 1 - epsilon)
#             neg_pair_ = torch.masked_select(sim_mat[i], labels[i] != labels)
#
#             neg_pair = torch.masked_select(neg_pair_, neg_pair_ > self._margin)
#
#             pos_loss = torch.sum(-pos_pair_ + 1)
#             if len(neg_pair) > 0:
#                 neg_loss = torch.sum(neg_pair)
#                 neg_count.append(len(neg_pair))
#             else:
#                 neg_loss = 0
#
#             loss.append(pos_loss + neg_loss)
#
#         loss = sum(loss) / batch_size
#         return loss
