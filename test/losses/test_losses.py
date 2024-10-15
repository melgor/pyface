import torch

from pyface.models.losses import FocalLoss


def test_focal_loss():
    loss_function = FocalLoss()
    num_classes = 100
    features = torch.randn(16, 100, requires_grad=True)
    targets = torch.randint(low=0, high=num_classes, size=(16,))
    loss = loss_function(features, targets)
    loss.backward()
