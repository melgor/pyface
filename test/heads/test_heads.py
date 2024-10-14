import torch

from pyface.models.heads import ClassificationLayer


def test_classification_layer():
    num_features = 4096
    num_classes = 100
    input_features = torch.randn(2, num_features)
    classification_layer = ClassificationLayer(num_features, num_classes)

    output = classification_layer(input_features)
    assert output.size(1) == num_classes
