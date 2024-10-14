import torch

from pyface.models.backbones import DeepFace


def test_deepface(input_size: int = 112):
    num_features = 4096
    input_image = torch.randn(2, 3, input_size, input_size)
    deepface_model = DeepFace(num_features)

    output = deepface_model(input_image)
    assert output.size(1) == num_features
