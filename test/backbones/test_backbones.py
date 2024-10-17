import torch

from pyface.models.backbones import DeepFace, DeepID2Plus


def test_deepface(input_size: int = 112):
    num_features = 4096
    input_image = torch.randn(2, 3, input_size, input_size)
    deepface_model = DeepFace(num_features)

    output = deepface_model(input_image)
    assert output.size(1) == num_features


def test_deepid2plus(input_size: tuple[int, int] = (55, 47)):
    num_features = 4096
    input_image = torch.randn(2, 3, input_size[0], input_size[1])
    deepid2plus_model = DeepID2Plus(num_features)

    output = deepid2plus_model(input_image)
    assert output.size(1) == num_features
