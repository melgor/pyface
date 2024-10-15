import torch

from pyface.config import load_config
from pyface.model import FaceRecognitionLightningModule, FaceRecognitionModel


def test_face_model():
    """Test if config can be loaded"""
    config_path = "test/assets/test_config.yaml"
    config = load_config(config_path)
    face_model = FaceRecognitionModel(config)
    data = torch.randn(4, 3, 112, 112)
    output = face_model(data)
    assert output is not None


def test_face_lighting_module():
    """Test if config can be loaded"""
    config_path = "test/assets/test_config.yaml"
    config = load_config(config_path)
    face_module = FaceRecognitionLightningModule(config)
    assert face_module is not None
