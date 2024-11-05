import torch

from pyface.config import load_config
from pyface.data import FaceDataModule
from pyface.face_models import FaceRecognitionModel
from pyface.model import FaceRecognitionLightningModule


def test_face_model():
    """Test if FaceRecognitionModel works and can pass data"""
    config_path = "test/assets/test_config.yaml"
    config = load_config(config_path)
    face_model = FaceRecognitionModel(config)
    data = torch.randn(4, 3, 112, 112)
    output = face_model(data)
    assert output is not None


def test_face_lighting_module():
    """Test if FaceRecognitionLightningModule works and is able to configure optimizers"""
    config_path = "test/assets/test_config.yaml"
    config = load_config(config_path)
    face_module = FaceRecognitionLightningModule(config)
    face_module.configure_optimizers()
    assert face_module is not None


def test_optimisation_loop_for_model():
    config_path = "test/assets/test_config.yaml"
    config = load_config(config_path)
    face_module = FaceRecognitionLightningModule(config)
    optimizers, _ = face_module.configure_optimizers()
    optimizer = optimizers[0]

    data_module = FaceDataModule(config)
    data_module.setup("fit")
    train_dataloader = data_module.train_dataloader()
    valid_dataloader = data_module.val_dataloader()

    face_module.train()
    for batch in train_dataloader:
        loss = face_module.training_step(batch, 0)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    face_module.eval()
    with torch.no_grad():
        for batch in valid_dataloader:
            loss = face_module.validation_step(batch, 0)
            assert loss is not None
