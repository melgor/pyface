from pyface.config import load_config
from pyface.model import DeepIDLightningModule
from pyface.trainer import FaceRecognitionTrainer


def test_trainer(tmpdir: str):
    """Run simple training"""
    config_path = "test/assets/test_config.yaml"
    config = load_config(config_path)
    config.logging_dir = tmpdir
    trainer = FaceRecognitionTrainer(config)
    trainer.train(DeepIDLightningModule)
