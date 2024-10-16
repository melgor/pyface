from pyface.config import load_config
from pyface.model import DeepIDLightningModule
from pyface.trainer import FaceRecognitionTrainer

# Press the green button in the gutter to run the script.
if __name__ == "__main__":
    config_path = "test/assets/test_config.yaml"
    config = load_config(config_path)
    trainer = FaceRecognitionTrainer(config)
    trainer.train(DeepIDLightningModule)
