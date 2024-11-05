import torch

from pyface.config import load_config
from pyface.trainer import FaceRecognitionTrainer

torch.backends.cuda.matmul.allow_tf32 = True
if __name__ == "__main__":
    config_path = "configs/deepface_config.yaml"
    config = load_config(config_path)
    trainer = FaceRecognitionTrainer(config)
    trainer.train()

    # config_path = "configs/deepid2_config.yaml"
    # config = load_config(config_path)
    # trainer = FaceRecognitionTrainer(config)
    # trainer.train(DeepIDLightningModule)
