import sys

import torch

from pyface.config import load_config
from pyface.trainer import FaceRecognitionTrainer

torch.backends.cuda.matmul.allow_tf32 = True

if __name__ == "__main__":
    config_path = sys.argv[1]
    config = load_config(config_path)
    trainer = FaceRecognitionTrainer(config)
    trainer.train()
