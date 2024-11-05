import torch

from pyface.config import load_config
from pyface.face_models import DeepIDFaceRecognitionModel
from pyface.model import CasiaNetLightningModule, DeepIDLightningModule
from pyface.models.backbones import DeepFace
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


# from torchsummary import summary
#
# from pyface.models.backbones import DeepID2Plus, DeepFace
#
# # mm = DeepID2Plus()
# # summary(mm, (3, 56, 56), batch_size=2,  device="cpu")
#
# mm = DeepFace()
# summary(mm, (3, 112, 112), batch_size=2,  device="cpu")
