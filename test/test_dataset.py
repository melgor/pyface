from pyface.config import load_config
from pyface.data import FaceRecognitionDataset, FaceDataModule


def test_dataset():
    test_csv = "test/assets/test_data.csv"
    root_dir = "test/assets/imgs"
    dataset = FaceRecognitionDataset(test_csv, root_dir)
    img, label = dataset[0]
    assert img is not None and label is not None


def test_data_module():
    config_path = "test/assets/test_config.yaml"
    config = load_config(config_path)
    data_module = FaceDataModule(config)
    data_module.setup('fit')
    train_dataloader = data_module.train_dataloader()
    valid_dataloader = data_module.val_dataloader()
    for imgs, labels in train_dataloader:
        assert imgs is not None
        assert labels is not None

    for imgs, labels in valid_dataloader:
        assert imgs is not None
        assert labels is not None
