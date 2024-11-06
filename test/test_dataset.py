from pyface.config import load_config
from pyface.datasets.data import FaceDataModule, FaceRecognitionDataset


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
    data_module.setup("fit")
    train_dataloader = data_module.train_dataloader()
    valid_dataloader = data_module.val_dataloader()[0]
    for imgs, labels in train_dataloader:
        assert imgs is not None
        assert labels is not None

    for imgs, labels in valid_dataloader:
        assert imgs is not None
        assert labels is not None


# def test_lfw():
#     lfw_paths = "test/assets/test_data.csv"
#     lfw_pairs = "test/assets/test_pairs.txt"
#     lfw_evaluator = LFWEvaluator(lfw_paths, lfw_pairs)
#     features = np.random.randn(len(lfw_evaluator.paths), 128)
#     accuracy = lfw_evaluator.evaluate(features)
#     print(accuracy)
