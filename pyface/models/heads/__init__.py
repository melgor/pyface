from .classification import ClassificationLayer, HeadLayer

HEADS: dict[str, type[HeadLayer]] = {"ClassificationLayer": ClassificationLayer}
