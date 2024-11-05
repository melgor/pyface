import os

from typing import Callable

import numpy as np
import pandas as pd

from loguru import logger
from scipy.spatial.distance import cosine
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import normalize


def load_pairs(pairs_filename: str) -> np.ndarray:
    """Parse pair data for LFW"""
    logger.info("Reading LFW pairs.")
    pairs = []
    with open(pairs_filename, "r") as f:
        for line in f.readlines()[1:]:
            pair = line.strip().split()
            pairs.append(pair)
    assert len(pairs) == 6000
    return np.asarray(pairs, dtype="object")


class LFWEvaluator:
    def __init__(self, lfw_list: str, lfw_pairs: str, method: str = "cosine"):
        paths = pd.read_csv(lfw_list)["img_path"].tolist()
        paths = map(os.path.basename, paths)  # Get the filename.

        # Remove the extension.
        self.paths = list(map(lambda path: os.path.splitext(path)[0], paths))
        self.pairs = load_pairs(lfw_pairs)
        if method == "l2":
            self._eval = self._evaluate_L2
        elif method == "eu":
            self._eval = self._evaluate_euclidean
        elif method == "cosine":
            self._eval = self._evaluate_cosine

    def evaluate(self, features: np.ndarray):
        return self._eval(features)

    def _evaluate_cosine(self, features: np.ndarray):
        embeddings = dict(zip(*[self.paths, features]))
        acc, std = verify_exp(self.pairs, embeddings, cosine=True)
        return acc

    def _evaluate_L2(self, features: np.ndarray) -> float:
        # normalize features before comparing them
        try:
            raw_embeddings = normalize(features)
        except ValueError:
            raw_embeddings = features

        embeddings = dict(zip(*[self.paths, raw_embeddings]))
        acc, std = verify_exp(self.pairs, embeddings)
        return acc

    def _evaluate_euclidean(self, features):
        embeddings = dict(zip(*[self.paths, features]))
        acc, std = verify_exp(self.pairs, embeddings)
        return acc


def get_pairs_embeddings(pair, embeddings) -> tuple[np.ndarray, np.ndarray, bool]:
    """Return embeddings of pair and their actual_same flag"""
    if len(pair) == 3:
        name1 = "{}_{}".format(pair[0], pair[1].zfill(4))
        name2 = "{}_{}".format(pair[0], pair[2].zfill(4))
        actual_same = True
    elif len(pair) == 4:
        name1 = "{}_{}".format(pair[0], pair[1].zfill(4))
        name2 = "{}_{}".format(pair[2], pair[3].zfill(4))
        actual_same = False
    else:
        raise Exception("Unexpected pair length: {}".format(len(pair)))

    (x1, x2) = (embeddings[name1], embeddings[name2])
    return x1, x2, actual_same


def squared_distance(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    diff = x1 - x2
    dist = np.dot(diff.T, diff)
    return dist


def cosine_distance(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    return cosine(x1, x2)


def get_distances(
    embeddings: np.ndarray,
    pairs_train: list[tuple[int, int]],
    distance_function: Callable[[np.ndarray, np.ndarray], np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Given embeddings and pairs, find distance between them"""
    list_dist = []
    y_true = []
    for pair in pairs_train:
        (x1, x2, actual_same) = get_pairs_embeddings(pair, embeddings)
        dist = distance_function(x1, x2)
        list_dist.append(dist)
        y_true.append(actual_same)

    return np.asarray(list_dist), np.array(y_true)


def eval_threshold_accuracy(embeddings: np.ndarray, pairs, threshold: float, cosine: bool):
    """Eval embeddings based on chosen threshold"""
    distances, y_true = get_distances(
        embeddings, pairs, distance_function=cosine_distance if cosine else squared_distance
    )
    y_predict = np.zeros(y_true.shape)
    y_predict[np.where(distances < threshold)] = 1

    y_true = np.array(y_true)
    accuracy = accuracy_score(y_true, y_predict)
    return accuracy, pairs[np.where(y_true != y_predict)]


def find_best_threshold(embeddings, pairsTrain, cosine, resolution=100):
    """Find the best threshold for given data"""
    bestThresh = bestThreshAcc = 0
    distances, y_true = get_distances(
        embeddings, pairsTrain, distance_function=cosine_distance if cosine else squared_distance
    )

    # define threshold using distance between features
    max_distance = np.max(distances)
    thresholds = np.linspace(0, max_distance, resolution)

    for threshold in thresholds:
        y_predlabels = np.zeros(y_true.shape)
        y_predlabels[np.where(distances < threshold)] = 1

        accuracy = accuracy_score(y_true, y_predlabels)
        if accuracy >= bestThreshAcc:
            bestThreshAcc = accuracy
            bestThresh = threshold
        else:
            # No further improvements.
            return bestThresh

    return bestThresh


def verify_exp(pairs, embeddings, cosine=False):
    logger.info("Computing LFW accuracy")
    folds = KFold(n_splits=10, shuffle=False)
    accuracies = list()
    for idx, (train, test) in enumerate(folds.split(range(6000))):
        bestThresh = find_best_threshold(embeddings, pairs[train], cosine=cosine)
        accuracy, pairs_bad = eval_threshold_accuracy(embeddings, pairs[test], bestThresh, cosine=cosine)
        accuracies.append(accuracy)

    avg = np.mean(accuracies)
    std = np.std(accuracies)
    return avg, std


# if __name__ == "__main__":
lfw_paths = "lfw2.csv"
lfw_pairs = "pairs.txt"
lfw_evaluator = LFWEvaluator(lfw_paths, lfw_pairs)
features = np.random.randn(len(lfw_evaluator.paths), 128)
accuracy = lfw_evaluator.evaluate(features)
print(accuracy)
