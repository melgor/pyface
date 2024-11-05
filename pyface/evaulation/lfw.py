import math
import os

from typing import Literal

import numpy as np

from loguru import logger
from scipy import interpolate
from sklearn.model_selection import KFold

DISTANCE_METRIC_TYPE = Literal["EU", "COSINE"]


# LFW functions taken from David Sandberg's FaceNet implementation
def distance(
    embeddings1: np.ndarray, embeddings2: np.ndarray, distance_metric: DISTANCE_METRIC_TYPE = "EU"
) -> np.ndarray:
    if distance_metric == "EU":
        # Euclidian distance
        diff = np.subtract(embeddings1, embeddings2)
        dist = np.sum(np.square(diff), 1)
    elif distance_metric == "COSINE":
        # Distance based on cosine similarity
        dot = np.sum(np.multiply(embeddings1, embeddings2), axis=1)
        norm = np.linalg.norm(embeddings1, axis=1) * np.linalg.norm(embeddings2, axis=1)
        similarity = dot / norm
        dist = np.arccos(similarity) / math.pi
    else:
        raise "Undefined distance metric %d" % distance_metric

    return dist


def calculate_roc(
    thresholds: list[float],
    embeddings1: np.ndarray,
    embeddings2: np.ndarray,
    actual_issame: np.ndarray,
    nrof_folds: int = 10,
    distance_metric: DISTANCE_METRIC_TYPE = "EU",
):
    assert embeddings1.shape[0] == embeddings2.shape[0]
    assert embeddings1.shape[1] == embeddings2.shape[1]
    nrof_pairs = min(len(actual_issame), embeddings1.shape[0])
    nrof_thresholds = len(thresholds)
    k_fold = KFold(n_splits=nrof_folds, shuffle=False)

    tprs = np.zeros((nrof_folds, nrof_thresholds))
    fprs = np.zeros((nrof_folds, nrof_thresholds))
    accuracy = np.zeros((nrof_folds))

    is_false_positive: list[float] = []
    is_false_negative: list[float] = []

    indices = np.arange(nrof_pairs)

    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):
        dist = distance(embeddings1, embeddings2, distance_metric)

        # Find the best threshold for the fold
        acc_train = np.zeros((nrof_thresholds))
        for threshold_idx, threshold in enumerate(thresholds):
            _, _, acc_train[threshold_idx], _, _ = calculate_accuracy(
                threshold, dist[train_set], actual_issame[train_set]
            )
        best_threshold_index = np.argmax(acc_train)
        for threshold_idx, threshold in enumerate(thresholds):
            tprs[fold_idx, threshold_idx], fprs[fold_idx, threshold_idx], _, _, _ = calculate_accuracy(
                threshold, dist[test_set], actual_issame[test_set]
            )
        _, _, accuracy[fold_idx], is_fp, is_fn = calculate_accuracy(
            thresholds[best_threshold_index], dist[test_set], actual_issame[test_set]
        )

        tpr = np.mean(tprs, 0)
        fpr = np.mean(fprs, 0)
        is_false_positive.extend(is_fp)
        is_false_negative.extend(is_fn)

    return tpr, fpr, accuracy, is_false_positive, is_false_negative


def calculate_accuracy(
    threshold: float, dist: np.ndarray, actual_issame: np.ndarray
) -> tuple[float, float, float, float, float]:
    predict_issame = np.less(dist, threshold)
    tp = np.sum(np.logical_and(predict_issame, actual_issame)).item()
    fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame))).item()
    tn = np.sum(np.logical_and(np.logical_not(predict_issame), np.logical_not(actual_issame))).item()
    fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame)).item()

    is_fp = np.logical_and(predict_issame, np.logical_not(actual_issame))
    is_fn = np.logical_and(np.logical_not(predict_issame), actual_issame)

    tpr = 0 if (tp + fn == 0) else float(tp) / float(tp + fn)
    fpr = 0 if (fp + tn == 0) else float(fp) / float(fp + tn)
    acc = float(tp + tn) / dist.size
    return tpr, fpr, acc, is_fp, is_fn


def calculate_val(
    thresholds,
    embeddings1,
    embeddings2,
    actual_issame,
    far_target,
    nrof_folds=10,
    distance_metric=0,
    subtract_mean=False,
):
    assert embeddings1.shape[0] == embeddings2.shape[0]
    assert embeddings1.shape[1] == embeddings2.shape[1]
    nrof_pairs = min(len(actual_issame), embeddings1.shape[0])
    nrof_thresholds = len(thresholds)
    k_fold = KFold(n_splits=nrof_folds, shuffle=False)

    val = np.zeros(nrof_folds)
    far = np.zeros(nrof_folds)

    indices = np.arange(nrof_pairs)
    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):
        if subtract_mean:
            mean = np.mean(np.concatenate([embeddings1[train_set], embeddings2[train_set]]), axis=0)
        else:
            mean = 0.0
        dist = distance(embeddings1 - mean, embeddings2 - mean, distance_metric)

        # Find the threshold that gives FAR = far_target
        far_train = np.zeros(nrof_thresholds)
        for threshold_idx, threshold in enumerate(thresholds):
            _, far_train[threshold_idx] = calculate_val_far(threshold, dist[train_set], actual_issame[train_set])
        if np.max(far_train) >= far_target:
            f = interpolate.interp1d(far_train, thresholds, kind="slinear")
            threshold = f(far_target)
        else:
            threshold = 0.0

        val[fold_idx], far[fold_idx] = calculate_val_far(threshold, dist[test_set], actual_issame[test_set])

    val_mean = np.mean(val)
    far_mean = np.mean(far)
    val_std = np.std(val)
    return val_mean, val_std, far_mean


def calculate_val_far(threshold, dist, actual_issame):
    predict_issame = np.less(dist, threshold)
    true_accept = np.sum(np.logical_and(predict_issame, actual_issame))
    false_accept = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    n_same = np.sum(actual_issame)
    n_diff = np.sum(np.logical_not(actual_issame))
    val = float(true_accept) / float(n_same)
    far = float(false_accept) / float(n_diff)
    return val, far


def evaluate(
    embeddings: np.ndarray,
    actual_issame: np.ndarray,
    nrof_folds: int = 10,
    distance_metric: DISTANCE_METRIC_TYPE = "EU",
):
    # Calculate evaluation metrics
    thresholds = np.arange(0, 4, 0.01)
    embeddings1 = embeddings[0::2]
    embeddings2 = embeddings[1::2]
    tpr, fpr, accuracy, fp, fn = calculate_roc(
        thresholds, embeddings1, embeddings2, actual_issame, nrof_folds=nrof_folds, distance_metric=distance_metric
    )
    return accuracy


def add_extension(path):
    if os.path.exists(path + ".jpg"):
        return path + ".jpg"
    elif os.path.exists(path + ".png"):
        return path + ".png"
    else:
        raise RuntimeError('No file "%s" with extension png or jpg.' % path)


class LFWVerificationEvaluator:
    def __init__(self, pairs_path: str, data_dir: str):
        self._pairs = self._read_pairs(pairs_path)
        self._path_list, self._is_same = self._get_paths(data_dir, self._pairs)

    def evaluate_lfw(self, embeddings_dict: dict[str, np.ndarray]) -> float:
        embeddings = np.array([embeddings_dict[path] for path in self._path_list])
        accuracy = evaluate(embeddings, self._is_same)
        return np.mean(accuracy)

    def _get_paths(self, lfw_dir: str, pairs: np.ndarray) -> tuple[list[str], np.ndarray]:
        nrof_skipped_pairs = 0
        path_list = []
        is_same_list: list[bool] = []
        for pair in pairs:
            if len(pair) == 3:
                path0 = add_extension(os.path.join(lfw_dir, pair[0], pair[0] + "_" + "%04d" % int(pair[1])))
                path1 = add_extension(os.path.join(lfw_dir, pair[0], pair[0] + "_" + "%04d" % int(pair[2])))
                is_same = True
            elif len(pair) == 4:
                path0 = add_extension(os.path.join(lfw_dir, pair[0], pair[0] + "_" + "%04d" % int(pair[1])))
                path1 = add_extension(os.path.join(lfw_dir, pair[2], pair[2] + "_" + "%04d" % int(pair[3])))
                is_same = False
            else:
                raise ValueError

            if os.path.exists(path0) and os.path.exists(path1):  # Only add the pair if both paths exist
                path_list += (path0, path1)
                is_same_list.append(is_same)
            else:
                nrof_skipped_pairs += 1
        if nrof_skipped_pairs > 0:
            logger.info(f"Skipped {nrof_skipped_pairs} image pairs")

        return path_list, np.asarray(is_same_list)

    def _read_pairs(self, pairs_filename: str) -> np.ndarray:
        pairs = []
        with open(pairs_filename, "r") as f:
            for line in f.readlines()[1:]:
                pair = line.strip().split()
                pairs.append(pair)
        return np.array(pairs, dtype=object)
