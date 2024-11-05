from dataclasses import dataclass

import faiss
import numpy as np

from sklearn.preprocessing import normalize
from tqdm import tqdm


@dataclass
class ModelFeatures:
    features: np.ndarray
    labels: np.ndarray


def get_class_ranks(query_embeddings: np.ndarray, gallery_embeddings: np.ndarray) -> np.ndarray:
    # query_embeddings, gallery_embeddings = postprocess(query_embeddings, gallery_embeddings)

    index = faiss.IndexFlatL2(gallery_embeddings.shape[1])
    index.add(gallery_embeddings)
    _, indices = index.search(query_embeddings, 1000)
    return indices


def compute_precision_at_k(ranked_targets: np.ndarray, k: int) -> float:
    """
    Computes the precision at k.
    Args:
        ranked_targets: A boolean array of retrieved targets, True if relevant and False otherwise.
        k: The number of examples to consider

    Returns: The precision at k
    """
    assert k >= 1
    assert ranked_targets.size >= k, ValueError("Relevance score length < k")
    return np.mean(ranked_targets[:k])


def compute_average_precision(ranked_targets: np.ndarray, gtp: int) -> float:
    """
    Computes the average precision.
    Args:
        ranked_targets: A boolean array of retrieved targets, True if relevant and False otherwise.
        gtp: ground truth positives.

    Returns:
        The average precision.
    """
    assert gtp >= 1
    # compute precision at rank only for positive targets
    out = [compute_precision_at_k(ranked_targets, k + 1) for k in range(ranked_targets.size) if ranked_targets[k]]
    if len(out) == 0:
        # no relevant targets in top1000 results
        return 0.0
    else:
        return np.sum(out) / gtp


def calculate_map(ranked_retrieval_results: np.ndarray, query_labels: np.ndarray, gallery_labels: np.ndarray) -> float:
    """
    Calculates the mean average precision.
    Args:
        ranked_retrieval_results: A 2D array of ranked retrieval results (shape: n_queries x 1000), because we use
                                top1000 retrieval results.
        query_labels: A 1D array of query class labels (shape: n_queries).
        gallery_labels: A 1D array of gallery class labels (shape: n_gallery_items).
    Returns:
        The mean average precision.
    """
    assert ranked_retrieval_results.ndim == 2
    assert ranked_retrieval_results.shape[1] == 1000

    class_average_precisions = []

    class_ids, class_counts = np.unique(gallery_labels, return_counts=True)
    class_id2quantity_dict = dict(zip(class_ids, class_counts))
    for gallery_indices, query_class_id in tqdm(zip(ranked_retrieval_results, query_labels), total=len(query_labels)):
        # Checking that no image is repeated in the retrival results
        assert len(np.unique(gallery_indices)) == len(gallery_indices), ValueError(
            f"Repeated images in retrieval results: {gallery_indices}"
        )

        current_retrieval = gallery_labels[gallery_indices] == query_class_id
        gpt = class_id2quantity_dict[query_class_id]

        class_average_precisions.append(compute_average_precision(current_retrieval, gpt))

    mean_average_precision = np.mean(class_average_precisions).item()
    return mean_average_precision


def postprocess(query_vecs: np.ndarray, reference_vecs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Postprocessing:
    1) Moving the origin of the feature space to the center of the feature vectors
    2) L2-normalization
    """
    # centralize
    query_vecs, reference_vecs = _normalize(query_vecs, reference_vecs)

    # l2 normalization
    query_vecs = normalize(query_vecs)
    reference_vecs = normalize(reference_vecs)

    return query_vecs, reference_vecs


def _normalize(v1: np.ndarray, v2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    concat = np.concatenate([v1, v2], axis=0)
    center = np.mean(concat, axis=0)
    std_feat = np.std(concat, axis=0)
    return (v1 - center) / std_feat, (v2 - center) / std_feat


def evaluate_identification_metric(query_features: ModelFeatures, gallery_features: ModelFeatures) -> float:
    example_ranks = get_class_ranks(query_features.features, gallery_features.features)
    mAP_score = calculate_map(example_ranks, query_features.labels, gallery_features.labels)
    return mAP_score
