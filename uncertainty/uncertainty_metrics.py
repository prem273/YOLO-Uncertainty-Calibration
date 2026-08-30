"""Independently testable Phase 6 uncertainty calculations."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np


def _normalized_distribution(probabilities: Sequence[float]) -> np.ndarray:
    values = np.asarray(probabilities, dtype=float).reshape(-1)
    if values.size == 0:
        raise ValueError("At least one class probability is required.")
    if not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("Probabilities must be finite and non-negative.")
    total = float(values.sum())
    if total <= 0.0:
        raise ValueError("At least one class probability must be positive.")
    return values / total


def calculate_entropy(probabilities: Sequence[float]) -> float:
    """Return stable Shannon entropy after normalizing non-negative class scores."""
    distribution = _normalized_distribution(probabilities)
    positive = distribution[distribution > 0.0]
    return float(-np.sum(positive * np.log(positive)))


def calculate_localization_variance(boxes: Sequence[Sequence[float]]) -> dict[str, float]:
    """Return population coordinate variances and their unnormalized mean."""
    array = np.asarray(boxes, dtype=float)
    if array.ndim != 2 or array.shape[1] != 4 or array.shape[0] == 0:
        raise ValueError("boxes must be a non-empty sequence of [x1, y1, x2, y2] boxes.")
    if not np.all(np.isfinite(array)):
        raise ValueError("Box coordinates must be finite.")
    variances = np.var(array, axis=0)  # ddof=0: variance across the available MC members.
    return {
        "variance_x1": float(variances[0]),
        "variance_y1": float(variances[1]),
        "variance_x2": float(variances[2]),
        "variance_y2": float(variances[3]),
        "localization_variance": float(np.mean(variances)),
    }


def calculate_persistence(pass_ids: Sequence[int], total_passes: int) -> float:
    """Calculate unique-pass detection frequency, bounded to the valid range."""
    if total_passes < 1:
        raise ValueError("total_passes must be positive.")
    unique_passes = set(int(pass_id) for pass_id in pass_ids)
    if any(pass_id < 1 or pass_id > total_passes for pass_id in unique_passes):
        raise ValueError("Pass IDs must be within the inclusive range [1, total_passes].")
    if len(unique_passes) > total_passes:
        raise ValueError("More unique pass IDs than total stochastic passes.")
    return len(unique_passes) / total_passes


def _cluster_probability_distribution(members: Sequence[dict[str, Any]]) -> list[float]:
    vectors = [member.get("class_probabilities") for member in members]
    if not vectors or any(vector is None for vector in vectors):
        raise ValueError(
            "Cluster lacks actual class probability vectors. Regenerate MC predictions with the Phase 6 pipeline."
        )
    array = np.asarray(vectors, dtype=float)
    if array.ndim != 2 or not np.all(np.isfinite(array)) or np.any(array < 0.0):
        raise ValueError("Cluster class probability vectors must be equally sized, finite, and non-negative.")
    # These are actual YOLO per-class sigmoid scores for matched candidates.
    # Average over MC members, then normalize once for categorical Shannon entropy.
    return [float(value) for value in _normalized_distribution(np.mean(array, axis=0))]


def calculate_cluster_uncertainty(
    cluster: dict[str, Any], total_passes: int, normalize_entropy: bool = True
) -> dict[str, Any]:
    """Calculate all Phase 6 signals for one IoU cluster without file I/O."""
    members = cluster.get("members", [])
    if not members:
        raise ValueError("A cluster must contain at least one member.")
    probability_distribution = _cluster_probability_distribution(members)
    semantic_entropy = calculate_entropy(probability_distribution)
    num_classes = len(probability_distribution)
    normalized_entropy = semantic_entropy / np.log(num_classes) if num_classes > 1 else 0.0
    coordinate_metrics = calculate_localization_variance([member["bbox"] for member in members])
    pass_ids = [member["pass_id"] for member in members]
    unique_passes = sorted(set(int(pass_id) for pass_id in pass_ids))

    return {
        "cluster_id": cluster["cluster_id"],
        "class_id": cluster["class_id"],
        "class_name": cluster.get("class_name"),
        "num_members": len(members),
        "unique_passes": unique_passes,
        "persistence": calculate_persistence(pass_ids, total_passes),
        "semantic_entropy": semantic_entropy,
        "normalized_semantic_entropy": float(normalized_entropy) if normalize_entropy else None,
        **coordinate_metrics,
        "mean_confidence": float(np.mean([float(member["confidence"]) for member in members])),
        "representative_bbox": [float(value) for value in cluster["representative_bbox"]],
        "class_probability_distribution": probability_distribution,
    }
