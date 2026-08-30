"""Apply a transparent configurable gate to existing cluster uncertainty records."""

from __future__ import annotations

import math
from typing import Any, Mapping


def _finite_cluster_value(cluster: Mapping[str, Any], field: str) -> float:
    if field not in cluster:
        raise ValueError(f"Cluster {cluster.get('cluster_id', '<unknown>')} lacks required field '{field}'.")
    value = float(cluster[field])
    if not math.isfinite(value):
        raise ValueError(f"Cluster {cluster.get('cluster_id', '<unknown>')} has non-finite '{field}'.")
    return value


def check_semantic_uncertainty(cluster: Mapping[str, Any], config: Mapping[str, Any]) -> str | None:
    """Return a rejection reason when normalized entropy exceeds its enabled limit."""
    section = config["entropy"]
    if not section.get("enabled", False):
        return None
    value = _finite_cluster_value(cluster, "normalized_semantic_entropy")
    if value > float(section["max_normalized_entropy"]):
        return "normalized_entropy_exceeds_max"
    return None


def check_localization_uncertainty(cluster: Mapping[str, Any], config: Mapping[str, Any]) -> str | None:
    """Return a rejection reason when raw localization variance exceeds its enabled limit."""
    section = config["localization"]
    if not section.get("enabled", False):
        return None
    value = _finite_cluster_value(cluster, "localization_variance")
    if value > float(section["max_variance"]):
        return "localization_variance_exceeds_max"
    return None


def check_detection_persistence(cluster: Mapping[str, Any], config: Mapping[str, Any]) -> str | None:
    """Return a rejection reason when persistence is below its enabled minimum."""
    section = config["persistence"]
    if not section.get("enabled", False):
        return None
    value = _finite_cluster_value(cluster, "persistence")
    if not 0.0 <= value <= 1.0:
        raise ValueError("persistence must be in [0, 1].")
    if value < float(section["min_persistence"]):
        return "persistence_below_min"
    return None


def apply_uncertainty_gate(cluster: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
    """Preserve one cluster while assigning a decision and every rejection reason."""
    reasons: list[str] = []
    if config.get("enabled", False):
        for check in (check_semantic_uncertainty, check_localization_uncertainty, check_detection_persistence):
            reason = check(cluster, config)
            if reason is not None:
                reasons.append(reason)

    # Preserve the requested analysis fields rather than silently dropping rejected clusters.
    decision = {
        "cluster_id": cluster["cluster_id"],
        "class_id": cluster["class_id"],
        "class_name": cluster.get("class_name"),
        "representative_bbox": cluster["representative_bbox"],
        "mean_confidence": _finite_cluster_value(cluster, "mean_confidence"),
        "entropy": _finite_cluster_value(cluster, "semantic_entropy"),
        "normalized_entropy": _finite_cluster_value(cluster, "normalized_semantic_entropy"),
        "localization_variance": _finite_cluster_value(cluster, "localization_variance"),
        "persistence": _finite_cluster_value(cluster, "persistence"),
        "accepted": not reasons,
        "rejection_reasons": reasons,
    }
    return decision


def filter_uncertainty_records(clusters: list[Mapping[str, Any]], config: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Apply the gate deterministically to every supplied uncertainty record."""
    return [apply_uncertainty_gate(cluster, config) for cluster in clusters]
