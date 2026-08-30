"""Deterministic, class-aware greedy IoU clustering for MC predictions.

Clusters are considered in their creation order. For a new detection, all
same-class clusters whose current mean bounding box has IoU at least the
configured threshold are candidates. The highest-IoU candidate is selected;
ties retain creation order. This is greedy association, not NMS or box fusion.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Iterable, Sequence

import numpy as np


def calculate_iou(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    """Calculate safe intersection-over-union for two ``[x1, y1, x2, y2]`` boxes."""
    if len(box_a) != 4 or len(box_b) != 4:
        raise ValueError("Each box must contain exactly four coordinates: [x1, y1, x2, y2].")
    ax1, ay1, ax2, ay2 = (float(value) for value in box_a)
    bx1, by1, bx2, by2 = (float(value) for value in box_b)

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    if area_a == 0.0 or area_b == 0.0:
        return 0.0

    intersection_width = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    intersection_height = max(0.0, min(ay2, by2) - max(ay1, by1))
    intersection = intersection_width * intersection_height
    union = area_a + area_b - intersection
    return float(max(0.0, min(1.0, intersection / union))) if union > 0.0 else 0.0


def _as_mapping(value: Any) -> dict[str, Any]:
    """Accept Phase 4 dataclasses as well as the JSON-compatible dictionaries."""
    if isinstance(value, dict):
        return value
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if hasattr(value, "__dict__"):
        return vars(value)
    raise TypeError(f"Expected a prediction mapping or dataclass, got {type(value).__name__}.")


def _passes_from_predictions(predictions: Any) -> tuple[list[dict[str, Any]], int]:
    source = _as_mapping(predictions) if not isinstance(predictions, (list, tuple)) else {"passes": predictions}
    passes = [_as_mapping(item) for item in source.get("passes", [])]
    total_passes = int(source.get("passes_requested", len(passes)))
    if total_passes < len(passes):
        total_passes = len(passes)
    if total_passes < 1 and passes:
        total_passes = len(passes)
    return passes, total_passes


def _representative_box(members: Iterable[dict[str, Any]]) -> list[float]:
    boxes = np.asarray([member["bbox"] for member in members], dtype=float)
    return [float(value) for value in boxes.mean(axis=0)]


def _finalize(cluster: dict[str, Any], total_passes: int) -> dict[str, Any]:
    members = cluster["members"]
    unique_passes = sorted({member["pass_id"] for member in members})
    return {
        "cluster_id": cluster["cluster_id"],
        "class_id": cluster["class_id"],
        "class_name": cluster["class_name"],
        "members": deepcopy(members),
        "num_members": len(members),
        "unique_passes": unique_passes,
        "persistence": len(unique_passes) / total_passes if total_passes else 0.0,
        "representative_bbox": _representative_box(members),
    }


def greedy_iou_clustering(predictions: Any, iou_threshold: float = 0.5) -> list[dict[str, Any]]:
    """Greedily associate same-class detections from Phase 4 MC predictions.

    ``predictions`` can be the JSON mapping written by Phase 4, its ``passes``
    list, or the corresponding Phase 4 dataclass. Individual member boxes,
    confidences, classes, and pass IDs are never discarded.
    """
    if not 0.0 <= iou_threshold <= 1.0:
        raise ValueError("iou_threshold must be in [0.0, 1.0].")
    passes, total_passes = _passes_from_predictions(predictions)
    active_clusters: list[dict[str, Any]] = []

    for pass_index, pass_prediction in enumerate(passes, start=1):
        pass_id = int(pass_prediction.get("pass_id", pass_index))
        for detection in pass_prediction.get("detections", []):
            detection = _as_mapping(detection)
            bbox = detection.get("box_xyxy", detection.get("bbox"))
            if bbox is None or len(bbox) != 4:
                raise ValueError("Each detection must provide box_xyxy or bbox with four coordinates.")
            member = {
                "pass_id": pass_id,
                "bbox": [float(value) for value in bbox],
                "confidence": float(detection["confidence"]),
                "class_id": int(detection["class_id"]),
                "class_name": detection.get("class_name"),
            }

            best_cluster: dict[str, Any] | None = None
            best_iou = -1.0
            for cluster in active_clusters:
                # Semantic class agreement is required before spatial matching.
                if cluster["class_id"] != member["class_id"]:
                    continue
                iou = calculate_iou(member["bbox"], _representative_box(cluster["members"]))
                if iou >= iou_threshold and iou > best_iou:
                    best_cluster, best_iou = cluster, iou

            if best_cluster is None:
                active_clusters.append(
                    {
                        "cluster_id": len(active_clusters) + 1,
                        "class_id": member["class_id"],
                        "class_name": member["class_name"],
                        "members": [member],
                    }
                )
            else:
                best_cluster["members"].append(member)

    return [_finalize(cluster, total_passes) for cluster in active_clusters]
