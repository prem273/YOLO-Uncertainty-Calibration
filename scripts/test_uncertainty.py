"""Integration test for Phase 6 uncertainty quantification."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from uncertainty.uncertainty_metrics import calculate_cluster_uncertainty


def main() -> int:
    input_path = ROOT / "results/predictions/iou_clusters.json"
    output_path = ROOT / "results/predictions/uncertainty_metrics.json"
    if not input_path.exists():
        raise FileNotFoundError(f"Run scripts/test_iou_clustering.py first: {input_path}")
    clusters_document = json.loads(input_path.read_text(encoding="utf-8"))
    config = yaml.safe_load((ROOT / "config/config.yaml").read_text(encoding="utf-8"))
    uncertainty_config = config["uncertainty"]
    if not all(section["enabled"] for section in uncertainty_config.values()):
        raise RuntimeError("All Phase 6 uncertainty signals must be enabled for this integration test.")

    total_passes = int(clusters_document["num_stochastic_passes"])
    records = [
        calculate_cluster_uncertainty(
            cluster, total_passes, normalize_entropy=bool(uncertainty_config["entropy"]["normalized"])
        )
        for cluster in clusters_document["clusters"]
    ]
    numeric_fields = ("semantic_entropy", "localization_variance", "persistence")
    for record in records:
        if not all(math.isfinite(record[field]) for field in numeric_fields):
            raise AssertionError(f"Non-finite uncertainty value in cluster {record['cluster_id']}.")
        if record["localization_variance"] < 0.0 or not 0.0 <= record["persistence"] <= 1.0:
            raise AssertionError(f"Invalid uncertainty range in cluster {record['cluster_id']}.")

    output_path.write_text(
        json.dumps(
            {
                "source_cluster_file": input_path.name,
                "num_stochastic_passes": total_passes,
                "num_clusters": len(records),
                "entropy_normalized": bool(uncertainty_config["entropy"]["normalized"]),
                "clusters": records,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Clusters processed: {len(records)}")
    for record in records:
        print(
            f"Cluster {record['cluster_id']}: class={record['class_name']}, "
            f"persistence={record['persistence']:.3f}, entropy={record['semantic_entropy']:.6f}, "
            f"normalized_entropy={record['normalized_semantic_entropy']:.6f}, "
            f"localization_variance={record['localization_variance']:.6f}, "
            f"mean_confidence={record['mean_confidence']:.6f}"
        )
    for field in ("semantic_entropy", "localization_variance", "persistence"):
        values = np.asarray([record[field] for record in records], dtype=float)
        print(f"{field}: mean={values.mean():.6f}, range=[{values.min():.6f}, {values.max():.6f}]")
    print(f"Saved uncertainty metrics: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
