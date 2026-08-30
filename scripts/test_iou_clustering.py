"""Integration test for Phase 5 greedy IoU clustering using saved MC predictions."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from clustering.iou_clustering import greedy_iou_clustering


def main() -> int:
    prediction_path = ROOT / "results/predictions/mc_dropout_predictions.json"
    output_path = ROOT / "results/predictions/iou_clusters.json"
    config_path = ROOT / "config/config.yaml"
    if not prediction_path.exists():
        raise FileNotFoundError(f"Run scripts/test_mc_dropout.py first: {prediction_path}")

    predictions = json.loads(prediction_path.read_text(encoding="utf-8"))
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    threshold = float(config["clustering"]["iou_threshold"])
    clusters = greedy_iou_clustering(predictions, iou_threshold=threshold)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            {
                "source_prediction_file": prediction_path.name,
                "iou_threshold": threshold,
                "num_stochastic_passes": predictions["passes_requested"],
                "total_raw_detections": sum(len(item["detections"]) for item in predictions["passes"]),
                "num_clusters": len(clusters),
                "clusters": clusters,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Stochastic passes: {predictions['passes_requested']}")
    print(f"Total raw detections: {sum(len(item['detections']) for item in predictions['passes'])}")
    print(f"IoU threshold: {threshold}")
    print(f"Resulting clusters: {len(clusters)}")
    for cluster in clusters:
        print(
            f"Cluster {cluster['cluster_id']}: class={cluster['class_name']} ({cluster['class_id']}), "
            f"members={cluster['num_members']}, passes={cluster['unique_passes']}, "
            f"persistence={cluster['persistence']:.3f}, "
            f"representative_bbox={[round(value, 2) for value in cluster['representative_bbox']]}"
        )
    print(f"Saved clusters: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
