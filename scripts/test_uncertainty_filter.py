"""Integration test for the uncertainty-gated filtering stage."""

from __future__ import annotations

import json
import math
import sys
from collections import Counter
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from filtering.uncertainty_filter import filter_uncertainty_records


def main() -> int:
    input_path = ROOT / "results/predictions/uncertainty_metrics.json"
    output_path = ROOT / "results/predictions/filtered_predictions.json"
    if not input_path.exists():
        raise FileNotFoundError(f"Run scripts/test_uncertainty.py first: {input_path}")
    source = json.loads(input_path.read_text(encoding="utf-8"))
    config = yaml.safe_load((ROOT / "config/config.yaml").read_text(encoding="utf-8"))["filtering"]
    decisions = filter_uncertainty_records(source["clusters"], config)

    required_numeric = ("entropy", "normalized_entropy", "localization_variance", "persistence")
    for decision in decisions:
        if not all(math.isfinite(float(decision[field])) for field in required_numeric):
            raise AssertionError(f"Non-finite filtering input for cluster {decision['cluster_id']}.")
        if not isinstance(decision["accepted"], bool) or not isinstance(decision["rejection_reasons"], list):
            raise AssertionError(f"Missing filtering decision for cluster {decision['cluster_id']}.")

    accepted = [decision for decision in decisions if decision["accepted"]]
    rejected = [decision for decision in decisions if not decision["accepted"]]
    reason_counts = Counter(reason for decision in rejected for reason in decision["rejection_reasons"])
    output_path.write_text(
        json.dumps(
            {
                "source_uncertainty_file": input_path.name,
                "filtering_enabled": bool(config["enabled"]),
                "thresholds": {
                    "max_normalized_entropy": config["entropy"]["max_normalized_entropy"],
                    "max_localization_variance": config["localization"]["max_variance"],
                    "min_persistence": config["persistence"]["min_persistence"],
                },
                "total_clusters": len(decisions),
                "accepted_clusters": len(accepted),
                "rejected_clusters": len(rejected),
                "acceptance_rate": len(accepted) / len(decisions) if decisions else 0.0,
                "rejection_count_by_reason": dict(reason_counts),
                "clusters": decisions,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Total clusters: {len(decisions)}")
    print(f"Accepted clusters: {len(accepted)}")
    print(f"Rejected clusters: {len(rejected)}")
    print(f"Acceptance rate: {len(accepted) / len(decisions):.3f}" if decisions else "Acceptance rate: 0.000")
    print(f"Rejection count by reason: {dict(reason_counts)}")
    print(f"Saved filtered predictions: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
