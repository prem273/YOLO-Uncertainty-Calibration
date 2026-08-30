"""Standard-library tests for configuration-driven uncertainty filtering."""

import copy
import unittest

from filtering.uncertainty_filter import apply_uncertainty_gate


CONFIG = {
    "enabled": True,
    "entropy": {"enabled": True, "max_normalized_entropy": 0.2},
    "localization": {"enabled": True, "max_variance": 10.0},
    "persistence": {"enabled": True, "min_persistence": 0.8},
}


def cluster(**updates):
    value = {
        "cluster_id": 1,
        "class_id": 0,
        "class_name": "person",
        "representative_bbox": [0.0, 0.0, 10.0, 10.0],
        "mean_confidence": 0.9,
        "semantic_entropy": 0.1,
        "normalized_semantic_entropy": 0.1,
        "localization_variance": 5.0,
        "persistence": 0.9,
    }
    value.update(updates)
    return value


class TestUncertaintyFilter(unittest.TestCase):
    def test_reliable_cluster_is_accepted(self):
        decision = apply_uncertainty_gate(cluster(), CONFIG)
        self.assertTrue(decision["accepted"])
        self.assertEqual(decision["rejection_reasons"], [])

    def test_high_entropy_is_rejected(self):
        decision = apply_uncertainty_gate(cluster(normalized_semantic_entropy=0.21), CONFIG)
        self.assertFalse(decision["accepted"])
        self.assertEqual(decision["rejection_reasons"], ["normalized_entropy_exceeds_max"])

    def test_high_localization_variance_is_rejected(self):
        decision = apply_uncertainty_gate(cluster(localization_variance=10.01), CONFIG)
        self.assertEqual(decision["rejection_reasons"], ["localization_variance_exceeds_max"])

    def test_low_persistence_is_rejected(self):
        decision = apply_uncertainty_gate(cluster(persistence=0.79), CONFIG)
        self.assertEqual(decision["rejection_reasons"], ["persistence_below_min"])

    def test_multiple_failures_preserve_all_reasons(self):
        decision = apply_uncertainty_gate(
            cluster(normalized_semantic_entropy=0.3, localization_variance=12.0, persistence=0.7), CONFIG
        )
        self.assertEqual(
            decision["rejection_reasons"],
            ["normalized_entropy_exceeds_max", "localization_variance_exceeds_max", "persistence_below_min"],
        )

    def test_disabled_condition_does_not_affect_acceptance(self):
        config = copy.deepcopy(CONFIG)
        config["entropy"]["enabled"] = False
        decision = apply_uncertainty_gate(cluster(normalized_semantic_entropy=0.99), config)
        self.assertTrue(decision["accepted"])

    def test_boundary_values_are_accepted(self):
        decision = apply_uncertainty_gate(
            cluster(normalized_semantic_entropy=0.2, localization_variance=10.0, persistence=0.8), CONFIG
        )
        self.assertTrue(decision["accepted"])


if __name__ == "__main__":
    unittest.main()
