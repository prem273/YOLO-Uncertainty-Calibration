"""Standard-library unit tests for Phase 6 uncertainty metrics."""

import math
import unittest

from uncertainty.uncertainty_metrics import (
    calculate_cluster_uncertainty,
    calculate_entropy,
    calculate_localization_variance,
    calculate_persistence,
)


class TestUncertaintyMetrics(unittest.TestCase):
    def test_deterministic_distribution_entropy_is_zero(self):
        self.assertEqual(calculate_entropy([1, 0, 0]), 0.0)

    def test_uniform_distribution_normalized_entropy_is_one(self):
        entropy = calculate_entropy([0.25, 0.25, 0.25, 0.25])
        self.assertAlmostEqual(entropy / math.log(4), 1.0)

    def test_zero_probabilities_are_safe(self):
        entropy = calculate_entropy([0, 0.5, 0.5, 0])
        self.assertTrue(math.isfinite(entropy))

    def test_identical_boxes_have_zero_variance(self):
        metrics = calculate_localization_variance([[1, 2, 3, 4], [1, 2, 3, 4]])
        self.assertEqual(metrics["localization_variance"], 0.0)

    def test_different_boxes_have_positive_variance(self):
        metrics = calculate_localization_variance([[0, 0, 10, 10], [2, 4, 14, 18]])
        self.assertGreater(metrics["localization_variance"], 0.0)

    def test_persistence(self):
        self.assertEqual(calculate_persistence([1, 2, 3, 4, 5], 5), 1.0)
        self.assertAlmostEqual(calculate_persistence([1, 3, 5], 5), 0.6)

    def test_combined_cluster_uncertainty(self):
        cluster = {
            "cluster_id": 1,
            "class_id": 0,
            "class_name": "person",
            "representative_bbox": [0.5, 0.5, 10.5, 10.5],
            "members": [
                {"pass_id": 1, "bbox": [0, 0, 10, 10], "confidence": 0.9, "class_probabilities": [0.8, 0.2]},
                {"pass_id": 3, "bbox": [1, 1, 11, 11], "confidence": 0.7, "class_probabilities": [0.6, 0.4]},
            ],
        }
        metrics = calculate_cluster_uncertainty(cluster, total_passes=5)
        self.assertEqual(metrics["persistence"], 0.4)
        self.assertTrue(math.isfinite(metrics["semantic_entropy"]))
        self.assertGreater(metrics["localization_variance"], 0.0)
        self.assertAlmostEqual(sum(metrics["class_probability_distribution"]), 1.0)


if __name__ == "__main__":
    unittest.main()
