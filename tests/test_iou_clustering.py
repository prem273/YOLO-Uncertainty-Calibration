"""Standard-library unit tests for Phase 5 greedy IoU clustering."""

import unittest

from clustering.iou_clustering import calculate_iou, greedy_iou_clustering


def prediction(*passes):
    return {"passes_requested": len(passes), "passes": list(passes)}


def pass_with(pass_id, *detections):
    return {"pass_id": pass_id, "detections": list(detections)}


def detection(box, class_id=0, confidence=0.9, class_name="person"):
    return {"box_xyxy": box, "class_id": class_id, "class_name": class_name, "confidence": confidence}


class TestIoUClustering(unittest.TestCase):
    def test_identical_boxes_have_iou_one(self):
        self.assertEqual(calculate_iou([0, 0, 10, 10], [0, 0, 10, 10]), 1.0)

    def test_separate_boxes_have_iou_zero(self):
        self.assertEqual(calculate_iou([0, 0, 10, 10], [20, 20, 30, 30]), 0.0)

    def test_partially_overlapping_boxes_have_correct_iou(self):
        self.assertAlmostEqual(calculate_iou([0, 0, 10, 10], [5, 5, 15, 15]), 25 / 175)

    def test_zero_area_boxes_are_safe(self):
        self.assertEqual(calculate_iou([0, 0, 0, 10], [0, 0, 10, 10]), 0.0)
        self.assertEqual(calculate_iou([2, 2, 2, 2], [2, 2, 2, 2]), 0.0)

    def test_same_class_high_iou_forms_one_cluster(self):
        clusters = greedy_iou_clustering(prediction(
            pass_with(1, detection([0, 0, 10, 10])), pass_with(2, detection([1, 0, 11, 10]))
        ))
        self.assertEqual(len(clusters), 1)
        self.assertEqual(clusters[0]["num_members"], 2)

    def test_same_class_low_iou_forms_separate_clusters(self):
        clusters = greedy_iou_clustering(prediction(
            pass_with(1, detection([0, 0, 10, 10])), pass_with(2, detection([6, 0, 16, 10]))
        ))
        self.assertEqual(len(clusters), 2)

    def test_different_classes_never_cluster_together(self):
        clusters = greedy_iou_clustering(prediction(
            pass_with(1, detection([0, 0, 10, 10], class_id=0)),
            pass_with(2, detection([0, 0, 10, 10], class_id=2, class_name="car")),
        ))
        self.assertEqual(len(clusters), 2)

    def test_multiple_passes_group_corresponding_detections_and_persistence(self):
        clusters = greedy_iou_clustering(prediction(
            pass_with(1, detection([0, 0, 10, 10])),
            pass_with(2, detection([1, 0, 11, 10])),
            pass_with(3, detection([0, 1, 10, 11])),
        ))
        self.assertEqual(len(clusters), 1)
        cluster = clusters[0]
        self.assertEqual(cluster["unique_passes"], [1, 2, 3])
        self.assertEqual(cluster["persistence"], 1.0)
        for observed, expected in zip(cluster["representative_bbox"], [1 / 3, 1 / 3, 31 / 3, 31 / 3]):
            self.assertAlmostEqual(observed, expected)


if __name__ == "__main__":
    unittest.main()
