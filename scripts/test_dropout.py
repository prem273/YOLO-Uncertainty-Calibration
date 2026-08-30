"""Verify that Phase 3 head-level MC dropout causes real YOLO predictions to vary."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.dropout_yolo import DropoutYOLO


def describe(result) -> np.ndarray:
    """Print detections and return an array for numerical cross-pass comparison."""
    boxes = result.boxes
    if len(boxes) == 0:
        print("  detections: 0")
        return np.empty((0, 6), dtype=np.float32)
    values = np.column_stack((
        boxes.cls.cpu().numpy(),
        boxes.conf.cpu().numpy(),
        boxes.xyxy.cpu().numpy(),
    ))
    print(f"  detections: {len(values)}")
    for index, (class_id, confidence, x1, y1, x2, y2) in enumerate(values):
        print(f"  {index}: class={int(class_id)}, conf={confidence:.6f}, xyxy=({x1:.2f}, {y1:.2f}, {x2:.2f}, {y2:.2f})")
    return values


def exact_difference(a: np.ndarray, b: np.ndarray) -> tuple[bool, float]:
    """Compare real post-NMS detections without synthesizing any values."""
    if a.shape != b.shape:
        return True, float("inf")
    if a.size == 0:
        return False, 0.0
    return not np.array_equal(a, b), float(np.max(np.abs(a - b)))


def main() -> int:
    with open(ROOT / "config/config.yaml", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    passes = int(config["mc_inference"]["passes"])
    baseline = config["baseline"]
    image = ROOT / "bus.jpg"

    model = DropoutYOLO(
        weights=ROOT / "yolov8n.pt",
        config_path=ROOT / "config/config.yaml",
        device="cpu",
    )
    assert len(model.dropout_layers) == 6, "Expected one dropout layer per cv2/cv3 scale branch."
    assert model.backbone_is_frozen, "Backbone model.model[0:10] was not frozen."
    print(f"Dropout layers: {len(model.dropout_layers)}, p={model.dropout_probability}")
    print(f"Backbone frozen: {model.backbone_is_frozen}")

    predict_args = dict(conf=baseline["conf_threshold"], imgsz=baseline["imgsz"], verbose=False)

    print("\nDeterministic inference (two identical passes):")
    deterministic_a = describe(model.predict(image, stochastic=False, **predict_args)[0])
    deterministic_b = describe(model.predict(image, stochastic=False, **predict_args)[0])
    deterministic_changed, deterministic_delta = exact_difference(deterministic_a, deterministic_b)
    print(f"Deterministic comparison: changed={deterministic_changed}, max_abs_delta={deterministic_delta:.8f}")
    if deterministic_changed:
        raise AssertionError("Deterministic inference produced unequal predictions.")

    print(f"\nStochastic inference ({passes} passes on the same image):")
    stochastic = []
    for pass_index in range(passes):
        print(f"Pass {pass_index + 1}:")
        stochastic.append(describe(model.predict(image, stochastic=True, **predict_args)[0]))

    comparisons = []
    for index in range(1, len(stochastic)):
        changed, max_delta = exact_difference(stochastic[0], stochastic[index])
        comparisons.append(changed)
        print(f"Pass 1 vs pass {index + 1}: changed={changed}, max_abs_delta={max_delta:.8f}")
    if not any(comparisons):
        raise AssertionError("No stochastic predictions differed; inserted dropout was not active.")

    print("\nPASS: deterministic inference is stable and real stochastic YOLO predictions vary.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
