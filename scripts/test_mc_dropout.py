"""Smoke-test Phase 4 true-batched MC-dropout inference."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from inference.mc_dropout import MCDropoutInference


def compact_array(prediction) -> np.ndarray:
    """Numerical representation of retained real detections for comparison."""
    return np.array(
        [[d.class_id, d.confidence, *d.box_xyxy] for d in prediction.detections], dtype=np.float32
    ).reshape((-1, 6))


def differs(first: np.ndarray, other: np.ndarray) -> bool:
    return first.shape != other.shape or not np.array_equal(first, other)


def main() -> int:
    image = ROOT / "bus.jpg"
    config_path = ROOT / "config/config.yaml"
    output_dir = ROOT / "results/predictions"
    output_dir.mkdir(parents=True, exist_ok=True)

    runner = MCDropoutInference(
        config_path=config_path, weights=ROOT / "yolov8n.pt", device="cpu"
    )
    assert runner.model.backbone_is_frozen, "The Phase 3 backbone freeze was not retained."
    print(f"Backbone frozen: {runner.model.backbone_is_frozen}")
    # T comes from configuration (5 by default); it is intentionally not hard-coded in the module.
    run = runner.run(image)
    output_path = output_dir / "mc_dropout_predictions.json"
    output_path.write_text(json.dumps(run.to_dict(), indent=2), encoding="utf-8")

    print(f"Batching: {run.batching_mode} (batch_size={run.batch_size})")
    print(f"Device: {run.device}")
    print(f"T={run.passes_requested}")
    for prediction in run.passes:
        print(
            f"Pass {prediction.pass_id}: detections={prediction.total_detections}, "
            f"mean_confidence={prediction.mean_confidence:.6f}, "
            f"allocated_time_ms={prediction.inference_time_ms:.2f}"
        )
    print(f"Total inference time: {run.total_inference_time_ms:.2f} ms")
    print(f"Average time per pass: {run.average_time_per_pass_ms:.2f} ms")
    print(f"Raw predictions: {output_path}")

    arrays = [compact_array(prediction) for prediction in run.passes]
    changed = [differs(arrays[0], candidate) for candidate in arrays[1:]]
    print(f"Stochastic variation vs pass 1: {changed}")
    if not any(changed):
        raise AssertionError("All stochastic pass predictions were identical.")

    # Deterministic DropoutYOLO inference must equal the ordinary pretrained baseline.
    deterministic = runner.model.predict(image, stochastic=False, conf=0.25, imgsz=640, verbose=False)[0]
    baseline = YOLO(str(ROOT / "yolov8n.pt")).predict(image, conf=0.25, imgsz=640, verbose=False)[0]
    deterministic_boxes = deterministic.boxes.data.cpu().numpy()
    baseline_boxes = baseline.boxes.data.cpu().numpy()
    # The inserted eval-mode dropout is an identity. A separately loaded model
    # can differ by tiny CPU floating-point fusion roundoff, so use a strict
    # numerical tolerance rather than byte equality.
    same_baseline = deterministic_boxes.shape == baseline_boxes.shape and np.allclose(
        deterministic_boxes, baseline_boxes, rtol=0.0, atol=1e-3
    )
    baseline_max_delta = (
        float(np.max(np.abs(deterministic_boxes - baseline_boxes)))
        if deterministic_boxes.shape == baseline_boxes.shape
        else float("inf")
    )
    print(
        f"Deterministic comparison: dropout_model_detections={len(deterministic.boxes)}, "
        f"baseline_detections={len(baseline.boxes)}, within_1e-3={same_baseline}, "
        f"max_abs_delta={baseline_max_delta:.8f}"
    )
    if not same_baseline:
        raise AssertionError("Deterministic dropout model no longer matches the baseline model.")

    print("PASS: batched stochastic variation retained; deterministic baseline behavior is available.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
