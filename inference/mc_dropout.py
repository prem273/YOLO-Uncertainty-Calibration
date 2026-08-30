"""Batched stochastic MC-dropout inference for the existing DropoutYOLO model.

For each chunk, one ``YOLO.predict`` call receives a list containing repeated
references to the same source image. Ultralytics preprocesses that list into a
single N-image tensor and invokes the underlying PyTorch model once. Since the
six head dropout modules remain in training mode, each tensor sample receives
its own dropout mask. Results are then retained separately by pass ID.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import yaml

from models.dropout_yolo import DropoutYOLO


@dataclass
class DetectionPrediction:
    """One post-NMS YOLO detection from one stochastic pass."""

    box_xyxy: list[float]
    class_id: int
    class_name: str | None
    confidence: float
    # YOLO Detect postprocessing exposes only the selected class/confidence;
    # complete class probability vectors are not available from Results.boxes.
    class_probabilities: None = None


@dataclass
class PassPrediction:
    """All retained predictions belonging to one MC pass."""

    pass_id: int
    image_id: str
    detections: list[DetectionPrediction]
    inference_time_ms: float

    @property
    def total_detections(self) -> int:
        return len(self.detections)

    @property
    def mean_confidence(self) -> float:
        if not self.detections:
            return 0.0
        return float(np.mean([detection.confidence for detection in self.detections]))

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["total_detections"] = self.total_detections
        data["mean_confidence"] = self.mean_confidence
        return data


@dataclass
class MCInferenceResult:
    """A full sequence of retained stochastic predictions."""

    image_id: str
    device: str
    passes_requested: int
    batch_size: int
    batching_mode: str
    total_inference_time_ms: float
    passes: list[PassPrediction]

    @property
    def average_time_per_pass_ms(self) -> float:
        return self.total_inference_time_ms / len(self.passes) if self.passes else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "image_id": self.image_id,
            "device": self.device,
            "passes_requested": self.passes_requested,
            "batch_size": self.batch_size,
            "batching_mode": self.batching_mode,
            "total_inference_time_ms": self.total_inference_time_ms,
            "average_time_per_pass_ms": self.average_time_per_pass_ms,
            "passes": [prediction.to_dict() for prediction in self.passes],
        }


class MCDropoutInference:
    """Run and retain batched head-level MC-dropout predictions."""

    def __init__(
        self,
        model: DropoutYOLO | None = None,
        config_path: str | Path = "config/config.yaml",
        weights: str | Path = "yolov8n.pt",
        device: str | int | None = None,
    ) -> None:
        config_path = Path(config_path)
        with config_path.open(encoding="utf-8") as config_file:
            self.config = yaml.safe_load(config_file) or {}
        self.config_path = config_path
        self.mc_config = self.config.get("mc_inference", {})
        self.baseline_config = self.config.get("baseline", {})
        self.model = model or DropoutYOLO(weights=weights, config_path=config_path, device=device)

    @staticmethod
    def _source_id(source: Any) -> str:
        return Path(source).name if isinstance(source, (str, Path)) else "in_memory_image"

    @staticmethod
    def _extract_pass(result: Any, pass_id: int, image_id: str, elapsed_ms: float) -> PassPrediction:
        detections: list[DetectionPrediction] = []
        names = result.names
        for box, confidence, class_id in zip(
            result.boxes.xyxy.cpu().numpy(),
            result.boxes.conf.cpu().numpy(),
            result.boxes.cls.cpu().numpy(),
        ):
            class_index = int(class_id)
            detections.append(
                DetectionPrediction(
                    box_xyxy=[float(value) for value in box],
                    class_id=class_index,
                    class_name=names.get(class_index) if isinstance(names, dict) else names[class_index],
                    confidence=float(confidence),
                )
            )
        return PassPrediction(pass_id, image_id, detections, elapsed_ms)

    def run(
        self,
        source: Any,
        passes: int | None = None,
        batch_size: int | None = None,
        **predict_overrides: Any,
    ) -> MCInferenceResult:
        """Run T MC passes using true image batches, returning all raw pass results."""
        passes = int(passes if passes is not None else self.mc_config.get("passes", 5))
        batch_size = int(batch_size if batch_size is not None else self.mc_config.get("batch_size", passes))
        if passes < 1 or batch_size < 1:
            raise ValueError("passes and batch_size must both be positive integers.")

        predict_args = {
            "conf": self.baseline_config.get("conf_threshold", 0.25),
            "imgsz": self.baseline_config.get("imgsz", 640),
            "verbose": False,
            **predict_overrides,
        }
        image_id = self._source_id(source)
        collected: list[PassPrediction] = []
        total_elapsed_ms = 0.0

        for start in range(0, passes, batch_size):
            current_batch_size = min(batch_size, passes - start)
            # A source list causes Ultralytics to make a single current_batch_size-image tensor.
            sources = [source] * current_batch_size
            started = perf_counter()
            results = self.model.predict(sources, stochastic=True, **predict_args)
            elapsed_ms = (perf_counter() - started) * 1000
            total_elapsed_ms += elapsed_ms
            if len(results) != current_batch_size:
                raise RuntimeError(f"Expected {current_batch_size} batch results, received {len(results)}.")
            per_pass_elapsed_ms = elapsed_ms / current_batch_size
            collected.extend(
                self._extract_pass(result, start + offset + 1, image_id, per_pass_elapsed_ms)
                for offset, result in enumerate(results)
            )

        model_device = str(next(self.model.model.parameters()).device)
        return MCInferenceResult(
            image_id=image_id,
            device=model_device,
            passes_requested=passes,
            batch_size=batch_size,
            batching_mode="true_batched_stochastic_inference",
            total_inference_time_ms=total_elapsed_ms,
            passes=collected,
        )
