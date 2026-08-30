"""Project-local YOLOv8n head-level Monte Carlo dropout support.

This module changes only an in-memory model instance.  It never modifies the
installed Ultralytics package or the pretrained checkpoint on disk.
"""

from pathlib import Path
from typing import Any

import torch.nn as nn
import yaml
from ultralytics import YOLO


class DropoutYOLO:
    """YOLOv8n with dropout immediately before each Detect projection layer."""

    def __init__(
        self,
        weights: str | Path = "yolov8n.pt",
        config_path: str | Path = "config/config.yaml",
        device: str | int | None = None,
    ) -> None:
        with open(config_path, "r", encoding="utf-8") as config_file:
            config = yaml.safe_load(config_file) or {}

        dropout_config = config.get("dropout", {})
        self.dropout_enabled = bool(dropout_config.get("enabled", True))
        self.dropout_probability = float(dropout_config.get("probability", 0.1))
        if not 0.0 <= self.dropout_probability < 1.0:
            raise ValueError("dropout.probability must be in [0.0, 1.0).")

        self.yolo = YOLO(str(weights))
        if device is not None:
            self.yolo.to(device)

        self.model = self.yolo.model
        self.detect_head = self._get_detect_head()
        self._insert_head_dropout()
        self._freeze_backbone()
        self.set_deterministic_mode()

    def _get_detect_head(self) -> nn.Module:
        """Return the verified YOLOv8 Detect module at model.model[22]."""
        head = self.model.model[-1]
        if not hasattr(head, "cv2") or not hasattr(head, "cv3"):
            raise RuntimeError("Expected the final YOLO module to expose cv2 and cv3 branches.")
        return head

    def _insert_head_dropout(self) -> None:
        """Insert dropout once before every scale's final cv2/cv3 Conv2d."""
        if not self.dropout_enabled:
            return

        for branch_name in ("cv2", "cv3"):
            for branch in getattr(self.detect_head, branch_name):
                if any(isinstance(layer, nn.Dropout) for layer in branch):
                    continue
                projection_index = len(branch) - 1
                projection = branch[projection_index]
                if not isinstance(projection, nn.Conv2d):
                    raise RuntimeError(
                        f"Expected {branch_name} final layer to be Conv2d, got {type(projection).__name__}."
                    )
                branch.insert(projection_index, nn.Dropout(p=self.dropout_probability))

    def _freeze_backbone(self) -> None:
        """Freeze verified backbone layers model.model[0:10]; leave neck/head trainable."""
        for layer in self.model.model[:10]:
            for parameter in layer.parameters():
                parameter.requires_grad = False

    @property
    def dropout_layers(self) -> list[nn.Dropout]:
        """Return precisely the six project-inserted head dropout layers."""
        return [
            layer
            for branch_name in ("cv2", "cv3")
            for branch in getattr(self.detect_head, branch_name)
            for layer in branch
            if isinstance(layer, nn.Dropout)
        ]

    @property
    def backbone_is_frozen(self) -> bool:
        return all(
            not parameter.requires_grad
            for layer in self.model.model[:10]
            for parameter in layer.parameters()
        )

    def set_deterministic_mode(self) -> None:
        """Run the entire model in evaluation mode, including inserted dropout."""
        self.model.eval()

    def set_stochastic_mode(self) -> None:
        """Keep batch norm deterministic while enabling only inserted dropout layers."""
        self.model.eval()
        for dropout in self.dropout_layers:
            dropout.train()

    def _ensure_predictor(self) -> None:
        """Construct Ultralytics' predictor before selecting a dropout mode.

        Predictor setup fuses eligible layers and calls ``eval()`` on the passed
        model. Constructing it lazily *after* ``set_stochastic_mode`` would
        therefore silently disable dropout for the first prediction call.
        """
        if self.yolo.predictor is not None:
            return
        overrides = {
            **self.yolo.overrides,
            "conf": 0.25,
            "batch": 1,
            "save": False,
            "mode": "predict",
            "rect": True,
            "embed": None,
        }
        predictor_type = self.yolo._smart_load("predictor")
        self.yolo.predictor = predictor_type(overrides=overrides, _callbacks=self.yolo.callbacks)
        self.yolo.predictor.setup_model(model=self.model, verbose=False)

    def predict(
        self, source: Any, stochastic: bool = False, capture_class_probabilities: bool = False, **kwargs: Any
    ):
        """Run prediction, optionally retaining real Detect-head class score vectors."""
        self._ensure_predictor()
        if stochastic:
            self.set_stochastic_mode()
        else:
            self.set_deterministic_mode()
        captured: list[Any] = []

        def capture_head_output(_module: nn.Module, _inputs: Any, output: Any) -> None:
            # Detect returns (decoded_predictions, feature_maps) for normal inference.
            decoded = output[0] if isinstance(output, tuple) else output
            captured.append(decoded.detach().clone())

        hook = self.detect_head.register_forward_hook(capture_head_output) if capture_class_probabilities else None
        try:
            results = self.yolo.predict(source=source, **kwargs)
        finally:
            if hook is not None:
                hook.remove()
        if capture_class_probabilities:
            if len(captured) != 1:
                raise RuntimeError(f"Expected one Detect forward output, captured {len(captured)}.")
            return results, captured[0]
        return results


def load_dropout_yolo(**kwargs: Any) -> DropoutYOLO:
    """Convenience factory for the Phase 3 model."""
    return DropoutYOLO(**kwargs)
