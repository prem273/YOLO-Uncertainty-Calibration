# Verified YOLOv8n architecture (Phase 2)

Inspection was performed on the project's local `yolov8n.pt` with
Ultralytics **8.4.135**, rather than reconstructed from a reference model.
The loaded object is `ultralytics.nn.tasks.DetectionModel`; its module list is
`model.model` and contains 23 modules (indices `0` through `22`).

## Backbone

The backbone is `model.model[0:10]` (indices `0`–`9`):

| Paths | Actual layer types |
| --- | --- |
| `model.model[0]`, `[1]`, `[3]`, `[5]`, `[7]` | `ultralytics.nn.modules.conv.Conv` |
| `model.model[2]`, `[4]`, `[6]`, `[8]` | `ultralytics.nn.modules.block.C2f` |
| `model.model[9]` | `ultralytics.nn.modules.block.SPPF` |

Phase 3 freezes parameters in exactly these modules (`model.model[:10]`).

## Neck

The neck is `model.model[10:22]` (indices `10`–`21`):

| Paths | Actual layer types |
| --- | --- |
| `[10]`, `[13]` | `torch.nn.Upsample` |
| `[11]`, `[14]`, `[17]`, `[20]` | `ultralytics.nn.modules.conv.Concat` |
| `[12]`, `[15]`, `[18]`, `[21]` | `ultralytics.nn.modules.block.C2f` |
| `[16]`, `[19]` | `ultralytics.nn.modules.conv.Conv` |

## Detection head and branch projections

The detection head is `model.model[22]`, an
`ultralytics.nn.modules.head.Detect` module with feature inputs from
`[15, 18, 21]`. Its branches are:

| Function | Exact path before Phase 3 | Branch structure | Final pretrained projection |
| --- | --- | --- | --- |
| Regression | `model.model[22].cv2` (`ModuleList`) | scales `[0]`, `[1]`, `[2]`, each `Sequential(Conv, Conv, Conv2d)` | `cv2[0][2]`, `cv2[1][2]`, `cv2[2][2]`: `Conv2d(64, 64, 1×1)` |
| Classification | `model.model[22].cv3` (`ModuleList`) | scales `[0]`, `[1]`, `[2]`, each `Sequential(Conv, Conv, Conv2d)` | `cv3[0][2]`, `cv3[1][2]`, `cv3[2][2]`: `Conv2d(80, 80, 1×1)` |

The first `cv2` convolution receives 64, 128, and 256 channels at the three
scales. The first `cv3` convolution receives the same inputs; each branch then
uses 80 channels for COCO classification. `Detect.dfl` is present separately,
but is not a `cv2` or `cv3` branch projection and is not changed.

## Phase 3 insertion points

`models/dropout_yolo.py` inserts `torch.nn.Dropout(p=0.1)` immediately before
every listed 1×1 final projection. After insertion the paths are
`cv2[i][2]` → `Dropout`, `cv2[i][3]` → original `Conv2d`, and likewise
`cv3[i][2]` → `Dropout`, `cv3[i][3]` → original `Conv2d`, for `i = 0, 1, 2`.
The original projection modules and their pretrained tensors are retained in
place; no Ultralytics source files or checkpoint weights are modified.

These points implement head-level MC dropout exactly at the activations fed
into each regression and classification output projection. The backbone and
neck are untouched, so the injected stochasticity is confined to the detection
head immediately before its requested output layers. Deterministic mode places
the entire model in evaluation mode. Stochastic mode keeps the model (and thus
BatchNorm) in evaluation mode and enables only these six inserted dropout
layers. The project-local wrapper constructs the Ultralytics predictor before
selecting either mode, because predictor initialization calls `eval()`; this
ensures dropout is active even for the first stochastic prediction.
