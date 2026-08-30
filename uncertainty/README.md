# Phase 6 uncertainty metrics

The MC pipeline now captures the actual 80 per-class sigmoid score vector from
the YOLO Detect output for every candidate retained by NMS. It does not derive
class probabilities from detection confidence. For each cluster, these vectors
are averaged across members and normalized by their sum to form the categorical
`class_probability_distribution` used for entropy. Shannon entropy is
`H = -sum(p * log(p))`, ignoring zero terms. With normalization enabled in the
configuration, `normalized_semantic_entropy = H / log(C)`, where `C` is the
length of this YOLO class-score vector (80 for this COCO YOLOv8n model).

Coordinate variance uses population variance (`ddof=0`) over member boxes:
`localization_variance = mean(Var(x1), Var(y1), Var(x2), Var(y2))`. These are
raw pixel-coordinate variances; they are not normalized. Persistence is
computed from unique stored pass IDs: `len(unique_passes) / T`.
