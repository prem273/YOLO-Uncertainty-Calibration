# Phase 5 greedy IoU clustering

`greedy_iou_clustering` consumes the retained Phase 4 per-pass detections in
their supplied order. For each detection, it considers only existing clusters
with the same `class_id`; different semantic classes are never spatially
associated, even if their boxes overlap. This prevents, for example, a person
and a car from becoming one cluster.

Among compatible clusters whose IoU with the current mean representative box
is at least the configured threshold, the highest-IoU cluster receives the
detection. Equal IoUs retain cluster creation order, making the output
deterministic. Otherwise a new cluster is created. Individual member boxes and
their pass IDs/confidences remain in the result. The representative box is the
unweighted mean coordinate box; it is used only for greedy association and is
not a replacement for stored member detections.

Persistence is currently `unique_passes / T`, where `T` is the number of
stochastic passes in the Phase 4 prediction file. No uncertainty metric or
filtering is performed here.
