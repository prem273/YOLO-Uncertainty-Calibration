# Phase 4 batched MC-dropout inference

`MCDropoutInference.run()` receives one image source and expands it into a list
of repeated sources for each batch chunk. Passing that list to
`DropoutYOLO.predict(..., stochastic=True)` makes Ultralytics preprocess the
copies into one batched tensor and execute one PyTorch forward pass for that
chunk. It is therefore true batched stochastic inference, not T sequential
single-image calls.

The `DropoutYOLO` instance supplies the existing Phase 3 model adaptation. Its
six head dropout layers are set to training mode for the complete batch, while
the rest of the model stays in evaluation mode. PyTorch dropout independently
samples a mask for every batch element. Ultralytics returns one post-NMS
`Results` object per image; all detections are retained in one `PassPrediction`
per stochastic pass.

If T exceeds `mc_inference.batch_size`, execution uses multiple true batches
(for example T=7 and batch size 5 uses batches of 5 and 2), not a sequential
fallback. The raw results contain selected class IDs/names and confidences.
Full class probability vectors are unavailable from Ultralytics post-NMS
`Results.boxes`, so `class_probabilities` is explicitly stored as `null`.
