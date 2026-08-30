# Drop-In Uncertainty Calibration for Ultra-Lightweight YOLO Detectors Under Natural Corruptions

## Project Overview

This research project implements uncertainty calibration mechanisms for ultra-lightweight YOLO detectors (YOLOv8n/YOLOv11n) to improve robustness and reliability under natural corruptions.

## Key Technical Specifications

### Model Architecture
- **Base Model**: YOLOv8n / YOLOv11n
- **Head-level Monte Carlo Dropout**: Enabled in detection head only
- **Dropout Probability**: p = 0.1
- **Backbone**: Frozen (no fine-tuning)
- **Dropout Placement**: 
  - Regression branch (cv2)
  - Classification branch (cv3)

### Inference Strategy
- **Batched Stochastic Inference**: Multiple forward passes with different dropout realizations
- **Number of Stochastic Passes**: T = 5 to 10
- **Post-processing**: Greedy IoU clustering with IoU threshold ≥ 0.5

### Uncertainty Quantification
- **Semantic Epistemic Entropy**: Class prediction uncertainty
- **Spatial Localization Variance**: Bounding box coordinate uncertainty
- **Detection Persistence Score**: Consistency across stochastic passes

### Filtering Strategy
- **Uncertainty-Gated Filtering**: Remove detections with high uncertainty

### Evaluation Protocol
- **Corruption Categories**: 15 types
- **Severity Tiers**: 5 levels per category
- **Performance Metrics**: Detection Expected Calibration Error (D-ECE)

### Performance Targets
- **Inference Latency**: < 20 ms on NVIDIA T4 (must be benchmarked, not assumed)

## Project Structure

```
.
├── models/              # Pretrained and fine-tuned models
├── inference/           # Inference pipelines
├── uncertainty/         # Uncertainty quantification modules
├── clustering/          # IoU-based NMS and clustering
├── evaluation/          # Evaluation metrics (D-ECE, etc.)
├── data/                # Datasets
├── results/             # Experiment results and outputs
├── scripts/             # Execution scripts
├── config/              # Configuration files
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

## Installation

1. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/Scripts/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Baseline Verification

Run the baseline inference script to verify setup:
```bash
python scripts/run_baseline.py
```

## Important Notes

- No MC Dropout is implemented yet; this baseline uses standard deterministic inference
- The backbone is not modified; only standard YOLOv8n inference is performed
- No uncertainty calculations are performed in the baseline
- No corruption dataset has been created
- No D-ECE evaluation is implemented

## Next Steps (Future Phases)

1. Implement head-level MC Dropout
2. Modify detection head for uncertainty-aware outputs
3. Create corrupted dataset with 15 corruption types × 5 severity tiers
4. Implement uncertainty quantification (epistemic entropy, localization variance)
5. Implement Greedy IoU clustering for NMS
6. Implement uncertainty-gated filtering
7. Implement D-ECE evaluation metric
8. Benchmark latency on NVIDIA T4
9. Run comprehensive evaluation across all corruption categories

## Authors

Research Project: YOLO Uncertainty Calibration

## References

- Ultralytics YOLOv8: https://github.com/ultralytics/ultralytics
- YOLOv11: https://github.com/ultralytics/ultralytics
