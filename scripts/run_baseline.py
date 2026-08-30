"""
Baseline Inference Script for YOLO Uncertainty Calibration Project

This script:
1. Loads pretrained YOLOv8n model
2. Runs deterministic inference on a sample image
3. Prints detected classes, confidence scores, and bounding boxes
4. Measures inference time
5. Saves annotated output image

No MC Dropout, no uncertainty calculations at this stage.
"""

import os
import sys
import time
import yaml
import torch
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

def load_config(config_path='config/config.yaml'):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def setup_output_dirs(output_dir):
    """Create output directories if they don't exist."""
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'annotated_images'), exist_ok=True)

def load_model(model_name, device='auto'):
    """Load pretrained YOLO model."""
    print(f"\n{'='*60}")
    print("LOADING MODEL")
    print(f"{'='*60}")
    print(f"Model: {model_name}")
    print(f"Device: {device}")
    
    model = YOLO(f'{model_name}.pt')
    
    # Move to device
    if device.lower() == 'auto':
        device = 0 if torch.cuda.is_available() else 'cpu'
    
    model = model.to(device)
    print(f"✓ Model loaded successfully")
    print(f"Model parameters: {sum(p.numel() for p in model.model.parameters()):,}")
    
    return model

def run_inference(model, image_source, conf_threshold=0.25, imgsz=640):
    """Run inference on an image."""
    print(f"\n{'='*60}")
    print("RUNNING INFERENCE")
    print(f"{'='*60}")
    print(f"Image source: {image_source}")
    print(f"Confidence threshold: {conf_threshold}")
    print(f"Image size: {imgsz}")
    
    # Record inference time
    start_time = time.time()
    results = model(image_source, conf=conf_threshold, imgsz=imgsz, verbose=False)
    inference_time = (time.time() - start_time) * 1000  # Convert to milliseconds
    
    print(f"✓ Inference completed in {inference_time:.2f} ms")
    
    return results[0], inference_time

def print_detections(result):
    """Print detected objects with their classes and confidence scores."""
    print(f"\n{'='*60}")
    print("DETECTION RESULTS")
    print(f"{'='*60}")
    
    detections = result.boxes
    num_detections = len(detections)
    
    print(f"Number of detections: {num_detections}\n")
    
    if num_detections == 0:
        print("No objects detected.")
        return
    
    print(f"{'ID':<4} {'Class':<20} {'Confidence':<12} {'Bounding Box (xyxy)':<40}")
    print("-" * 80)
    
    for idx, (box, conf, cls_id) in enumerate(zip(
        detections.xyxy.cpu().numpy(),
        detections.conf.cpu().numpy(),
        detections.cls.cpu().numpy()
    )):
        class_name = result.names[int(cls_id)]
        x1, y1, x2, y2 = box
        print(f"{idx+1:<4} {class_name:<20} {conf:<12.4f} ({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f})")
    
    print()

def save_annotated_image(result, output_path):
    """Save annotated image with detections."""
    annotated_image = result.plot()
    cv2.imwrite(output_path, annotated_image)
    print(f"✓ Annotated image saved to: {output_path}")
    
    return annotated_image

def main():
    """Main execution function."""
    print("\n" + "="*60)
    print("BASELINE INFERENCE - YOLO UNCERTAINTY CALIBRATION")
    print("="*60)
    
    # Check Python and PyTorch versions
    print(f"\nPython version: {sys.version.split()[0]}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Load configuration
    config = load_config('config/config.yaml')
    baseline_config = config.get('baseline', {})
    output_config = config.get('output', {})
    
    # Setup output directories
    output_dir = output_config.get('output_dir', 'results/')
    setup_output_dirs(output_dir)
    
    try:
        # Load model
        model = load_model(
            baseline_config.get('model', 'yolov8n'),
            baseline_config.get('device', 'auto')
        )
        
        # Use Ultralytics sample image (bus.jpg)
        # This will download automatically from Ultralytics
        image_source = 'https://ultralytics.com/images/bus.jpg'
        
        # Run inference
        result, inference_time = run_inference(
            model,
            image_source,
            conf_threshold=baseline_config.get('conf_threshold', 0.25),
            imgsz=baseline_config.get('imgsz', 640)
        )
        
        # Print detections
        print_detections(result)
        
        # Save annotated image
        if output_config.get('visualize', True):
            output_image_path = os.path.join(
                output_dir,
                'annotated_images',
                'baseline_inference.jpg'
            )
            annotated_image = save_annotated_image(result, output_image_path)
        
        # Summary
        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        print(f"Model: {baseline_config.get('model', 'yolov8n')}")
        print(f"Inference time: {inference_time:.2f} ms")
        print(f"Number of detections: {len(result.boxes)}")
        print(f"Output saved to: {output_dir}")
        print(f"✓ Baseline inference completed successfully!")
        print(f"{'='*60}\n")
        
        return 0
        
    except Exception as e:
        print(f"\n✗ Error during inference: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    sys.exit(main())
