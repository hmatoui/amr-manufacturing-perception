# AMR Manufacturing Perception

A computer vision project for object detection and distance estimation in manufacturing environments using YOLO models and depth estimation with DepthPro.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Dataset](#dataset)
- [Models](#models)
- [Usage](#usage)
- [Results](#results)
- [Requirements](#requirements)

## Overview

This project provides an intelligent perception system for Autonomous Mobile Robots (AMR) in manufacturing settings. It combines:

- **Object Detection**: Uses YOLO models to detect objects, people, and equipment in industrial environments
- **Depth Estimation**: Employs DepthPro for accurate distance measurement to detected objects
- **Real-time Processing**: Processes images to provide annotated outputs with bounding boxes and distance information

The system is designed to help AMRs navigate safely and make informed decisions in complex manufacturing environments.

## Features

- ✅ Object detection using state-of-the-art YOLO models (YOLOv8, YOLO11)
- ✅ Depth estimation using DepthPro for accurate distance measurements
- ✅ Support for multiple model formats (PyTorch, ONNX, TensorRT)
- ✅ Batch processing of test images
- ✅ Visualization with bounding boxes and distance labels
- ✅ GPU acceleration support
- ✅ Google Cloud Platform Vision AI integration

## Project Structure

```
amr-manufacturing-perception/
├── datasets/
│   └── sew-dataset-2025/
│       └── images/
│           ├── train/
│           │   ├── 02_yolo_rgb/          # YOLO format training annotations
│           │   ├── 02_yolo_rgb_distance/ # YOLO annotations with distance
│           │   └── 05_rgb/               # RGB training images
│           └── test/
│               ├── 02_yolo_rgb/          # YOLO format test annotations
│               ├── 02_yolo_rgb_distance/ # YOLO test annotations with distance
│               └── 05_rgb/               # RGB test images
├── models/
│   ├── depth_pro/
│   │   └── depth_pro.pt                  # DepthPro model weights
│   └── yolo/
│       ├── yolo11n.onnx                  # YOLO11 Nano ONNX format
│       ├── yolo11n.pt                    # YOLO11 Nano PyTorch weights
│       └── yolov8s.pt                    # YOLOv8 Small PyTorch weights
├── results/
│   ├── annotated_depth_images/           # Output images with depth annotations
│   ├── annotated_images/                 # Output images with detections only
│   └── depthPro_input_images/            # Intermediate processing images
├── scripts/
│   ├── object_distance_estimation.py     # Main inference script
│   └── draw_detections.py       # Draw detections given results
utilities
├── requirements.txt                      # Python dependencies
├── LICENSE
└── README.md
```

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/hmatoui/amr-manufacturing-perception.git
cd amr-manufacturing-perception
```

### 2. Python Virtual Environment Setup

#### On Windows:

```bash
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
.\venv\Scripts\activate
```

#### On Linux/macOS:

```bash
# Create a virtual environment
python3 -m venv venv

# Activate the virtual environment
source venv/bin/activate
```

### 3. Install Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install required packages
pip install -r requirements.txt
```

### 4. GPU Support (Optional but Recommended)

For NVIDIA GPU acceleration, ensure you have:
- CUDA Toolkit installed (compatible with your PyTorch version)
- cuDNN library
- TensorRT (for optimized inference)

The `requirements.txt` includes GPU-enabled packages (`onnxruntime-gpu`, `tensorrt`, `pycuda`).

## Dataset

### SEW Dataset 2025

The project uses the SEW (Smart Electronic Works) Dataset 2025, which contains industrial scene images for training and testing.

**Dataset Structure:**
- **Training Set**: Located in `datasets/sew-dataset-2025/images/train/`
  - RGB images in `05_rgb/`
  - YOLO format annotations in `02_yolo_rgb/`
  - Distance-annotated labels in `02_yolo_rgb_distance/`

- **Test Set**: Located in `datasets/sew-dataset-2025/images/test/`
  - RGB images in `05_rgb/`
  - YOLO format annotations in `02_yolo_rgb/`
  - Distance-annotated labels in `02_yolo_rgb_distance/`

**YOLO Annotation Format:**
Each `.txt` file contains bounding box annotations in YOLO format:
```
<class_id> <x_center> <y_center> <width> <height>
```
All values are normalized to [0, 1] relative to image dimensions.

## Models

### YOLO Models

The project supports multiple YOLO model variants for object detection:

1. **YOLO11 Nano** (`yolo11n.pt`, `yolo11n.onnx`)
   - Source: [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
   - Lightweight model for fast inference
   - Suitable for real-time applications

2. **YOLOv8 Small** (`yolov8s.pt`)
   - Source: [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
   - Balanced accuracy and speed
   - Better accuracy than Nano variant

**Downloading Models:**
```bash
# Using Ultralytics CLI (after installing requirements)
yolo task=detect mode=export model=yolo11n.pt format=onnx
```

Or download pre-trained weights from [Ultralytics](https://github.com/ultralytics/ultralytics).

### DepthPro Model

**DepthPro** is used for monocular depth estimation:
- Model: `depth_pro.pt`
- Source: [Apple DepthPro](https://github.com/apple/ml-depth-pro)
- Provides accurate metric depth estimation from single RGB images
- Configuration: DINOv2-L16 backbone with 384x384 patches

**Installation:**
```bash
pip install git+https://github.com/apple/ml-depth-pro.git
```

Download the pre-trained model weights and place them in `models/depth_pro/`.

## Usage

### Object Detection with Distance Estimation

The main script `object_distance_estimation.py` performs:
1. Object detection using YOLO
2. Depth estimation using DepthPro
3. Distance calculation for each detected object
4. Visualization with annotated bounding boxes and depth labels

**Run the script:**

```bash
python scripts/object_distance_estimation.py
```

**Configuration:**

Edit the paths in the script's `__main__` section:

```python
MODEL_PATH = "models/yolo/yolo11n.pt"
DepthPRO_MODEL_PATH = "models/depth_pro/depth_pro.pt"
TEST_IMAGES_PATH = "datasets/sew-dataset-2025/images/test/05_rgb/industrial"
RESULTS_PATH = "results"
```

**Parameters:**
- `MODEL_PATH`: Path to YOLO model weights (.pt or .onnx)
- `DepthPRO_MODEL_PATH`: Path to DepthPro model weights
- `TEST_IMAGES_PATH`: Directory containing test images
- `RESULTS_PATH`: Output directory for results

### Draw Detections Given Results

The `draw_detections.py` script provides utilities for drawing detections resulted from Google Cloud Platform Vision AI services.

```bash
python scripts/draw_detections.py
```

## Results

The inference pipeline generates three types of outputs in the `results/` directory:

1. **annotated_images/**: Images with object detection bounding boxes only
   - Shows detected objects with class labels
   - Confidence scores displayed

2. **depthPro_input_images/**: Intermediate images with detection boxes
   - Used as input for depth estimation
   - Green bounding boxes around detected objects

3. **annotated_depth_images/**: Final output with depth information
   - Object detection bounding boxes
   - Distance measurements in meters (e.g., "Depth: 4.52m")
   - Black background with white text for depth labels

**Example Output:**
Each detected object shows:
- Bounding box (green rectangle)
- Object class label
- Distance from camera in meters
- Confidence score

## Requirements

Key dependencies (see `requirements.txt` for complete list):

- **numpy**: Numerical computing
- **matplotlib**: Visualization
- **pandas**: Data manipulation
- **ultralytics**: YOLO models
- **torch**: PyTorch deep learning framework
- **opencv-python**: Computer vision operations
- **onnx**: ONNX format support
- **onnxruntime-gpu**: GPU-accelerated ONNX inference
- **tensorrt**: NVIDIA TensorRT optimization
- **pycuda**: CUDA Python bindings

## License

See [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) - Object detection models
- [Apple DepthPro](https://github.com/apple/ml-depth-pro) - Depth estimation
- SEW Dataset contributors

## Contact

For questions or issues, please open an issue on GitHub or contact the project maintainers.