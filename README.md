# Teeth Segmentation and Whitening Pipeline

This repository contains a complete end-to-end pipeline for teeth segmentation and whitening using deep learning. The pipeline uses a Compact Attention U-Net model for precise teeth segmentation followed by advanced LAB color space whitening.

## Overview

The project implements:
- **Compact Attention U-Net**: An optimized U-Net architecture with attention gates for efficient teeth segmentation
- **Advanced Teeth Whitening**: LAB color space-based whitening with adaptive parameters
- **Complete Pipeline**: End-to-end processing from raw images to whitened results

## Model Architecture

### Compact Attention U-Net
- **Architecture**: U-Net with attention gates optimized for teeth segmentation
- **Model Size**: 76.3 MB (optimized for efficiency)
- **Input Size**: 256x256 pixels
- **Depth**: 4 levels
- **Base Channels**: 48
- **Channel Progression**: 48 → 96 → 192 → 288 → 288

### Model Selection Rationale
Classical U-Net architectures were tested:
- **Classical U-Net (Depth 4, 64 base channels)**: 200 MB, minimal validation loss improvement
- **Classical U-Net (Depth 5, 64 base channels)**: 260 MB, minimal validation loss improvement

The Compact Attention U-Net was selected for its optimal balance of model size (76.3 MB) and performance.

## Training Results

From `Teeth-Segmentation_Whitening_training-inference_pipeline.ipynb`:
- **Final Training Loss**: 0.2818
- **Final Validation Loss**: 0.2818
- **Validation Dice Score**: 0.8993
- **Validation IoU**: 0.8173
- **Training Epoch**: 5 (early stopping)

## Files Description

### Main Training Notebook
- **`Teeth-Segmentation_Whitening_training-inference_pipeline.ipynb`**: Complete training and inference pipeline
  - Model definition and training
  - Dataset preprocessing
  - Training loop with early stopping
  - Validation metrics calculation
  - Inference pipeline implementation
  - Teeth whitening algorithm

### Inference Scripts
- **`Inference-whitening-pipeline.ipynb`**: Interactive notebook for inference and testing
  - Load pre-trained model
  - Process single images or directories
  - Visualization tools
  - Parameter tuning interface

- **`Inference-whitening-pipeline.py`**: Command-line inference script
  - Standalone Python script for production use
  - CLI arguments for batch processing
  - Configurable whitening parameters
  - Progress tracking and error handling

## Quick Start

### Training (Google Colab)
1. Upload `Teeth-Segmentation_Whitening_training-inference_pipeline.ipynb` to Google Colab
2. Set runtime to GPU
3. Upload your dataset
4. Run all cells to train the model

### Inference Options

#### Option 1: Interactive Notebook
```bash
# Open Inference-whitening-pipeline.ipynb in Jupyter/Colab
# Follow the cells to load model and process images
```

#### Option 2: Command Line Script
```bash
# Single image processing
python Inference-whitening-pipeline.py --image path/to/image.jpg

# Directory processing
python Inference-whitening-pipeline.py --directory path/to/images/

# Custom parameters
python Inference-whitening-pipeline.py --image image.jpg --intensity 1.5 --output results/
```


## Configuration

The pipeline uses `config.yaml` for configuration:
```yaml
model:
  base_channels: 48
  depth: 4
  image_size: 256

whitening:
  lightness_increase: 40
  yellowness_decrease: 25
  blur_kernel_size: [21, 21]
```

## Features

### Teeth Segmentation
- Attention-based U-Net architecture
- Efficient memory usage (76.3 MB model)
- High accuracy (Dice: 0.8993, IoU: 0.8173)

### Teeth Whitening
- LAB color space processing
- Adaptive whitening parameters
- Natural-looking results with soft blending
- Customizable intensity levels

### Processing Options
- Single image processing
- Batch directory processing
- Configurable output formats
- Progress tracking and error handling

## Usage Examples
### Command Line
```bash
# Basic usage
python Inference-whitening-pipeline.py --image teeth.jpg

# Advanced usage with custom parameters
python Inference-whitening-pipeline.py \
  --directory /path/to/images \
  --output /path/to/results \
  --intensity 1.2 \
  --max-images 100
```

## Testing and Evaluation

Use the inference files for further testing and evaluation:
- **`Inference-whitening-pipeline.ipynb`**: Interactive testing with visualization
- **`Inference-whitening-pipeline.py`**: Automated batch processing and evaluation

Both files support:
- Custom intensity settings
- Adaptive vs fixed whitening parameters
- Batch processing capabilities
- Performance metrics calculation

## Solution Approach

### Attempted Methods and Architectures

#### Unsuccessful Approaches
- **Classical U-Net (Depth 4, 64 base channels)**: 200 MB model size with minimal validation loss improvement
- **Classical U-Net (Depth 5, 64 base channels)**: 260 MB model size with minimal validation loss improvement

#### Final Chosen Approach
The Compact Attention U-Net architecture was selected based on optimal performance-to-size ratio:

**Preprocessing**:
- Image resizing to 256x256 pixels
- ImageNet normalization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
- Comlex Data augmentation icluding AdaptiveColorJitter, RandomNoise, RandomBlur, etc.

**Model Architecture**:
- Compact Attention U-Net with attention gates
- 4-level depth encoder-decoder structure
- 48 base channels with progressive channel multiplication
- Bilinear upsampling for memory efficiency
- Channel progression: 48 → 96 → 192 → 288 → 288

**Training Strategy**:
- Early stopping based on validation loss
- Learning Rate Scheduler
- Mixed Loss (Dice + BCE)
- Dice and IoU metrics for segmentation evaluation
- Model checkpointing for best validation performance

**Post-processing**:
- LAB color space conversion for natural whitening
- Adaptive whitening parameters based on teeth color analysis
- Gaussian blur mask feathering for smooth blending
- Morphological operations for mask refinement

### Future Improvements and Alternative Approaches

**Model Architecture Enhancements**:
- More Robust Architecture level optimization
- Ensemble methods combining multiple model predictions
- Progressive training with increasing input resolutions

**Data and Training Improvements**:
- More ddvanced data augmentation strategies
- Transfer learning from medical imaging pretrained models

**Whitening Algorithm Enhancements**:
- Machine learning-based whitening parameter optimization
- Real-time intensity adjustment based on facial features
- Integration with facial detection for context-aware processing

**Performance Optimizations**:
- Model quantization for mobile deployment
- ONNX conversion for cross-platform inference
- Batch processing optimizations for large-scale deployment

## Reproducibility Instructions

### Environment Setup

**Requirements**:

**Environment Creation**:
```bash
# Using pip
pip install -r requirements.txt

# Using conda
conda create -n teeth-whitening python=3.8
conda activate teeth-whitening
pip install -r requirements.txt
```

### Training Instructions

**Step 1: Dataset Preparation**
- Use ```dentalai-DatasetNinja``` dir and set it as ```base_data_path``` in config.yaml if want to create a new pocessed_data

**Step 2: Configuration**
- Modify `config.yaml` for custom training parameters
- Adjust model architecture settings if needed

**Step 3: Run Training**
Have to start it in ```Teeth-Segmentation_Whitening_training-inference_pipeline.ipynb```

```bash
# Google Colab training
# Upload Teeth-Segmentation_Whitening_training-inference_pipeline.ipynb
# Set runtime to GPU and execute all cells
```

### Inference Instructions

**Command Line Inference**:
```bash
# Single image processing
python Inference-whitening-pipeline.py --image path/to/image.jpg

# Directory processing
python Inference-whitening-pipeline.py --directory path/to/images/ --output results/

# Custom parameters
python Inference-whitening-pipeline.py --image image.jpg --intensity 1.2 --no-adaptive
```

**Interactive Inference**:
```bash
# Launch Jupyter notebook
jupyter notebook Inference-whitening-pipeline.ipynb

# Or use Google Colab
# Upload the notebook and execute cells sequentially
```

### Model Files
- Pre-trained model: `models/best_model.pth`
- Configuration: `config.yaml`
- Place model file in `models/` directory before running inference

### Expected Output Structure
```
results/
├── whitened_image1.jpg
├── whitened_image2.jpg
└── ...
```
