# Corner-Detection
A Python implementation combining Holistic Edge Detection (HED) and corner detection algorithms. Uses PyTorch and CNN for pre-processing images to improve corner detection accuracy in computer vision tasks.

## Overview

This project implements a two-stage corner detection pipeline:
1. Holistic Edge Detection (HED) for edge enhancement
2. Corner detection using traditional algorithms

The key innovation is using HED as a pre-processing step, which improves the quality of corner detection by reducing noise and enhancing relevant edge features.

## Features

- HED-based edge enhancement using deep learning
- Custom CNN architecture with 5 sequential layers
- Support for multiple image formats
- Interactive visualizations of intermediate and final results
- Fixed input size optimization (480x320 pixels)
- Comparative output with and without HED pre-processing

## Requirements

- Python 3.7+
- PyTorch
- NumPy
- PIL (Python Imaging Library)
- SciPy

## Installation

```bash
git clone https://github.com/yourusername/corner-detection-hed.git
cd corner-detection-hed
pip install -r requirements.txt
```

## Usage

```python
# Basic usage
python run.py --input try.png --output out.png --model bsds500

# Parameters
--input: Input image path (PNG format)
--output: Output image path
--model: Model type (default: bsds500)
```

## Model Architecture

### HED Network Structure
- 5 sequential layers
- First 2 layers: width 2
- Remaining 3 layers: width 3
- Final sigmoid-activated convolutional layer
- Side output layers for multi-scale feature extraction

### Image Processing Pipeline
1. Image preprocessing and normalization
2. HED edge detection
3. Corner detection on enhanced edges

## Results

The implementation shows significant improvements in corner detection accuracy when using HED pre-processing:
- Better handling of noisy images
- More accurate corner localization
- Reduced false positives
- Improved performance on complex scenes

## Research Background

This implementation is based on research into combining traditional corner detection with deep learning approaches. Key references:
- "Corner Detection using Deep Learning" (2017)
- "End-to-End Corner Detection with Deep Neural Networks" (2018)
- "Fast Corner Detection Based on Deep Learning" (2019)
