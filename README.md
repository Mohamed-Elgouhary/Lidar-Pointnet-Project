# LiDAR-Based 3D Object Classification for Pattern Recognition Using a PointNet-Style Neural Network

## Overview
This project performs 3D object classification on LiDAR point clouds from the KITTI object dataset.
The three target classes are:
- Car
- Pedestrian
- Cyclist

A balanced subset is created from KITTI, each crop is normalized and resampled to 1024 points, and a simple PointNet-style classifier is trained.

## Project structure
- `src/` : preprocessing, dataset, model, training, evaluation
- `data/metadata/` : metadata CSV
- `outputs/checkpoints/` : saved model weights
- `outputs/logs/` : training history
- `outputs/figures/` : generated figures
- `report/` : LaTeX report and compiled PDF

## Requirements
Install the main dependencies with:

```bash
python -m pip install numpy matplotlib scikit-learn torch torchvision tqdm
