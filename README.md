# Project README

## Overview
This project is a computer vision pipeline designed to detect and classify cracks in images. It uses deep learning techniques for both image classification (detecting whether a crack exists) and segmentation (identifying exactly where the crack is).

## Project Structure
1.app.py - Main entry point  runs the application (web app).

2.backend.py - Handles backend logic such as model loading, inference, or API services.

3.train_classifier.py - Trains a model to classify images (e.g., crack vs no crack).

4.train_segmentation.py - Trains a segmentation model (like UNet) to detect crack regions pixel-by-pixel.

5.train_automl.py - Uses AutoML (e.g., AutoKeras) to automatically search for the best model.

6.predict_random.py - Runs predictions on random images from the dataset.

7.create_labels.py - Generates label files for training.

8.check_labels.py - Helps verify and validate dataset labels.

9.dataset_reduction.py - Reduces dataset size for faster experimentation.

10.classification_labels.csv - Stores labels for classification tasks.

11.label_map.json / autokeras_label_map.json -
Map label indices to class names.

12.best_crack_classifier.pth / best_unet_model.pth
Saved trained models ready for inference.

13.automl_runs- Stores AutoML experiment results.

14.reduced_dataset - Contains processed dataset splits:
train/
val/
test/
Each split includes:
IMG/ → input images
GT/ → ground truth labels

15.uploads/ - Likely used for storing images uploaded for prediction 


## How to Use

### 1. Environment Setup

Create and activate a Python virtual environment:

python -m venv .venv
.venv\Scripts\activate

Install dependencies 
pip install torch torchvision autokeras

### 2.Training models

Train Classification Model:
python train_classifier.py

Train Segmentation Model:
python train_segmentation.py

Train with AutoML:
python train_automl.py

### 3. Prediction Command

Run Prediction on Random Samples:
python predict_random.py

### 4. Dataset Management Commands

Create Labels:
python create_labels.py

Check Labels:
python check_labels.py

Reduce Dataset:
python dataset_reduction.py

### 5. Frontend command 
Frontend and backend should run in parallel
streamlit run app.py

### 6. Backend command
uvicorn backend:app

### 7. Model Files
Trained models are saved as `.pth` files and can be loaded for inference.

### 8. Data Structure
Datasets are organized under `reduced_dataset/` with `train`, `val`, and `test` splits, each containing `IMG` (images) and `GT` (ground truth) folders.

## Comments and Documentation
All scripts are commented for clarity. Please refer to the comments within each script for detailed explanations of functions, parameters, and workflow.

## Requirements
- Python 3.8+
- PyTorch
- Torchvision
- AutoKeras (if using AutoML)
- Other dependencies as required by individual scripts

