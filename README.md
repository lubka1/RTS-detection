#  Retrogressive Thaw Slump Detection Using Deep Learning

This project focuses on the detection of **retrogressive thaw slumps (RTS)** using binary semantic segmentation. A **U-Net** architecture is employed to segment satellite imagery, specifically leveraging **Sentinel-1** (SAR) and **Sentinel-2** (optical) datasets. The project includes data fusion techniques, preprocessing tools, model training, and prediction pipelines.

Permafrost degradation is a critical issue in Arctic environments, and RTS detection is vital for monitoring landscape changes. This project builds a reproducible deep learning pipeline capable of detecting these slumps from multi-source satellite imagery.

##  Project Summary

- **Model**: U-Net  
- **Task**: Binary Semantic Segmentation  
- **Data**: Sentinel-1 (SAR) + Sentinel-2 (Optical), ArcticDEM  
- **Fusion**: Early, Middle and Late data fusion 
- **Framework**: Keras

## 📁 Project Structure
. ├── custom_classification_models/ # Custom model definitions ├── segmentation_models/ # Prebuilt segmentation architectures ├── ArcticDEM.ipynb # Notebook for ArcticDEM analysis ├── config.py # Configurations and hyperparameters ├── data_utils.py # Dataset loading and preparation ├── fusion.py # Data fusion logic for S1 and S2 ├── predict.py # Inference script ├── test.py # Evaluation script ├── train.py # Model training pipeline ├── utils.py # Utility functions ├── requirements.txt # Python dependencies └── .gitignore


---



