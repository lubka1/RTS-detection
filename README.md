#  Retrogressive Thaw Slump Detection Using Deep Learning

This project focuses on the detection of **retrogressive thaw slumps (RTS)** using binary semantic segmentation. A **U-Net** architecture is employed to segment satellite imagery, specifically leveraging **Sentinel-1** (SAR) and **Sentinel-2** (optical) datasets. The project includes data fusion techniques, preprocessing tools, model training, and prediction pipelines.

Permafrost degradation is a critical issue in Arctic environments, and RTS detection is vital for monitoring landscape changes. This project builds a reproducible deep learning pipeline capable of detecting these slumps from multi-source satellite imagery.

##  Project Summary

- **Model**: U-Net  
- **Task**: Binary Semantic Segmentation  
- **Data**: Sentinel-1 (SAR) + Sentinel-2 (Optical), ArcticDEM  
- **Fusion**: Early, Middle and Late data fusion 
- **Framework**: Keras

## Project Structure
```       

├── src/                    # Source code
│   ├── utils/      # Data loading, normalization, and formatting
│   ├── training/           # Model architecture, training loop, and loss functions
│   ├── inference/          # Prediction and visualization scripts
│   └── fusion/             # Fusion techniques (early, middle, late)
├── config.py               # Configuration file for paths and hyperparameters
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

## Environment Setup (Conda)

```bash
conda create -n rts-detection python=3.8
conda activate rts-detection
pip install -r requirements.txt
```
Add dependencies:
```bash
conda install -c conda-forge gdal
```

## Train Model
To train the model, use the `train.py` script with a specified fusion mode:

```bash
python train.py --fusion early
```
- `--fusion` *(required)*  
  Type: `str`  
  Choices: `early`, `middle`, `late`  
  **Description**: Selects the data fusion strategy used in the model architecture.


## Run Inference
```bash
python  predict.py
```

## Credits

This project is adapted from [qubvel/segmentation_models](https://github.com/qubvel/segmentation_models), licensed under the MIT License, with modifications to support early, middle, and late fusion strategies, attention mechanisms, and custom training pipelines.  qubvel/segmentation_models, licensed under the MIT License.





