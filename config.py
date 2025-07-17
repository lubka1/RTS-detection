import os

# The backend must be configured before importing Keras, and the backend cannot be changed after the package has been imported. https://keras.io/getting_started/#configuring-your-backend
os.environ["SM_FRAMEWORK"] = "tf.keras"
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0" # oneDNN can introduce floating-point variability, turn it off

import tensorflow as tf
import tensorflow.keras as keras
import segmentation_models as sm
import numpy as np
import random

SEED = 42  
np.random.seed(SEED)# Set seed for NumPy
random.seed(SEED)# Set seed for Python random
tf.random.set_seed(SEED)# Set seed for TensorFlow (and Keras)

# Environment setting
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')

S1_train_dir = os.path.join(DATA_DIR, 'train', 'S1')
S1_valid_dir = os.path.join(DATA_DIR, 'val', 'S1')

DEM_train = os.path.join(DATA_DIR, 'train', 'DEM')
DEM_val = os.path.join(DATA_DIR, 'val', 'DEM')

S2_train_dir = os.path.join(DATA_DIR, 'train', 'S2')
S2_valid_dir = os.path.join(DATA_DIR, 'val', 'S2')

y_train_dir = os.path.join(DATA_DIR, 'train', 'Masks')
y_valid_dir = os.path.join(DATA_DIR, 'val', 'Masks')

# Define test paths
S1_test = os.path.join(DATA_DIR, 'test', 'S1')
S2_test = os.path.join(DATA_DIR, 'test', 'S2')
y_test = os.path.join(DATA_DIR, 'test', 'Masks')
DEM_test = os.path.join(DATA_DIR, 'test', 'DEM')

# Training Parameters
BACKBONE = 'resnet50' 
BATCH_SIZE = 16
LR = 0.0005 
EPOCHS = 50

activation = 'sigmoid' 
optim = keras.optimizers.Adam(LR)

alpha = 0.9  # Class 1 (thaw slump) gets higher weight
gamma = 4.0   # Focus more on hard-to-classify areas
focal_loss = sm.losses.BinaryFocalLoss(alpha=alpha, gamma=gamma)
dice_loss = sm.losses.DiceLoss()
total_loss = focal_loss + dice_loss 

