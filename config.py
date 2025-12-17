import os
import warnings

# The backend must be configured before importing Keras, and the backend cannot be changed after the package has been imported. https://keras.io/getting_started/#configuring-your-backend
os.environ["SM_FRAMEWORK"] = "tf.keras"
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0" # oneDNN can introduce floating-point variability, turn it off
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Disable most TensorFlow logs
warnings.filterwarnings("ignore")          # Disable Python warnings
os.environ["TF_XLA_FLAGS"] = "--tf_xla_enable_xla_devices=false"
 
import tensorflow as tf
import tensorflow.keras as keras
import segmentation_models as sm
import numpy as np
import random

SEED = 42  
np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)

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
BATCH_SIZE = 32   #32        16         16
LR = 0.001    #0.0005      0.0001        0.001     0.00001
EPOCHS = 50

patienceRLR = 6 # patience for ReduceLearningRate
factor = 0.5
patienceES = 10 # patience for early stopping

activation = 'sigmoid' 
optim = keras.optimizers.Adam(LR)

# add attention, fusion, and unet xnet options here?
TL = True  # True False
ATTENTION = False    # True False
STRATEGY = 'concat'  # 'concat' 'average'

# the authors write "we found γ=2 to work best in our experiments."
alpha = 0.8   #0.9   Class 1 (thaw slump) gets higher weight
gamma = 3.0   # Focus more on hard-to-classify areas when higher, default 2, 
focal_loss = sm.losses.BinaryFocalLoss(alpha=alpha, gamma=gamma)
dice_loss = sm.losses.DiceLoss()
total_loss = focal_loss + dice_loss 

#total_loss= masked_total_loss