import os


os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ["SM_FRAMEWORK"] = "tf.keras"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0" # oneDNN can introduce floating-point variability, turn it off
# Environment variables must be set before importing tensorflow or segmentation_models, otherwise, they won’t take effect.

import tensorflow.keras as keras
import segmentation_models as sm



# Environment setting
DATA_DIR = r"C:\Users\smola\Documents\MASTER\SEN12"

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
BATCH_SIZE = 8
LR = 0.0005 
EPOCHS = 50

activation = 'sigmoid' 
optim = keras.optimizers.Adam(LR)

alpha = 0.9  # Class 1 (thaw slump) gets higher weight
gamma = 4.0   # Focus more on hard-to-classify areas
focal_loss = sm.losses.BinaryFocalLoss(alpha=alpha, gamma=gamma)
dice_loss = sm.losses.DiceLoss()
total_loss = focal_loss + dice_loss 

