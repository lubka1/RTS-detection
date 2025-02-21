import os
import sys
import time
from datetime import datetime
import tensorflow as tf
import tensorflow.keras as keras
import segmentation_models as sm
import matplotlib.pyplot as plt
import utils

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ["SM_FRAMEWORK"] = "tf.keras"


# Environment setting
DATA_DIR = r"C:\Users\smola\Documents\MASTER\TRY_Dataset"

S1_train_dir = os.path.join(DATA_DIR,'train', 'S1')
S1_valid_dir = os.path.join(DATA_DIR, 'val', 'S1')

S2_train_dir = os.path.join(DATA_DIR, 'train', 'S2')
S2_valid_dir = os.path.join(DATA_DIR, 'val', 'S2')

y_train_dir = os.path.join(DATA_DIR, 'train','Masks')
y_valid_dir = os.path.join(DATA_DIR, 'val', 'Masks')

# Set parameters
MODEL = 's1' # option: ['s1', 's2', 'early', 'late']
BACKBONE = 'resnet34'
BATCH_SIZE = 8
LR = 0.001  # toto alebo 0.0001???
EPOCHS = 3

# Dataset for train images
train_dataset = utils.Dataset(
    S2_train_dir, 
    y_train_dir, 
    fusion=True, 
    images_dir2=S1_train_dir,
    classes=['rts'], 
    augmentation=utils.get_training_augmentation(),
    preprocessing=utils.get_preprocessing(preprocess_input),
)
print('Number of training data: {}'.format(len(train_dataset.ids)))

# Dataset for validation images
val_dataset = utils.Dataset(
    S2_valid_dir, 
    y_valid_dir, 
    fusion=True, 
    images_dir2=S1_valid_dir,
    classes=['rts'], 
    preprocessing=utils.get_preprocessing(preprocess_input),
)

# Determine the number of channels in the image
image, mask = train_dataset[12]  
N = image.shape[-1]

train_dataloader = utils.Dataloder(train_dataset, batch_size=BATCH_SIZE, shuffle=True)   # should this be in the other function? load_data?
valid_dataloader = utils.Dataloder(val_dataset, batch_size=1, shuffle=False)

# check shapes for errors
assert train_dataloader[0][0].shape == (BATCH_SIZE, 256, 256, N)
assert train_dataloader[0][1].shape == (BATCH_SIZE, 256, 256, 1)

# create model
if MODEL == 's1':
    model = sm.Unet(BACKBONE, encoder_weights=None,classes = 1, activation='sigmoid', input_shape=(None, None, N))
elif MODEL == 's2':
    model = sm.FPN(BACKBONE, encoder_weights=None,classes = 1, activation='sigmoid', input_shape=(None, None, N))
elif MODEL == 'early':
    model = sm.Linknet(BACKBONE, encoder_weights=None,classes = 1, activation='sigmoid', input_shape=(None, None, N))
elif MODEL == 'late':
    model = sm.Linknet(BACKBONE, encoder_weights=None,classes = 1, activation='sigmoid', input_shape=(None, None, N))
else:
    raise Exception("Models not found")

# Define callbacks for learning rate scheduling and best checkpoints saving
callbacks = [
    keras.callbacks.ModelCheckpoint(some_dir+'best_'+MODEL+ '_weights.h5', save_weights_only=True, save_best_only=True, mode='min'),
    keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=8, verbose=1, min_lr=1e-6),   # reduces  learnig rate when the metric has stopped imroving
    keras.callbacks.EarlyStopping(monitor='val_loss', patience=10,)
    keras.callbacks.TensorBoard(log_dir=ckpts_dir+'tensorboard_'+MODEL+"_"+BACKBONE+'/'),
]

# Compile keras model with defined optimizer, loss and metrics
optim = keras.optimizers.Adam(LR)
total_loss = sm.losses.binary_focal_dice_loss
model.compile(optim, total_loss, 
              metrics=[keras.metrics.BinaryIoU(target_class_ids=[1], threshold=0.5)])

model.compile(
    optimizer=keras.optimizers.Adam(LR),
    loss=sm.losses.binary_focal_dice_loss,
    metrics=[keras.metrics.BinaryIoU(target_class_ids=[1], threshold=0.5)])
    #metrics=[sm.metrics.iou_score, sm.metrics.f1_score, sm.metrics.precision, sm.metrics.recall])

# Train model
start_time = time.time()
history = model.fit(
      train_dataloader, 
      #steps_per_epoch=len(train_dataloader)//BATCH_SIZE, 
      epochs=EPOCHS, 
      callbacks=callbacks, 
      validation_data=valid_dataloader, 
    #  validation_steps=len(valid_dataloader)//BATCH_SIZE, # https://stackoverflow.com/questions/59864408/tensorflowyour-input-ran-out-of-data  
  )
elapsed_time = time.time() - start_time
print('Training complete. Elapsed time: '+str(elapsed_time))


utils.plot_history(history)
