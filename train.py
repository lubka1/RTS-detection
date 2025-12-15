
import config
import utils
import data_utils

import time
import argparse
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
tf.debugging.set_log_device_placement(True)
import tensorflow.keras as keras
import segmentation_models as sm
from tensorflow.keras import layers, Model, Input
import wandb
from wandb.integration.keras import WandbMetricsLogger, WandbModelCheckpoint

print("[INFO] TensorFlow version:", tf.__version__)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"[INFO] GPU(s) detected: {[gpu.name for gpu in gpus]}")
else:
    print("[INFO] No GPU detected. Training will use CPU.")

BACKBONE = config.BACKBONE
BATCH_SIZE = config.BATCH_SIZE
LR = config.LR
EPOCHS = config.EPOCHS
patienceRLR = config.patienceRLR
patienceES = config.patienceES
factor = config.factor
preprocess_input = sm.get_preprocessing(BACKBONE)

STRATEGY = config.STRATEGY
ATTENTION = config.ATTENTION
TL = config.TL


def train_model(fusion_type):
        
    """
    Train the fusion model based on selected configuration.

    Args:
        fusion_type (str): Fusion type - one of ['early', 'middle', 'late'].
        strategy (str): Fusion strategy - e.g. 'concat', 'average'.
        attention (str or None): Attention mechanism - 'grid', 'channel', or None.
        transfer_learning (bool): Whether to use pretrained weights.

    Returns:
        float: Best validation IoU score.
    """
    print(f"\nStarting training with fusion type: {fusion_type}")
    print(f"[DEBUG] Strategy: {STRATEGY}, Attention: {ATTENTION}, Transfer Learning: {TL}")

    # Initialize wandb
    wandb.init(
        project="fusion-unet",  
        name=f"train_{fusion_type}_{STRATEGY}_{ATTENTION}",  
        config={
            "fusion_type": fusion_type,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LR,
            "backbone": BACKBONE,
            "strategy": STRATEGY,
            "attention": ATTENTION,
            "transfer_learning": TL
        },
        sync_tensorboard=False,
        reinit=True,
        #settings=wandb.Settings(_disable_stats=True)  # Asynchronous upload and no summary stats (system metrics like CPU/GPU usage, memory usage)
    )

    train_dataloader, val_dataloader, N, M = data_utils.get_data(fusion_type)

    model = utils.load_model(fusion_type, N, M)  

    model.compile(
        config.optim, 
        config.total_loss, 
        metrics = [
            keras.metrics.BinaryIoU(target_class_ids=[1], threshold=0.5),
            utils.f_score  
        ]
    )
    
    callbacks = [
        keras.callbacks.ModelCheckpoint(f'best_{fusion_type}{STRATEGY}{ATTENTION}.weights.h5', save_weights_only=True, save_best_only=True, monitor='val_binary_io_u', mode='max'),
        #keras.callbacks.ModelCheckpoint('best_f1.weights.h5', monitor='val_f_score', save_best_only=True, save_weights_only=True, mode='max'),
        keras.callbacks.ReduceLROnPlateau(monitor='val_binary_io_u', factor=factor, patience=patienceRLR, verbose=1, min_lr=5e-5),   
        keras.callbacks.EarlyStopping(monitor='val_binary_io_u', patience=patienceES),
        WandbMetricsLogger(),
        ]
    
    # Train
    start_time = time.time()
    print('Started training')
    history = model.fit(
        train_dataloader, 
        epochs=EPOCHS, 
        callbacks=callbacks, 
        validation_data=val_dataloader, 
        workers=0,
        use_multiprocessing=False,
    )
    elapsed_time = time.time() - start_time
    print('Training complete. Elapsed time: '+str(elapsed_time))

    # Extract best metric
    try:
        best_val_iou = max(history.history['val_binary_io_u'])  # Adjust metric name if necessary
    except KeyError as e:
        print(f"Error: Metric not found.")
        print("Available metrics:")
        for key in history.history.keys():
            print(key)

    wandb.log({"final_val_iou": max(history.history.get("val_binary_io_u", [0]))})
    wandb.finish()

    #utils.plot_history(history)

    keras.backend.clear_session()

    return best_val_iou

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fusion", type=str, required=True, choices=["early", "middle", "late"], help="Select fusion mode")

    args = parser.parse_args()

    best_iou = train_model(args.fusion)

    print(f"[RESULT] Best Validation IoU for fusion type '{args.fusion}': {best_iou:.4f}")

