import config
import utils
import data_utils

import time
import argparse
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras as keras
import segmentation_models as sm
from tensorflow.keras import layers, Model, Input
import wandb
from wandb.integration.keras import WandbMetricsLogger, WandbModelCheckpoint


BACKBONE = config.BACKBONE
BATCH_SIZE = config.BATCH_SIZE
LR = config.LR
EPOCHS = config.EPOCHS
preprocess_input = sm.get_preprocessing(BACKBONE)


def train_model(fusion_type, strategy='concat', attention=None , transfer_learning=False):
        
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
    print(f"[DEBUG] Strategy: {strategy}, Attention: {attention}, Transfer Learning: {transfer_learning}")

    # Initialize wandb
    wandb.init(
        project="fusion-unet",  
        name=f"train_{fusion_type}_{strategy}_{attention}",  
        config={
            "fusion_type": fusion_type,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LR,
            "backbone": BACKBONE,
            "strategy": strategy,
            "attention": attention,
            "transfer_learning": transfer_learning
        },
        sync_tensorboard=False,
        reinit=True,
        #settings=wandb.Settings(_disable_stats=True)  # Asynchronous upload and no summary stats (system metrics like CPU/GPU usage, memory usage)
    )

    # Load data
    train_dataloader, val_dataloader, N, M = data_utils.get_data(fusion_type)

    # Build model
    model = utils.load_model(
        fusion_type, N, M, 
        strategy=strategy, 
        attention=attention, 
        transfer_learning=transfer_learning
    )  

    model.compile(
        config.optim, 
        config.total_loss, 
        metrics = [
            keras.metrics.BinaryIoU(target_class_ids=[1], threshold=0.5),
            utils.f_score  
        ]
    )
    
    callbacks = [
        keras.callbacks.ModelCheckpoint(f'best_{fusion_type}{strategy}{attention}.weights.h5', save_weights_only=True, save_best_only=True, monitor='val_binary_io_u', mode='max'),
        keras.callbacks.ReduceLROnPlateau(monitor='val_binary_io_u', factor=0.5, patience=6, verbose=1, min_lr=5e-5),   
        keras.callbacks.EarlyStopping(monitor='val_binary_io_u', patience=10),
        WandbMetricsLogger(),
        ]
    
    # Train
    start_time = time.time()
    history = model.fit(
        train_dataloader, 
        steps_per_epoch=len(train_dataloader), 
        epochs=EPOCHS, 
        callbacks=callbacks, 
        validation_data=val_dataloader, 
        validation_steps=len(val_dataloader), 
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
    parser.add_argument("--strategy", type=str, choices=["concat", "average"], default="concat", help="Select fusion strategy")
    parser.add_argument("--attention", type=str, choices=["None", "grid", "channel"], default="None", help="Select attention strategy")
    parser.add_argument("--transfer_learning", action='store_true', help="Use transfer learning (pretrained weights)") 

    args = parser.parse_args()
    attention = None if args.attention == "None" else args.attention

    print(f"[INFO] STarting training fusion: {args.fusion}, Strategy: {args.strategy}, Attention: {attention}, Transfer Learning: {args.transfer_learning}")

    best_iou = train_model(args.fusion, strategy=args.strategy, attention=args.attention, transfer_learning=args.transfer_learning)

    print(f"[RESULT] Best Validation IoU for fusion type '{args.fusion}': {best_iou:.4f}")

