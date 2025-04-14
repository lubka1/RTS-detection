import config  # Import settings from config.py

import os
import sys
import time
import argparse
from datetime import datetime
import tensorflow as tf
import tensorflow.keras as keras
import segmentation_models as sm
import matplotlib.pyplot as plt
import utils
import data_utils
from tensorflow.keras import layers, Model, Input
import wandb
from wandb.integration.keras import WandbMetricsLogger, WandbModelCheckpoint
import fusion
import numpy as np
import random


# Set the random seed
seed_value = 42  
np.random.seed(seed_value)# Set seed for NumPy
random.seed(seed_value)# Set seed for Python random
tf.random.set_seed(seed_value)# Set seed for TensorFlow (and Keras)


BACKBONE = config.BACKBONE
BATCH_SIZE = config.BATCH_SIZE
LR = config.LR
EPOCHS = config.EPOCHS
preprocess_input = sm.get_preprocessing(BACKBONE)


def train_model(fusion_type, strategy='concat', attention=None ,transfer_learning=False):
        
        """
    Trains the model using the specified fusion strategy.

    Args:
        fusion_type (str): One of ['s1', 's2', 'early', 'late'].

    Returns:
        float: Best validation accuracy.
        str: Path to the saved model.
    """
        
        
        print(f"\nStarting training with fusion type: {fusion_type}")
        print("[DEBUG] transfer_learning inside train_model:", transfer_learning)


        # Initialize wandb
        wandb.init(
            project="fusion-unet",  # Change this to your project name
            name=f"train_{fusion_type}_{strategy}_{attention}",  # Experiment name
            config={
                "fusion_type": fusion_type,
                "epochs": EPOCHS,
                "batch_size": BATCH_SIZE,
                "learning_rate": LR,
                "backbone": BACKBONE,
            },
            sync_tensorboard=False,
            reinit=True,
            #settings=wandb.Settings(_disable_stats=True)  # Asynchronous upload and no summary stats (system metrics like CPU/GPU usage, memory usage)

        )

        train_dataloader, val_dataloader, N, M = data_utils.get_data(fusion_type)

        model = utils.load_model(fusion_type, N, M, strategy=strategy, attention=attention, transfer_learning=transfer_learning)  

        model.compile(config.optim, config.total_loss, 
                      metrics = [
                            keras.metrics.BinaryIoU(target_class_ids=[1], threshold=0.5),
                            utils.f_score  ])
            #        metrics=[keras.metrics.BinaryIoU(target_class_ids=[1], threshold=0.5)])
            #metrics=[sm.metrics.iou_score, sm.metrics.f1_score, sm.metrics.precision, sm.metrics.recall])
        
        callbacks = [
            # keras.callbacks.ModelCheckpoint('best_'+fusion_type+strategy+str(transfer_learning)+ '.weights.h5', save_weights_only=True, save_best_only=True, mode='min'),
            keras.callbacks.ModelCheckpoint(f'best_{fusion_type}{strategy}{attention}.weights.h5', save_weights_only=True, save_best_only=True, monitor='val_binary_io_u', mode='max'),
            keras.callbacks.ReduceLROnPlateau(monitor='val_binary_io_u', factor=0.5, patience=6, verbose=1, min_lr=5e-5),   
            keras.callbacks.EarlyStopping(monitor='val_binary_io_u', patience=10),
            WandbMetricsLogger(),
            ]
        
        # Train model
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

        # log to wandb
        wandb.log({"final_val_iou": max(history.history.get("val_binary_io_u", [0]))})
        wandb.finish()

        try:
            best_val_acc = max(history.history['val_binary_io_u'])  # Adjust metric name if necessary
        except KeyError as e:
            print(f"Error: Metric not found.")
            print("Available metrics:")
            for key in history.history.keys():
                print(key)

        #utils.plot_history(history)

        keras.backend.clear_session()

        return best_val_acc

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fusion", type=str, required=True, choices=["early", "middle", "late"], help="Select fusion mode")
    parser.add_argument("--strategy", type=str, choices=["concat", "average"], help="Select fusion strategy")
    parser.add_argument("--attention", type=str, choices=["None", "grid", "channel"], help="Select attention strategy")
    parser.add_argument("--transfer_learning", action='store_true', help="Use transfer learning (pretrained weights)") # default ziadne TL, staci pouzit --tran_lea... a je aplikvoane


    args = parser.parse_args()
    print("Transfer learning flag:", args.transfer_learning)


    best_acc = train_model(args.fusion, strategy=args.strategy, attention=args.attention, transfer_learning=args.transfer_learning)
    print(f"Best accuracy for {args.fusion}: {best_acc:.4f}")

    #print(f"Best accuracy for {args.fusion}: {best_acc:.4f}, Model saved at: {best_model_path}")