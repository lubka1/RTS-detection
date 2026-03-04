import wandb
import config
import data_utils
import utils
import tensorflow.keras as keras
from wandb.integration.keras import WandbMetricsLogger
import segmentation_models as sm
import sweep

# https://docs.wandb.ai/models/sweeps



def sweep_train():

    wandb.init(
        project="fusion-unet",  
        name=f"grid_search",  
        config={
            "fusion_type": "early",
            "epochs": 30,
            "backbone": 'resnet50',
            "transfer_learning": False
        },
        sync_tensorboard=False,
        reinit=True,
    )

    batch_size = wandb.config.batch_size
    lr = wandb.config.lr
    alpha = wandb.config.alpha
    gamma = wandb.config.gamma

    train_dataloader, val_dataloader, N, M = data_utils.get_data("early", batch_size)

    model = utils.load_model("early", N, M) 

    focal_loss = sm.losses.BinaryFocalLoss(alpha=alpha, gamma=gamma)
    dice_loss = sm.losses.DiceLoss()
    total_loss = focal_loss + dice_loss 

    optim = keras.optimizers.Adam(lr)

    model.compile(
        optim, 
        total_loss, 
        metrics = [
            keras.metrics.BinaryIoU(target_class_ids=[1], threshold=0.5),
            utils.f_score  
        ]
    )
    callbacks = [
        keras.callbacks.ModelCheckpoint(f'grid_search.weights.h5', save_weights_only=True, save_best_only=True, monitor='val_binary_io_u', mode='max'),
        keras.callbacks.ReduceLROnPlateau(monitor='val_binary_io_u', factor=0.5, patience=3, verbose=1, min_lr=5e-5),   
        keras.callbacks.EarlyStopping(monitor='val_binary_io_u', patience=5),
        WandbMetricsLogger(),
        ]

    history = model.fit(
        train_dataloader, 
        epochs=wandb.config.epochs, 
        callbacks=callbacks, 
        validation_data=val_dataloader, 
    )

sweep_id = wandb.sweep(sweep.sweep_config, project="fusion-unet")
wandb.agent(sweep_id, function=sweep_train)
