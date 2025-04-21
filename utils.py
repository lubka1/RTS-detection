"""
utils.py

Utility functions and classes for model construction, training, evaluation,
and data augmentation for semantic segmentation tasks.
"""

import config  
import fusion

import numpy as np
import matplotlib.pyplot as plt
import albumentations as A
import tensorflow.keras as keras
import segmentation_models as sm
import tensorflow as tf
from tensorflow.keras import layers, Model, Input
import logging
logging.basicConfig(level=logging.INFO)


BACKBONE = config.BACKBONE
BATCH_SIZE = config.BATCH_SIZE
LR = config.LR
EPOCHS = config.EPOCHS
preprocess_input = sm.get_preprocessing(BACKBONE)



def load_model(fusion_type, N, M, strategy='concat', attention=None,transfer_learning=False, model_path=None):
    """
    Load and return a segmentation model based on fusion type and other settings.

    Args:
        fusion_type (str): 'early', 'middle', or 'late' fusion model type.
        N (int): Number of input channels for input2.
        M (int): Number of input channels for input1.
        strategy (str): Fusion strategy for late fusion ('concat', 'average'). 
        attention (str, optional): Attention mechanism to use ('grid', 'channel').
        transfer_learning (bool): Whether to use pretrained ImageNet weights.
        model_path (str, optional): Path to load saved weights.

    Returns:
        keras.Model: Constructed and optionally preloaded model.
    """
     
    encoder_weights = None  # default

    if transfer_learning:
        encoder_weights = 'imagenet'
        logging.info("Using Imagenet pre-trained weights")
    
    if fusion_type == 'early':
        model = sm.Unet(
            BACKBONE,
            encoder_weights=encoder_weights,
            classes=1,
            activation=config.activation,
            input_shape=(None, None, N)
        )
    elif fusion_type == 'middle':
        model = sm.MiddleUnet(
            'midresnet50',
            'resnet50',
            encoder_weights=encoder_weights,
            classes=1,
            activation=config.activation,
            input_shape1=(None, None, M),
            input_shape2=(None, None, N)
        )
    elif fusion_type == 'late':
        model = construct_late_unet(M, N, strategy, attention, encoder_weights)
    else:
        raise ValueError("Fusion type not recognized.")
    
    '''
    # Get the first convolutional layer
    first_layer = model.layers[0]

    # If transfer_learning is True and N > 3, modify the first layer weights
    if transfer_learning and N > 3:
        # Reinitialize the first layer's weights (since it doesn't match pretrained weights for N > 3)
        first_layer.set_weights([tf.random.normal(w.shape) for w in first_layer.get_weights()])  # Random initialization
        # Load the pretrained weights for all layers except the first layer
    if encoder_weights == 'imagenet' and N <= 3:
        model.load_weights(model_path, by_name=True, skip_mismatch=True)
    '''
    
    # Load saved weights if provided, for testing purposes
    if model_path:
        model.load_weights(model_path)

    return model


def construct_late_unet(M, N, strategy='concat', attention=None, encoder_weights=None):
    """
    Construct a U-Net model using a late fusion strategy.

    Args:
        M (int): Number of input channels for first input.
        N (int): Number of input channels for second input.
        strategy (str): Fusion strategy ('concat', 'average').
        attention (str, optional): Attention mechanism ('grid' or 'channel').
        encoder_weights (str, optional): Encoder weights ('imagenet' or None).

    Returns:
        keras.Model: A compiled Keras model with late fusion.
    """

    model1 = sm.Unet(BACKBONE, encoder_weights=encoder_weights,classes = 1, activation=config.activation, input_shape=(None, None, M), late_fusion=True, input_shape2=(None, None, N))   # S1
    model2 = sm.Unet(BACKBONE, encoder_weights=encoder_weights,classes = 1, activation=config.activation, input_shape=(None, None, N), late_fusion=True, input_shape2=(None, None, M))   # S2
    
    input1 = Input(shape=(None, None, M))  # S1
    input2 = Input(shape=(None, None, N))  # S2

    features1 = model1(input1)  
    features2 = model2(input2)

    if attention == 'grid': 
        attention_layer = fusion.GridAttention() # GridAttention is a cross-attention mechanism — it takes two inputs
        features1 = attention_layer(features1, features2)
        features2 = attention_layer(features2, features1)

    elif attention == 'channel': # ChannelGate is a self-attention mechanism — it only needs its own feature map to decide which channels are important.
        channel_gate1 = fusion.ChannelGate(features1.shape[-1])
        channel_gate2 = fusion.ChannelGate(features2.shape[-1])
        features1 = channel_gate1(features1)
        features2 = channel_gate2(features2)

    x = [features1, features2]
    if strategy == 'concat':
        fusion_output = layers.Concatenate(axis=-1, name="concat_features")(x)
    elif strategy == 'average':
        fusion_output = fusion.WeightedAverage(n_output=len(x))(x)

    # final convolution layer
    output = layers.Conv2D(
        filters=1,  # Number of output classes 
        kernel_size=(3, 3),
        padding='same',
        use_bias=True,
        kernel_initializer='glorot_uniform',
        name='final_conv'
    )(fusion_output)

    output = layers.Activation(config.activation)(output)

    return Model(inputs=[input1, input2], outputs=output)


def plot_history(history):
    """
    Plot training and validation metrics from a training history.

    Args:
        history (keras.callbacks.History): The history object from model.fit().

    """
    #print("Available metrics in history:", history.history.keys())
    
    # Extract metric names
    metric_1 = list(history.history.keys())[0]  # First metric (binary iou)
    metric_2 = list(history.history.keys())[1]  # loss
    
    val_metric_1 = list(history.history.keys())[2]  # Validation for the first metric
    val_metric_2 = list(history.history.keys())[3]  # Validation for the second metric
    
    plt.figure(figsize=(30, 5))
    
    plt.subplot(121)
    plt.plot(history.history[metric_1])
    plt.plot(history.history[val_metric_1])
    plt.title(metric_1)
    plt.ylabel(metric_1)
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    plt.subplot(122)
    plt.plot(history.history[metric_2])
    plt.plot(history.history[val_metric_2])
    plt.title(metric_2)
    plt.ylabel(metric_2)
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    plt.show()

def f_score(y_true, y_pred):
    """
    Compute the F1 score.

    Args:
        y_true (tf.Tensor): Ground truth binary labels.
        y_pred (tf.Tensor): Predicted binary labels.

    Returns:
        tf.Tensor: Computed F1 score.
    """
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    tp = tf.reduce_sum(y_true * y_pred)
    fp = tf.reduce_sum((1 - y_true) * y_pred)
    fn = tf.reduce_sum(y_true * (1 - y_pred))
    return tp / (tp + 0.5 * (fp + fn))


# AUGMENTATION            

def round_clip_0_1(x, **kwargs):
    return x.round().clip(0, 1)

train_transform = [
        A.HorizontalFlip(p=0.5),        # Randomly flip images horizontally
        A.VerticalFlip(p=0.5),          # Randomly flip images vertically
        A.RandomRotate90(p=0.5),        # Randomly rotate images by multiples of 90°
        A.Blur(blur_limit=3, p=0.5),    # Randomly blur images
    ]

def get_training_augmentation(train_transform=train_transform):
    """
    Compose and return a training augmentation pipeline. Including a second image if provided by dataloader.

    Args:
        train_transform (list): List of Albumentations transforms.

    Returns:
        A.Compose: Albumentations pipeline.
    """
    return A.Compose(train_transform, additional_targets={'image2': 'image'})

def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform
    
    Args:
        preprocessing_fn (callable): Preprocessing function for a given model backbone.

    Return:
        transform: albumentations.Compose
    
    """
    _transform = [
        A.Lambda(image=preprocessing_fn),
    ]
    return A.Compose(_transform)

class RandomChoice(A.BasicTransform):
    """
    Apply one of several random transformations to the input.

    Args:
        transforms (list): List of Albumentations transforms to choose from.
        always_apply (bool): Whether to always apply this transform.
        p (float): Probability of applying the transformation.
    """
    def __init__(self, transforms, always_apply=False, p=1.0):
        super().__init__(always_apply, p)
        self.transforms = transforms

    def apply(self, image, **params):
        transform = np.random.choice(self.transforms)
        return transform(image=image)['image']

    def __call__(self, imgs):
        transform = np.random.choice(self.transforms)
        return [transform(image=img)['image'] for img in imgs]

    def get_transform_init_args_names(self):
        return ('transforms',)


