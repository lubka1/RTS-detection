from keras.layers import Conv2DTranspose
from keras.layers import UpSampling2D
from keras.layers import Conv2D
from keras.layers import BatchNormalization
from keras.layers import Activation
from keras.layers import Concatenate
from keras.models import Model
import numpy as np
import fusion

import tensorflow as tf
from tensorflow.keras import layers

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

def handle_block_names(stage):
    conv_name = 'decoder_stage{}_conv'.format(stage)
    bn_name = 'decoder_stage{}_bn'.format(stage)
    relu_name = 'decoder_stage{}_relu'.format(stage)
    up_name = 'decoder_stage{}_upsample'.format(stage)
    return conv_name, bn_name, relu_name, up_name


def ConvRelu(filters, kernel_size, use_batchnorm=False, conv_name='conv', bn_name='bn', relu_name='relu'):
    def layer(x):
        x = Conv2D(filters, kernel_size, padding="same", name=conv_name, use_bias=not(use_batchnorm))(x)
        if use_batchnorm:
            x = BatchNormalization(name=bn_name)(x)
        x = Activation('relu', name=relu_name)(x)
        return x
    return layer


def Upsample2D_block(filters, stage, kernel_size=(3,3), upsample_rate=(2,2),
                     use_batchnorm=False, skip=None):

    def layer(input_tensor):

        conv_name, bn_name, relu_name, up_name = handle_block_names(stage)

        x = UpSampling2D(size=upsample_rate, name=up_name)(input_tensor)

        if skip is not None:
            x = Concatenate()([x, skip])

        x = ConvRelu(filters, kernel_size, use_batchnorm=use_batchnorm,
                     conv_name=conv_name + '1', bn_name=bn_name + '1', relu_name=relu_name + '1')(x)

        x = ConvRelu(filters, kernel_size, use_batchnorm=use_batchnorm,
                     conv_name=conv_name + '2', bn_name=bn_name + '2', relu_name=relu_name + '2')(x)

        return x
    return layer


def Transpose2D_block(filters, stage, kernel_size=(3,3), upsample_rate=(2,2),
                      transpose_kernel_size=(4,4), use_batchnorm=False, skip=None):

    def layer(input_tensor):

        conv_name, bn_name, relu_name, up_name = handle_block_names(stage)

        x = Conv2DTranspose(filters, transpose_kernel_size, strides=upsample_rate,
                            padding='same', name=up_name, use_bias=not(use_batchnorm))(input_tensor)
        if use_batchnorm:
            x = BatchNormalization(name=bn_name+'1')(x)
        x = Activation('relu', name=relu_name+'1')(x)

        if skip is not None:
            x = Concatenate()([x, skip])

        x = ConvRelu(filters, kernel_size, use_batchnorm=use_batchnorm,
                     conv_name=conv_name + '2', bn_name=bn_name + '2', relu_name=relu_name + '2')(x)

        return x
    return layer

'''
def build_unet(backbone, classes, skip_connection_layers,
               decoder_filters=(256,128,64,32,16),
               upsample_rates=(2,2,2,2,2),
               n_upsample_blocks=5,
               block_type='upsampling',
               activation='sigmoid',
               use_batchnorm=True):

    input = backbone.input
    x = backbone.output

    if block_type == 'transpose':
        up_block = Transpose2D_block
    else:
        up_block = Upsample2D_block

    # convert layer names to indices
    skip_connection_idx = ([get_layer_number(backbone, l) if isinstance(l, str) else l
                               for l in skip_connection_layers])

    for i in range(n_upsample_blocks):

        # check if there is a skip connection
        skip_connection = None
        if i < len(skip_connection_idx):
            skip_connection = backbone.layers[skip_connection_idx[i]].output

        upsample_rate = to_tuple(upsample_rates[i])

        x = up_block(decoder_filters[i], i, upsample_rate=upsample_rate,
                     skip=skip_connection, use_batchnorm=use_batchnorm)(x)

    x = Conv2D(classes, (3,3), padding='same', name='final_conv')(x)
    x = Activation(activation, name=activation)(x)

    model = Model(input, x)

    return model
'''

DEFAULT_SKIP_CONNECTIONS = {
    'vgg16':            ('block5_conv3', 'block4_conv3', 'block3_conv3', 'block2_conv2', 'block1_conv2'),
    'vgg19':            ('block5_conv4', 'block4_conv4', 'block3_conv4', 'block2_conv2', 'block1_conv2'),
    'resnet18':         ('stage4_unit1_relu1', 'stage3_unit1_relu1', 'stage2_unit1_relu1', 'relu0'), # check 'bn_data'
    'resnet34':         ('stage4_unit1_relu1', 'stage3_unit1_relu1', 'stage2_unit1_relu1', 'relu0'),
    'resnet50':         ('stage4_unit1_relu1', 'stage3_unit1_relu1', 'stage2_unit1_relu1', 'relu0'),
    'midresnet50':         ('mid_stage4_unit1_relu1', 'mid_stage3_unit1_relu1', 'mid_stage2_unit1_relu1', 'mid_relu0'),
    'resnet101':        ('stage4_unit1_relu1', 'stage3_unit1_relu1', 'stage2_unit1_relu1', 'relu0'),
    'resnet152':        ('stage4_unit1_relu1', 'stage3_unit1_relu1', 'stage2_unit1_relu1', 'relu0'),
    'resnext50':        ('stage4_unit1_relu1', 'stage3_unit1_relu1', 'stage2_unit1_relu1', 'relu0'),
    'resnext101':       ('stage4_unit1_relu1', 'stage3_unit1_relu1', 'stage2_unit1_relu1', 'relu0'),
    'inceptionv3':          (228, 86, 16, 9),
    'inceptionresnetv2':    (594, 260, 16, 9),
    'densenet121':          (311, 139, 51, 4),
    'densenet169':          (367, 139, 51, 4),
    'densenet201':          (479, 139, 51, 4),
}


def MiddleUnet(backbone_name1='vgg16', backbone_name2='vgg16',
         input_shape1=(None, None, 2), input_shape2=(None, None, 11),
         input_tensor=None,
         encoder_weights='imagenet',
         freeze_encoder=False,
         skip_connections='default',
         decoder_block_type='upsampling',
         decoder_filters=(256,128,64,32,16),
         decoder_use_batchnorm=True,
         n_upsample_blocks=5,
         upsample_rates=(2,2,2,2,2),
         classes=1,
         activation='sigmoid',
         strategy='concat'):
    """

    Args:
        backbone_name: (str) look at list of available backbones.
        input_shape:  (tuple) dimensions of input data (H, W, C)
        input_tensor: keras tensor
        encoder_weights: one of `None` (random initialization), 
            'imagenet' (pre-training on ImageNet), 
            'dof' (pre-training on DoF)
        freeze_encoder: (bool) Set encoder layers weights as non-trainable. Useful for fine-tuning
        skip_connections: if 'default' is used take default skip connections,
            else provide a list of layer numbers or names starting from top of model
        decoder_block_type: (str) one of 'upsampling' and 'transpose' (look at blocks.py)
        decoder_filters: (int) number of convolution layer filters in decoder blocks
        decoder_use_batchnorm: (bool) if True add batch normalisation layer between `Conv2D` ad `Activation` layers
        n_upsample_blocks: (int) a number of upsampling blocks
        upsample_rates: (tuple of int) upsampling rates decoder blocks
        classes: (int) a number of classes for output
        activation: (str) one of keras activations for last model layer

    Returns:
        keras.models.Model instance

    """
    print('input_shape1 in middle ',input_shape1)
    print('input_shape2 in middle ',input_shape2)
    backbone1 = get_backbone(backbone_name1,
                            input_shape=input_shape1,
                            input_tensor=input_tensor,
                            weights=encoder_weights,
                            include_top=False)

    
    backbone2 = get_backbone(backbone_name2,   # has to be MidResNet50
                            input_shape=input_shape2,
                            input_tensor=input_tensor,
                            weights=encoder_weights,
                            include_top=False)

    if skip_connections == 'default':
        skip_connections1 = DEFAULT_SKIP_CONNECTIONS[backbone_name1]
        skip_connections2 = DEFAULT_SKIP_CONNECTIONS[backbone_name2]
    else:
        skip_connections1 = skip_connections
        skip_connections2 = skip_connections

    
    # Convert layer names to indices
    skip_connection_idx1 = [get_layer_number(backbone1, l) if isinstance(l, str) else l
                           for l in skip_connections1]
    skip_connection_idx2 = [get_layer_number(backbone2, l) if isinstance(l, str) else l
                           for l in skip_connections2]

    # Extract skip connections from both backbones
    skip1 = [backbone1.layers[idx].output for idx in skip_connection_idx1]
    skip2 = [backbone2.layers[idx].output for idx in skip_connection_idx2]

    # Combine skip connections (e.g., concatenate)
    combined_skips = [Concatenate()([s1, s2]) for s1, s2 in zip(skip1, skip2)]

    # Build the decoder using combined skip connections

    x = [backbone1.output, backbone2.output]
    if strategy == 'concat':
        x = Concatenate()(x)
    elif strategy == 'average':
        fusion.WeightedAverage(n_output=len(x))(x)

        
    for i in range(n_upsample_blocks):
        # Use combined skip connections
        skip_connection = combined_skips[i] if i < len(combined_skips) else None

        upsample_rate = to_tuple(upsample_rates[i])

        if decoder_block_type == 'transpose':
            x = Transpose2D_block(decoder_filters[i], i, upsample_rate=upsample_rate,
                                  skip=skip_connection, use_batchnorm=decoder_use_batchnorm)(x)
        else:
            x = Upsample2D_block(decoder_filters[i], i, upsample_rate=upsample_rate,
                                 skip=skip_connection, use_batchnorm=decoder_use_batchnorm)(x)

    # Final layers
    x = Conv2D(classes, (3, 3), padding='same', name='final_conv')(x)
    x = Activation(activation, name=activation)(x)

    # Create the model
    model = Model([backbone1.input, backbone2.input], x)
 

    # lock encoder weights for fine-tuning
  #  if freeze_encoder:
   #     freeze_model(backbone)

    return model


######### utils

def freeze_model(model):
    for layer in model.layers:
        layer.trainable = False
    return

def to_tuple(x):
    if isinstance(x, tuple):
        if len(x) == 2:
            return x
    elif np.isscalar(x):
        return (x, x)

    raise ValueError('Value should be tuple of length 2 or int value, got "{}"'.format(x))

def get_layer_number(model, layer_name):
    """
    Help find layer in Keras model by name
    Args:
        model: Keras `Model`
        layer_name: str, name of layer

    Returns:
        index of layer

    Raises:
        ValueError: if model does not contains layer with such name
    """
    for i, l in enumerate(model.layers):
        if l.name == layer_name:
            return i
    raise ValueError('No layer with name {} in  model {}.'.format(layer_name, model.name))


from custom_classification_models.classification_models import MidResNet50, ResNet50 #ResNet18, ResNet34, ResNet101, ResNet152
#from .classification_models.classification_models import ResNeXt50, ResNeXt101


backbones = {
  #  "resnet18": ResNet18,
    "resnet50": ResNet50,
    "midresnet50": MidResNet50,
  #  "resnet101": ResNet101,
  #  "resnet152": ResNet152,
    #"resnext50": ResNeXt50,
   # "resnext101": ResNeXt101,

}
def get_backbone(name, *args, **kwargs):
    return backbones[name](*args, **kwargs)


# FUSION

class GridAttention(Model):
    def __init__(self, inter_channels):
        super(GridAttention, self).__init__()
        self.inter_channels = inter_channels
        
        self.theta_x = layers.Conv2D(inter_channels, kernel_size=2, strides=2, padding='same')
        self.upsample = layers.UpSampling2D(size=(2, 2))
        self.phi_g = layers.Conv2D(inter_channels, kernel_size=1, strides=1, padding='same')
        self.add = layers.Add()
        self.relu = layers.Activation('relu')
        self.attention_conv = layers.Conv2D(1, kernel_size=1)
        self.sigmoid = layers.Activation('sigmoid')
        self.multiply = layers.Multiply()
    
    def call(self, features, gating):
        # Upsample features to match gating size
        theta_x = self.theta_x(features)
        theta_x = self.upsample(theta_x)
        
        # Apply phi_g
        phi_g = self.phi_g(gating)
        
        # Add and apply attention
        add = self.add([theta_x, phi_g])
        attention_map = self.relu(add)
        attention_map = self.attention_conv(attention_map)
        attention_map = self.sigmoid(attention_map)
        
        # Apply attention to features
        attention_weighted = self.multiply([features, attention_map])
        
        return attention_weighted