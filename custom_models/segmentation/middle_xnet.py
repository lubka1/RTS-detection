# https://github.com/MrGiovanni/UNetPlusPlus/tree/master/keras/segmentation_models/xnet
# blocks, builder and model in one

from keras.layers import Conv2DTranspose
from keras.layers import UpSampling2D
from keras.layers import Conv2D
from keras.layers import BatchNormalization
from keras.layers import Activation
from keras.layers import Concatenate
from keras.models import Model
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

import fusion
import cbam 

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))   

# blocks.py
def handle_block_names(stage, cols):
    conv_name = 'decoder_stage{}-{}_conv'.format(stage, cols)
    bn_name = 'decoder_stage{}-{}_bn'.format(stage, cols)
    relu_name = 'decoder_stage{}-{}_relu'.format(stage, cols)
    up_name = 'decoder_stage{}-{}_upsample'.format(stage, cols)
    merge_name = 'merge_{}-{}'.format(stage, cols)
    return conv_name, bn_name, relu_name, up_name, merge_name


def ConvRelu(filters, kernel_size, use_batchnorm=False, conv_name='conv', bn_name='bn', relu_name='relu'):
    def layer(x):
        x = Conv2D(filters, kernel_size, padding="same", name=conv_name, use_bias=not(use_batchnorm))(x)
        if use_batchnorm:
            x = BatchNormalization(name=bn_name)(x)
        x = Activation('relu', name=relu_name)(x)
        return x
    return layer


def Upsample2D_block(filters, stage, cols, kernel_size=(3,3), upsample_rate=(2,2),
                     use_batchnorm=False, skip=None):

    def layer(input_tensor):

        conv_name, bn_name, relu_name, up_name, merge_name = handle_block_names(stage, cols)

        x = UpSampling2D(size=upsample_rate, name=up_name)(input_tensor)

        if (type(skip) != list and skip is not None) or (type(skip) == list and None not in skip):
            if type(skip) is list:
                x = Concatenate(name=merge_name)([x] + skip)
            else:
                x = Concatenate(name=merge_name)([x, skip])

        x = ConvRelu(filters, kernel_size, use_batchnorm=use_batchnorm,
                     conv_name=conv_name + '1', bn_name=bn_name + '1', relu_name=relu_name + '1')(x)

        x = ConvRelu(filters, kernel_size, use_batchnorm=use_batchnorm,
                     conv_name=conv_name + '2', bn_name=bn_name + '2', relu_name=relu_name + '2')(x)

        return x
    return layer


def Transpose2D_block(filters, stage, cols, kernel_size=(3,3), upsample_rate=(2,2),
                      transpose_kernel_size=(4,4), use_batchnorm=False, skip=None):

    def layer(input_tensor):

        conv_name, bn_name, relu_name, up_name, merge_name = handle_block_names(stage, cols)

        x = Conv2DTranspose(filters, transpose_kernel_size, strides=upsample_rate,
                            padding='same', name=up_name, use_bias=not(use_batchnorm))(input_tensor)
        if use_batchnorm:
            x = BatchNormalization(name=bn_name+'1')(x)
        x = Activation('relu', name=relu_name+'1')(x)

        if (type(skip) != list and skip is not None) or (type(skip) == list and None not in skip):
            # print("\nskip = {}".format(skip))
            if type(skip) is list:
                merge_list = []
                merge_list.append(x)
                for l in skip:
                    merge_list.append(l)
                x = Concatenate(name=merge_name)(merge_list)
            else:
                x = Concatenate(name=merge_name)([x, skip])

        x = ConvRelu(filters, kernel_size, use_batchnorm=use_batchnorm,
                     conv_name=conv_name + '2', bn_name=bn_name + '2', relu_name=relu_name + '2')(x)

        return x
    return layer


# model.py
DEFAULT_SKIP_CONNECTIONS = {
 
    'resnet18':         ('stage4_unit1_relu1', 'stage3_unit1_relu1', 'stage2_unit1_relu1', 'relu0'), 
    'resnet34':         ('stage4_unit1_relu1', 'stage3_unit1_relu1', 'stage2_unit1_relu1', 'relu0'),
# we are using resnet50
    'resnet50':         ('one_stage4_unit1_relu1', 'one_stage3_unit1_relu1', 'one_stage2_unit1_relu1', 'one_relu0'), 
    'midresnet50':         ('mid_stage4_unit1_relu1', 'mid_stage3_unit1_relu1', 'mid_stage2_unit1_relu1', 'mid_relu0'),
}


def MiddleXnet(backbone_name1='vgg16', backbone_name2='vgg16', 
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
         strategy='concat',
         attention=False):
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

    
    backbone2 = get_backbone(backbone_name2,   
                            input_tensor=input_tensor,
                            input_shape=input_shape2,
                            weights=encoder_weights,
                            include_top=False)

    if skip_connections == 'default':
        skip_connection_layers_1 = DEFAULT_SKIP_CONNECTIONS[backbone_name1]
        skip_connection_layers_2 = DEFAULT_SKIP_CONNECTIONS[backbone_name2]
    else:
        skip_connection_layers_1 = skip_connections
        skip_connection_layers_2 = skip_connections

    # builder.py
    #  bez transpose
    
    up_block = Upsample2D_block
    
    if len(skip_connection_layers_1) > n_upsample_blocks:
        downsampling_layers_1 = skip_connection_layers_1[int(len(skip_connection_layers_1)/2):]
        skip_connection_layers_1 = skip_connection_layers_1[:int(len(skip_connection_layers_1)/2)]
    else:
        downsampling_layers_1 = skip_connection_layers_1

    if len(skip_connection_layers_2) > n_upsample_blocks:
        downsampling_layers_2 = skip_connection_layers_2[int(len(skip_connection_layers_2)/2):]
        skip_connection_layers_2 = skip_connection_layers_2[:int(len(skip_connection_layers_2)/2)]
    else:
        downsampling_layers_2 = skip_connection_layers_2

    # Convert layer names to indices
    skip_connection_idx_1 = [get_layer_number(backbone1, l) if isinstance(l, str) else l
                           for l in skip_connection_layers_1]
    skip_connection_idx_2 = [get_layer_number(backbone2, l) if isinstance(l, str) else l
                           for l in skip_connection_layers_2]

    # Extract skip connections from both backbones
    skip_layers_list_1 = [backbone1.layers[skip_connection_idx_1[i]].output for i in range(len(skip_connection_idx_1))]
    skip_layers_list_2 = [backbone2.layers[skip_connection_idx_2[i]].output for i in range(len(skip_connection_idx_2))]

    downsampling_idx_1 = ([get_layer_number(backbone1, l) if isinstance(l, str) else l
                               for l in downsampling_layers_1])
    downsampling_list_1 = [backbone1.layers[downsampling_idx_1[i]].output for i in range(len(downsampling_idx_1))]

    downsampling_idx_2 = ([get_layer_number(backbone2, l) if isinstance(l, str) else l
                               for l in downsampling_layers_2])
    downsampling_list_2 = [backbone2.layers[downsampling_idx_2[i]].output for i in range(len(downsampling_idx_2))]

    downterm_1 = [None] * (n_upsample_blocks+1)
    for i in range(len(downsampling_idx_1)):
        if downsampling_list_1[0] == backbone1.output:
            downterm_1[n_upsample_blocks-i] = downsampling_list_1[i]
        else:
            downterm_1[n_upsample_blocks-i-1] = downsampling_list_1[i]
    downterm_1[-1] = backbone1.output

    downterm_2 = [None] * (n_upsample_blocks+1)
    for i in range(len(downsampling_idx_2)):
        if downsampling_list_2[0] == backbone2.output:
            downterm_2[n_upsample_blocks-i] = downsampling_list_2[i]
        else:
            downterm_2[n_upsample_blocks-i-1] = downsampling_list_2[i]
    downterm_2[-1] = backbone2.output

    downterm = [Concatenate()([d1, d2]) if d1 is not None else d2 for d1, d2 in zip(downterm_1, downterm_2)]

    # interm is a 2-dimensional grid of intermediate decoder features in X-Net.
    interm_1 = [None] * (n_upsample_blocks+1) * (n_upsample_blocks+1)
    for i in range(len(skip_connection_idx_1)):
        interm_1[-i*(n_upsample_blocks+1)+(n_upsample_blocks+1)*(n_upsample_blocks-1)] = skip_layers_list_1[i]
    interm_1[(n_upsample_blocks+1)*n_upsample_blocks] = backbone1.output

    interm_2 = [None] * (n_upsample_blocks+1) * (n_upsample_blocks+1)
    for i in range(len(skip_connection_idx_2)):
        interm_2[-i*(n_upsample_blocks+1)+(n_upsample_blocks+1)*(n_upsample_blocks-1)] = skip_layers_list_2[i]
    interm_2[(n_upsample_blocks+1)*n_upsample_blocks] = backbone2.output

    # Apply attention to each element of the intermediate grid
    if attention:
        interm_1 = [cbam.attach_attention_module(f) if f is not None else None for f in interm_1]
        interm_2 = [cbam.attach_attention_module(f) if f is not None else None for f in interm_2]
        print('With Attention')

    if strategy == 'average':
        interm = [
            fusion.WeightedAverage(n_output=sum(x is not None for x in [x1, x2]))([x for x in [x1, x2] if x is not None])
            if x1 is not None or x2 is not None else None
            for x1, x2 in zip(interm_1, interm_2)
        ]
    else: #strategy == 'concat'
        interm = [Concatenate()(inputs=[x1, x2]) if x1 is not None else x2 for x1, x2 in zip(interm_1, interm_2)]

    for j in range(n_upsample_blocks):
        for i in range(n_upsample_blocks-j):
            upsample_rate = to_tuple(upsample_rates[i])
            
            if i == 0 and j < n_upsample_blocks-1 and len(skip_connection_layers_1) < n_upsample_blocks:   
                interm[(n_upsample_blocks+1)*i+j+1] = None
            elif j == 0:
                if downterm[i+1] is not None:
                    interm[(n_upsample_blocks+1)*i+j+1] = up_block(decoder_filters[n_upsample_blocks-i-2], 
                                      i+1, j+1, upsample_rate=upsample_rate,
                                      skip=interm[(n_upsample_blocks+1)*i+j], 
                                      use_batchnorm=decoder_use_batchnorm)(downterm[i+1])  
                else:
                    interm[(n_upsample_blocks+1)*i+j+1] = None
            else:
                interm[(n_upsample_blocks+1)*i+j+1] = up_block(decoder_filters[n_upsample_blocks-i-2], 
                                  i+1, j+1, upsample_rate=upsample_rate,
                                  skip=interm[(n_upsample_blocks+1)*i : (n_upsample_blocks+1)*i+j+1], 
                                  use_batchnorm=decoder_use_batchnorm)(interm[(n_upsample_blocks+1)*(i+1)+j])

    x = Conv2D(classes, (3,3), padding='same', name='final_conv')(interm[n_upsample_blocks])  # interm[n_upsample_blocks]) is just one “diagonal” element of the 2D interm grid.  OR instead fuse all elements from the last row or column of interm before the final conv
    x = Activation(activation, name=activation)(x)

    model = Model([backbone1.input, backbone2.input], x)

    # lock encoder weights for fine-tuning
    #if freeze_encoder:
    #    freeze_model(backbone)

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


from custom_models.classification.models import MidResNet50, ResNet50 


backbones = {

    "resnet50": ResNet50,
    "midresnet50": MidResNet50,


}
def get_backbone(name, *args, **kwargs):
    return backbones[name](*args, **kwargs)


