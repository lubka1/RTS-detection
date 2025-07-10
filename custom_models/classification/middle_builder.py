import tensorflow.keras.backend as K
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, BatchNormalization, 
    Activation, GlobalAveragePooling2D, ZeroPadding2D, Dense
)
from tensorflow.keras.models import Model
from tensorflow.keras.utils import get_source_inputs

import tensorflow as tf

from distutils.version import StrictVersion


from .params import get_conv_params
from .params import get_bn_params
from .middle_blocks import basic_conv_block
from .middle_blocks import basic_identity_block
from .middle_blocks import conv_block as usual_conv_block
from .middle_blocks import identity_block as usual_identity_block

def get_input_shape(input_shape=None):
    if input_shape is None:
        input_shape = (256, 256, 3)  # Example input shape (height, width, channels)
    #print('input_shape: ', input_shape)
    return input_shape

def mid_build_resnet(
    repetitions=(2, 2, 2, 2),
    include_top=True,
    input_tensor=None,
    input_shape=None,
    classes=1000,
    block_type='usual',
    name_prefix='mid'  # here we will have two different prefexies
):

    input_shape = get_input_shape(input_shape)

    if input_tensor is None:
        img_input = Input(shape=input_shape, name=f'{name_prefix}_data')
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    # get parameters for model layers
    no_scale_bn_params = get_bn_params(scale=False)
    bn_params = get_bn_params()
    conv_params = get_conv_params()
    init_filters = 64

    if block_type == 'basic':
        conv_block = basic_conv_block
        identity_block = basic_identity_block
    else:
        conv_block = usual_conv_block
        identity_block = usual_identity_block

    # resnet bottom
    x = BatchNormalization(name=f'{name_prefix}_bn_data', **no_scale_bn_params)(img_input)
    x = ZeroPadding2D(padding=(3, 3))(x)
    x = Conv2D(init_filters, (7, 7), strides=(2, 2), name=f'{name_prefix}_conv0', **conv_params)(x)
    x = BatchNormalization(name=f'{name_prefix}_bn0', **bn_params)(x)
    x = Activation('relu', name=f'{name_prefix}_relu0')(x)
    x = ZeroPadding2D(padding=(1, 1))(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='valid', name=f'{name_prefix}_pooling0')(x)

    # resnet body
    for stage, rep in enumerate(repetitions):
        for block in range(rep):
            filters = init_filters * (2 ** stage)

            if block == 0 and stage == 0:
                x = conv_block(filters, stage, block,name_prefix, strides=(1, 1))(x)
            elif block == 0:
                x = conv_block(filters, stage, block,name_prefix, strides=(2, 2))(x)
            else:
                x = identity_block(filters, stage, block, name_prefix)(x)  

    x = BatchNormalization(name=f'{name_prefix}_bn1', **bn_params)(x)
    x = Activation('relu', name=f'{name_prefix}_relu1')(x)

    # resnet top
    if include_top:
        x = GlobalAveragePooling2D(name=f'{name_prefix}_pool1')(x)
        x = Dense(classes, name=f'{name_prefix}_fc1')(x)
        x = Activation('softmax', name=f'{name_prefix}_softmax')(x)

    # finalize model
    inputs = get_source_inputs(input_tensor) if input_tensor is not None else img_input
    model = Model(inputs, x)

    return model
