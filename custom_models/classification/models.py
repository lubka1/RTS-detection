#from ..utils import load_model_weights
#from ..weights import weights_collection
from .middle_builder import mid_build_resnet

# these two functions call the same funcion, redundant?

def ResNet50(input_shape, input_tensor=None, weights=None, classes=1000, include_top=True):
    model = mid_build_resnet(input_tensor=input_tensor,
                         input_shape=input_shape,
                         repetitions=(3, 4, 6, 3),
                         classes=classes,
                         include_top=include_top,
                         name_prefix='one')  
    model.name = 'resnet50'
    print('Initialize resnet50')

    return model

def MidResNet50(input_shape, input_tensor=None, weights=None, classes=1000, include_top=True):
    model = mid_build_resnet(input_tensor=input_tensor,
                         input_shape=input_shape,
                         repetitions=(3, 4, 6, 3),
                         classes=classes,
                         include_top=include_top,
                         name_prefix='mid')
    model.name = 'midresnet50'
    print('initialize MIDresnet50')

    return model
