import os
import numpy as np
import matplotlib.pyplot as plt
import rasterio
import albumentations as A
import keras

def read_tif(file_path, as_gray=False, normalize=True):
    with rasterio.open(file_path) as src:
        if as_gray:
            # Read the first band only for grayscale
            data = src.read(1)
        else:
            # Read all bands
            data = src.read()
            if data.shape[0] > 1:  # If the data has multiple 
                data = np.transpose(data, (1, 2, 0))  # Reordero to (height, width, channels)

            if normalize:
                # Normalize each band separately
                data = data.astype(np.float32)
                for i in range(data.shape[-1]):  # Normalize each band
                    band = data[..., i]
                    data[..., i] = normalize_image(band)    
        return data  
    
"""def normalize_image(image):
    Normalize the image to the range [0, 1].
    image_min = image.min()
    image_max = image.max()
    return (image - image_min) / (image_max - image_min)"""

def normalize_image(image):
    """Normalize the image to the range [0, 1]."""
    image_min = image.min()
    image_max = image.max()
    if image_min != image_max:
        return (image - image_min) / (image_max - image_min)
    else:
        return image-image_min

# helper function for data visualization    
def denormalize(x):
    """Scale image to range 0..1 for correct plot"""
    x_max = np.percentile(x, 98)
    x_min = np.percentile(x, 2)    
    x = (x - x_min) / (x_max - x_min)
    x = x.clip(0, 1)
    return x
    
def calculate_ndvi( nir_band, red_band):
    ndvi = (nir_band - red_band) / (nir_band + red_band + 1e-10)  # Adding a small constant to avoid division by zero
    return ndvi


# classes for data loading and preprocessing
class Dataset:
    """ Dataset. Read images, apply augmentation and preprocessing transformations.
    
    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        class_values (list): values of classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transfromation pipeline 
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing 
            (e.g. noralization, shape manipulation, etc.)
        fusion (bool): whether to use fusion with another set of images
        images_dir2 (str): path to the second set of images folder (for fusion)    
        ndvi (bool): whether to calculate and include NDVI as an additional channel
    """
    # in case of fusion and ndvi the order of the bands is: S2, S1, ndvi

    CLASSES = ['rts']
    
    def __init__(
            self, 
            images_dir, 
            masks_dir, 
            classes=None, 
            augmentation=None, 
            preprocessing=None,
            fusion=False,
            images_dir2=None,
            ndvi=False,
    ):
        self.ids = os.listdir(images_dir)
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids]

        self.fusion = fusion
        if self.fusion:
            assert images_dir2 is not None, "When fusion is enabled, images_dir2 must be provided."
            self.images_fps2 = [os.path.join(images_dir2, image_id) for image_id in self.ids]
        
        # convert str names to class values on masks
        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes] # check if I shouldnt remove this?
        
        self.augmentation = augmentation
        self.preprocessing = preprocessing
        self.ndvi = ndvi
    
    def __getitem__(self, i):
 # takze tuto citam uz normalized image, asi uz nepotrebujem to dat na float
        image1 = read_tif(self.images_fps[i]).astype(np.float32) # np.uint16

        # fusion of S2 and S1
        if self.fusion:
            image2 = read_tif(self.images_fps2[i]).astype(np.float32) # np.uint16
            image = np.concatenate((image1, image2), axis=-1)
        else:
            image = image1

        # Concatenate NDVI as a new channel if required
        if self.ndvi:
            red_band = image1[:, :, 3]  # Band 4
            nir_band = image1[:, :, 7]  # Band 8
            ndvi = calculate_ndvi(nir_band, red_band)
            ndvi = np.expand_dims(ndvi, axis=2)  # Expand dims to add as a channel
            #print(red_band.shape,ndvi.shape)
            image = np.concatenate((image, ndvi), axis=-1)    

        mask = read_tif(self.masks_fps[i], as_gray=True)
        mask_array = np.array(mask)
        mask = (mask_array / 250).astype(np.uint16)
        #masks = [(mask == v) for v in self.class_values]
        masks = [(~(mask == v)) for v in self.class_values]  # Inverting the boolean mask
        mask = np.stack(masks, axis=-1).astype('float')
        
        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        
        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
            
        return image, mask
        
    def __len__(self):
        return len(self.ids)
    
    
class Dataloder(keras.utils.Sequence):
    """Load data from dataset and form batches
    
    Args:
        dataset: instance of Dataset class for image loading and preprocessing.
        batch_size: Integet number of images in batch.
        shuffle: Boolean, if `True` shuffle image indexes each epoch.
    """
    
    def __init__(self, dataset, batch_size=1, shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(dataset))

        self.on_epoch_end()

    def __getitem__(self, i):
        
        # collect batch data
        start = i * self.batch_size
        stop = (i + 1) * self.batch_size
        data = []
        for j in range(start, stop):
            data.append(self.dataset[j])
        
        # transpose list of lists
        batch = [np.stack(samples, axis=0) for samples in zip(*data)]
        
        return batch
    
    
    def __len__(self):
        """Denotes the number of batches per epoch"""
        return len(self.indexes) // self.batch_size
    
    def on_epoch_end(self):
        """Callback function to shuffle indexes each epoch"""
        if self.shuffle:
            self.indexes = np.random.permutation(self.indexes)   



class MyDataGenerator:
    def __init__(self, shuffle=True, seed=42):
        self.shuffle = shuffle
        self.random_state = np.random.RandomState(seed)  # Create a local random state

    def on_epoch_end(self):
        """Shuffle indexes each epoch using a fixed random state"""
        if self.shuffle:
            self.indexes = self.random_state.permutation(self.indexes)


# AUGMENTATION            

def round_clip_0_1(x, **kwargs):
    return x.round().clip(0, 1)

train_transform = [

    A.HorizontalFlip(p=0.5),

    A.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0),

    A.PadIfNeeded(min_height=256, min_width=256, always_apply=True, border_mode=0, value=0 ),
    
    A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),  
    A.Perspective(p=0.5),

    A.OneOf(
        [
            A.RandomGamma(p=1),
            A.RandomCrop(height=256, width=256),
        ],
        p=0.9,
    ),

    A.OneOf(
        [
            A.Sharpen(p=1),
            A.Blur(blur_limit=3, p=1),
            A.MotionBlur(blur_limit=3, p=1),
        ],
        p=0.9,
    ),

    A.OneOf(
        [
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5,) 
        ],
        p=0.9,
    ),
    A.Lambda(mask=round_clip_0_1)
]

# define heavy augmentations
def get_training_augmentation(train_transform=train_transform):
    return A.Compose(train_transform)


def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform
    
    Args:
        preprocessing_fn (callable): data normalization function 
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose
    
    """
    
    _transform = [
        A.Lambda(image=preprocessing_fn),
    ]
    return A.Compose(_transform)

class RandomChoice(A.BasicTransform):
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


