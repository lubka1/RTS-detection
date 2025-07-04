import config  
import utils

import os
import numpy as np
import matplotlib.pyplot as plt
import rasterio
import segmentation_models as sm
import tensorflow.keras as keras

BACKBONE = config.BACKBONE
BATCH_SIZE = config.BATCH_SIZE
preprocess_input = sm.get_preprocessing(BACKBONE)

# DATA LOADING

def get_data(fusion_type):  
        
        if fusion_type == 'early':

            train_dataset = EarlyDataset(
                s1_dir=config.S1_train_dir, 
                s2_dir=config.S2_train_dir, 
                dem_dir=config.DEM_train, 
                masks_dir=config.y_train_dir, 
                augmentation=utils.get_training_augmentation(),
                preprocessing=utils.get_preprocessing(preprocess_input),
                classes=['rts'],
               # ndvi=True,  because i set it explicuely in dataset?????? wtf
            )
            print('Number of training data: {}'.format(len(train_dataset.ids)))

            val_dataset = EarlyDataset(
                s1_dir=config.S1_valid_dir,
                s2_dir=config.S2_valid_dir, 
                masks_dir=config.y_valid_dir, 
                dem_dir=config.DEM_val,
                augmentation=None,
                preprocessing=utils.get_preprocessing(preprocess_input),
                classes=['rts'],
             #   ndvi=True,
            )

            # Determine the number of channels in the image
            image, mask = train_dataset[12]  
            N = image.shape[-1]
            M = None
            print(N)

            train_dataloader = Dataloder(train_dataset, batch_size=BATCH_SIZE, shuffle=True)   
            valid_dataloader = Dataloder(val_dataset, batch_size=1, shuffle=False)
            print(train_dataloader[0][1].shape)

            if train_dataloader[0][0].shape != (BATCH_SIZE, 256, 256, N):
                raise ValueError(f"Input shape: {train_dataloader[0][0].shape}, expected: (BATCH_SIZE, 256, 256, {N})")
            if train_dataloader[0][1].shape != (BATCH_SIZE, 256, 256, 1):
                raise ValueError(f"Output shape: {train_dataloader[0][1].shape}, expected: (BATCH_SIZE, 256, 256, 1)")
            
        elif fusion_type in ['middle', 'late']:

            train_dataset = FusionDataset(
                images_dir1=config.S1_train_dir,  
                images_dir2=config.S2_train_dir,  
                masks_dir=config.y_train_dir,  
              #  dem_dir=config.DEM_train, 
                classes=['rts'],  
               augmentation=utils.get_training_augmentation(),  
                preprocessing=utils.get_preprocessing(preprocess_input),  
             ndvi=True  # Set True to include NDVI as an additional channel
            )

            val_dataset =  FusionDataset(
                images_dir1=config.S1_valid_dir,  
                images_dir2=config.S2_valid_dir,  
                masks_dir=config.y_valid_dir,  
              #  dem_dir=config.DEM_val,
                classes=['rts'],
                augmentation=None,  # No augmentation for validation
                preprocessing=utils.get_preprocessing(preprocess_input), 
             ndvi=True  # Include NDVI if used in training
            )

            train_dataloader = FusionDataloder(train_dataset, BATCH_SIZE, shuffle=True)
            valid_dataloader = FusionDataloder(val_dataset, 1, shuffle=False)  # No shuffling for validation

            # Determine the number of channels in the images
            (images1, images2), mask = train_dataset[0]
            M = images1.shape[-1]  
            N = images2.shape[-1]  

            if train_dataloader[0][0][0].shape != (BATCH_SIZE, 256, 256, M):
                raise ValueError(f"Expected image1 shape {(BATCH_SIZE, 256, 256, M)}, but got {train_dataloader[0][0][0].shape}")

            if train_dataloader[0][0][1].shape != (BATCH_SIZE, 256, 256, N):
                raise ValueError(f"Expected image2 shape {(BATCH_SIZE, 256, 256, N)}, but got {train_dataloader[0][0][1].shape}")

            if train_dataloader[0][1].shape != (BATCH_SIZE, 256, 256, 1):
                raise ValueError(f"Expected mask shape {(BATCH_SIZE, 256, 256, 1)}, but got {train_dataloader[0][1].shape}")

        return train_dataloader, valid_dataloader, N, M
'''
def read_tif(file_path, as_gray=False, normalize=True):
    with rasterio.open(file_path) as src:
        if as_gray:
            # Read the first band only for grayscale
            data = src.read(1)
            data = data.astype(np.float32)
        else:
            # Read all bands
            data = src.read()
            if data.shape[0] > 1:  # If the data has multiple 
                data = np.transpose(data, (1, 2, 0))  # Reorder to (height, width, channels)

            if normalize:
                # Normalize each band separately
                data = data.astype(np.float32)
                for i in range(data.shape[-1]):  
                    band = data[..., i]
                    data[..., i] = normalize_image(band)    
        return data  
'''  
def read_tif(file_path, as_gray=False, normalize=True, bands=None):
    with rasterio.open(file_path) as src:
        if as_gray:
            data = src.read(1).astype(np.float32)
        else:
            if bands is not None:
                # Convert band indices from human-readable (1-based) to 0-based indexing
                bands_zero_indexed = [b for b in bands]
                data = np.stack([src.read(b).astype(np.float32) for b in bands_zero_indexed], axis=-1)
            else:
                data = src.read().astype(np.float32)
                if data.shape[0] > 1:
                    data = np.transpose(data, (1, 2, 0))

            if normalize:
                if bands is not None:
                    for i in range(data.shape[-1]):
                        band = data[..., i]
                        data[..., i] = normalize_image(band)
                else:
                    for i in range(data.shape[-1]):
                        band = data[..., i]
                        data[..., i] = normalize_image(band)

        return data  

def normalize_image(image):
    image_min = image.min()
    image_max = image.max()
    if image_max - image_min == 0:
        return np.zeros_like(image, dtype=np.float32)  # or np.ones_like(image)
    else:
        return (image - image_min) / (image_max - image_min)

# helper function for data visualization    
def denormalize(x):
    """Scale image to range 0..1 for correct plot"""
    x_max = np.percentile(x, 98)
    x_min = np.percentile(x, 2)    
    x = (x - x_min) / (x_max - x_min)
    x = x.clip(0, 1)
    return x
    
def calculate_ndvi(nir_band, red_band):
    """Calculate NDVI given the Red and NIR bands."""
    nir_band = nir_band.astype(np.float32)
    red_band = red_band.astype(np.float32)
    epsilon = 1e-6  # Small constant to prevent division by zero
    return (nir_band - red_band) / (nir_band + red_band + epsilon)

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
            self.images_fps2 = [os.path.join(images_dir2, image_id) for image_id in self.ids]
        
        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes] 
        
        self.augmentation = augmentation
        self.preprocessing = preprocessing
        self.ndvi = ndvi
    
    def __getitem__(self, i):
        image1 = read_tif(self.images_fps[i]).astype(np.float32) 

        # fusion of S2 and S1
        if self.fusion:
            image2 = read_tif(self.images_fps2[i]).astype(np.float32) 
            image = np.concatenate((image1, image2), axis=-1)
        else:
            image = image1

        # Concatenate NDVI as a new channel if required
        if self.ndvi:
            red_band = image1[:, :, 3]  # Band 4
            nir_band = image1[:, :, 7]  # Band 8
            ndvi = calculate_ndvi(nir_band, red_band)
            ndvi = np.expand_dims(ndvi, axis=2)  # Expand dims to add as a channel
            image = np.concatenate((image, ndvi), axis=-1)    

        mask = read_tif(self.masks_fps[i], as_gray=True)
        mask_array = np.array(mask)
        mask = (mask_array / 250).astype(np.float32)
        masks = [(mask == v) for v in self.class_values]
        #masks = [(~(mask == v)) for v in self.class_values]  # Inverting the boolean mask
        mask = np.stack(masks, axis=-1).astype('float')
        #print(f"final mask shape: {mask.shape}")  
        
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
    
class EarlyDataset:
    """ 
    Dataset. Read images, apply augmentation and preprocessing transformations.
    Add NDVI band to Sentinel2 and Slope and Elevation band to Sentinel1.
    
    Args:
        s1_dir (str): path to Sentinel-1 images folder
        s2_dir (str): path to Sentinel-2 images folder
        masks_dir (str): path to segmentation masks folder
        dem_dir (str): path to slope and elevation images folder
        augmentation (albumentations.Compose): data transformation pipeline 
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing 
            (e.g. normalization, shape manipulation, etc.)
    """
    CLASSES = ['rts']
    
    def __init__(
            self, 
            s1_dir,         # Sentinel-1 directory
            s2_dir,         # Sentinel-2 directory
            dem_dir,
            masks_dir, 
            augmentation=None, 
            preprocessing=None,
            classes=None,
    ):

        # Load Sentinel-1 and Sentinel-2 image file paths
        self.ids = os.listdir(s1_dir)
        self.s1_fps = [os.path.join(s1_dir, image_id) for image_id in self.ids]
        self.s2_fps = [os.path.join(s2_dir, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids]
        self.dem_fps = [os.path.join(dem_dir, image_id) for image_id in self.ids]  

        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes] 

        self.augmentation = augmentation
        self.preprocessing = preprocessing
        
    def __getitem__(self, i):
        # Read Sentinel-1 image (assumed to be float32 after normalization)
        s1_image = read_tif(self.s1_fps[i]).astype(np.float32) 
        # add slope and elevation
        dem_data = read_tif(self.dem_fps[i]).astype(np.float32)
        s1_image = np.concatenate((s1_image, dem_data), axis=-1)

        s2_image = read_tif(self.s2_fps[i]).astype(np.float32)

        # Calculate and concatenate NDVI as a new channel to Sentinel2, we have bands 2, 3, 4, 5, 6, 7, 8, 8A, 11,12
        red_band = s2_image[:, :, 2]  # Band 4 (Red)
        nir_band = s2_image[:, :, 6]  # Band 8 (NIR)
        ndvi = calculate_ndvi(nir_band, red_band)
        ndvi = np.expand_dims(ndvi, axis=2)  # Expand dims to add as a channel
        s2_image = np.concatenate((s2_image, ndvi), axis=-1)

        image = np.concatenate((s1_image, s2_image), axis=-1)
        

        mask = read_tif(self.masks_fps[i], as_gray=True)
        mask_array = np.array(mask)
        mask = (mask_array / 250).astype(np.float32)
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
        shuffle: Boolean, if True shuffle image indexes each epoch.
    """
    
    def __init__(self, dataset, batch_size=1, shuffle=False):
        super().__init__()
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

class FusionDataset(Dataset):
    def __init__(
        self,
        images_dir1,
        images_dir2,
        masks_dir,
        dem_dir=None,
        classes=None,
        augmentation=None,
        preprocessing=None,
        ndvi=False,
    ):
        super().__init__(images_dir1, masks_dir, classes, augmentation, preprocessing)
        self.images_fps2 = [os.path.join(images_dir2, image_id) for image_id in self.ids]
        self.ndvi = ndvi
        self.dem_dir = dem_dir
        if self.dem_dir is not None:
            self.dem_fps = [os.path.join(dem_dir, image_id) for image_id in self.ids]  

    def __getitem__(self, i):

        image1 = read_tif(self.images_fps[i]).astype(np.float32)

        if self.dem_dir is not None:
            dem_data = read_tif(self.dem_fps[i], as_gray=False).astype(np.float32)  
            #dem_data = dem_data[..., np.newaxis]  # CHANGE
            image1 = np.concatenate((image1, dem_data), axis=-1)

    
        image2 = read_tif(self.images_fps2[i]).astype(np.float32)  

        if self.ndvi:
            red_band = image2[:, :, 2]
            nir_band = image2[:, :, 6]
            ndvi = calculate_ndvi(nir_band, red_band)
            ndvi = np.expand_dims(ndvi, axis=2)
            image2 = np.concatenate((image2, ndvi), axis=-1)  
        
        mask = read_tif(self.masks_fps[i], as_gray=True)
        mask_array = np.array(mask)
        mask = (mask_array / 250).astype(np.float32)
        masks = [(~(mask == v)) for v in self.class_values]
        mask = np.stack(masks, axis=-1).astype('float')
        
        # Apply augmentation (same to both images)
        if self.augmentation:
            sample = self.augmentation(image=image1, image2=image2, mask=mask)
            image1, image2, mask = sample['image'], sample['image2'], sample['mask']
        
        if self.preprocessing:
            sample = self.preprocessing(image=image1, mask=mask)
            image1, mask = sample['image'], sample['mask']
            sample2 = self.preprocessing(image=image2)
            image2 = sample2['image']
        
        return (image1, image2), mask
 
    
class FusionDataloder(Dataloder):
    """Loads data from FusionDataset and forms batches for middle fusion.
    
    Inherits:
        Dataloder: Base data loader class.

    Args:
        dataset: Instance of FusionDataset for loading and preprocessing images.
        batch_size: Integer, number of images per batch.
        shuffle: Boolean, if True, shuffles indexes each epoch.
    """
    def __init__(self, dataset, batch_size=1, shuffle=False, **kwargs):
        # Initialize parent class Dataloder
        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle, **kwargs)
    
    def __getitem__(self, index):
        # Collect batch data
        start = index * self.batch_size
        stop = (index + 1) * self.batch_size
        data = [self.dataset[i] for i in self.indexes[start:stop]]
        
        images1, images2, masks = [], [], []
        
        for (image1, image2), mask in data:
            images1.append(image1)
            images2.append(image2)
            masks.append(mask)
        
        # Convert lists to numpy arrays
        images1 = np.array(images1)
        images2 = np.array(images2)
        masks = np.stack(masks, axis=0)
        
        return (images1, images2), masks

 
    
class FusionDataloder(Dataloder):
    """Loads data from FusionDataset and forms batches for middle fusion.
    
    Inherits:
        Dataloder: Base data loader class.

    Args:
        dataset: Instance of FusionDataset for loading and preprocessing images.
        batch_size: Integer, number of images per batch.
        shuffle: Boolean, if True, shuffles indexes each epoch.
    """
    def __init__(self, dataset, batch_size=1, shuffle=False, **kwargs):
        # Initialize parent class Dataloder
        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle, **kwargs)
    
    def __getitem__(self, index):
        # Collect batch data
        start = index * self.batch_size
        stop = (index + 1) * self.batch_size
        data = [self.dataset[i] for i in self.indexes[start:stop]]
        
        images1, images2, masks = [], [], []
        
        for (image1, image2), mask in data:
            images1.append(image1)
            images2.append(image2)
            masks.append(mask)
        
        # Convert lists to numpy arrays
        images1 = np.array(images1)
        images2 = np.array(images2)
        masks = np.stack(masks, axis=0)
        
        return (images1, images2), masks

