import os
import rasterio
import numpy as np
import keras
import segmentation_models as sm
from sklearn.preprocessing import MinMaxScaler


# predict on bigger images??

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Environment setting
DATA_DIR = r"C:\Users\smola\Documents\MASTER\TRY_Dataset"

S1_test = os.path.join(DATA_DIR, 'test', 'S1')
S2_test = os.path.join(DATA_DIR, 'test', 'S2')
y_test = os.path.join(DATA_DIR, 'test', 'Masks')

test_dataset = Dataset(
    S2_valid_dir, # test
    y_valid_dir,  # test
    fusion=True, images_dir2=S1_valid_dir,
    classes=['rts'], 
    preprocessing=get_preprocessing(preprocess_input),
)

# Load model

# Load best weights

# predict on n images
n = 5
ids = np.random.choice(np.arange(len(test_dataset)), size=n)

for i in ids:
    
    image, gt_mask = test_dataset[i]
    image = np.expand_dims(image, axis=0)
    pr_mask = model.predict(image).round()
    
    image=denormalize(image.squeeze())
    gt_mask=gt_mask[..., 0].squeeze()
    pr_mask=pr_mask[..., 0].squeeze()

        # Display the selected band from the image
    plt.subplot(1, 3, 1)
    plt.imshow(image, cmap='gray')
    plt.title('Image S2')

    # Display the selected band from the image
    plt.subplot(1, 3, 2)
    plt.imshow(gt_mask, cmap='gray')
    plt.title('Ground Truth')

    # Display the selected band from the ground truth mask
    plt.subplot(1, 3, 3)
    plt.imshow(pr_mask, cmap='gray')
    plt.title('Predicted Mask')