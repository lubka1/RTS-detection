import os
import rasterio
import numpy as np
import tensorflow.keras as keras
import segmentation_models as sm
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from utils import Dataset, load_model, get_preprocessing, denormalize  # Ensure these are correctly implemented



# predict on bigger images??

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Environment setting
DATA_DIR = r"C:\Users\smola\Documents\MASTER\TRY_Dataset"

S1_test = os.path.join(DATA_DIR, 'test', 'S1')
S2_test = os.path.join(DATA_DIR, 'test', 'S2')
y_test = os.path.join(DATA_DIR, 'test', 'Masks')

def main(fusion_type, backbone, model_path,n=None):
    """Main function to load the model, predict on test data, and visualize results."""
    
    test_dataset = Dataset(
        S2_test, # test
        y_test,  # test
        fusion=True, images_dir2=S1_test,
        classes=['rts'], 
        preprocessing=get_preprocessing(preprocess_input),
    )

    # Load model with weights
    model = load_model(fusion_type, backbone, model_path)


    # predict on n images
    if n==None:
        n = len(test_dataset)
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

if __name__ == "__main__":
    # Example usage - make sure to set the right values or pass them as arguments
    FUSION_TYPE = "s1"  
    BACKBONE = "resnet34" # if i use only one, can be in the begning of the code
    MODEL_PATH = "best.weights.h5"

    main(FUSION_TYPE, BACKBONE, MODEL_PATH)        