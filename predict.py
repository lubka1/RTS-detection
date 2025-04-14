import config
import os
import argparse
import segmentation_models as sm
import utils
import data_utils
import tensorflow.keras as keras
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import save_img  # Optional: to save predictions as images

# Preprocessing function (same as during training)
preprocess_input = sm.get_preprocessing(config.BACKBONE)

# this one was made by chatgpt mine was different?????
def denormalize(image):
    """Denormalize image if necessary (scaling back to [0, 255] range)."""
    return image * 255.0

def main(fusion_type, model_path, strategy='concat', n=None):
    """Loads a trained model and predicts on n images from the test set."""
    
    # Load test dataset
    test_dataset = data_utils.FusionDataset(
                images_dir1=config.S1_test,  
                images_dir2=config.S2_test,  
                masks_dir=config.y_test,  
                dem_dir=config.DEM_test, 
                classes=['rts'],  
                preprocessing=utils.get_preprocessing(preprocess_input),  
                ndvi=True  
            )
    
    # Determine the number of channels in the images
    (images1, images2), _ = test_dataset[0]
    M = images1.shape[-1]  
    N = images2.shape[-1]  
    
    # Load the model with the trained weights
    model = utils.load_model(fusion_type, N, M, strategy, model_path)

    # Compile the model (same as during training setup)
    model.compile(optimizer=config.optim,  
                  loss=config.total_loss, 
                  metrics=[keras.metrics.BinaryIoU(target_class_ids=[1], threshold=0.5)])

    # If no specific number of images is given, predict on all available images
    if n is None:
        n = len(test_dataset)
    
    ids = np.random.choice(np.arange(len(test_dataset)), size=n)

    # Set up the figure with subplots to display all images together in a grid
    fig, axes = plt.subplots(n, 3, figsize=(15, 5 * n))  # Create a subplot grid for n images
    axes = axes.reshape(n, 3)  # Ensure axes is a 2D array (n, 3)

    # Predict on selected images
    for i, ax_row in zip(ids, axes):
        # Load the image and ground truth mask
        (images1, images2), gt_mask = test_dataset[i]
        images1 = np.expand_dims(images1, axis=0)
        images2 = np.expand_dims(images2, axis=0)

        # Predict mask from model
        pr_mask = model.predict([images1, images2]).round()

        # Denormalize images (optional if they are normalized)
        image = data_utils.denormalize(images1.squeeze())
        gt_mask = gt_mask[..., 0].squeeze()
        pr_mask = pr_mask[..., 0].squeeze()

        # Display the selected image, ground truth, and predicted mask
        ax_row[0].imshow(image, cmap='gray')
        ax_row[0].set_title('Image S2')
        ax_row[0].axis('off')

        ax_row[1].imshow(gt_mask, cmap='gray')
        ax_row[1].set_title('Ground Truth')
        ax_row[1].axis('off')

        ax_row[2].imshow(pr_mask, cmap='gray')
        ax_row[2].set_title('Predicted Mask')
        ax_row[2].axis('off')

    # Adjust the layout to make sure everything fits
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fusion", type=str, required=True, choices=["early", "middle", "late"], help="Fusion type used for training")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model weights")
    parser.add_argument("--n", type=int, help="Number of random images to predict on", default=None)

    args = parser.parse_args()
    main(args.fusion, args.model_path, n=args.n)
