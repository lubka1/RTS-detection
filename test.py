
import config

import os
import argparse
import segmentation_models as sm
import utils
import data_utils
import utils
import tensorflow.keras as keras



preprocess_input = sm.get_preprocessing(config.BACKBONE)


def main( fusion_type, model_path, strategy='concat'):
    """Loads a trained model and evaluates it on the test set."""
    
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
    
    test_dataloader = data_utils.FusionDataloder(test_dataset, batch_size=1, shuffle=False)
    
    # Determine the number of channels in the images
    (images1, images2), _ = test_dataset[0]
    M = images1.shape[-1]  
    N = images2.shape[-1]  
    
    # Load the model with weights
    model = utils.load_model(fusion_type, N, M, strategy, model_path)

    # make sure its the same from training
    model.compile(optimizer=config.optim,  #
              loss=config.total_loss, 
              metrics=[keras.metrics.BinaryIoU(target_class_ids=[1], threshold=0.5)])

    # Evaluate the model
    scores = model.evaluate(test_dataloader)

    print(f"Evaluating model: {model_path}")
    print("Loss: {:.5}".format(scores[0]))
    print("Mean IOU: {:.5}".format(scores[1]))  # IOU is the first metric after loss


    return scores[1]  # Assuming accuracy is at index 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fusion", type=str, required=True, choices=["early", "middle","late"], help="Fusion type used for training")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model weights")
    parser.add_argument("--strategy", type=str, choices=["concat", "average"], help="Select fusion strategy")

    args = parser.parse_args()
    main(args.fusion, args.model_path, args.strategy )
