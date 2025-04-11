
import os
import argparse
import segmentation_models as sm
from utils import load_model, get_preprocessing
from utils import Dataset, Dataloder  # Ensure these are correctly implemented

# Set dataset directory
DATA_DIR = r"C:\Users\smola\Documents\MASTER\TRY_Dataset"

# Define test paths
S1_test = os.path.join(DATA_DIR, 'test', 'S1')
S2_test = os.path.join(DATA_DIR, 'test', 'S2')
y_test = os.path.join(DATA_DIR, 'test', 'Masks')

def main(model_path, fusion_type, backbone="resnet34"):
    """Loads a trained model and evaluates it on the test set."""
    
    # Load test dataset
    test_dataset = Dataset(
        S2_test,  # Sentinel-2 images
        y_test,   # Ground truth masks
        fusion=True, images_dir2=S1_test,  # Sentinel-1 images
        classes=['rts'], 
        preprocessing=get_preprocessing(sm.get_preprocessing(backbone)),
    )

    test_dataloader = Dataloder(test_dataset, batch_size=1, shuffle=False)

    # Load the model with weights
    model = load_model(fusion_type, backbone, model_path)

    # Evaluate the model
    scores = model.evaluate(test_dataloader)

    # Metrics
    metrics = [sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)]

    # Print results
    print(f"Evaluating model: {model_path}")
    print("Loss: {:.5}".format(scores[0]))
    for metric, value in zip(metrics, scores[1:]):
        print("mean {}: {:.5}".format(metric.name, value))

    return scores[1]  # Assuming accuracy is at index 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model weights")
    parser.add_argument("--fusion_type", type=str, required=True, choices=["s1", "s2", "early", "late"], help="Fusion type used for training")
    parser.add_argument("--backbone", type=str, default="resnet34", help="Backbone network for segmentation model")

    args = parser.parse_args()
    main(args.model_path, args.fusion_type, args.backbone)
