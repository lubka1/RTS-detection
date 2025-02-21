
import os
import numpy as np
import keras
import segmentation_models as sm
import utils

DATA_DIR = r"C:\Users\smola\Documents\MASTER\TRY_Dataset"

S1_test = os.path.join(DATA_DIR, 'test', 'S1')
S2_test = os.path.join(DATA_DIR, 'test', 'S2')
y_test = os.path.join(DATA_DIR, 'test', 'Masks')

# Set parameters
MODEL = 's1' # option: ['s1', 's2', 'early', 'late']
BACKBONE = 'resnet34'
LR = 0.001  # toto alebo 0.0001???

test_dataset = Dataset(
    S2_valid_dir, # test
    y_valid_dir,  # test
    fusion=True, images_dir2=S1_valid_dir,
    classes=['rts'], 
    preprocessing=get_preprocessing(preprocess_input),
)

test_dataloader = Dataloder(test_dataset, batch_size=1, shuffle=False)

# Load model

# Load best weights

model.load_weights('best.weights.h5')

scores = model.evaluate(test_dataloader)

metrics = [sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)]

print("Loss: {:.5}".format(scores[0]))
for metric, value in zip(metrics, scores[1:]):
    print("mean {}: {:.5}".format(metric.name, value))