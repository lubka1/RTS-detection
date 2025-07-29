# Image Dataset Description

This folder contains the image dataset used for training, validation, and testing. The data is structured into remote sensing inputs from Sentinel-1 (S1), Sentinel-2 (S2), and DEM (digital elevation model) sources.

---

## Folder Structure


Each of `train/`, `val/`, and `test/` has the same internal structure.

---

## Sentinel-1 (S1)

- Located in the `S1/` folder.
- Each sample includes two bands:
  - **VV** 
  - **VH** 

---

##  Sentinel-2 (S2)

- Located in the `S2/` folder.
- Includes the following 12 bands (channels):
  - `B2`, `B3`, `B4`, `B5`, `B6`, `B7`, `B8`, `B8A`, `B10`, `B11`, `B12`

---

## DEM (Digital Elevation Model)

- Located in the `DEM/` folder.
- Contains two types of data:
  - **Slope**
  - **Elevation**

---

## Masks (optional)

- Located in the `Masks/` folder (for each split).
- These are the ground truth labels (e.g., for segmentation or classification tasks).
- Format and class mapping should be explained here if applicable.

---

## Preprocessing

## Source

##  Notes

- Total samples: X in train, Y in val, Z in test
