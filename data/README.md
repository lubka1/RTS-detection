This dataset includes remote sensing images organized into `train/`, `val/`, and `test/` folders. Each split contains four subfolders:

- **S1/**: Sentinel-1 radar images with two bands — VV and VH 
- **S2/**: Sentinel-2 optical images with 12 bands — B2, B3, B4, B5, B6, B7, B8, B8A, B10, B11, B12
- **DEM/**: Digital Elevation Model data — elevation and slope
- **Masks/**: Ground truth labels (used for training/validation)

add Preprocessing, Source, Notes
- Total samples: X in train, Y in val, Z in test
