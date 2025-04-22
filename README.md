# CS25_CopperCoreAI_ML
Machine Learning Model for Porphyry Copper Deposit

# CopperCore AI - Porphyry Copper Deposit Detection

This project aims to develop a full end-to-end machine learning pipeline for detecting potential porphyry copper deposits using geospatial datasets (GeoTIFF, shapefiles, etc.).


## Data Access

Due to large file sizes, all datasets are hosted externally. Please download the datasets via the links below:

[Onedrive: COMP5703-CS25-CopperCoreAI-Datasets](https://unisydneyedu-my.sharepoint.com/:f:/g/personal/fjia3080_uni_sydney_edu_au/EsmTTWAEUAhFllvxBn_h1YgBBUSFvqinmp0PuI-UrGDU5A?e=XhwrZE)

You can find detailed data descriptions in `data/metadata/`.


## Project Structure

```
.
├── backend/                           # Backend service directory
├── model_development/                # ML model development pipeline
│   ├── 00_sample_construction/       # Sample generation: positive, other, blank
│   ├── 01_data_diagnostics/          # Raw GeoTIFF data inspection (quality, coverage)
│   ├── 02_geospatial_feature_extraction/  # Extract geophysical and geochemical features from raster
│   ├── 03_EDA_and_preprocessing/         # EDA on the training dataset and preorocessing based on EDA results
│   ├── 04_model_training_and_hypertuning/    # ML model training and validation           
│   ├── 05_prediction_and_visualisation/         # Prediction generation and export
├── data/                             # Raw and processed dataset storage
├── models/                           # Trained model artifacts
├── scripts/                          # Utility and backend-facing scripts
├── geoenv/                           # Python virtual environment (excluded from Git)
├── .dist/                            # Temporary distribution outputs
├── requirements.txt                  # Python dependency list
└── LICENSE.txt                       # Project license
```

## Directory Descriptions

- **`backend/`**: Django + PostGIS backend implementation with API integration.
- **`model_development/`**: Primary directory for machine learning pipeline, organized into modular stages:
  - **`00_sample_construction/`**: Builds labeled sample points using positive (porphyry), other deposit, and blank regions.
  - **`01_data_diagnostics/`**: Inspects raw GeoTIFF layers (gravity, magnetics, AEM, radiometrics, geochem) to assess usability for modeling.
  - **`02_geospatial_feature_extraction/`**: Extracts numerical features from raster data based on sample coordinates.
  - **`03_EDA_and_preprocessing/`**: Explores the labeled training data via statistical summaries, correlation matrices, and feature distributions.
  - **`04_model_training_and_hypertuning`**: Builds and validates machine learning models (Random Forest, XGBoost, etc.).
  - **`05_prediction_and_visualisation/`**: Generates prediction probabilities and exports model results for integration.
- **`data/`**: Contains source GeoTIFFs, CSV samples, and intermediate data outputs.
- **`models/`**: Saves serialized model objects (e.g., `.pkl`, `.joblib`) and checkpoints.
- **`scripts/`**: Contains reusable preprocessing, model loading, and prediction scripts for backend use.
- **`geoenv/`**: Local Python virtual environment (excluded via `.gitignore`).
- **`.dist/`**: Output distribution files for packaging or sharing.
- **`requirements.txt`**: Lists all Python dependencies required for this project.
- **`LICENSE.txt`**: Project license and usage permissions.


