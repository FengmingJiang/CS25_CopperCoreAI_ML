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
├── backend/                 # Backend service directory
├── model_dev/              # Model development directory
│   ├── 0_data_understanding/    # Initial data analysis and understanding
│   ├── 1_sample_construction/   # Sample dataset construction
│   ├── 2_feature_extraction/    # Feature engineering and extraction
│   ├── 3_data_exploration/      # Data exploration and visualization
│   ├── 4_model_training/        # Model training and evaluation
│   └── 5_prediction/           # Prediction pipeline
├── data/                   # Data storage directory
├── models/                 # Trained models storage
├── scripts/               # Utility scripts
├── geoenv/                # Python virtual environment
├── .dist/                 # Distribution files
├── requirements.txt       # Python dependencies
└── LICENSE.txt           # Project license
```

## Directory Descriptions

- **backend/**: Contains the backend service implementation
- **model_dev/**: Main directory for model development, following a structured pipeline:
  - **0_data_understanding/**: Initial data analysis and understanding
  - **1_sample_construction/**: Sample dataset construction
  - **2_feature_extraction/**: Feature engineering and extraction
  - **3_data_exploration/**: Data exploration and visualization
  - **4_model_training/**: Model training and evaluation
  - **5_prediction/**: Prediction pipeline implementation
- **data/**: Storage for datasets and processed data
- **models/**: Storage for trained model artifacts
- **scripts/**: Utility scripts for various tasks
- **geoenv/**: Python virtual environment
- **.dist/**: Distribution files
- **requirements.txt**: List of Python package dependencies
- **LICENSE.txt**: Project license information


