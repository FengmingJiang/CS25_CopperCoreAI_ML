# CopperCore AI - Machine Learning Model Development Notebooks

## Introduction

This directory contains the Jupyter Notebooks detailing the machine learning model development process for the CopperCore AI project (COMP5703 Capstone Project CS25). These notebooks serve as a record of the experimentation, data processing, feature engineering, model training, and evaluation steps undertaken.

**Important:** The code presented in these notebooks represents the development and exploration phase. The final, production-ready code (refactored into Python modules, functions, and classes) used by the operational CopperCore AI Django web application resides within the main `cs25` Django project repository, specifically under the `cs25_coppercore/ml_logic/` directory. These notebooks are primarily for documentation, reference, and potential future experimentation.

The overall goal of this ML workflow is to develop models capable of predicting porphyry copper deposit prospectivity based on various geospatial data sources, initially focusing on Australian datasets.

## ML Workflow Overview

The model development process is broken down into the following stages:

1.  [Stage 0: Sample Construction](#stage-0-sample-construction)
2.  [Stage 1: Data Diagnostics](#stage-1-data-diagnostics)
3.  [Stage 2: Geospatial Feature Extraction](#stage-2-geospatial-feature-extraction)
4.  [Stage 3: Training Data Exploration and Preprocessing](#stage-3-training-data-exploration-and-preprocessing)
5.  [Stage 4: Model-Driven Feature Extraction](#stage-4-model-driven-feature-extraction)
6.  [Stage 5: Model Training](#stage-5-model-training)
7.  [Stage 6: Prediction and Visualization Preparation](#stage-6-prediction-and-visualisation-preparation)

---

## Stage 0: Sample Construction

* **Objective:** To generate positive (known deposit locations) and negative (non-deposit locations) sample points for model training. This involves cleaning deposit data, creating spatial representations (patches/buffers), and sampling background locations.
* **Notebooks:**
    * [`./0_sample_construction/01_generate_positvie_aus_porpyhry_samples.ipynb`](./0_sample_construction/01_generate_positvie_aus_porpyhry_samples.ipynb): Processes known Australian porphyry deposit data.
    * [`./0_sample_construction/02_generate_positive_patch_samples.ipynb`](./0_sample_construction/02_generate_positive_patch_samples.ipynb): Generates positive samples based on spatial patches around known deposits.
    * [`./0_sample_construction/03_generate_negative_other_deposit_samples.ipynb`](./0_sample_construction/03_generate_negative_other_deposit_samples.ipynb): Generates negative samples using locations of other deposit types.
    * [`./0_sample_construction/04_generate_negative_blank_samples.ipynb`](./0_sample_construction/04_generate_negative_blank_samples.ipynb): Generates negative samples from random "blank" locations away from known deposits.
    * [`./0_sample_construction/05_generate_all_samples.ipynb`](./0_sample_construction/05_generate_all_samples.ipynb): Combines all generated positive and negative samples.
* **Key Inputs:**
    * Raw porphyry deposit data (e.g., CSV from `../data/raw/`).
    * Data for other deposit types (if used for negative samples).
    * Geospatial boundary data (e.g., Australia extent).
* **Key Outputs:**
    * `../data/processed/positive_core_clean.csv`: Cleaned core positive sample points.
    * `../data/processed/positive_augmented_patch.csv`: Positive samples derived from patches.
    * `../data/processed/negative_other_deposit_samples.csv`: Negative samples from other deposit locations.
    * `../data/processed/negative_blank_samples.csv`: Negative samples from blank locations.
    * `../data/processed/model_input_samples.csv`: Combined initial set of positive and negative sample points with coordinates and labels.

---

## Stage 1: Data Diagnostics

* **Objective:** To understand the characteristics, distributions, formats, and potential issues within the various raw data sources (deposits, gravity, magnetics, radiometrics, etc.).
* **Notebooks:**
    * [`./1_data_diagnostics/11_data_understand_porphyry_datasheet.ipynb`](./1_data_diagnostics/11_data_understand_porphyry_datasheet.ipynb)
    * [`./1_data_diagnostics/12_data_understand_gravity.ipynb`](./1_data_diagnostics/12_data_understand_gravity.ipynb)
    * [`./1_data_diagnostics/13_data_understand_earthchem.ipynb`](./1_data_diagnostics/13_data_understand_earthchem.ipynb) (Note: EarthChem is geochemical, may or may not be used in final feature set for this project focus)
    * [`./1_data_diagnostics/14_data_understand_aem.ipynb`](./1_data_diagnostics/14_data_understand_aem.ipynb) (Note: AEM is electromagnetic, may or may not be used)
    * [`./1_data_diagnostics/15_data_understand_magnetic.ipynb`](./1_data_diagnostics/15_data_understand_magnetic.ipynb)
    * *(Radiometrics understanding might be implicitly included or missing)*
* **Key Inputs:** Raw data files (GeoTIFFs, CSVs) from `../data/raw/Dataset/GA`.
* **Key Outputs:** Primarily insights, plots, and understanding that inform subsequent feature extraction and preprocessing steps. No major processed data files typically generated here.

---

## Stage 2: Geospatial Feature Extraction

* **Objective:** To extract relevant feature values (gravity, magnetics, radiometrics, potentially derived features like distance or gradients) for each sample point generated in Stage 0.
* **Notebooks:**
    * [`./2_geospatial_feature_extraction/20_combine_features.ipynb`](./2_geospatial_feature_extraction/20_combine_features.ipynb)
* **Key Inputs:**
    * `../data/processed/finnal_samples_balanced.csv` (Sample point coordinates and labels).
    * Raw or pre-aligned GeoTIFF layers (gravity, magnetics, radiometrics, etc.) from `../data/raw/Dataset/GA`.
* **Key Outputs:**
    * `../data/processed/final_samples_with_gravity.csv`: Samples with gravity features joined.
    * `../data/processed/final_samples_with_gravity_magnetics.csv`: Samples with gravity and magnetic features joined.
    * `../data/processed/final_samples_with_gravity_magnetics_radiometric.csv`: Samples with gravity, magnetic, and radiometric features joined (likely the main feature set).
    * `../data/processed/model_input_samples.csv`: Samples are ready to input in the model training stage.
---

## Stage 3: Training Data Exploration and Preprocessing

* **Objective:** To explore the combined feature dataset, handle missing values, address outliers, format data types, and scale features in preparation for model training. This stage might also involve balancing the dataset.
* **Notebooks:**
    * [`./3_training_data_exploration_and_preprocessing/30_eda_model_input.ipynb`](./3_training_data_exploration_and_preprocessing/30_eda_model_input.ipynb): Exploratory Data Analysis on the combined feature set.
    * [`./3_training_data_exploration_and_preprocessing/31_data_preprocessing_formmating.ipynb`](./3_training_data_exploration_and_preprocessing/31_data_preprocessing_formmating.ipynb): Data type formatting.
    * [`./3_training_data_exploration_and_preprocessing/32_data_preprocessing_missing_value_imputation.ipynb`](./3_training_data_exploration_and_preprocessing/32_data_preprocessing_missing_value_imputation.ipynb): Handling missing values.
    * [`./3_training_data_exploration_and_preprocessing/33_data_preprocessing_outlier_transform.ipynb`](./3_training_data_exploration_and_preprocessing/33_data_preprocessing_outlier_transform.ipynb): Handling outliers.
    * [`./3_training_data_exploration_and_preprocessing/34_data_preprocessing_feature_scaling.ipynb`](./3_training_data_exploration_and_preprocessing/34_data_preprocessing_feature_scaling.ipynb): Scaling numerical features (e.g., StandardScaler).
* **Key Inputs:**
    * `../data/processed/model_input_samples.csv` .
* **Key Outputs:**
    * `../data/processed/train_dataset_formatted.csv`
    * `../data/processed/train_dataset_formatted_no_missing.csv`
    * `../data/processed/train_dataset_formatted_no_missing_transformed.csv`
    * `../data/processed/train_dataset_scaled.csv`: Fully preprocessed and scaled dataset ready for modeling.
    * `../data/processed/final_samples_balanced.csv`: (Potentially) A balanced version of the dataset using techniques like SMOTE or undersampling.
    * Scaler objects (e.g., saved as `.pkl` files, though not listed in files).

---

## Stage 4: Model-Driven Feature Extraction

* **Objective:** To potentially engineer new features or select a subset of features based on insights gained from initial modeling attempts or domain knowledge.
* **Notebooks:**
    * [`./4_model_driven_feature_extraction/40_feature_engineer.ipynb`](./4_model_driven_feature_extraction/40_feature_engineer.ipynb)
* **Key Inputs:** Preprocessed dataset (e.g., `../data/processed/train_dataset_scaled.csv`). Feature importance results from previous modeling might also be input.
* **Key Outputs:**
    * `../data/processed/train_dataset_selected.csv`: Dataset with final selected/engineered features.

---

## Stage 5: Model Training

* **Objective:** To train various machine learning models (Baseline, Random Forest, Gradient Boosting) using the final preprocessed and feature-selected dataset. Includes hyperparameter tuning and model evaluation.
* **Notebooks:**
    * [`./5_model_training/50_MVP_model_pipeline.ipynb`](./5_model_training/50_MVP_model_pipeline.ipynb): An overview or initial pipeline attempt.
    * [`./5_model_training/51_model_logistic_regression_baseline.ipynb`](./5_model_training/51_model_logistic_regression_baseline.ipynb): Training a baseline model.
    * [`./5_model_training/52_model_random_forest_model.ipynb`](./5_model_training/52_model_random_forest_model.ipynb): Training Random Forest.
    * [`./5_model_training/53_model_xgboost_lightgbm.ipynb`](./5_model_training/53_model_xgboost_lightgbm.ipynb): Training Gradient Boosting models.
* **Key Inputs:**
    * `../data/processed/train_dataset_selected.csv` (or the final preprocessed dataset used).
* **Key Outputs:**
    * Trained model files (e.g., `RF_model.pkl`, `GB_model.pkl` saved to a models directory - **These are the `.pkl` files used in Mode A**).
    * Model performance metrics (AUC, F1, Precision, Recall, Confusion Matrix).
    * Feature importance plots.

---

## Stage 6: Prediction and Visualisation Preparation

* **Objective:** To prepare the necessary datasets and logic for making predictions on new areas (inference) and visualizing the results. This stage in the notebooks likely focuses on preparing data structures suitable for prediction, whereas the actual *raster map prediction* logic is implemented in the refactored backend code.
* **Notebooks:**
    * [`./6_prediction_and_visualisation/60_inference_ready_dataset_construction.ipynb`](./6_prediction_and_visualisation/60_inference_ready_dataset_construction.ipynb)
* **Key Inputs:**
    * Aligned GeoTIFF layers covering the area of interest.
    * Scaler object (fitted during preprocessing).
* **Key Outputs:**
    * Logic/understanding of how to prepare features for a prediction grid.
    * The actual prediction map generation happens within the Django application's backend tasks using the saved model (`.pkl`) and scaler.

---

## Data and Environment

* Raw input data is expected to be in the `../data/raw/` directory.
* Processed intermediate and final data files are stored in `../data/processed/`.
* Metadata descriptions may be found in `../data/metadata/`.
* The Python environment requirements for running these notebooks are listed in `../requirements.txt`.

---

**Disclaimer:** As mentioned, these notebooks document the development journey. The operational code integrated into the `cs25` Django project is a refactored and potentially optimized version of the logic presented here.