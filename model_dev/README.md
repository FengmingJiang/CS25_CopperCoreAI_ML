# Porphyry Copper Prospectivity - Machine Learning Model Development Notebooks

## Introduction

This directory contains the Jupyter Notebooks detailing the machine learning model development process for predicting porphyry copper deposit prospectivity, primarily focusing on Australian datasets. These notebooks serve as a record of the experimentation, data processing, feature engineering, model training, and evaluation steps undertaken during the development phase.

**Important:** The code presented in these notebooks represents the development and exploration phase. While they document the core logic, production-ready code (potentially refactored into Python modules, functions, and classes) might reside elsewhere or require further refinement for operational deployment. These notebooks are primarily for documentation, reference, reproducibility, and future experimentation.

The overall goal of this ML workflow is to develop models capable of predicting porphyry copper deposit prospectivity based on various geospatial data sources.

## ML Workflow Overview

The model development process is broken down into the following stages, reflected in the numbered directories:

* **Stage 0: Sample Construction**
* **Stage 1: Data Diagnostics**
* **Stage 2: Geospatial Feature Extraction & Engineering**
* **Stage 3: EDA and Data Preprocessing**
* **Stage 4: Model Training and Hypertuning**
* **Stage 5: Prediction and Visualisation**

---

## Stage 0: Sample Construction

**Objective:** To generate positive (known deposit locations) and negative (non-deposit locations) sample points for model training. This involves cleaning deposit data, defining sample areas (e.g., patches/buffers), and sampling background/other deposit locations.

**Notebooks:**

* `./0_sample_construction/01_generate_positvie_aus_porpyhry_samples.ipynb`: Processes known Australian porphyry deposit data.
* `./0_sample_construction/02_generate_positive_patch_samples.ipynb`: Generates positive samples based on spatial patches around known deposits.
* `./0_sample_construction/03_generate_negative_other_deposit_samples.ipynb`: Generates negative samples using locations of other deposit types.
* `./0_sample_construction/04_generate_negative_blank_samples.ipynb`: Generates negative samples from random "blank" locations away from known deposits.
* `./0_sample_construction/05_generate_all_samples_balanced_manually.ipynb`: Explores manual balancing (potentially not the final approach).
* `./0_sample_construction/06_generate_all_samples_unbalanced_only_merge_blank_samples.ipynb`: Merges positive samples with only blank negative samples.
* `./0_sample_construction/07_generate_all_samples_unbalanced_merge_blank_and_other_deposit.ipynb`: Merges positive samples with both blank and other-deposit negative samples (likely leads to the initial input dataset).

**Key Inputs:**

* Raw porphyry deposit data (e.g., CSVs, shapefiles).
* Data for other deposit types (if used for negative samples).
* Geospatial boundary data.
* Configuration parameters (buffer sizes, sampling strategy).

**Key Outputs:**

* Processed positive sample coordinates/labels.
* Processed negative sample coordinates/labels (from various sources).
* Combined initial set of positive and negative sample points with coordinates and labels (e.g., `model_input_samples.csv` or similar).

---

## Stage 1: Data Diagnostics

**Objective:** To understand the characteristics, distributions, formats, spatial coverage, and potential issues within the various raw input data sources (deposits, gravity, magnetics, radiometrics, etc.).

**Notebooks:**

* `./1_data_diagnostics/11_data_understand_porphyry_datasheet.ipynb`
* `./1_data_diagnostics/12_data_understand_gravity.ipynb`
* `./1_data_diagnostics/13_data_understand_earthchem.ipynb` (*Note: Geochemical data, usage depends on final feature set*)
* `./1_data_diagnostics/14_data_understand_aem.ipynb` (*Note: Electromagnetic data, usage depends on final feature set*)
* `./1_data_diagnostics/15_data_understand_magnetic.ipynb`
* `./1_data_diagnostics/16_data_understand_radiometric.ipynb`

**Key Inputs:**

* Raw data files (GeoTIFFs, CSVs, etc.) for various geospatial layers.
* Deposit datasheet files.

**Key Outputs:**

* Primarily insights, plots (histograms, maps), and understanding that inform subsequent feature extraction and preprocessing steps. Limited processed data files typically generated.

---

## Stage 2: Geospatial Feature Extraction & Engineering

**Objective:** To align all raster datasets to a common grid, extract feature values (gravity, magnetics, radiometrics) for each sample point generated in Stage 0, and engineer derived features (e.g., texture, gradients, ratios).

**Notebooks:**

* `./2_geospatial_feature_extraction/20_geotiff_alignment.ipynb`: Aligns input GeoTIFF raster layers.
* `./2_geospatial_feature_extraction/21_feature_engineer.ipynb`: Calculates derived features (e.g., standard deviation filters, THD, ratios).
* `./2_geospatial_feature_extraction/22_combine_features.ipynb`: Extracts values from all feature layers for the sample points and combines them.

**Key Inputs:**

* Sample point coordinates and labels (from Stage 0 output).
* Raw or pre-aligned GeoTIFF layers (gravity, magnetics, radiometrics, etc.).

**Key Outputs:**

* Aligned GeoTIFF layers.
* Combined feature table containing sample points with their corresponding features extracted from all layers (e.g., `model_input_features.csv` or similar). This forms the input for EDA and preprocessing.

---

## Stage 3: EDA and Data Preprocessing

**Objective:** To explore the combined feature dataset, handle missing values, perform feature transformations (e.g., log transform for skewness), scale features, split data into training/validation/test sets, and address class imbalance.

**Notebooks:**

* `./3_EDA_and_data_preprocessing/30_EDA_model_input.ipynb`: Exploratory Data Analysis on the combined feature set.
* `./3_EDA_and_data_preprocessing/31_data_preprocessing_formmating.ipynb`: Basic data type formatting, column renaming.
* `./3_EDA_and_data_preprocessing/32_data_preprocessing_missing_value_and_duplicate.ipynb`: Handles missing values (e.g., median imputation) and checks for duplicates.
* `./3_EDA_and_data_preprocessing/33_data_preprocessing_feature_transformation.ipynb`: Applies transformations (e.g., Log1p) to skewed features.
* `./3_EDA_and_data_preprocessing/34_data_preprocessing_data_split.ipynb`: Splits data into training, validation, and test sets.
* `./3_EDA_and_data_preprocessing/35_data_preprocessing_feature_scaling.ipynb`: Scales numerical features (e.g., RobustScaler).
* `./3_EDA_and_data_preprocessing/36_data_preprocessing_imbalance_handling.ipynb`: Addresses class imbalance on the training set (e.g., using SMOTE).
* `./3_EDA_and_data_preprocessing/37_data_preprocessing_feature_selection.ipynb`: Explores feature selection techniques (e.g., based on model importance).

**Key Inputs:**

* Combined feature table from Stage 2.

**Key Outputs:**

* Preprocessed and scaled training, validation, and test datasets (e.g., saved as CSVs or pickle files).
* Fitted scaler object (e.g., `RobustScaler.pkl`).
* List of selected features (if feature selection is applied).
* Insights from EDA (distributions, correlations, imbalance ratio).

---

## Stage 4: Model Training and Hypertuning

**Objective:** To train various machine learning models (e.g., Random Forest, Gradient Boosting) using the final preprocessed (and potentially feature-selected) training dataset. Includes hyperparameter tuning using cross-validation and evaluation on validation/test sets.

**Notebooks:**

* `./4_model_training_and_hypertuning/40a_model_training_rf_with_all_features_transformed.ipynb`: RF, All Features, Transformed.
* `./4_model_training_and_hypertuning/40b_model_training_rf_with_selected_features_transformed.ipynb`: RF, Selected Features, Transformed.
* `./4_model_training_and_hypertuning/40c_model_training_rf_with_all_features_not_transformed.ipynb`: RF, All Features, Not Transformed.
* `./4_model_training_and_hypertuning/40d_model_training_rf_with_selected_features_no_transformed.ipynb`: RF, Selected Features, Not Transformed.
* `./4_model_training_and_hypertuning/41a_model_training_gb_with_all_features_transformed.ipynb`: GB, All Features, Transformed.
* `./4_model_training_and_hypertuning/41b_model_training_gb_with_selected_features_transformed.ipynb`: GB, Selected Features, Transformed (Potentially best model).
* `./4_model_training_and_hypertuning/41c_model_training_gb_with_all_features_no_transformed copy.ipynb`: GB, All Features, Not Transformed.
* `./4_model_training_and_hypertuning/41d_model_training_gb_with_selected_features_no_transformed copy.ipynb`: GB, Selected Features, Not Transformed.
* `./4_model_training_and_hypertuning/README.md`: (This file might contain additional notes on the modeling experiments).

**Key Inputs:**

* Preprocessed training, validation, and test datasets from Stage 3.

**Key Outputs:**

* Trained model files (e.g., `RF_model_selected_transformed.pkl`, `GB_model_selected_transformed.pkl` saved to a models directory).
* Model performance metrics (AUC, F1, Precision, Recall, Average Precision, Confusion Matrix) for validation and test sets.
* Feature importance plots.
* Cross-validation results and selected hyperparameters.

---

## Stage 5: Prediction and Visualisation

**Objective:** To apply the final trained model to predict prospectivity scores on new, unseen geographic areas (inference) and visualize these predictions, potentially including confidence estimates.

**Notebooks:**

* `./5_prediction_and_visualisation/50_inference_ready_dataset_construction.ipynb`: Prepares the feature data for a target prediction area (e.g., QLD) using the same preprocessing steps (loading scaler).
* `./5_prediction_and_visualisation/51_inference_prediction_visualisation_qld.ipynb`: Loads the trained model and scaler, makes predictions on the target area grid, and performs initial visualizations.
* `./5_prediction_and_visualisation/52_inference_heatmap_prediction.ipynb`: Focuses on generating heatmap visualizations of the prediction scores.
* `./5_prediction_and_visualisation/53_inference_confidence_range.ipynb`: Explores methods for estimating prediction confidence or uncertainty.

**Key Inputs:**

* Aligned GeoTIFF layers covering the target prediction area.
* Fitted scaler object from Stage 3.
* Best trained model file (.pkl) from Stage 4.

**Key Outputs:**

* Prospectivity prediction scores/probabilities for the target area grid.
* Prediction maps (e.g., GeoTIFFs or plots).
* Confidence/uncertainty maps (if generated).

---

## Data and Environment

* Raw input data is expected to be in a parallel `../data/raw/` directory structure (relative to the notebooks).
* Processed intermediate and final data files are typically saved to a parallel `../data/processed/` directory.
* Trained models and scalers are typically saved to a parallel `../models/` directory.
* The Python environment requirements for running these notebooks should be listed in a `requirements.txt` file (ideally at the root of the repository or within the `model_dev` directory).

## Disclaimer

As mentioned, these notebooks document the development journey, including experimentation and iterative refinement. The final operational code used for any deployed application may be a refactored, optimized, and potentially slightly different version of the logic presented here.