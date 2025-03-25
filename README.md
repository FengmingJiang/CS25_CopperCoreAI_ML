# CS25_CopperCoreAI_ML
Machine Learning Model for Porphyry Copper Deposit

# 🪙 CopperCore AI - Porphyry Copper Deposit Detection

This project aims to develop a full end-to-end machine learning pipeline for detecting potential porphyry copper deposits using geospatial datasets (GeoTIFF, shapefiles, etc.).

---

## 📌 Project Structure

CopperCore-AI/
├── README.md
├── .gitignore
├── data/                        # raw data, cleaned data（ limits 50MB）
│   ├── raw/
│   ├── processed/
│   └── metadata/               # datasets description markdown
├── notebooks/                  # Jupyter Notebooks（data exploration, preprocessing）
│   ├── 01_data_exploration.ipynb
├── src/                        # all source code（Python moduler）
│   ├── preprocessing/
│   │   ├── geotiff_loader.py
│   │   └── vector_loader.py
│   ├── features/
│   ├── training/
│   ├── evaluation/
│   └── utils/
├── models/                     # final models
├── requirements.txt            # Python packages
└── LICENSE



---

## 📁 Data Access

Due to large file sizes, all datasets are hosted externally. Please download the datasets via the links below:

https://unisydneyedu-my.sharepoint.com/:f:/g/personal/fjia3080_uni_sydney_edu_au/Er6O8y-CU7pJl7jC_JKD-zUBcYEroGcXQNx0osWUyY9F0g?e=hpNIip

You can find detailed data descriptions in `data/metadata/`.

---
