[ raw coords (CSV) ]
        ↓
[ inference_preprocessor.py ] → uses → [ feature_pipeline.py ]
        ↓
[ inference_ready_dataset.csv ]
        ↓
[ predict_and_export.py ] → uses → [ xgboost_final_model.pkl ]
        ↓
[ prediction_result.csv ]

# 1. Generate Model Input Data
python scripts/inference_preprocessor.py --input data/blank_area_samples.csv

# 2. Predict by calling trained models
python scripts/predict_and_export.py --model models/xgboost_final_model.pkl
