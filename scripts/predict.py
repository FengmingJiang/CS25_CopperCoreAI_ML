import pandas as pd
import numpy as np
import joblib
import os
import json
import matplotlib
matplotlib.use("Agg")
import plotly.express as px
from plotly.utils import PlotlyJSONEncoder
from django.conf import settings

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'gb_model_selected_features_transform.pkl')
SELECTED_FEATURES_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'selected_columns_transform.csv')
LOG1P_COLS = ['gravity_cscba_stddev3x3', 'mag_uc_2_4km_thd', 'mag_uc_2_4km_stddev3x3', 'radio_th_k_ratio']

def clean_and_preprocess(df):
    features = [
        'radio_k_pct', 'radio_th_ppm', 'radio_u_ppm',
        'radio_th_k_ratio', 'radio_u_k_ratio', 'radio_u_th_ratio',
        'gravity_iso_residual', 'gravity_cscba',
        'mag_uc_1_2km', 'mag_uc_2_4km', 'mag_uc_4_8km',
        'mag_uc_8_12km', 'mag_uc_12_16km'
    ]
    if 'label' not in df.columns:
        df['label'] = 0
    for col in features:
        if col in df.columns:
            df[col] = df.groupby('label')[col].transform(lambda x: x.fillna(x.median()))
    return df

def run_prediction(file_path):
    try:
        original_df = pd.read_csv(file_path)
    except Exception as e:
        return {"error": f"Failed to read file: {e}"}

    df = clean_and_preprocess(original_df.copy())

    for col in LOG1P_COLS:
        df[col] = pd.to_numeric(df.get(col), errors="coerce")
        df[f"{col}_log1p"] = np.log1p(df[col].clip(lower=0).fillna(0))
    df.drop(columns=LOG1P_COLS, inplace=True, errors='ignore')

    try:
        selected_columns = pd.read_csv(SELECTED_FEATURES_PATH).squeeze().tolist()
        features_df = df[selected_columns]
    except Exception as e:
        return {"error": f"Column filtering failed: {e}"}

    try:
        model = joblib.load(MODEL_PATH)
        proba = model.predict_proba(features_df)[:, 1] * 100  # scale to 0–100%
        preds = (proba >= 50).astype(int)  # apply threshold of 50%
    except Exception as e:
        return {"error": f"Prediction failed: {e}"}

    original_df['probability'] = proba
    original_df['prediction'] = preds  # new column for binary classification


    output_path = file_path.replace('.csv', '_predictions.csv')
    original_df.to_csv(output_path, index=False)

    # Save prediction path to text file
    latest_path = os.path.join(settings.MEDIA_ROOT, "latest_prediction.txt")
    with open(latest_path, "w") as f:
        f.write(output_path)

    all_plots = generate_plotly_charts(original_df)

    return {
        "output_path": output_path,
        "num_predictions": len(original_df),
        "preview": original_df.head().to_dict(orient='records'),
        "all_plots": all_plots
    }

def generate_plotly_charts(df):
    charts = []

    # 1. Prediction Map (Mapbox)
    if all(col in df.columns for col in ['latitude', 'longitude', 'probability']):
        map_df = df[['latitude', 'longitude', 'probability']].dropna()
        if not map_df.empty:
            fig = px.scatter_mapbox(
                map_df, lat="latitude", lon="longitude", color="probability",
                color_continuous_scale="YlOrRd", zoom=4, height=400,
                title="Copper Porphyry Probability Map (%)"
            ).update_layout(mapbox_style="carto-positron")
            charts.append(fig)

    # 2. Histogram of Probabilities
    if 'probability' in df.columns:
        prob_df = df[['probability']].dropna()
        fig = px.histogram(prob_df, x='probability', nbins=20,
                           title="Distribution of Predicted Copper Probability (%)")
        charts.append(fig)

    # 3. Scatter Plot (first 2 numeric cols vs probability)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'probability' in df.columns and len(numeric_cols) >= 2:
        x, y = numeric_cols[0], numeric_cols[1]
        scatter_df = df[[x, y, 'probability']].dropna()
        if not scatter_df.empty:
            fig = px.scatter(scatter_df, x=x, y=y, color='probability',
                             title=f"{x} vs {y} with Copper Probability (%)")
            charts.append(fig)

    # 4. Box Plot of Probabilities
    if 'probability' in df.columns:
        box_df = df[['probability']].dropna()
        fig = px.box(box_df, y='probability', title="Copper Probability Spread (%)")
        charts.append(fig)

    # 5. Probability Stats Summary
    if 'probability' in df.columns:
        summary = df['probability'].dropna().describe()
        model_perf = pd.DataFrame({
            "Metric": summary.index.astype(str),
            "Value": summary.values
        })
        fig = px.bar(model_perf, x='Metric', y='Value', title="Summary of Copper Probability Stats")
        charts.append(fig)

    return [json.dumps(fig, cls=PlotlyJSONEncoder) for fig in charts]
