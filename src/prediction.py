import joblib
import json
import pandas as pd
import numpy as np
import os

EXPECTED_FEATURES = [
    "radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean", "compactness_mean", "concavity_mean", "concave points_mean", "symmetry_mean", "fractal_dimension_mean",
    "radius_se", "texture_se", "perimeter_se", "area_se", "smoothness_se", "compactness_se", "concavity_se", "concave points_se", "symmetry_se", "fractal_dimension_se",
    "radius_worst", "texture_worst", "perimeter_worst", "area_worst", "smoothness_worst", "compactness_worst", "concavity_worst", "concave points_worst", "symmetry_worst", "fractal_dimension_worst"
]

def load_artifacts(model_path='final_model_pipeline.joblib', features_path='feature_names.joblib', meta_path='model_metadata.json'):
    if not os.path.exists(model_path) or not os.path.exists(features_path) or not os.path.exists(meta_path):
        raise FileNotFoundError("Missing required artifact files. Ensure final_model_pipeline.joblib, feature_names.joblib, and model_metadata.json exist.")
    
    model = joblib.load(model_path)
    feature_names = joblib.load(features_path)
    
    with open(meta_path, 'r') as f:
        metadata = json.load(f)
        
    if len(feature_names) != 30:
        raise ValueError(f"Expected 30 features in feature_names.joblib, but found {len(feature_names)}.")
        
    for expected, actual in zip(EXPECTED_FEATURES, feature_names):
        if expected != actual:
            raise ValueError(f"Feature mismatch. Expected '{expected}', got '{actual}'.")
            
    optimal_threshold = metadata.get("optimal_threshold", 0.5)
    class_mapping = metadata.get("class_mapping", {'Malignant': 1, 'Benign': 0})
    
    return model, feature_names, optimal_threshold, class_mapping, metadata

def validate_input(input_data, feature_names):
    if not isinstance(input_data, (pd.DataFrame, dict)):
        raise TypeError("input_data must be a pandas DataFrame or dictionary.")
        
    df = pd.DataFrame([input_data]) if isinstance(input_data, dict) else input_data.copy()
    
    missing_cols = [col for col in feature_names if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required features: {missing_cols}")
        
    unexpected_cols = [col for col in df.columns if col not in feature_names]
    
    if unexpected_cols:
        raise ValueError(
            f"Unexpected features provided: {unexpected_cols}"
        )
        
    # Enforce order exactly as in feature_names.joblib
    df = df[feature_names]
    
    # Check numeric types
    if not all(pd.api.types.is_numeric_dtype(df[col]) for col in df.columns):
        raise ValueError("All features must be numeric.")
        
    if df.isnull().values.any():
        raise ValueError("Input contains missing or NaN values.")
        
    if np.isinf(df.values).any():
        raise ValueError("Input contains infinite values.")
        
    return df

def predict_breast_cancer(input_data, model=None, feature_names=None, threshold=None):
    if model is None or feature_names is None or threshold is None:
        model, feature_names, threshold, class_mapping, _ = load_artifacts()
    
    df_valid = validate_input(input_data, feature_names)
    
    # Verify predict_proba exists
    model_step = model.named_steps.get('model', model)
    if not hasattr(model_step, "predict_proba"):
        raise AttributeError("The loaded model does not support predict_proba().")
        
    probability = model.predict_proba(df_valid)[:, 1][0]
    
    prediction_val = int(probability >= threshold)
    class_label = "Malignant" if prediction_val == 1 else "Benign"
    
    return {
        "prediction": prediction_val,
        "probability": float(probability),
        "threshold": float(threshold),
        "class_label": class_label
    }
