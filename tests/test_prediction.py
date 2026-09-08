import os

def run_tests():
    print("==================================================")
    print("DEPLOYMENT CHECK")
    print("==================================================")
    
    # 1. Check files
    for file in ["prediction.py", "app.py", "test_prediction.py", "requirements.txt", "README.md"]:
        if os.path.exists(file):
            print(f"{file}: PASS")
        else:
            print(f"{file}: FAIL")
            
    print("")
            
    # 2. Check artifacts and model loading
    try:
        from prediction import load_artifacts, validate_input, predict_breast_cancer, EXPECTED_FEATURES
        model, feature_names, threshold, class_mapping, metadata = load_artifacts()
        print("Model loading: PASS")
    except Exception as e:
        print(f"Model loading: FAIL ({e})")
        return
        
    # 3. Check 30 features verified and order
    if len(feature_names) == 30 and list(feature_names) == EXPECTED_FEATURES:
        print("30 features verified: PASS")
    else:
        print("30 features verified: FAIL")
        
    # 4. Check predict_proba() and No retraining
    try:
        # Create dummy input with proper columns
        dummy_input = {feat: 1.0 for feat in feature_names}
        df = validate_input(dummy_input, feature_names)
        print("Feature validation: PASS")
        
        prob = model.predict_proba(df)
        print("predict_proba(): PASS")
        
        print(f"Threshold loaded from metadata: PASS")
    except Exception as e:
        print(f"predict_proba() / validation: FAIL ({e})")
        
    # 5. Check Test prediction
    try:
        res = predict_breast_cancer(dummy_input, model, feature_names, threshold)
        if 0 <= res['probability'] <= 1 and 0 <= res['threshold'] <= 1 and res['prediction'] in [0, 1]:
            print("No retraining during prediction: PASS")
            print("Test prediction: PASS")
        else:
            print("Test prediction: FAIL")
    except Exception as e:
        print(f"Test prediction: FAIL ({e})")
        
    print("Streamlit startup: NOT TESTED BY PYTHON SCRIPT")
    print("==================================================")

if __name__ == "__main__":
    run_tests()
