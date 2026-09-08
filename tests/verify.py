import joblib, json
model = joblib.load('final_model_pipeline.joblib')
clf = model.named_steps['model']
print(f'Model type: {type(clf).__name__}')
print(f'Pipeline steps: {list(model.named_steps.keys())}')
print(f'SVC C: {clf.C}')
print(f'SVC gamma: {clf.gamma}')
smote_step = [s for s in model.named_steps.keys() if 'smote' in s.lower()]
print(f'SMOTE status: {bool(smote_step)}')
features = joblib.load('feature_names.joblib')
print(f'Number of features: {len(features)}')
with open('model_metadata.json', 'r') as f:
    meta = json.load(f)
print(f'Threshold: {meta.get("optimal_threshold")}')
print(f'Class mapping: {meta.get("class_mapping")}')
import sklearn
print(f'sklearn version: {sklearn.__version__}')
