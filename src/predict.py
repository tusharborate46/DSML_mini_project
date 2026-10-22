"""Load saved model pipelines and predict from the shared feature schema."""
import joblib
import pandas as pd
from .config import MODELS_DIR, MODEL_FILES, FEATURES, LABELS

def load_model(name):
    if name not in MODEL_FILES: raise ValueError(f"Unknown model: {name}")
    path=MODELS_DIR/MODEL_FILES[name]
    if not path.exists(): raise FileNotFoundError("Trained models are not available. From the project root, run: python -m src.train_models")
    return joblib.load(path)

def predict(name, record):
    missing=set(FEATURES)-set(record)
    if missing: raise ValueError(f"Missing prediction fields: {', '.join(sorted(missing))}")
    row=pd.DataFrame([{key:record[key] for key in FEATURES}],columns=FEATURES)
    model=load_model(name); label=str(model.predict(row)[0])
    probabilities={str(k):float(v) for k,v in zip(model.classes_,model.predict_proba(row)[0])} if hasattr(model,"predict_proba") else {}
    if label not in LABELS: raise ValueError(f"Model returned unsupported label: {label}")
    return label,probabilities
