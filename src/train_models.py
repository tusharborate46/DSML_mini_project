"""Chronological training/evaluation entry point: python -m src.train_models"""
import json, time
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from .config import DATA_PATH, MODELS_DIR, RESULTS_DIR, REPORTS_DIR, RANDOM_STATE, TEST_FRACTION, FEATURES, NUMERIC_FEATURES, CATEGORICAL_FEATURES, LABELS, MODEL_FILES
from .data_loader import load_data
from .preprocess import prepare_features

def make_pipeline(model):
    prep = ColumnTransformer([
        ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), list(NUMERIC_FEATURES)),
        ("categorical", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("encode", OneHotEncoder(handle_unknown="ignore"))]), list(CATEGORICAL_FEATURES)),
    ])
    return Pipeline([("preprocess", prep), ("model", model)])

def train_and_evaluate(data_path=DATA_PATH, models_dir=MODELS_DIR, results_dir=RESULTS_DIR):
    raw = load_data(data_path)
    X, y, cleaned = prepare_features(raw)
    valid = y.notna()
    X, y, cleaned = X.loc[valid].reset_index(drop=True), y.loc[valid].reset_index(drop=True), cleaned.loc[valid].reset_index(drop=True)
    order = cleaned.date_time.argsort(kind="stable").to_numpy()
    X, y, cleaned = X.iloc[order].reset_index(drop=True), y.iloc[order].reset_index(drop=True), cleaned.iloc[order].reset_index(drop=True)
    split = int(len(X) * (1 - TEST_FRACTION))
    if split < 2 or len(X) - split < 1:
        raise ValueError("At least 3 valid observations are required for chronological training and testing.")
    X_train, X_test, y_train, y_test = X.iloc[:split], X.iloc[split:], y.iloc[:split], y.iloc[split:]
    models = {
        "Decision Tree": DecisionTreeClassifier(max_depth=12, min_samples_leaf=5, class_weight="balanced", random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=160, max_depth=16, min_samples_leaf=3, class_weight="balanced_subsample", n_jobs=-1, random_state=RANDOM_STATE),
        "Logistic Regression": LogisticRegression(max_iter=1200, solver="lbfgs", class_weight="balanced"),
    }
    reports_dir = results_dir / "classification_reports"
    models_dir.mkdir(parents=True, exist_ok=True); results_dir.mkdir(parents=True, exist_ok=True); reports_dir.mkdir(parents=True, exist_ok=True)
    rows=[]; outputs={}
    for name, estimator in models.items():
        pipe=make_pipeline(estimator); started=time.perf_counter(); pipe.fit(X_train,y_train); seconds=time.perf_counter()-started
        pred=pipe.predict(X_test)
        joblib.dump(pipe, models_dir / MODEL_FILES[name])
        report=classification_report(y_test,pred,labels=LABELS,output_dict=True,zero_division=0)
        cm=confusion_matrix(y_test,pred,labels=LABELS)
        importance = None
        if hasattr(pipe.named_steps["model"], "feature_importances_"):
            names = pipe.named_steps["preprocess"].get_feature_names_out()
            scores = pipe.named_steps["model"].feature_importances_
            importance = sorted([{"feature": str(feature), "importance": float(score)} for feature, score in zip(names, scores)], key=lambda item: item["importance"], reverse=True)[:15]
        outputs[name]={"report":report,"confusion_matrix":cm.tolist(),"labels":list(LABELS),"feature_importance":importance}
        rows.append({"model":name,"accuracy":accuracy_score(y_test,pred),"macro_precision":precision_score(y_test,pred,labels=LABELS,average="macro",zero_division=0),"macro_recall":recall_score(y_test,pred,labels=LABELS,average="macro",zero_division=0),"macro_f1":f1_score(y_test,pred,labels=LABELS,average="macro",zero_division=0),"training_seconds":seconds})
        with open(reports_dir / (MODEL_FILES[name].replace(".joblib",".json")),"w",encoding="utf-8") as f: json.dump({"classification_report":report,"confusion_matrix":cm.tolist(),"labels":list(LABELS)},f,indent=2)
    metrics=pd.DataFrame(rows).sort_values("macro_f1",ascending=False)
    metrics.to_csv(results_dir/"model_comparison.csv",index=False)
    class_counts = {part: {str(k): int(v) for k, v in series.value_counts().items()} for part, series in (("train", y_train), ("test", y_test))}
    with open(results_dir/"evaluation_details.json","w",encoding="utf-8") as f: json.dump({"models":outputs,"split":{"strategy":"chronological","train_fraction":1-TEST_FRACTION,"train_rows":len(X_train),"test_rows":len(X_test),"train_end":str(cleaned.iloc[split-1].date_time),"test_start":str(cleaned.iloc[split].date_time)},"class_counts":class_counts,"dataset_rows":len(raw),"duplicate_rows":int(raw.duplicated().sum())},f,indent=2)
    cleaned.assign(congestion=y).to_csv(results_dir/"processed_observations.csv",index=False)
    return metrics

if __name__ == "__main__":
    result=train_and_evaluate(); print(result.to_string(index=False))
