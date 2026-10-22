"""Shared project configuration."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DATA_PATH = DATA_DIR / "Metro_Interstate_Traffic_Volume.csv"
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
REPORTS_DIR = RESULTS_DIR / "classification_reports"
RANDOM_STATE = 42
TEST_FRACTION = 0.20
# Project-defined label boundaries; these are not official traffic standards.
THRESHOLDS = (1000, 3000, 5000)
LABELS = ("Low", "Moderate", "High", "Very High")
NUMERIC_FEATURES = ("temp", "rain_1h", "snow_1h", "clouds_all", "hour", "day_of_week", "month", "is_weekend")
CATEGORICAL_FEATURES = ("holiday", "weather_main")
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
MODEL_FILES = {
    "Decision Tree": "decision_tree.joblib",
    "Random Forest": "random_forest.joblib",
    "Logistic Regression": "logistic_regression.joblib",
}
