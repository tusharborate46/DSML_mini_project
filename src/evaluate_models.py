"""Evaluation outputs are generated with training so all models share one test set."""
import json
from .config import RESULTS_DIR

def load_evaluation_details():
    path = RESULTS_DIR / "evaluation_details.json"
    if not path.exists():
        raise FileNotFoundError("Evaluation results are missing. Run python -m src.train_models first.")
    return json.loads(path.read_text(encoding="utf-8"))
