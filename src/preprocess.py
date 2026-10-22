"""Reusable feature preparation; target source is excluded from X."""
import numpy as np
import pandas as pd
from .config import FEATURES, LABELS, THRESHOLDS

def congestion_labels(volumes):
    values = pd.to_numeric(pd.Series(volumes), errors="coerce")
    result = pd.cut(values, bins=[-np.inf, *THRESHOLDS, np.inf], labels=LABELS, right=False)
    return result.astype("object").where(values.notna(), np.nan)

def prepare_features(frame: pd.DataFrame, *, require_target=True):
    """Return feature matrix, optional target, and cleaned rows.

    Duplicate rows are reported by callers and retained: repeated hourly
    observations may be genuine. Invalid dates/target volumes are excluded.
    Missing feature values are left for pipeline imputers.
    """
    data = frame.copy()
    data.columns = [str(c).strip().lower() for c in data.columns]
    if "date_time" not in data:
        raise ValueError("Required column 'date_time' is missing.")
    data["date_time"] = pd.to_datetime(data["date_time"], errors="coerce")
    data = data.dropna(subset=["date_time"])
    if require_target:
        if "traffic_volume" not in data:
            raise ValueError("Required target column 'traffic_volume' is missing.")
        data["traffic_volume"] = pd.to_numeric(data["traffic_volume"], errors="coerce")
        data = data.dropna(subset=["traffic_volume"])
        y = congestion_labels(data["traffic_volume"])
    else:
        y = None
    data["hour"] = data.date_time.dt.hour
    data["day_of_week"] = data.date_time.dt.dayofweek
    data["month"] = data.date_time.dt.month
    data["is_weekend"] = (data.day_of_week >= 5).astype(int)
    # UCI temperature is Kelvin (typically ~270-310); GUI temperature is Celsius.
    temp = pd.to_numeric(data.get("temp", np.nan), errors="coerce")
    data["temp"] = temp.where(temp <= 100, temp - 273.15)
    for col in ("rain_1h", "snow_1h", "clouds_all"):
        data[col] = pd.to_numeric(data.get(col, np.nan), errors="coerce")
    for col in ("holiday", "weather_main"):
        data[col] = data.get(col, "Unknown").fillna("Unknown").astype(str).str.strip()
        data.loc[data[col].eq(""), col] = "Unknown"
    X = data.loc[:, FEATURES].copy()
    # traffic_volume is never in FEATURES: labels are derived solely as target.
    if "traffic_volume" in X.columns:
        raise AssertionError("Target leakage detected")
    return X, y, data
