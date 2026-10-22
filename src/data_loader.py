"""CSV loading and schema validation for Metro Interstate Traffic Volume."""
from pathlib import Path
import pandas as pd

REQUIRED_COLUMNS = {"date_time", "traffic_volume", "temp", "rain_1h", "snow_1h", "clouds_all", "weather_main", "holiday"}

def load_data(path: str | Path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}. Download the Metro Interstate Traffic Volume CSV from https://archive.ics.uci.edu/dataset/492/metro+interstate+traffic+volume and save it as {path.name} in the data folder.")
    frame = pd.read_csv(path)
    frame.columns = [str(c).strip().lower() for c in frame.columns]
    # Accept common alternate spellings while keeping one internal schema.
    aliases = {"date time": "date_time", "weather": "weather_main", "traffic volume": "traffic_volume"}
    frame = frame.rename(columns=aliases)
    missing = sorted(REQUIRED_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {', '.join(missing)}. Check the CSV header and UCI download.")
    return frame
