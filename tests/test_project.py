import tempfile
import unittest
from pathlib import Path
import pandas as pd
from src.data_loader import load_data
from src.preprocess import prepare_features, congestion_labels
from src.config import FEATURES, LABELS, ROOT
from src.train_models import make_pipeline
from src.predict import predict
from sklearn.tree import DecisionTreeClassifier

def fixture(n=60):
    dates=pd.date_range("2018-01-01",periods=n,freq="h")
    volumes=([500,1500,3500,5500]*(n//4+1))[:n]
    return pd.DataFrame({"date_time":dates.astype(str),"traffic_volume":volumes,"temp":280.0,"rain_1h":0.0,"snow_1h":0.0,"clouds_all":20,"weather_main":"Clear","holiday":"None"})

class ProjectTests(unittest.TestCase):
    def test_csv_schema_and_missing_file(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as d:
            path=Path(d)/"fixture.csv"; fixture().to_csv(path,index=False)
            self.assertEqual(len(load_data(path)),60)
            with self.assertRaises(FileNotFoundError): load_data(Path(d)/"missing.csv")
            bad=Path(d)/"bad.csv"; pd.DataFrame({"x":[1]}).to_csv(bad,index=False)
            with self.assertRaises(ValueError): load_data(bad)

    def test_labels_boundaries(self):
        self.assertEqual(list(congestion_labels([999,1000,3000,5000])),["Low","Moderate","High","Very High"])

    def test_datetime_features_imputation_and_no_leakage(self):
        raw=fixture(); raw.loc[0,"temp"]=None
        X,y,clean=prepare_features(raw)
        self.assertEqual(X.iloc[0].hour,0); self.assertEqual(X.iloc[0].day_of_week,0)
        self.assertIn("temp",X); self.assertNotIn("traffic_volume",X.columns)
        self.assertEqual(float(X.iloc[1].temp),280-273.15)
        self.assertEqual(len(y),len(X)); self.assertTrue(X.temp.isna().any())

    def test_fit_predict_and_pipeline_serialization(self):
        X,y,_=prepare_features(fixture())
        model=make_pipeline(DecisionTreeClassifier(random_state=42)).fit(X.iloc[:45],y.iloc[:45])
        pred=model.predict(X.iloc[45:]); self.assertTrue(set(pred).issubset(set(LABELS)))
        with tempfile.TemporaryDirectory(dir=ROOT) as d:
            import joblib
            path=Path(d)/"model.joblib"; joblib.dump(model,path); loaded=joblib.load(path)
            self.assertEqual(list(loaded.predict(X.iloc[45:])),list(pred))

    def test_full_training_generates_metrics(self):
        from src.train_models import train_and_evaluate
        with tempfile.TemporaryDirectory(dir=ROOT) as d:
            path=Path(d)/"fixture.csv"; fixture(100).to_csv(path,index=False)
            metrics=train_and_evaluate(path,Path(d)/"models",Path(d)/"results")
            self.assertEqual(len(metrics),3)
            self.assertTrue({"accuracy","macro_precision","macro_recall","macro_f1"}.issubset(metrics.columns))

if __name__=="__main__": unittest.main()
