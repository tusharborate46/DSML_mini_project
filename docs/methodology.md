# TrafficSense methodology

## Data and preprocessing
Use the UCI Metro Interstate Traffic Volume CSV. The loader normalizes headers, checks required fields, and reports actionable missing-file/schema errors. Invalid timestamps and invalid target volumes are excluded. Duplicate rows are retained because repeated hourly values may be meaningful. Missing numeric and category values are imputed inside each model pipeline, fitted on training observations only. The source temperature is expected in Kelvin and is converted to Celsius when values exceed 100; GUI inputs are Celsius.

## Feature engineering and target
Features are temperature, rain, snow, cloud cover, hour, weekday (Monday=0), month, weekend indicator, holiday, and main weather category. Congestion categories come from the original traffic volume using configurable boundaries: <1000 Low, 1000–2999 Moderate, 3000–4999 High, and >=5000 Very High. These are project choices, not official thresholds. `traffic_volume` and any derivative are excluded from model inputs.

## Algorithms and split
Decision Tree uses bounded depth and leaf size; Random Forest combines 160 bounded trees; Logistic Regression provides a scaled multiclass baseline. All share one ColumnTransformer pipeline and the same data. Valid observations are sorted by timestamp and split into the earliest 80% for training and latest 20% for testing. Preprocessing is fit on training data only. Rare/missing test classes should be considered when interpreting metrics; there is no tuning against the final test segment.

## Evaluation and comparison
Training writes accuracy, macro precision, macro recall, macro F1, elapsed training time, per-class reports, and confusion matrices. Macro metrics weight each label equally. Use macro F1 as a useful comparison signal, alongside class support, confusion matrices, and the intended use; it is not proof that one model is universally best.

## Prediction workflow
The GUI gathers time and weather inputs, constructs the shared feature schema, and loads the fitted joblib pipeline. The pipeline applies the same imputation, encoding, scaling, and classifier learned during training. Predictions are historical-pattern estimates, not live traffic forecasts.
