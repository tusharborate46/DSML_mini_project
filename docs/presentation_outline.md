# Presentation outline and demonstration

## Slides
1. Title, team, and project aim
2. Congestion problem and SDG 11 relevance
3. UCI dataset and its US interstate scope
4. Cleaning, time/weather features, and configurable labels
5. Chronological train/test methodology and leakage prevention
6. Decision Tree, Random Forest, Logistic Regression
7. Actual results: comparison table, macro metrics, confusion matrices
8. TrafficSense dashboard and historical analysis
9. Limitations, conclusion, and future scope

## Live demonstration
1. From the project folder run `streamlit run app.py`.
2. Home: point out dataset status, four project-defined thresholds, and offline/US dataset limitation.
3. Traffic Prediction: select 8 AM, Tuesday, January, holiday `None`, 15°C, Clear, 0 mm rain, 0 mm snow, and 40% cloud cover; choose Random Forest and click **Predict Traffic**. Explain the displayed category and probability as model output, not a guarantee. If model files have not yet been trained, show the actionable training instruction.
4. Model Comparison: show measured metrics and confusion matrices; explain why macro F1 and class support complement accuracy.
5. Traffic Data Analysis: distinguish historical chart observations from the model prediction.

## Viva quick answers
1. **Problem?** Classify historical hourly traffic into four congestion categories from time and weather inputs.
2. **Why traffic?** Congestion affects travel and planning; patterns can help study mobility.
3. **Dataset?** Metro Interstate Traffic Volume from UCI, representing a US interstate.
4. **Features and labels?** Features are model inputs; the label is the volume-derived congestion class.
5. **Preprocessing?** Parse time, extract calendar fields, convert Kelvin to Celsius, and impute/encode/scale in a fitted pipeline.
6. **Why four categories?** They make volume easier to interpret; boundaries are configurable project choices.
7. **Decision Tree?** It asks a sequence of feature questions that lead to a class.
8. **Random Forest?** It combines predictions from many trees, usually reducing reliance on one tree.
9. **Logistic Regression?** A linear classifier that estimates multiclass probabilities; despite its name it is used for classification.
10. **Why these models?** Interpretable tree, tree ensemble, and a linear baseline provide distinct comparisons.
11. **Training vs testing?** Training fits model parameters; testing estimates performance on held-out later observations.
12. **Accuracy?** Fraction of test predictions that are correct.
13. **Precision, recall, F1?** Precision measures correctness among predicted class cases; recall measures found true cases; F1 balances them.
14. **Confusion matrix?** Counts true versus predicted classes and shows which categories are confused.
15. **Why chronological split?** It preserves time order and better reflects using past data to predict later observations.
16. **Data leakage?** Information unavailable at prediction time contaminates training; traffic volume is excluded after label creation.
17. **GUI and models?** Streamlit builds the same named feature record and passes it to the saved preprocessing/model pipeline.
18. **Why Streamlit?** It creates an interactive Python dashboard with little UI boilerplate.
19. **Limitations?** One US interstate dataset, project-defined boundaries, no live traffic feed, and no validation for Indian roads.
20. **Extension?** Collect local road observations, validate labels/models there, then integrate responsibly sourced real-time feeds.
