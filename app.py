"""TrafficSense Streamlit dashboard."""
import json
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from src.config import DATA_PATH, RESULTS_DIR, LABELS
from src.data_loader import load_data
from src.preprocess import prepare_features
from src.predict import predict

st.set_page_config(page_title="TrafficSense", page_icon="🚦", layout="wide")
st.markdown("<style>.block-container{padding-top:1.8rem}.hero{padding:1.4rem 1.8rem;background:linear-gradient(120deg,#102a43,#167d9a);border-radius:16px;color:white}</style>",unsafe_allow_html=True)
st.sidebar.title("🚦 TrafficSense")
page=st.sidebar.radio("Navigate",["Home","Traffic Prediction","Model Comparison","Traffic Data Analysis","About the Project"])

@st.cache_data
def read_raw(): return load_data(DATA_PATH)

st.sidebar.caption("Historical US interstate dataset · Offline predictions")
if page=="Home":
    st.markdown("<div class='hero'><h1>TrafficSense</h1><p>Explore historical traffic patterns and predict a congestion category from time and weather conditions.</p></div>",unsafe_allow_html=True)
    a,b,c=st.columns(3); a.metric("Prediction classes","4"); b.metric("Compared algorithms","3"); c.metric("Dataset", "Metro Interstate")
    st.subheader("Project objectives")
    st.markdown("- Prepare hourly traffic and weather observations for supervised learning.\n- Compare Decision Tree, Random Forest, and Logistic Regression using a chronological holdout.\n- Provide a local dashboard for predictions and historical analysis.")
    st.info("SDG 11 — Sustainable Cities and Communities: historical traffic analysis can inform mobility planning. This project does not measure or claim environmental improvements.")
    st.subheader("Congestion categories")
    st.write("Low (<1,000), Moderate (1,000–2,999), High (3,000–4,999), Very High (≥5,000 vehicles/hour). These are project-defined thresholds, not official standards.")
    try:
        raw=read_raw(); st.success(f"Dataset ready: {len(raw):,} rows · {len(raw.columns)} columns · {int(raw.duplicated().sum()):,} duplicate rows retained.")
        with st.expander("Dataset schema and missing values"):
            st.dataframe(pd.DataFrame({"dtype":raw.dtypes.astype(str),"missing":raw.isna().sum()}),use_container_width=True)
    except Exception as e: st.warning(str(e))
    st.caption("Models are trained once from the command line; opening this dashboard does not retrain them.")
elif page=="Traffic Prediction":
    st.title("Traffic prediction")
    st.write("Enter conditions available at prediction time. Results reflect historical patterns, not guaranteed future traffic.")
    with st.form("prediction"):
        left,right=st.columns(2)
        with left:
            hour=st.slider("Hour of day",0,23,8); day=st.selectbox("Day of week",list(range(7)),format_func=lambda x:["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"][x]); month=st.selectbox("Month",list(range(1,13)),format_func=lambda x:pd.Timestamp(2024,x,1).strftime("%B")); holiday=st.text_input("Holiday",value="None",help="Use None when it is not a named holiday.")
            temp=st.number_input("Temperature (°C)",min_value=-60.0,max_value=60.0,value=15.0); weekend=day>=5
            st.caption(f"Weekend derived from selected day: {'Yes' if weekend else 'No'}")
        with right:
            weather=st.selectbox("Weather category",["Clear","Clouds","Rain","Drizzle","Mist","Snow","Fog","Thunderstorm","Smoke","Haze","Unknown"]); rain=st.number_input("Rainfall (mm in previous hour)",min_value=0.0,max_value=500.0,value=0.0); snow=st.number_input("Snowfall (mm in previous hour)",min_value=0.0,max_value=500.0,value=0.0); clouds=st.slider("Cloud cover (%)",0,100,40); model_name=st.selectbox("Model",["Decision Tree","Random Forest","Logistic Regression"])
        submitted=st.form_submit_button("Predict Traffic",type="primary")
    if submitted:
        record={"temp":temp,"rain_1h":rain,"snow_1h":snow,"clouds_all":clouds,"hour":hour,"day_of_week":day,"month":month,"is_weekend":int(weekend),"holiday":holiday.strip() or "None","weather_main":weather}
        try:
            label,probs=predict(model_name,record); st.subheader(f"Predicted congestion: {label}"); st.write(f"Selected model: **{model_name}**")
            if probs: st.bar_chart(pd.Series(probs,name="Estimated class probability"))
            st.info("This is a model estimate from historical Metro Interstate observations, not live traffic or a guarantee of actual conditions.")
        except Exception as e: st.error(str(e))
elif page=="Model Comparison":
    st.title("Model comparison")
    metrics_path=RESULTS_DIR/"model_comparison.csv"; details_path=RESULTS_DIR/"evaluation_details.json"
    if not metrics_path.exists() or not details_path.exists(): st.warning("Evaluation results are not available. Add the UCI dataset, then run `python -m src.train_models` from the project root.")
    else:
        metrics=pd.read_csv(metrics_path); details=json.loads(details_path.read_text(encoding="utf-8")); st.dataframe(metrics,hide_index=True,use_container_width=True)
        best=metrics.sort_values("macro_f1",ascending=False).iloc[0]; st.success(f"Recommended by macro F1: {best.model} (macro F1 {best.macro_f1:.3f}). Macro F1 gives each class equal weight; consider class support and intended use as well.")
        cols=st.columns(2)
        for col,metric,title in zip(cols,["accuracy","macro_f1"],["Accuracy","Macro F1"]):
            with col: st.bar_chart(metrics.set_index("model")[[metric]],y_label=title)
        st.caption(f"Chronological split: {details['split']['train_rows']:,} earlier training observations and {details['split']['test_rows']:,} later test observations. Test period begins {details['split']['test_start']}.")
        for name,item in details["models"].items():
            with st.expander(f"{name} · classification report and confusion matrix"):
                st.dataframe(pd.DataFrame(item["report"]).T, use_container_width=True)
                fig,ax=plt.subplots(); sns.heatmap(item["confusion_matrix"],annot=True,fmt="d",cmap="Blues",xticklabels=item["labels"],yticklabels=item["labels"],ax=ax); ax.set(xlabel="Predicted",ylabel="Actual"); st.pyplot(fig); plt.close(fig)
                if item.get("feature_importance"):
                    importance=pd.DataFrame(item["feature_importance"]).set_index("feature")
                    st.subheader("Tree feature importance"); st.bar_chart(importance)
elif page=="Traffic Data Analysis":
    st.title("Historical traffic analysis")
    try:
        raw=read_raw(); X,y,clean=prepare_features(raw); clean=clean.assign(congestion=y).dropna(subset=["traffic_volume"]); clean["day_name"]=clean.date_time.dt.day_name()
        st.caption("Charts show historical observations from the UCI dataset; they are not model predictions.")
        charts=[("Average traffic volume by hour",clean.groupby("hour").traffic_volume.mean()),("Average traffic volume by weekday",clean.groupby("day_name").traffic_volume.mean().reindex(["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"])),("Average traffic volume by month",clean.groupby("month").traffic_volume.mean()),("Average traffic volume by weather",clean.groupby("weather_main").traffic_volume.mean().sort_values(ascending=False).head(12)),("Congestion category distribution",clean.congestion.value_counts().reindex(LABELS,fill_value=0))]
        for title,series in charts:
            st.subheader(title); st.bar_chart(series)
    except Exception as e: st.warning(str(e))
elif page=="About the Project":
    st.title("About TrafficSense")
    st.markdown("""### Problem and objectives
Traffic congestion can cause delays, fuel use, and planning challenges. This academic prototype classifies hourly historical traffic into four project-defined categories using time, weather, and holiday features.

### Dataset and limits
The Metro Interstate Traffic Volume dataset describes a US interstate. It does not validate predictions for Indian roads and has no live traffic feed. The label boundaries are configurable project choices. Missing feature values are imputed inside each fitted pipeline; duplicate observations are retained. Invalid timestamps and traffic-volume targets are excluded.

### Algorithms
- **Decision Tree:** interpretable sequence of feature-based splits.
- **Random Forest:** combines many trees to reduce dependence on one tree's decisions.
- **Logistic Regression:** scaled multiclass linear baseline.

### Methodology and SDG 11
Valid records are sorted by timestamp; the earliest 80% train and latest 20% test. The preprocessing pipeline is fitted only on training data. Models are compared with accuracy and macro precision, recall, and F1, plus class reports and confusion matrices. Traffic pattern analysis can support sustainable mobility planning under SDG 11, but this project has not measured emissions or mobility impact.

### Future scope
Validate on local road data, add carefully sourced live inputs, test alternate label boundaries, and monitor model drift.""")
