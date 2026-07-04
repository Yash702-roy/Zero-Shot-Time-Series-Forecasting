import os
import sys
import tempfile
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# Bootstrap path settings
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from preprocessing.loader import TimeSeriesLoader
from preprocessing.scaler import TimeSeriesScaler
from preprocessing.serializer import TimeSeriesSerializer
from models.openrouter_model import OpenRouterModel
from forecasting.predictor import LLMTimePredictor
from forecasting.baselines import BaselineForecaster
from evaluation.metrics import calculate_metrics, calculate_prediction_intervals
from evaluation.visualization import plot_forecast_interactive

# Streamlit Page Config
st.set_page_config(
    page_title="LLMTIME Zero-Shot Forecasting Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Premium Custom CSS Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;600;800&family=Inter:wght@300;400;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    h1, h2, h3, h4 {
        font-family: 'Outfit', sans-serif;
        font-weight: 800;
    }
    .main-title {
        font-size: 2.8rem;
        background: linear-gradient(135deg, #ff7f0e 0%, #ffc078 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 5px;
    }
    .metric-card {
        background-color: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        margin: 10px 0;
    }
    .metric-val {
        font-size: 2rem;
        font-weight: bold;
        color: #ff7f0e;
        font-family: 'Outfit', sans-serif;
    }
    .metric-lbl {
        font-size: 0.85rem;
        color: #888;
        text-transform: uppercase;
        margin-top: 5px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='main-title'>📈 LLMTIME Forecasting Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<p style='color:#777; margin-bottom: 25px;'>Research-Grade Zero-Shot Time Series Forecasting using Large Language Models with OpenRouter</p>", unsafe_allow_html=True)

# Sidebar configurations
st.sidebar.markdown("## ⚙️ Configuration")

# Model Selection
model_option = st.sidebar.selectbox(
    "Select LLM Model",
    ["google/gemma-4-31b-it:free", "deepseek/deepseek-chat", "qwen/qwen-2.5-72b-instruct"],
    index=0
)

# Hyperparameters
st.sidebar.markdown("### Hyperparameters")
temp = st.sidebar.slider("Temperature", min_value=0.1, max_value=1.5, value=0.7, step=0.1)
horizon = st.sidebar.number_input("Forecast Horizon", min_value=5, max_value=100, value=24, step=1)
num_samples = st.sidebar.number_input("Parallel Samples", min_value=2, max_value=50, value=10, step=1)
max_workers = st.sidebar.number_input("Max Workers (Threads)", min_value=1, max_value=10, value=5, step=1)

# Scaling config
scaling_method = st.sidebar.selectbox("Scaling Method", ["percentile", "standard", "minmax"])

# File Uploader
st.markdown("### 📂 Data Source")
uploaded_file = st.file_uploader("Upload Time Series CSV file (e.g. ETTh1.csv)", type=["csv"])

if uploaded_file is not None:
    # Save uploaded file to temp file to read with loader
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_filepath = tmp_file.name
        
    try:
        # Load and preprocess
        loader = TimeSeriesLoader(tmp_filepath)
        df_raw = loader.load_data()
        
        col1, col2 = st.columns(2)
        with col1:
            timestamp_col = st.selectbox("Timestamp Column", df_raw.columns, index=0)
        with col2:
            target_col = st.selectbox("Target Column", [c for c in df_raw.columns if c != timestamp_col], index=len(df_raw.columns)-2)
            
        loader.detect_columns(suggested_timestamp=timestamp_col, suggested_target=target_col)
        loader.preprocess_data(fill_method="interpolate")
        
        # Split Data
        train_series, test_series = loader.split_chronologically(test_size=int(horizon))
        
        st.success(f"Successfully loaded and parsed dataset. History steps: {len(train_series)}, Test steps: {len(test_series)}")
        
        # Display dataset preview
        with st.expander("🔍 Dataset Preview"):
            st.dataframe(df_raw.head(10))
            
        # Run Forecast Button
        if st.button("🚀 Run Zero-Shot Forecast Pipeline"):
            with st.spinner("Executing pipeline (scaling → serializing → calling OpenRouter → decoding)..."):
                
                # Scaler & Serializer
                scaler = TimeSeriesScaler(method=scaling_method)
                scaler.fit(train_series)
                serializer = TimeSeriesSerializer(precision=2)
                
                # Model (automatic mock fallback if API key is not present)
                model = OpenRouterModel(
                    model_name=model_option,
                    temperature=temp,
                    max_tokens=128
                )
                
                predictor = LLMTimePredictor(model=model, scaler=scaler, serializer=serializer)
                
                # Run forecast
                forecast_start_time = time_ns = st.time_ns() if hasattr(st, "time_ns") else 0
                import time
                start_t = time.time()
                median_pred, all_samples = predictor.forecast(
                    history=train_series,
                    horizon=int(horizon),
                    num_samples=int(num_samples),
                    max_workers=int(max_workers)
                )
                elapsed = time.time() - start_t
                
                # Get prediction intervals
                intervals = calculate_prediction_intervals(all_samples)
                
                # Evaluate metrics
                y_true = test_series.values
                metrics = calculate_metrics(y_true, median_pred)
                
                # Compute baselines
                baselines = BaselineForecaster(train_series)
                arima_pred = baselines.forecast_arima(int(horizon))
                arima_metrics = calculate_metrics(y_true, arima_pred)
                
                lr_pred = baselines.forecast_linear_regression(int(horizon), lags=12)
                lr_metrics = calculate_metrics(y_true, lr_pred)
                
                st.markdown("## 📊 Forecast Visualization")
                
                # Plotly interactive figure
                fig = plot_forecast_interactive(
                    history=train_series,
                    y_true=test_series,
                    y_pred=median_pred,
                    intervals=intervals,
                    save_path=os.path.join(tempfile.gettempdir(), "forecast.html")
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Metrics grid
                st.markdown("## 📈 Performance Metrics (LLMTIME)")
                m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)
                with m_col1:
                    st.markdown(f"<div class='metric-card'><div class='metric-val'>{metrics['MAE']:.4f}</div><div class='metric-lbl'>MAE</div></div>", unsafe_allow_html=True)
                with m_col2:
                    st.markdown(f"<div class='metric-card'><div class='metric-val'>{metrics['RMSE']:.4f}</div><div class='metric-lbl'>RMSE</div></div>", unsafe_allow_html=True)
                with m_col3:
                    st.markdown(f"<div class='metric-card'><div class='metric-val'>{metrics['sMAPE']:.2f}%</div><div class='metric-lbl'>sMAPE</div></div>", unsafe_allow_html=True)
                with m_col4:
                    st.markdown(f"<div class='metric-card'><div class='metric-val'>{metrics['MedAE']:.4f}</div><div class='metric-lbl'>MedAE</div></div>", unsafe_allow_html=True)
                with m_col5:
                    st.markdown(f"<div class='metric-card'><div class='metric-val'>{elapsed:.2f}s</div><div class='metric-lbl'>Runtime</div></div>", unsafe_allow_html=True)
                
                # Baselines Comparison Table
                st.markdown("## ⚖️ Baseline Model Comparison")
                comp_data = {
                    "Model": ["LLMTIME (Median)", "ARIMA Baseline", "Linear Regression (AR-12)"],
                    "MAE": [metrics["MAE"], arima_metrics["MAE"], lr_metrics["MAE"]],
                    "RMSE": [metrics["RMSE"], arima_metrics["RMSE"], lr_metrics["RMSE"]],
                    "sMAPE": [f"{metrics['sMAPE']:.2f}%", f"{arima_metrics['sMAPE']:.2f}%", f"{lr_metrics['sMAPE']:.2f}%"],
                    "MedAE": [metrics["MedAE"], arima_metrics["MedAE"], lr_metrics["MedAE"]]
                }
                st.table(pd.DataFrame(comp_data))
                
                # Expose downloads
                st.markdown("## 📥 Download Predictions")
                forecast_df = pd.DataFrame(index=test_series.index)
                forecast_df["actual"] = y_true
                forecast_df["median_forecast"] = median_pred
                forecast_df["p10"] = intervals["p10"]
                forecast_df["p90"] = intervals["p90"]
                
                csv_data = forecast_df.to_csv().encode('utf-8')
                st.download_button(
                    label="📥 Download Forecast CSV",
                    data=csv_data,
                    file_name="llmtime_forecast.csv",
                    mime="text/csv"
                )
                
    except Exception as e:
        st.error(f"Error parsing data/running pipeline: {e}")
        logger.exception(e)
    finally:
        # Cleanup temp file
        if os.path.exists(tmp_filepath):
            os.remove(tmp_filepath)
else:
    st.info("Please upload a CSV file containing date and target time series columns to begin.")
