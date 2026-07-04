import os
import sys
import time
import json
import shutil
import logging
import numpy as np
import pandas as pd
from datetime import datetime

# Bootstrap path settings to enable running from any working directory
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Import project modules
from config import AppConfig
from utils.logger import setup_logger
from utils.helper import save_json, save_txt
from preprocessing.loader import TimeSeriesLoader
from preprocessing.scaler import TimeSeriesScaler
from preprocessing.serializer import TimeSeriesSerializer
from models.openrouter_model import OpenRouterModel
from forecasting.predictor import LLMTimePredictor
from evaluation.metrics import calculate_metrics, calculate_prediction_intervals
from evaluation.visualization import plot_forecast_static, plot_forecast_interactive

def run_pipeline(config_path: str = "config.yaml") -> dict:
    """
    Runs the complete zero-shot time series forecasting pipeline.
    
    Args:
        config_path (str): Path to config YAML file.
        
    Returns:
        dict: Computed evaluation metrics.
    """
    start_time = time.time()
    
    # 1. Initialize Logger
    os.makedirs("logs", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)
    log_filepath = "logs/pipeline.log"
    setup_logger(log_file=log_filepath, level=logging.INFO)
    
    logger = logging.getLogger(__name__)
    logger.info("=" * 60)
    logger.info("Starting Zero-Shot Time Series Forecasting Pipeline (LLMTIME)")
    logger.info(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)
    
    # 2. Load Configuration
    config = AppConfig(config_path)
    config.save_json("outputs/config.json")
    
    # Extract configs
    data_path = config.get("data.filepath", "data/ETTh1.csv")
    ts_col = config.get("data.timestamp_col", "date")
    target_col = config.get("data.target_col", "OT")
    test_size = config.get("data.test_size", 24)
    fill_method = config.get("data.fill_method", "interpolate")
    
    scaling_method = config.get("scaling.method", "percentile")
    low_perc = config.get("scaling.low_percentile", 5.0)
    high_perc = config.get("scaling.high_percentile", 95.0)
    
    precision = config.get("serialization.precision", 2)
    separator = config.get("serialization.separator", ", ")
    
    model_name = config.get("model.name", "google/gemma-4-31b-it:free")
    temperature = config.get("model.temperature", 0.7)
    top_p = config.get("model.top_p", 0.9)
    max_tokens = config.get("model.max_tokens", 128)
    
    num_samples = config.get("forecasting.num_samples", 20)
    max_workers = config.get("forecasting.max_workers", 5)
    
    # 3. Load & Clean Data
    loader = TimeSeriesLoader(data_path)
    loader.load_data()
    loader.detect_columns(suggested_timestamp=ts_col, suggested_target=target_col)
    loader.preprocess_data(fill_method=fill_method)
    
    # Chronological Split
    train_series, test_series = loader.split_chronologically(test_size=test_size)
    # The forecasting horizon is the length of the test set
    horizon = len(test_series)
    
    # History context (we pass the train series)
    history = train_series.values
    
    # 4. Fit Scaler on Training History
    scaler = TimeSeriesScaler(method=scaling_method, low_percentile=low_perc, high_percentile=high_perc)
    scaler.fit(train_series)
    
    # Log scaling factors
    logger.info(f"Scaling parameters: {scaler.params}")
    
    # 5. Initialize Serializer & Model & Predictor
    serializer = TimeSeriesSerializer(precision=precision, separator=separator)
    
    model = OpenRouterModel(
        model_name=model_name,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens
    )
    
    # Expose the prompt in main for outputs saving
    scaled_history = scaler.transform(history)
    serialized_prompt = serializer.serialize(scaled_history)
    save_txt(serialized_prompt, "outputs/serialized_prompt.txt")
    
    # Predictor initialization
    predictor = LLMTimePredictor(model=model, scaler=scaler, serializer=serializer)
    
    # 6. Forecasting execution
    logger.info("Executing forecasting...")
    forecast_start_time = time.time()
    median_pred, all_samples = predictor.forecast(
        history=train_series, 
        horizon=horizon, 
        num_samples=num_samples, 
        max_workers=max_workers
    )
    forecast_elapsed = time.time() - forecast_start_time
    logger.info(f"Forecasting finished in {forecast_elapsed:.2f} seconds.")
    
    # 7. Evaluate Metrics
    y_true = test_series.values
    metrics = calculate_metrics(y_true, median_pred)
    metrics["runtime_seconds"] = round(forecast_elapsed, 2)
    save_json(metrics, "outputs/metrics.json")
    
    # Calculate Prediction Intervals
    intervals = calculate_prediction_intervals(all_samples)
    
    # 8. Visualizations
    logger.info("Generating plots...")
    plot_forecast_static(
        history=train_series,
        y_true=test_series,
        y_pred=median_pred,
        intervals=intervals,
        save_path="outputs/forecast_plot.png"
    )
    
    plot_forecast_interactive(
        history=train_series,
        y_true=test_series,
        y_pred=median_pred,
        intervals=intervals,
        save_path="outputs/forecast_plot_interactive.html"
    )
    
    # 9. Save Outputs
    # Save samples.json
    samples_list = all_samples.tolist()
    save_json(samples_list, "outputs/samples.json")
    
    # Save forecast.csv
    forecast_df = pd.DataFrame(index=test_series.index)
    forecast_df["actual"] = y_true
    forecast_df["median_forecast"] = median_pred
    forecast_df["p10"] = intervals["p10"]
    forecast_df["p50"] = intervals["p50"]
    forecast_df["p90"] = intervals["p90"]
    forecast_df.to_csv("outputs/forecast.csv")
    logger.info("Saved forecast.csv containing forecast values and confidence intervals.")
    
    total_elapsed = time.time() - start_time
    logger.info(f"Zero-shot forecasting pipeline completed successfully in {total_elapsed:.2f} seconds.")
    logger.info("=" * 60)
    
    # Copy log file to outputs folder for compliance
    try:
        shutil.copy(log_filepath, "outputs/pipeline.log")
        logger.info("Copied log file to outputs/pipeline.log")
    except Exception as e:
        logger.error(f"Failed to copy log file to outputs: {e}")
        
    return metrics

if __name__ == "__main__":
    run_pipeline()
