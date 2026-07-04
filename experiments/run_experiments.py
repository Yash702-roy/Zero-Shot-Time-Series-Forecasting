import os
import sys
import time
import logging
import json
import numpy as np
import pandas as pd

# Bootstrap path settings
current_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(current_dir)
parent_dir = os.path.dirname(project_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
if project_dir not in sys.path:
    sys.path.insert(0, project_dir)

from config import AppConfig
from preprocessing.loader import TimeSeriesLoader
from preprocessing.scaler import TimeSeriesScaler
from preprocessing.serializer import TimeSeriesSerializer
from models.openrouter_model import OpenRouterModel
from forecasting.predictor import LLMTimePredictor
from forecasting.baselines import BaselineForecaster
from evaluation.metrics import calculate_metrics, calculate_prediction_intervals
from utils.logger import setup_logger
from utils.report_generator import generate_html_report

logger = logging.getLogger(__name__)

def run_experiment_suite(config_path: str = "config.yaml") -> dict:
    """
    Runs the experiment suite comparing multiple LLM models (Gemma, DeepSeek, Qwen)
    alongside traditional statistical baselines (ARIMA, Linear Regression).
    Tracks hyperparameters and logs performance metrics.
    """
    os.makedirs("experiments", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)
    setup_logger(log_file="logs/pipeline.log", level=logging.INFO)
    
    logger.info("=" * 60)
    logger.info("Starting LLMTIME Model Comparison and Experiment Suite")
    logger.info("=" * 60)
    
    # 1. Load Configurations
    config = AppConfig(config_path)
    data_path = config.get("data.filepath", "data/ETTh1.csv")
    ts_col = config.get("data.timestamp_col", "date")
    target_col = config.get("data.target_col", "OT")
    test_size = config.get("data.test_size", 24)
    fill_method = config.get("data.fill_method", "interpolate")
    
    scaling_method = config.get("scaling.method", "percentile")
    precision = config.get("serialization.precision", 2)
    separator = config.get("serialization.separator", ", ")
    
    num_samples = config.get("forecasting.num_samples", 20)
    max_workers = config.get("forecasting.max_workers", 5)
    temperature = config.get("model.temperature", 0.7)
    
    # Models to compare
    models_to_test = {
        "Gemma": "google/gemma-4-31b-it:free",
        "DeepSeek": "deepseek/deepseek-chat",
        "Qwen": "qwen/qwen-2.5-72b-instruct"
    }
    
    # 2. Loader & Preprocessing
    loader = TimeSeriesLoader(data_path)
    loader.load_data()
    loader.detect_columns(suggested_timestamp=ts_col, suggested_target=target_col)
    loader.preprocess_data(fill_method=fill_method)
    
    train_series, test_series = loader.split_chronologically(test_size=test_size)
    horizon = len(test_series)
    y_true = test_series.values
    history = train_series.values
    
    # 3. Scaler & Serializer Setup
    scaler = TimeSeriesScaler(method=scaling_method)
    scaler.fit(train_series)
    serializer = TimeSeriesSerializer(precision=precision, separator=separator)
    
    experiments_results = []
    comparison_table = {}
    
    # --- Run LLM models ---
    for model_label, model_name in models_to_test.items():
        logger.info(f"Running experiments for {model_label} ({model_name})...")
        
        # Instantiate model (will handle mock mode if key is missing)
        model_wrapper = OpenRouterModel(
            model_name=model_name,
            temperature=temperature,
            max_tokens=128
        )
        
        # If in mock mode, add slight variation to execution times and values for realistic comparison
        if model_wrapper.is_mock:
            # Seed based on model name for reproducible mock behavior
            np.random.seed(len(model_label))
            
        predictor = LLMTimePredictor(model=model_wrapper, scaler=scaler, serializer=serializer)
        
        start_time = time.time()
        median_pred, all_samples = predictor.forecast(
            history=train_series,
            horizon=horizon,
            num_samples=num_samples,
            max_workers=max_workers
        )
        elapsed = time.time() - start_time
        
        # Calculate metrics
        metrics = calculate_metrics(y_true, median_pred)
        intervals = calculate_prediction_intervals(all_samples)
        
        # Record results
        run_record = {
            "model": model_label,
            "model_name": model_name,
            "temperature": temperature,
            "runtime_seconds": round(elapsed, 2),
            "metrics": metrics,
            "forecast": median_pred.tolist(),
            "p10": intervals["p10"].tolist(),
            "p90": intervals["p90"].tolist(),
            "configuration": config.config_dict
        }
        experiments_results.append(run_record)
        
        comparison_table[model_label] = {
            "MAE": metrics["MAE"],
            "RMSE": metrics["RMSE"],
            "Runtime (s)": round(elapsed, 2)
        }
        
    # --- Run Baselines ---
    baselines = BaselineForecaster(train_series)
    
    # 1. ARIMA Baseline
    arima_start = time.time()
    arima_pred = baselines.forecast_arima(horizon)
    arima_elapsed = time.time() - arima_start
    arima_metrics = calculate_metrics(y_true, arima_pred)
    comparison_table["ARIMA"] = {
        "MAE": arima_metrics["MAE"],
        "RMSE": arima_metrics["RMSE"],
        "Runtime (s)": round(arima_elapsed, 4)
    }
    experiments_results.append({
        "model": "ARIMA",
        "model_name": "ARIMA(1,1,1)",
        "runtime_seconds": round(arima_elapsed, 4),
        "metrics": arima_metrics,
        "forecast": arima_pred.tolist()
    })
    
    # 2. Linear Regression Baseline
    lr_start = time.time()
    lr_pred = baselines.forecast_linear_regression(horizon, lags=12)
    lr_elapsed = time.time() - lr_start
    lr_metrics = calculate_metrics(y_true, lr_pred)
    comparison_table["Linear Regression"] = {
        "MAE": lr_metrics["MAE"],
        "RMSE": lr_metrics["RMSE"],
        "Runtime (s)": round(lr_elapsed, 4)
    }
    experiments_results.append({
        "model": "Linear Regression",
        "model_name": "AR-12 Linear Regression",
        "runtime_seconds": round(lr_elapsed, 4),
        "metrics": lr_metrics,
        "forecast": lr_pred.tolist()
    })
    
    # Save experiments log to files
    with open("experiments/experiments.json", "w", encoding="utf-8") as f:
        json.dump(experiments_results, f, indent=4)
    with open("outputs/experiment.json", "w", encoding="utf-8") as f:
        json.dump(experiments_results, f, indent=4)
        
    logger.info("Experiments logs successfully saved.")
    
    # Print Comparison Table
    df_comp = pd.DataFrame(comparison_table).T
    print("\n" + "=" * 50)
    print("              MODEL COMPARISON SUMMARY")
    print("=" * 50)
    print(df_comp.to_string())
    print("=" * 50 + "\n")
    
    # Save comparison table as CSV
    df_comp.to_csv("outputs/model_comparison.csv")
    logger.info("Saved outputs/model_comparison.csv")
    
    # Generate HTML Report
    generate_html_report("outputs/experiment.json", "outputs/report.html")
    
    return comparison_table

if __name__ == "__main__":
    run_experiment_suite()
