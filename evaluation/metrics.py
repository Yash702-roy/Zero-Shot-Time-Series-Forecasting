import logging
import numpy as np
from typing import Dict, Union

logger = logging.getLogger(__name__)

def calculate_metrics(y_true: Union[np.ndarray, list], y_pred: Union[np.ndarray, list]) -> Dict[str, float]:
    """
    Computes standard regression evaluation metrics for time series forecasting.
    
    Args:
        y_true (np.ndarray or list): True target values.
        y_pred (np.ndarray or list): Predicted target values.
        
    Returns:
        Dict[str, float]: Dictionary containing MAE, RMSE, MAPE, sMAPE, and MedAE.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    
    if len(y_true) != len(y_pred):
        raise ValueError(f"Lengths of y_true ({len(y_true)}) and y_pred ({len(y_pred)}) do not match.")
        
    # 1. Mean Absolute Error (MAE)
    mae = float(np.mean(np.abs(y_true - y_pred)))
    
    # 2. Root Mean Squared Error (RMSE)
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    
    # 3. Mean Absolute Percentage Error (MAPE)
    # Handle division by zero using np.where
    mape_denominator = np.where(y_true == 0.0, 1e-8, y_true)
    mape = float(np.mean(np.abs((y_true - y_pred) / mape_denominator)) * 100)
    
    # 4. Symmetric Mean Absolute Percentage Error (sMAPE)
    # Standard definition: 200 * |y - y_hat| / (|y| + |y_hat|)
    smape_denominator = np.abs(y_true) + np.abs(y_pred)
    smape_denominator = np.where(smape_denominator == 0.0, 1e-8, smape_denominator)
    smape = float(np.mean(2.0 * np.abs(y_true - y_pred) / smape_denominator) * 100)
    
    # 5. Median Absolute Error (MedAE)
    medae = float(np.median(np.abs(y_true - y_pred)))
    
    metrics = {
        "MAE": round(mae, 4),
        "RMSE": round(rmse, 4),
        "MAPE": round(mape, 4),
        "sMAPE": round(smape, 4),
        "MedAE": round(medae, 4)
    }
    
    logger.info(f"Computed metrics: {metrics}")
    return metrics

def calculate_prediction_intervals(samples: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Calculates prediction intervals (quantiles) from generative forecast samples.
    Computes the 10th, 50th, and 90th percentiles.
    
    Args:
        samples (np.ndarray): 2D array of shape (num_samples, horizon) containing decoded forecasts.
        
    Returns:
        Dict[str, np.ndarray]: Dict with keys 'p10', 'p50' (median), and 'p90'.
    """
    samples_arr = np.asarray(samples)
    if samples_arr.ndim != 2:
        raise ValueError("Samples must be a 2D array of shape (num_samples, horizon).")
        
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        p10 = np.nanpercentile(samples_arr, 10, axis=0)
        p50 = np.nanpercentile(samples_arr, 50, axis=0)
        p90 = np.nanpercentile(samples_arr, 90, axis=0)
        
        # Also calculate 5% and 95% quantiles for standard 90% confidence boundaries
        p05 = np.nanpercentile(samples_arr, 5, axis=0)
        p95 = np.nanpercentile(samples_arr, 95, axis=0)
        
    # Replace any NaNs with 0.0 as fallback
    p10 = np.nan_to_num(p10, nan=0.0)
    p50 = np.nan_to_num(p50, nan=0.0)
    p90 = np.nan_to_num(p90, nan=0.0)
    p05 = np.nan_to_num(p05, nan=0.0)
    p95 = np.nan_to_num(p95, nan=0.0)
    
    intervals = {
        "p10": p10,
        "p50": p50,
        "p90": p90,
        "p05": p05,
        "p95": p95
    }
    
    logger.info(f"Calculated prediction intervals (p10, p50, p90) for horizon of length {samples_arr.shape[1]}")
    return intervals
