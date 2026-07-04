import logging
import numpy as np
import pandas as pd
from typing import Union, Dict, Any

logger = logging.getLogger(__name__)

class TimeSeriesScaler:
    """
    Scaler for normalizing time series data before digit serialization.
    Supports Percentile Scaling, Standard Scaling, and MinMax Scaling.
    """
    
    def __init__(self, method: str = "percentile", **kwargs):
        """
        Initializes the scaler with the selected method.
        
        Args:
            method (str): Scaling method. Options: 'percentile', 'standard', 'minmax'.
            **kwargs: Extra parameters:
                - low_percentile (float): Lower percentile for percentile scaling. Default: 5.0
                - high_percentile (float): Upper percentile for percentile scaling. Default: 95.0
        """
        self.method = method.lower()
        self.low_percentile = kwargs.get("low_percentile", 5.0)
        self.high_percentile = kwargs.get("high_percentile", 95.0)
        
        # Fit parameters
        self.params: Dict[str, float] = {}
        self.is_fitted = False
        
        if self.method not in ["percentile", "standard", "minmax"]:
            raise ValueError(f"Unknown scaling method: {self.method}")
            
    def fit(self, data: Union[pd.Series, np.ndarray]) -> "TimeSeriesScaler":
        """
        Fits the scaler on the training data. Computes and stores the scaling parameters.
        
        Args:
            data (pd.Series or np.ndarray): Training data to fit parameters.
            
        Returns:
            TimeSeriesScaler: The fitted scaler instance.
        """
        # Convert to numpy array and remove NaNs for robust statistics
        arr = np.asarray(data)
        arr = arr[~np.isnan(arr)]
        
        if len(arr) == 0:
            raise ValueError("Cannot fit scaler on empty or all-NaN array.")
            
        if self.method == "percentile":
            q_low = np.percentile(arr, self.low_percentile)
            q_high = np.percentile(arr, self.high_percentile)
            
            # Prevent division by zero
            if q_high == q_low:
                logger.warning("Lower and upper percentiles are identical. Using default scale factor of 1.0.")
                q_high = q_low + 1e-8
                
            self.params = {
                "q_low": float(q_low),
                "q_high": float(q_high)
            }
            logger.info(f"Fitted Percentile Scaler (P{self.low_percentile}={q_low:.4f}, P{self.high_percentile}={q_high:.4f})")
            
        elif self.method == "standard":
            mean = np.mean(arr)
            std = np.std(arr)
            
            if std == 0:
                logger.warning("Standard deviation is zero. Setting std to 1e-8 to avoid division by zero.")
                std = 1e-8
                
            self.params = {
                "mean": float(mean),
                "std": float(std)
            }
            logger.info(f"Fitted Standard Scaler (mean={mean:.4f}, std={std:.4f})")
            
        elif self.method == "minmax":
            data_min = np.min(arr)
            data_max = np.max(arr)
            
            if data_max == data_min:
                logger.warning("Min and max values are identical. Setting max = min + 1e-8.")
                data_max = data_min + 1e-8
                
            self.params = {
                "min": float(data_min),
                "max": float(data_max)
            }
            logger.info(f"Fitted MinMax Scaler (min={data_min:.4f}, max={data_max:.4f})")
            
        self.is_fitted = True
        return self
        
    def transform(self, data: Union[pd.Series, np.ndarray, float]) -> Union[np.ndarray, float]:
        """
        Transforms the input data using the fitted parameters.
        
        Args:
            data (pd.Series, np.ndarray, or float): Input data to scale.
            
        Returns:
            np.ndarray or float: Scaled data.
        """
        if not self.is_fitted:
            raise ValueError("Scaler has not been fitted yet. Call fit() first.")
            
        is_scalar = isinstance(data, (int, float, np.number))
        arr = np.asarray([data]) if is_scalar else np.asarray(data)
        
        if self.method == "percentile":
            scaled = (arr - self.params["q_low"]) / (self.params["q_high"] - self.params["q_low"])
        elif self.method == "standard":
            scaled = (arr - self.params["mean"]) / self.params["std"]
        elif self.method == "minmax":
            scaled = (arr - self.params["min"]) / (self.params["max"] - self.params["min"])
        else:
            raise ValueError(f"Invalid scaling method: {self.method}")
            
        return float(scaled[0]) if is_scalar else scaled
        
    def fit_transform(self, data: Union[pd.Series, np.ndarray]) -> np.ndarray:
        """
        Fits and transforms the input training data in one step.
        
        Args:
            data (pd.Series or np.ndarray): Training data.
            
        Returns:
            np.ndarray: Scaled training data.
        """
        return self.fit(data).transform(data)
        
    def inverse_transform(self, data: Union[pd.Series, np.ndarray, float]) -> Union[np.ndarray, float]:
        """
        Performs inverse scaling (descaling) to restore data to its original scale.
        
        Args:
            data (pd.Series, np.ndarray, or float): Scaled data to revert.
            
        Returns:
            np.ndarray or float: Descaled data in the original unit.
        """
        if not self.is_fitted:
            raise ValueError("Scaler has not been fitted yet. Call fit() first.")
            
        is_scalar = isinstance(data, (int, float, np.number))
        arr = np.asarray([data]) if is_scalar else np.asarray(data)
        
        if self.method == "percentile":
            descaled = arr * (self.params["q_high"] - self.params["q_low"]) + self.params["q_low"]
        elif self.method == "standard":
            descaled = arr * self.params["std"] + self.params["mean"]
        elif self.method == "minmax":
            descaled = arr * (self.params["max"] - self.params["min"]) + self.params["min"]
        else:
            raise ValueError(f"Invalid scaling method: {self.method}")
            
        return float(descaled[0]) if is_scalar else descaled
