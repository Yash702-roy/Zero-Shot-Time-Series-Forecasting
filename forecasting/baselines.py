import logging
import numpy as np
import pandas as pd
from typing import Tuple, Union
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.arima.model import ARIMA

logger = logging.getLogger(__name__)

class BaselineForecaster:
    """
    Implements standard statistical and ML baselines (ARIMA, Lag-based Linear Regression)
    for comparing against the LLMTIME zero-shot model.
    """
    
    def __init__(self, history: Union[pd.Series, np.ndarray]):
        """
        Initializes baseline forecaster with history data.
        
        Args:
            history (pd.Series or np.ndarray): Training history.
        """
        self.history = np.asarray(history, dtype=float)
        
    def forecast_arima(self, horizon: int, order: Tuple[int, int, int] = (1, 1, 1)) -> np.ndarray:
        """
        Fits an ARIMA model on history and forecasts next steps.
        
        Args:
            horizon (int): Forecasting steps.
            order (Tuple[int, int, int]): ARIMA (p, d, q) order. Default: (1, 1, 1).
            
        Returns:
            np.ndarray: Predicted values of length horizon.
        """
        logger.info(f"Fitting ARIMA{order} baseline...")
        try:
            # Fit ARIMA
            model = ARIMA(self.history, order=order)
            fitted_model = model.fit()
            
            # Forecast
            forecast = fitted_model.forecast(steps=horizon)
            logger.info("ARIMA fitting and forecasting completed.")
            return np.asarray(forecast)
        except Exception as e:
            logger.error(f"ARIMA forecasting failed: {e}. Falling back to persistence.")
            # Persistence fallback: repeat the last value
            return np.full(horizon, self.history[-1])

    def forecast_linear_regression(self, horizon: int, lags: int = 12) -> np.ndarray:
        """
        Fits an Autoregressive Lag-based Linear Regression model and forecasts recursively.
        
        Args:
            horizon (int): Forecasting steps.
            lags (int): Number of lagged features. Default: 12.
            
        Returns:
            np.ndarray: Predicted values of length horizon.
        """
        logger.info(f"Fitting Lag-based Linear Regression (lags={lags}) baseline...")
        try:
            # Create autoregressive training data
            n_samples = len(self.history)
            if n_samples <= lags:
                raise ValueError("History size is smaller than or equal to lags.")
                
            X, y = [], []
            for i in range(lags, n_samples):
                X.append(self.history[i - lags:i])
                y.append(self.history[i])
                
            X = np.array(X)
            y = np.array(y)
            
            # Fit Linear Regression
            lr = LinearRegression()
            lr.fit(X, y)
            
            # Multi-step recursive forecasting
            predictions = []
            curr_window = list(self.history[-lags:])
            
            for _ in range(horizon):
                pred_input = np.array([curr_window]).reshape(1, -1)
                pred = float(lr.predict(pred_input)[0])
                predictions.append(pred)
                
                # Slide window
                curr_window.pop(0)
                curr_window.append(pred)
                
            logger.info("Linear Regression forecasting completed.")
            return np.array(predictions)
        except Exception as e:
            logger.error(f"Linear Regression forecasting failed: {e}. Falling back to persistence.")
            return np.full(horizon, self.history[-1])
