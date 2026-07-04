import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Union
from models.openrouter_model import OpenRouterModel
from preprocessing.scaler import TimeSeriesScaler
from preprocessing.serializer import TimeSeriesSerializer
from forecasting.decoder import TimeSeriesDecoder

logger = logging.getLogger(__name__)

class LLMTimePredictor:
    """
    Main coordinator for the LLMTIME Zero-Shot Forecasting pipeline.
    Orchestrates scaling, serialization, prompt creation, parallel generative sampling,
    decoding, and forecast aggregation (median and intervals).
    """
    
    def __init__(self, 
                 model: OpenRouterModel, 
                 scaler: TimeSeriesScaler, 
                 serializer: TimeSeriesSerializer):
        """
        Initializes the predictor.
        
        Args:
            model (OpenRouterModel): The OpenRouter API wrapper.
            scaler (TimeSeriesScaler): Fitted scaler.
            serializer (TimeSeriesSerializer): Serializer instance.
        """
        self.model = model
        self.scaler = scaler
        self.serializer = serializer
        self.decoder = TimeSeriesDecoder(scaler, serializer)

    def forecast(self, 
                 history: Union[pd.Series, np.ndarray, List[float]], 
                 horizon: int, 
                 num_samples: int = 20, 
                 max_workers: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Runs the LLMTIME zero-shot forecasting pipeline.
        
        Args:
            history: Historical time series values.
            horizon (int): Number of steps to forecast.
            num_samples (int): Number of parallel sampling runs. Default: 20.
            max_workers (int): Number of parallel threads for OpenRouter queries. Default: 5.
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: 
                - median_forecast: 1D array of length `horizon` (median across samples)
                - all_samples: 2D array of shape `(num_samples, horizon)` (raw sampled forecasts)
        """
        logger.info(f"Starting forecast pipeline. Horizon: {horizon}, Samples: {num_samples}")
        
        # 1. Scaling: transform history using the fitted scaler
        # Note: Scaler must be already fitted on the training history
        history_arr = np.asarray(history)
        history_scaled = self.scaler.transform(history_arr)
        
        # 2. Serialization: Convert scaled array to LLMTIME digit format
        # e.g., [330.35, 330.36] -> "3 3 0 3 5, 3 3 0 3 6"
        serialized_prompt = self.serializer.serialize(history_scaled)
        
        # Logging stage: Serialized prompt
        logger.info(f"Serialized prompt (scaled history): {serialized_prompt}")
        logger.info(f"Serialized prompt length (chars): {len(serialized_prompt)}")
        
        # 3. Call OpenRouter to generate multiple samples in parallel
        # The prompt contains ONLY the serialized numeric history, matching LLMTIME exactly.
        raw_samples = self.model.generate_samples(
            prompt=serialized_prompt, 
            num_samples=num_samples, 
            max_workers=max_workers
        )
        
        # Save the RAW OpenRouter completions before decoding to outputs/raw_completion.txt
        import os
        os.makedirs("outputs", exist_ok=True)
        with open("outputs/raw_completion.txt", "w", encoding="utf-8") as f:
            for idx, sample in enumerate(raw_samples):
                f.write(f"=== Sample {idx + 1} ===\n")
                f.write(sample)
                f.write("\n\n")
        logger.info("Saved raw completions to outputs/raw_completion.txt")
        
        # Logging stage: Raw completion
        for idx, sample in enumerate(raw_samples):
            logger.info(f"Raw Completion for Sample {idx + 1}:\n{sample}")
            
        # 4. Decoding: Convert completion strings back to numeric scale
        # Returns shape: (num_samples, horizon)
        all_samples = self.decoder.decode_samples(raw_samples, expected_length=horizon)
        
        # Verify that the model actually returns multiple forecast values instead of one repeated value
        for idx, sample_forecast in enumerate(all_samples):
            non_nan_vals = sample_forecast[~np.isnan(sample_forecast)]
            if len(non_nan_vals) > 1:
                unique_vals = np.unique(non_nan_vals)
                if len(unique_vals) == 1:
                    logger.warning(f"Sample {idx + 1} has collapsed to a single constant value: {unique_vals[0]}")
                else:
                    logger.info(f"Sample {idx + 1} is valid with {len(unique_vals)} unique values (variance: {np.var(non_nan_vals):.4f}).")
            elif len(non_nan_vals) == 1:
                logger.warning(f"Sample {idx + 1} has only one non-NaN value: {non_nan_vals[0]}")
            else:
                logger.warning(f"Sample {idx + 1} has no valid numeric values.")
                
        # 5. Median Prediction: Aggregate across samples using nanmedian to ignore NaNs
        median_forecast = np.nanmedian(all_samples, axis=0)
        
        # Handle case where all samples failed at some timesteps (returns NaN)
        if np.any(np.isnan(median_forecast)):
            logger.warning("Median forecast contains NaNs (all samples failed). Falling back to historical last value.")
            last_val = history_arr[-1]
            nan_mask = np.isnan(median_forecast)
            median_forecast[nan_mask] = last_val
            
        # Logging stage: Final aggregate forecast
        logger.info(f"Aggregate Median Forecast: {median_forecast.tolist()}")
        
        logger.info("Successfully completed forecasting pipeline and aggregation.")
        return median_forecast, all_samples
