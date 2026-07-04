import logging
import numpy as np
from typing import List, Union
from preprocessing.scaler import TimeSeriesScaler
from preprocessing.serializer import TimeSeriesSerializer

logger = logging.getLogger(__name__)

class TimeSeriesDecoder:
    """
    Decodes LLM digit completions back into original-scale numeric forecasts.
    """
    
    def __init__(self, scaler: TimeSeriesScaler, serializer: TimeSeriesSerializer):
        """
        Initializes the decoder with the fitted scaler and serializer.
        
        Args:
            scaler (TimeSeriesScaler): The fitted scaler containing scaling parameters.
            serializer (TimeSeriesSerializer): The serializer containing precision and separators.
        """
        self.scaler = scaler
        self.serializer = serializer

    def decode_completion(self, completion: str, expected_length: int) -> np.ndarray:
        """
        Decodes a single raw completion string from the LLM into a numeric array.
        Example: ', 3 3 0 3 8 0 0, 3 3 0 3 9 1 0' -> [330.38, 330.39]
        
        Args:
            completion (str): The raw string output from the LLM.
            expected_length (int): The expected number of steps in the forecast horizon.
            
        Returns:
            np.ndarray: Decoded and inverse-scaled numpy array of length expected_length.
        """
        import re
        
        # Clean markdown formatting if present
        completion_cleaned = re.sub(r'```(?:[a-zA-Z0-9]+)?\n?', '', completion)
        completion_cleaned = completion_cleaned.replace('```', '').strip()
        
        # Regex to find individual serialized numeric patterns (e.g. "3 3 0", "- 1 2", "NaN")
        number_pattern = re.compile(r'(?:-?\s*\d+(?:\s+\d+)*|NaN|nan|NAN)', re.IGNORECASE)
        matches = list(number_pattern.finditer(completion_cleaned))
        
        # Group consecutive matches separated only by valid delimiters (commas, newlines, spaces, semicolons, dots)
        valid_sep_pattern = re.compile(r'^[,\s;\n\r.]*$')
        
        groups = []
        current_group = []
        
        for idx, match in enumerate(matches):
            if not current_group:
                current_group.append(match)
            else:
                prev_match = current_group[-1]
                sep_text = completion_cleaned[prev_match.end():match.start()]
                if valid_sep_pattern.match(sep_text):
                    current_group.append(match)
                else:
                    groups.append(current_group)
                    current_group = [match]
        if current_group:
            groups.append(current_group)
            
        # Select the group with the most elements
        best_group = max(groups, key=len) if groups else []
        
        # Convert matches in the best group to float values
        decoded_values = []
        for m in best_group:
            val_str = m.group(0).strip()
            # Remove internal spaces in digits: "- 3 3 0" -> "-330"
            no_spaces = re.sub(r'\s+', '', val_str)
            if not no_spaces or no_spaces.upper() == "NAN":
                decoded_values.append(np.nan)
            else:
                try:
                    ival = float(no_spaces)
                    decoded_values.append(ival / self.serializer.scale_factor)
                except ValueError:
                    logger.warning(f"Could not parse numeric match: '{val_str}'")
                    decoded_values.append(np.nan)
                    
        scaled_forecast = np.array(decoded_values, dtype=float)
        
        # Enforce minimum sequence length and handle length discrepancies
        if len(scaled_forecast) < 3:
            logger.warning(f"Decoded sequence has only {len(scaled_forecast)} elements. Marking entire completion as NaN.")
            scaled_forecast = np.full(expected_length, np.nan)
        elif len(scaled_forecast) > expected_length:
            logger.debug(f"Decoded completion has {len(scaled_forecast)} steps. Truncating to {expected_length}.")
            scaled_forecast = scaled_forecast[:expected_length]
        elif len(scaled_forecast) < expected_length:
            logger.warning(f"Decoded completion has {len(scaled_forecast)} steps. Expected {expected_length}. Padding with NaNs.")
            padding = np.full(expected_length - len(scaled_forecast), np.nan)
            scaled_forecast = np.concatenate([scaled_forecast, padding])
            
        # Logging stage: parsed numeric values
        logger.info(f"Parsed numeric values (scaled): {scaled_forecast.tolist()}")
        
        # Apply inverse scaling to restore original unit scale
        original_forecast = self.scaler.inverse_transform(scaled_forecast)
        
        # Logging stage: inverse-scaled forecast
        logger.info(f"Inverse-scaled forecast: {original_forecast.tolist()}")
        
        return original_forecast

    def decode_samples(self, completions: List[str], expected_length: int) -> np.ndarray:
        """
        Decodes a list of multiple completion samples into a 2D array of forecasts.
        
        Args:
            completions (List[str]): List of completion strings from OpenRouter.
            expected_length (int): Expected forecasting horizon length.
            
        Returns:
            np.ndarray: 2D array of shape (num_samples, expected_length).
        """
        forecasts = []
        for idx, completion in enumerate(completions):
            try:
                decoded = self.decode_completion(completion, expected_length)
                forecasts.append(decoded)
            except Exception as e:
                logger.error(f"Failed to decode sample {idx + 1}: {e}")
                # Fallback to a NaN-filled array
                forecasts.append(np.full(expected_length, np.nan))
                
        return np.array(forecasts)
