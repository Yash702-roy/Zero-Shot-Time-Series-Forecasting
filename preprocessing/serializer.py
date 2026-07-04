import logging
import numpy as np
import pandas as pd
from typing import Union, List

logger = logging.getLogger(__name__)

class TimeSeriesSerializer:
    """
    Handles the serialization of numeric time series into space-separated digit sequences
    and the deserialization of LLM completions back into numeric values.
    """
    
    def __init__(self, precision: int = 2, separator: str = ", "):
        """
        Initializes the serializer.
        
        Args:
            precision (int): Number of decimal places to preserve. Default: 2.
            separator (str): Separator between different time steps. Default: ", ".
        """
        self.precision = precision
        self.separator = separator
        self.scale_factor = 10 ** precision

    def serialize_value(self, val: float) -> str:
        """
        Serializes a single float value into a space-separated digit string.
        Example: 33035.10 -> 3303510 -> '3 3 0 3 5 1 0'
        Example: -12.34 -> -1234 -> '- 1 2 3 4'
        
        Args:
            val (float): Numeric value to serialize.
            
        Returns:
            str: Space-separated digit representation.
        """
        if pd.isna(val) or np.isnan(val):
            return "NaN"
            
        # Multiply by scale factor and round to integer
        ival = int(round(val * self.scale_factor))
        
        if ival >= 0:
            digits = str(ival)
            return " ".join(digits)
        else:
            digits = str(abs(ival))
            return "- " + " ".join(digits)

    def deserialize_value(self, serialized_val: str) -> float:
        """
        Deserializes a single space-separated digit string back to a float.
        Example: '3 3 0 3 5 1 0' -> 33035.10
        Example: '- 1 2 3 4' -> -12.34
        
        Args:
            serialized_val (str): Space-separated digit string.
            
        Returns:
            float: Numeric value, or np.nan if malformed.
        """
        cleaned = serialized_val.strip()
        if not cleaned or cleaned.upper() == "NAN":
            return np.nan
            
        # Remove spaces
        # Handle cases like "- 1 2" -> "-12"
        no_spaces = cleaned.replace(" ", "")
        
        try:
            ival = float(no_spaces)
            return ival / self.scale_factor
        except ValueError:
            logger.warning(f"Could not deserialize value: '{serialized_val}'. Returning NaN.")
            return np.nan

    def serialize(self, data: Union[pd.Series, np.ndarray, List[float], float]) -> str:
        """
        Serializes a sequence of values (or a single value) into a string.
        
        Args:
            data: Numeric data (scalar or array-like).
            
        Returns:
            str: Serialized sequence.
        """
        if isinstance(data, (int, float, np.number)):
            return self.serialize_value(float(data))
            
        arr = np.asarray(data)
        serialized_items = [self.serialize_value(val) for val in arr]
        return self.separator.join(serialized_items)

    def deserialize(self, serialized_str: str) -> np.ndarray:
        """
        Deserializes a separator-split string of digit sequences back into a numpy array of floats.
        
        Args:
            serialized_str (str): Serialized sequence of digits (e.g., '1 2 3, 4 5 6').
            
        Returns:
            np.ndarray: Decoded numeric array of floats.
        """
        if not serialized_str:
            return np.array([], dtype=float)
            
        # Split by separator
        items = serialized_str.split(self.separator.strip())
        
        decoded_values = []
        for item in items:
            item_stripped = item.strip()
            if not item_stripped:
                continue
            decoded_values.append(self.deserialize_value(item_stripped))
            
        return np.array(decoded_values, dtype=float)
