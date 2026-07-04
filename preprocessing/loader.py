import logging
import pandas as pd
from typing import Tuple, Optional, List

logger = logging.getLogger(__name__)

class TimeSeriesLoader:
    """
    DataLoader for research-grade zero-shot time series forecasting.
    Handles loading, parsing, sorting, cleaning, and splitting of time series datasets.
    """
    
    def __init__(self, filepath: str):
        """
        Initializes the loader with the dataset path.
        
        Args:
            filepath (str): Path to the time series CSV file.
        """
        self.filepath = filepath
        self.df: Optional[pd.DataFrame] = None
        self.timestamp_col: Optional[str] = None
        self.target_col: Optional[str] = None
        
    def load_data(self) -> pd.DataFrame:
        """
        Reads the CSV file into a pandas DataFrame.
        
        Returns:
            pd.DataFrame: The raw loaded data.
        
        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file is empty or cannot be parsed.
        """
        try:
            logger.info(f"Loading data from {self.filepath}")
            self.df = pd.read_csv(self.filepath)
            logger.info(f"Successfully loaded data with shape {self.df.shape}")
            return self.df
        except FileNotFoundError as e:
            logger.error(f"File not found at {self.filepath}")
            raise e
        except Exception as e:
            logger.error(f"Error loading CSV file: {e}")
            raise ValueError(f"Failed to read CSV: {e}") from e

    def detect_columns(self, 
                       suggested_timestamp: Optional[str] = None, 
                       suggested_target: Optional[str] = None) -> Tuple[str, str]:
        """
        Detects the timestamp and target columns in the dataset.
        If suggested names are provided, uses them if they exist in the columns.
        Otherwise, uses heuristics:
        - Timestamp: looks for 'date', 'time', 'timestamp', or column with datetime-like strings.
        - Target: looks for 'OT', 'target', 'value', 'y', or defaults to the last numeric column.
        
        Args:
            suggested_timestamp (str, optional): Suggested timestamp column name.
            suggested_target (str, optional): Suggested target column name.
            
        Returns:
            Tuple[str, str]: (timestamp_col, target_col)
            
        Raises:
            ValueError: If appropriate columns cannot be identified.
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
            
        columns = list(self.df.columns)
        
        # 1. Detect Timestamp Column
        if suggested_timestamp and suggested_timestamp in columns:
            self.timestamp_col = suggested_timestamp
        else:
            # Look for common patterns
            datetime_keywords = ['date', 'time', 'timestamp', 'datetime', 'epoch']
            for col in columns:
                if any(kw in col.lower() for kw in datetime_keywords):
                    self.timestamp_col = col
                    break
            
            # Fallback: check if any column is of type datetime or can be converted
            if not self.timestamp_col:
                for col in columns:
                    try:
                        pd.to_datetime(self.df[col].head(10))
                        self.timestamp_col = col
                        break
                    except (ValueError, TypeError):
                        continue
                        
        if not self.timestamp_col:
            raise ValueError("Could not detect a valid timestamp column. Please provide suggested_timestamp.")
            
        # 2. Detect Target Column
        if suggested_target and suggested_target in columns:
            self.target_col = suggested_target
        else:
            # Look for common patterns
            target_keywords = ['ot', 'target', 'value', 'y', 'close', 'price', 'demand', 'load']
            for col in columns:
                if col != self.timestamp_col and any(kw == col.lower() for kw in target_keywords):
                    self.target_col = col
                    break
            
            # Fallback: choose the last numeric column that is not the timestamp
            if not self.target_col:
                numeric_cols = [c for c in columns if c != self.timestamp_col and pd.api.types.is_numeric_dtype(self.df[c])]
                if numeric_cols:
                    self.target_col = numeric_cols[-1]
                else:
                    raise ValueError("Could not detect any numeric target column.")
                    
        logger.info(f"Detected columns - Timestamp: '{self.timestamp_col}', Target: '{self.target_col}'")
        return self.timestamp_col, self.target_col

    def preprocess_data(self, fill_method: str = "interpolate") -> pd.DataFrame:
        """
        Preprocesses the dataset:
        1. Ensures the timestamp column is parsed as datetime.
        2. Sorts the DataFrame chronologically.
        3. Sets timestamp as index.
        4. Handles missing values in the target column.
        
        Args:
            fill_method (str): Method to handle missing values. Options: 'interpolate', 'ffill', 'bfill', 'drop'.
            
        Returns:
            pd.DataFrame: Cleaned, sorted, and indexed DataFrame.
        """
        if self.df is None or not self.timestamp_col or not self.target_col:
            raise ValueError("Data or columns not initialized. Run load_data() and detect_columns() first.")
            
        # Convert timestamp to datetime
        self.df[self.timestamp_col] = pd.to_datetime(self.df[self.timestamp_col])
        
        # Sort chronologically
        self.df = self.df.sort_values(by=self.timestamp_col).reset_index(drop=True)
        
        # Keep only date and target, set date as index
        processed_df = self.df[[self.timestamp_col, self.target_col]].copy()
        processed_df.set_index(self.timestamp_col, inplace=True)
        
        # Handle missing values
        missing_count = processed_df[self.target_col].isnull().sum()
        if missing_count > 0:
            logger.warning(f"Found {missing_count} missing values in target column '{self.target_col}'.")
            if fill_method == "interpolate":
                processed_df[self.target_col] = processed_df[self.target_col].interpolate(method="linear")
                logger.info("Imputed missing values using linear interpolation.")
            elif fill_method == "ffill":
                processed_df[self.target_col] = processed_df[self.target_col].ffill()
                logger.info("Imputed missing values using forward-fill.")
            elif fill_method == "bfill":
                processed_df[self.target_col] = processed_df[self.target_col].bfill()
                logger.info("Imputed missing values using backward-fill.")
            elif fill_method == "drop":
                processed_df.dropna(subset=[self.target_col], inplace=True)
                logger.info("Dropped rows with missing target values.")
            else:
                raise ValueError(f"Unknown missing value fill method: {fill_method}")
        else:
            logger.info("No missing values found.")
            
        self.df = processed_df
        return self.df

    def split_chronologically(self, 
                               test_size: float = 0.2) -> Tuple[pd.Series, pd.Series]:
        """
        Performs a chronological train-test split of the time series.
        Since this is sequential time series data, we MUST NOT shuffle.
        
        Args:
            test_size (float): Proportion of the dataset to include in the test split. 
                               Must be between 0.0 and 1.0. If test_size is an integer, 
                               it is treated as the exact number of test steps.
                               
        Returns:
            Tuple[pd.Series, pd.Series]: (train_series, test_series)
        """
        if self.df is None or self.target_col is None:
            raise ValueError("DataFrame has not been preprocessed. Call preprocess_data() first.")
            
        series = self.df[self.target_col]
        total_len = len(series)
        
        if isinstance(test_size, float):
            if not (0.0 < test_size < 1.0):
                raise ValueError("test_size as a float must be strictly between 0.0 and 1.0")
            split_idx = int(total_len * (1 - test_size))
        elif isinstance(test_size, int):
            if not (0 < test_size < total_len):
                raise ValueError(f"test_size as an integer must be between 1 and {total_len - 1}")
            split_idx = total_len - test_size
        else:
            raise TypeError("test_size must be either float or int")
            
        train_series = series.iloc[:split_idx]
        test_series = series.iloc[split_idx:]
        
        logger.info(f"Split data: Train size = {len(train_series)}, Test size = {len(test_series)}")
        return train_series, test_series
