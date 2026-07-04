import os
import yaml
import logging

logger = logging.getLogger(__name__)

DEFAULT_CONFIG = {
    "model": {
        "name": "google/gemma-4-31b-it:free",
        "temperature": 0.7,
        "top_p": 0.9,
        "max_tokens": 128
    },
    "data": {
        "filepath": "data/ETTh1.csv",
        "timestamp_col": "date",
        "target_col": "OT",
        "test_size": 24,
        "fill_method": "interpolate"
    },
    "scaling": {
        "method": "percentile",
        "low_percentile": 5.0,
        "high_percentile": 95.0
    },
    "serialization": {
        "precision": 2,
        "separator": ", "
    },
    "forecasting": {
        "num_samples": 20,
        "max_workers": 5
    }
}

class AppConfig:
    """
    Manages application configuration, reading from a config.yaml file if it exists,
    otherwise falling back to defaults. Supports overrides.
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = config_path
        self.config_dict = self._load_config()
        
    def _load_config(self) -> dict:
        """Loads configuration from file or returns default config."""
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    config = yaml.safe_load(f)
                logger.info(f"Loaded configuration from {self.config_path}")
                # Merge loaded config with default config to ensure all fields exist
                merged = self._deep_merge(DEFAULT_CONFIG, config)
                return merged
            except Exception as e:
                logger.error(f"Error reading config file {self.config_path}: {e}. Using defaults.")
                return DEFAULT_CONFIG
        else:
            logger.info(f"Config file {self.config_path} not found. Using default configurations.")
            return DEFAULT_CONFIG

    def _deep_merge(self, base: dict, override: dict) -> dict:
        """Recursively merges two dictionaries."""
        merged = base.copy()
        for k, v in override.items():
            if isinstance(v, dict) and k in merged and isinstance(merged[k], dict):
                merged[k] = self._deep_merge(merged[k], v)
            else:
                merged[k] = v
        return merged

    def get(self, key_path: str, default=None):
        """
        Gets a configuration value using dot notation (e.g. 'model.name').
        """
        keys = key_path.split(".")
        val = self.config_dict
        for key in keys:
            if isinstance(val, dict) and key in val:
                val = val[key]
            else:
                return default
        return val

    def save_json(self, save_path: str) -> None:
        """Saves current configuration as a JSON file (required output)."""
        import json
        try:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(self.config_dict, f, indent=4)
            logger.info(f"Saved configuration JSON to {save_path}")
        except Exception as e:
            logger.error(f"Failed to save configuration JSON: {e}")
