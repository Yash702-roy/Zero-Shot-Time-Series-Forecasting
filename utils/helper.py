import os
import json
import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

def save_json(data: Any, filepath: str) -> None:
    """Saves a Python object as a JSON file."""
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)
        logger.info(f"Successfully saved JSON output to {filepath}")
    except Exception as e:
        logger.error(f"Failed to save JSON file {filepath}: {e}")

def save_txt(text: str, filepath: str) -> None:
    """Saves a string to a text file."""
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(text)
        logger.info(f"Successfully saved text output to {filepath}")
    except Exception as e:
        logger.error(f"Failed to save text file {filepath}: {e}")
