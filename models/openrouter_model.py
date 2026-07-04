import os
import time
import logging
import requests
import numpy as np
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

class OpenRouterModel:
    """
    API Wrapper for OpenRouter, specifically tuned for LLMTIME.
    Supports timeout, retries with exponential backoff, parallel requests for sampling,
    and a robust mock fallback if no API key is provided.
    """
    
    def __init__(self, 
                 model_name: str = "google/gemma-4-31b-it:free",
                 temperature: float = 0.1,
                 top_p: float = 1.0,
                 max_tokens: int = 512,
                 timeout: float = 30.0,
                 max_retries: int = 5,
                 backoff_factor: float = 2.0,
                 initial_delay: float = 1.0):
        """
        Initializes the OpenRouter model wrapper.
        """
        self.model_name = model_name
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.initial_delay = initial_delay
        
        # Load API key
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        
        # Determine if we are in mock mode
        self.is_mock = False
        if not self.api_key or self.api_key == "your_api_key_here" or self.api_key.strip() == "":
            self.is_mock = True
            logger.warning("No valid OPENROUTER_API_KEY found in environment. Running in MOCK MODE.")
        else:
            logger.info(f"OpenRouter Model initialized. Target model: {self.model_name}")

        self.api_url = "https://openrouter.ai/api/v1/chat/completions"

    def _call_api_with_retry(self, prompt: str) -> str:
        """
        Sends a single POST request to OpenRouter with timeout and exponential backoff retries.
        
        Args:
            prompt (str): The serialized history prompt.
            
        Returns:
            str: The raw text completion.
            
        Raises:
            Exception: If all retries fail or if API returns an error.
        """
        # Headers as required by OpenRouter documentation
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/Yash702-roy/Demo",
            "X-Title": "LLMTime Zero-Shot Forecasting"
        }
        
        # Message payload - LLMTIME prompt contains only serialized numeric history
        # Ensure the user prompt ends with the separator (e.g. ", ") to prompt direct sequence continuation
        prompt_with_separator = prompt
        if not prompt_with_separator.endswith(", "):
            if prompt_with_separator.endswith(","):
                prompt_with_separator += " "
            else:
                prompt_with_separator += ", "

        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a precise time series forecasting assistant. Your task is to continue the given sequence of "
                        "space-separated digits. Output ONLY the continuation of the sequence in the exact same serialized "
                        "format (space-separated digits, separated by commas). "
                        "CRITICAL: Do NOT include any markdown code blocks (e.g. ``` or ```text), natural language explanations, "
                        "notes, introductory text, or concluding text. Output ONLY the raw space-separated numbers."
                    )
                },
                {
                    "role": "user",
                    "content": prompt_with_separator
                }
            ],
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens
        }
        
        delay = self.initial_delay
        last_exception = None
        
        for attempt in range(1, self.max_retries + 1):
            try:
                start_time = time.time()
                logger.info(f"Sending API request to OpenRouter (Attempt {attempt}/{self.max_retries})...")
                
                response = requests.post(
                    self.api_url, 
                    headers=headers, 
                    json=payload, 
                    timeout=self.timeout
                )
                
                latency = time.time() - start_time
                logger.info(f"API request completed in {latency:.2f} seconds. Status code: {response.status_code}")
                
                if response.status_code == 200:
                    result = response.json()
                    completion = result["choices"][0]["message"]["content"]
                    return completion
                elif response.status_code == 429:
                    logger.warning(f"Rate limited (429). Retrying after delay...")
                elif 500 <= response.status_code < 600:
                    logger.warning(f"Server error ({response.status_code}). Retrying after delay...")
                else:
                    # Non-retriable error
                    logger.error(f"API error: {response.status_code} - {response.text}")
                    response.raise_for_status()
                    
            except requests.exceptions.RequestException as e:
                logger.warning(f"Request exception on attempt {attempt}: {e}")
                last_exception = e
                
            # Wait with exponential backoff
            if attempt < self.max_retries:
                time.sleep(delay)
                delay *= self.backoff_factor
                
        raise Exception(f"Failed to get response from OpenRouter after {self.max_retries} attempts.") from last_exception

    def _generate_mock_completion(self, prompt: str) -> str:
        """
        Generates a mock digit completion when no API key is available.
        Parse the last values in the prompt, fit a simple autoregressive model (with noise),
        and serialize the output.
        """
        # Parse the prompt values
        # The prompt is a string like "3 3 0 3 5 1 0, 3 3 0 3 6 2 0"
        try:
            items = prompt.split(",")
            values = []
            for item in items:
                cleaned = item.replace(" ", "").strip()
                if cleaned:
                    values.append(float(cleaned))
                    
            if not values:
                values = [1000.0]  # Fallback
        except Exception:
            values = [1000.0]
            
        # Simulate forecast horizon (self.max_tokens can fit about 10-20 steps of length 7)
        # We will generate a series of length 24 (which is the default forecast horizon for ETTh1 usually)
        horizon = 24
        
        # Fit a simple random walk with drift based on the last 10 points
        history = values[-10:] if len(values) >= 10 else values
        differences = np.diff(history) if len(history) > 1 else [0.0]
        mean_drift = np.mean(differences) if len(differences) > 0 else 0.0
        std_drift = np.std(differences) if len(differences) > 0 else 10.0
        if std_drift == 0:
            std_drift = 10.0
            
        last_val = values[-1]
        simulated_vals = []
        for _ in range(horizon):
            # Add some randomness to drift
            drift = mean_drift + np.random.normal(0, std_drift * 0.5)
            next_val = last_val + drift
            simulated_vals.append(next_val)
            last_val = next_val
            
        # Serialize simulated values back to the space-separated digit format
        serialized_items = []
        for val in simulated_vals:
            ival = int(round(val))
            if ival >= 0:
                serialized_items.append(" ".join(str(ival)))
            else:
                serialized_items.append("- " + " ".join(str(abs(ival))))
                
        # Mimic OpenRouter output format: typically returns comma-separated or space-separated digit continuations
        # A typical completion would be ", 3 3 0 3 8 0 0, 3 3 0 3 9 1 0" etc.
        completion = ", " + ", ".join(serialized_items)
        # Sleep for a tiny amount of time to simulate network latency
        time.sleep(0.1)
        return completion

    def generate_sample(self, prompt: str) -> str:
        """
        Generates a single sample from the model.
        
        Args:
            prompt (str): Serialized input history.
            
        Returns:
            str: Generated text sample.
        """
        if self.is_mock:
            return self._generate_mock_completion(prompt)
        else:
            return self._call_api_with_retry(prompt)

    def generate_samples(self, 
                         prompt: str, 
                         num_samples: int = 20, 
                         max_workers: int = 5) -> List[str]:
        """
        Generates multiple samples in parallel.
        
        Args:
            prompt (str): Serialized input history.
            num_samples (int): Total number of samples to generate. Default: 20.
            max_workers (int): Number of concurrent API requests. Default: 5.
            
        Returns:
            List[str]: List of generated sample completions.
        """
        start_time = time.time()
        logger.info(f"Generating {num_samples} samples in parallel with {max_workers} workers...")
        
        samples: List[str] = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all sample generation tasks
            futures = [executor.submit(self.generate_sample, prompt) for _ in range(num_samples)]
            
            for idx, future in enumerate(as_completed(futures)):
                try:
                    sample = future.result()
                    samples.append(sample)
                    logger.info(f"Received sample {idx + 1}/{num_samples}")
                except Exception as e:
                    logger.error(f"Error generating sample {idx + 1}: {e}")
                    
        total_time = time.time() - start_time
        logger.info(f"Successfully generated {len(samples)}/{num_samples} samples in {total_time:.2f} seconds.")
        
        if not samples:
            raise RuntimeError("All parallel sample generation calls failed.")
            
        return samples
