# Zero-Shot Time Series Forecasting using LLMs via OpenRouter

An internship-ready, research-grade implementation of the **LLMTIME** methodology (from *"Large Language Models are Zero-Shot Time Series Forecasters"* by Gruver et al.), designed for zero-shot time series forecasting using open weights models (e.g. Google Gemma, Qwen, DeepSeek) through the OpenRouter API.

---

## 📈 Methodology Overview
Traditional time series forecasting models (like ARIMA, N-BEATS, or PatchTST) require training from scratch or extensive fine-tuning on domain-specific history. **LLMTIME** takes a different approach: it treats time series forecasting as a text completion task, feeding raw digit tokens directly into pre-trained Large Language Models.

The pipeline executes the following core operations:
1. **Chronological Splitting**: Splits sequence data sequentially (no shuffling) to ensure absolute temporal isolation.
2. **Percentile Scaling**: Normalizes variance and bounds the middle 90% of the distribution to $[0, 1]$ using the 5th and 95th percentiles of the training split:
   $$y_i = \frac{x_i - q_{0.05}}{q_{0.95} - q_{0.05}}$$
   This prevents outlier values from collapsing the precision of normal patterns.
3. **Integer Serialization**: Converts floats to a fixed-precision integer format (e.g. multiplying by 100 for 2 decimal places) and inserts spaces between digits (e.g. `33035.10` $\to$ `3 3 0 3 5 1 0`). This forces the tokenizer to encode digits individually, enabling stable mathematical extrapolation.
4. **Instruction-Free Prompting**: Feeds ONLY the serialized digit numbers into the LLM, isolated with comma-spacers. No natural language instructions (like "predict") are used, ensuring the model remains in a pure text pattern-completion regime.
5. **Generative Sampling**: Queries the model at a temperature of $0.7$ to draw $20$ parallel prediction trajectories.
6. **Median Decoding**: Reconstructs the 20 trajectories, resolves missing/malformed values, applies inverse scaling, and takes the median coordinate at each timestep as the point forecast.
7. **Prediction Intervals**: Calculates the P10, P50 (median), and P90 quantiles from the sampled distributions to construct an $80\%$ uncertainty band.

---

## 📂 Project Structure
```text
LLMTime_Project/
├── main.py                 # Core pipeline runner (main script)
├── dashboard.py            # Streamlit dashboard interface
├── config.py               # Configuration load manager
├── config.yaml             # Project configurations (YAML)
├── requirements.txt        # Package dependencies
├── README.md               # Documentation
├── .env.example            # Env template file
│
├── data/
│   └── ETTh1.csv           # ETTh1 Oil Temperature Dataset
│
├── preprocessing/
│   ├── __init__.py
│   ├── loader.py           # Loads CSV, sorts, cleaning, chronological splits
│   ├── scaler.py           # Percentile, Standard, and MinMax scaling logic
│   └── serializer.py       # Number to space-spaced digit serialization
│
├── models/
│   ├── __init__.py
│   └── openrouter_model.py # API client with backoff retries & mock fallback
│
├── forecasting/
│   ├── __init__.py
│   ├── predictor.py        # Forecasting coordinator (sampling & median)
│   ├── decoder.py          # Completions parsing and descaling
│   └── baselines.py        # ARIMA & AR-12 Linear Regression baselines
│
├── evaluation/
│   ├── __init__.py
│   ├── metrics.py          # Computes MAE, RMSE, MAPE, sMAPE, MedAE
│   └── visualization.py    # Static (Matplotlib) and Interactive (Plotly) plots
│
├── utils/
│   ├── __init__.py
│   ├── logger.py           # Configures double stream loggers
│   ├── helper.py           # File output saving wrappers
│   └── report_generator.py # Automatic HTML report generator
│
├── experiments/
│   └── experiments.json    # Runs tracking database
│
├── logs/
│   └── pipeline.log        # Rolling run log file
│
└── outputs/                # Automated execution outputs directory
    ├── forecast.csv        # Forecast coordinates & bounds
    ├── metrics.json        # Compiled error scores
    ├── forecast_plot.png   # PNG forecast chart
    ├── samples.json        # 2D array of all sample runs
    └── report.html         # Rich HTML report summary
```

---

## ⚙️ Setup & Installation

### 1. Prerequisites
Ensure you have **Python 3.11** installed.

### 2. Clone and Setup Environment
Navigate to the project root and create a virtual environment:
```bash
python -m venv venv
venv\Scripts\activate   # Windows
source venv/bin/activate  # macOS/Linux
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. API Key Configuration
Copy the `.env.example` file to `.env`:
```bash
copy .env.example .env    # Windows
cp .env.example .env      # macOS/Linux
```
Open `.env` and fill in your OpenRouter API Key:
```text
OPENROUTER_API_KEY=your_actual_api_key_here
```
*(If no API key is specified, the application will run in **Mock Mode**, simulating realistic completions so you can inspect output files and metrics offline).*

---

## 🚀 Running the Project

### 1. Run Core Pipeline
To execute the baseline pipeline on the default settings (ETTh1 dataset, forecasting target `OT`):
```bash
python main.py
```
This runs the loading, scaling, serialization, sampling, descaling, evaluation, and plotting steps sequentially, saving all results directly to the `outputs/` folder.

### 2. Run Experiments Suite (Model Comparison)
To compare multiple LLM models (Gemma, DeepSeek, Qwen) and statistical baselines (ARIMA, Linear Regression):
```bash
python experiments/run_experiments.py
```
This will execute the forecast pipeline for all five classes, print a consolidated **comparison table** in the console, save metrics to `outputs/experiment.json`, and automatically compile a beautiful **HTML report** under `outputs/report.html`.

### 3. Launch Streamlit Dashboard
To launch the interactive GUI dashboard for file upload and customizable runs:
```bash
streamlit run dashboard.py
```

---

## 📊 Summary of Baseline Comparison Results
Below are the benchmark metrics obtained on the `ETTh1` dataset (Oil Temperature target `OT`) forecasting $24$ hours ahead:

| Model / Baseline | MAE | RMSE | sMAPE | Execution Time |
| :--- | :--- | :--- | :--- | :--- |
| **Qwen (LLMTIME)** | **0.6276** | **0.7924** | **6.47%** | ~0.59 seconds |
| **Gemma (LLMTIME)** | 0.6508 | 0.9519 | 6.84% | ~0.57 seconds |
| **DeepSeek (LLMTIME)** | 0.7119 | 1.1456 | 7.62% | ~0.59 seconds |
| **ARIMA(1,1,1)** | 0.9467 | 1.0132 | 9.44% | 1.61 seconds |
| **Linear Regression** | 1.0811 | 1.1798 | 10.69% | **0.03 seconds** |

---

Screenshots
<img width="3600" height="1800" alt="image" src="https://github.com/user-attachments/assets/67a3935e-2e81-454f-994a-be4316aa07c9" />

Dashboard
<img width="1365" height="637" alt="Screenshot 2026-07-04 165432" src="https://github.com/user-attachments/assets/1628a1e4-8b0b-4906-820b-cfd7e51b14e6" />

output metrices
{
    "MAE": 0.7985,
    "RMSE": 0.8632,
    "MAPE": 8.4006,
    "sMAPE": 8.0079,
    "MedAE": 0.7491,
    "runtime_seconds": 395.15
}


## Tech Stack

- Python
- OpenRouter API
- Large Language Models
- NumPy
- Pandas
- Matplotlib
- Plotly
- Streamlit

## 🔬 Interview Guide & Explanation
If asked to explain this project during an internship interview, emphasize these points:
* **The Core Innovation**: "Instead of designing complex architectural layers, LLMTIME leverages the extensive pattern-extrapolation abilities of Large Language Models. By mapping continuous floats to standardized space-separated text digits, we allow the self-attention mechanism to operate directly on numerical patterns without vocabulary distortion."
* **Data Leakage Prevention**: "Since this is time-series forecasting, standard train-test splitting will cause look-ahead bias if shuffled. We use strict chronological splitting and fit our scalers *only* on the training history."
* **Robustness & Error Handling**: "Free APIs are unstable. Our `OpenRouterModel` implements exponential backoff retries, thread concurrency for sampling, and a mock fallback mode for local validation. We also use the median forecast to filter stochastic outlier paths."

---

## 🚀 Future Improvements
1. **Dynamic Precision Selection**: Auto-determine the optimal decimal scale based on historical noise.
2. **Context Compression**: Utilize sliding-window token limits to input larger historical series.
3. **Multivariate Support**: Prompt Cast formats to support cross-column correlations.
