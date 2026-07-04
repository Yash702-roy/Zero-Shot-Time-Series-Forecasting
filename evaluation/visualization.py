import logging
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from typing import Dict, Union, Optional

logger = logging.getLogger(__name__)

def plot_forecast_static(history: Union[pd.Series, np.ndarray], 
                         y_true: Union[pd.Series, np.ndarray], 
                         y_pred: np.ndarray, 
                         intervals: Dict[str, np.ndarray], 
                         save_path: str,
                         title: str = "Zero-Shot Time Series Forecast (LLMTIME)") -> None:
    """
    Generates a static matplotlib plot showing historical values, actual values, 
    median predictions, and shaded prediction intervals (80% and 90%).
    Saves the plot as a PNG.
    
    Args:
        history: Historical data series.
        y_true: Actual future values (test set).
        y_pred: Median forecast values.
        intervals: Dictionary of prediction interval arrays ('p10', 'p90', 'p05', 'p95').
        save_path: Path to save the PNG file.
        title: Title of the chart.
    """
    plt.figure(figsize=(12, 6))
    
    # 1. Align time indexes if inputs are pandas Series
    # If they are numpy arrays, generate sequential integer indexes
    if isinstance(history, pd.Series):
        history_idx = history.index
        future_idx = y_true.index if isinstance(y_true, pd.Series) else pd.date_range(start=history_idx[-1], periods=len(y_pred) + 1, freq=history_idx.freq)[1:]
    else:
        history_idx = np.arange(len(history))
        future_idx = np.arange(len(history), len(history) + len(y_pred))
        
    y_true_vals = y_true.values if isinstance(y_true, pd.Series) else np.asarray(y_true)
    history_vals = history.values if isinstance(history, pd.Series) else np.asarray(history)
    
    # 2. Plot lines
    # Plot history (last 3-4x of horizon length for readability)
    plot_history_len = min(len(history_vals), len(y_pred) * 4)
    plt.plot(history_idx[-plot_history_len:], history_vals[-plot_history_len:], label="History", color="#1f77b4", linewidth=1.5)
    
    # Plot true values
    plt.plot(future_idx, y_true_vals, label="Actuals (Ground Truth)", color="#2ca02c", linewidth=2.0, linestyle="--")
    
    # Plot median forecast
    plt.plot(future_idx, y_pred, label="Median Forecast (P50)", color="#ff7f0e", linewidth=2.0)
    
    # 3. Plot shaded intervals
    # 80% interval (P10 to P90)
    if "p10" in intervals and "p90" in intervals:
        plt.fill_between(future_idx, intervals["p10"], intervals["p90"], 
                         color="#ff7f0e", alpha=0.25, label="80% Prediction Interval (P10-P90)")
                         
    # 90% interval (P05 to P95)
    if "p05" in intervals and "p95" in intervals:
        plt.fill_between(future_idx, intervals["p05"], intervals["p95"], 
                         color="#ff7f0e", alpha=0.10, linestyle=":", label="90% Prediction Interval (P05-P95)")

    # 4. Stylize
    plt.title(title, fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Timestamp" if isinstance(history, pd.Series) else "Index", fontsize=11, labelpad=10)
    plt.ylabel("Value", fontsize=11, labelpad=10)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="upper left", frameon=True, shadow=False, facecolor="white", edgecolor="#ddd")
    
    plt.tight_layout()
    
    # Save file
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    logger.info(f"Saved static forecast plot to {save_path}")


def plot_forecast_interactive(history: Union[pd.Series, np.ndarray], 
                             y_true: Union[pd.Series, np.ndarray], 
                             y_pred: np.ndarray, 
                             intervals: Dict[str, np.ndarray], 
                             save_path: str,
                             title: str = "Zero-Shot Time Series Forecast (LLMTIME)") -> go.Figure:
    """
    Generates an interactive Plotly graph showing history, actuals, median predictions, 
    and shaded prediction intervals. Saves the graph as an HTML file.
    
    Args:
        history: Historical data series.
        y_true: Actual future values (test set).
        y_pred: Median forecast values.
        intervals: Dictionary of prediction interval arrays ('p10', 'p90', 'p05', 'p95').
        save_path: Path to save the HTML file.
        title: Title of the chart.
        
    Returns:
        go.Figure: The generated Plotly figure.
    """
    if isinstance(history, pd.Series):
        history_idx = history.index
        future_idx = y_true.index if isinstance(y_true, pd.Series) else pd.date_range(start=history_idx[-1], periods=len(y_pred) + 1, freq=history_idx.freq)[1:]
    else:
        history_idx = np.arange(len(history))
        future_idx = np.arange(len(history), len(history) + len(y_pred))
        
    y_true_vals = y_true.values if isinstance(y_true, pd.Series) else np.asarray(y_true)
    history_vals = history.values if isinstance(history, pd.Series) else np.asarray(history)
    plot_history_len = min(len(history_vals), len(y_pred) * 4)

    fig = go.Figure()

    # 1. Shaded Interval (90% Interval P05-P95) - Plot first to lay behind lines
    if "p05" in intervals and "p95" in intervals:
        fig.add_trace(go.Scatter(
            x=list(future_idx) + list(future_idx)[::-1],
            y=list(intervals["p95"]) + list(intervals["p05"])[::-1],
            fill='toself',
            fillcolor='rgba(255, 127, 14, 0.1)',
            line=dict(color='rgba(255, 100, 0, 0.05)', width=0),
            hoverinfo="skip",
            showlegend=True,
            name="90% Prediction Interval (P05-P95)"
        ))

    # 2. Shaded Interval (80% Interval P10-P90)
    if "p10" in intervals and "p90" in intervals:
        fig.add_trace(go.Scatter(
            x=list(future_idx) + list(future_idx)[::-1],
            y=list(intervals["p90"]) + list(intervals["p10"])[::-1],
            fill='toself',
            fillcolor='rgba(255, 127, 14, 0.25)',
            line=dict(color='rgba(255, 127, 14, 0)'),
            hoverinfo="skip",
            showlegend=True,
            name="80% Prediction Interval (P10-P90)"
        ))

    # 3. Plot History
    fig.add_trace(go.Scatter(
        x=history_idx[-plot_history_len:],
        y=history_vals[-plot_history_len:],
        mode='lines',
        name='History',
        line=dict(color='#1f77b4', width=2)
    ))

    # 4. Plot Actuals
    fig.add_trace(go.Scatter(
        x=future_idx,
        y=y_true_vals,
        mode='lines',
        name='Actuals (Ground Truth)',
        line=dict(color='#2ca02c', width=2, dash='dash')
    ))

    # 5. Plot Forecast
    fig.add_trace(go.Scatter(
        x=future_idx,
        y=y_pred,
        mode='lines',
        name='Median Forecast (P50)',
        line=dict(color='#ff7f0e', width=3)
    ))

    # Update Layout
    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color='#333', family='Outfit, Inter, sans-serif'), pad=dict(b=10)),
        xaxis=dict(title="Timestamp" if isinstance(history, pd.Series) else "Index", gridcolor='#f0f0f0', showgrid=True),
        yaxis=dict(title="Value", gridcolor='#f0f0f0', showgrid=True),
        plot_bgcolor='white',
        paper_bgcolor='white',
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(l=40, r=40, t=80, b=40)
    )

    # Save to file
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.write_html(save_path)
    logger.info(f"Saved interactive forecast plot to {save_path}")

    return fig
