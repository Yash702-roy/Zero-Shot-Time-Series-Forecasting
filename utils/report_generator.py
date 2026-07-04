import os
import json
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

def generate_html_report(experiments_path: str = "outputs/experiment.json", 
                         output_path: str = "outputs/report.html") -> None:
    """
    Generates a professional research-grade HTML report of the zero-shot forecasting
    experiments and baseline comparisons.
    """
    logger.info(f"Generating HTML report from {experiments_path}...")
    try:
        if not os.path.exists(experiments_path):
            raise FileNotFoundError(f"Experiments results file {experiments_path} not found.")
            
        with open(experiments_path, "r", encoding="utf-8") as f:
            runs = json.load(f)
            
        # Parse comparison stats
        table_rows = ""
        model_details = ""
        
        for run in runs:
            model_label = run.get("model")
            model_name = run.get("model_name")
            runtime = run.get("runtime_seconds")
            metrics = run.get("metrics", {})
            mae = metrics.get("MAE", "N/A")
            rmse = metrics.get("RMSE", "N/A")
            mape = metrics.get("MAPE", "N/A")
            smape = metrics.get("sMAPE", "N/A")
            medae = metrics.get("MedAE", "N/A")
            
            # Format row for summary table
            table_rows += f"""
            <tr>
                <td><strong>{model_label}</strong></td>
                <td>{model_name}</td>
                <td>{mae}</td>
                <td>{rmse}</td>
                <td>{smape}%</td>
                <td>{runtime}s</td>
            </tr>
            """
            
            # Format details for individual runs
            config_str = ""
            if "configuration" in run:
                config_str = f"""
                <div class="config-section">
                    <h4>Hyperparameters</h4>
                    <pre><code>{json.dumps(run["configuration"], indent=2)}</code></pre>
                </div>
                """
                
            model_details += f"""
            <div class="model-card">
                <h3>{model_label} (Model Details)</h3>
                <p><strong>Registry Name:</strong> <code>{model_name}</code></p>
                <div class="metrics-grid">
                    <div class="metric-box"><span class="val">{mae}</span><span class="lbl">MAE</span></div>
                    <div class="metric-box"><span class="val">{rmse}</span><span class="lbl">RMSE</span></div>
                    <div class="metric-box"><span class="val">{mape}%</span><span class="lbl">MAPE</span></div>
                    <div class="metric-box"><span class="val">{smape}%</span><span class="lbl">sMAPE</span></div>
                    <div class="metric-box"><span class="val">{medae}</span><span class="lbl">MedAE</span></div>
                    <div class="metric-box"><span class="val">{runtime}s</span><span class="lbl">Runtime</span></div>
                </div>
                {config_str}
            </div>
            """

        timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # HTML Content template with premium styling (rich dark gradient, clean fonts, glassmorphism)
        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LLMTIME Forecasting Report</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&family=Outfit:wght@400;600;800&display=swap" rel="stylesheet">
    <style>
        :root {{
            --bg-color: #0b0f19;
            --card-bg: rgba(255, 255, 255, 0.03);
            --border-color: rgba(255, 255, 255, 0.08);
            --accent-color: #ff7f0e;
            --text-primary: #f3f4f6;
            --text-secondary: #9ca3af;
            --success-color: #10b981;
        }}
        
        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}
        
        body {{
            background-color: var(--bg-color);
            color: var(--text-primary);
            font-family: 'Inter', sans-serif;
            line-height: 1.6;
            padding: 40px 20px;
        }}
        
        .container {{
            max-width: 1100px;
            margin: 0 auto;
        }}
        
        header {{
            margin-bottom: 40px;
            text-align: center;
        }}
        
        h1 {{
            font-family: 'Outfit', sans-serif;
            font-size: 2.5rem;
            font-weight: 800;
            background: linear-gradient(135deg, #ff7f0e 0%, #ffc078 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 10px;
        }}
        
        .subtitle {{
            color: var(--text-secondary);
            font-size: 1.1rem;
        }}
        
        .meta {{
            font-size: 0.85rem;
            color: var(--text-secondary);
            margin-top: 15px;
        }}
        
        .section {{
            background: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 16px;
            padding: 30px;
            margin-bottom: 30px;
            backdrop-filter: blur(10px);
        }}
        
        h2 {{
            font-family: 'Outfit', sans-serif;
            font-size: 1.6rem;
            margin-bottom: 20px;
            border-left: 4px solid var(--accent-color);
            padding-left: 12px;
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 15px;
            margin-bottom: 15px;
        }}
        
        th, td {{
            padding: 14px 16px;
            text-align: left;
            border-bottom: 1px solid var(--border-color);
        }}
        
        th {{
            background-color: rgba(255, 255, 255, 0.05);
            font-weight: 600;
            color: var(--text-primary);
        }}
        
        tr:hover {{
            background-color: rgba(255, 255, 255, 0.01);
        }}
        
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
            gap: 15px;
            margin-top: 15px;
            margin-bottom: 15px;
        }}
        
        .metric-box {{
            background: rgba(255, 255, 255, 0.02);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 15px;
            text-align: center;
        }}
        
        .metric-box .val {{
            display: block;
            font-size: 1.4rem;
            font-weight: 700;
            color: var(--accent-color);
            font-family: 'Outfit', sans-serif;
        }}
        
        .metric-box .lbl {{
            font-size: 0.75rem;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-top: 4px;
            display: block;
        }}
        
        .config-section {{
            margin-top: 15px;
            background: rgba(0, 0, 0, 0.2);
            padding: 15px;
            border-radius: 8px;
            border: 1px solid rgba(255, 255, 255, 0.05);
        }}
        
        pre {{
            font-family: 'Courier New', Courier, monospace;
            font-size: 0.85rem;
            overflow-x: auto;
            color: #d1d5db;
        }}
        
        .plot-container {{
            margin-top: 20px;
            text-align: center;
        }}
        
        .plot-container img {{
            max-width: 100%;
            height: auto;
            border-radius: 8px;
            border: 1px solid var(--border-color);
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
        }}
        
        .model-card {{
            border-top: 1px dashed var(--border-color);
            padding-top: 25px;
            margin-top: 25px;
        }}
        
        .model-card:first-of-type {{
            border-top: none;
            padding-top: 0;
            margin-top: 0;
        }}
        
        .footer {{
            text-align: center;
            color: var(--text-secondary);
            font-size: 0.8rem;
            margin-top: 50px;
            padding-top: 20px;
            border-top: 1px solid var(--border-color);
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>LLMTIME Zero-Shot Forecasting Report</h1>
            <p class="subtitle">Benchmarking LLMs as Zero-Shot Forecasters against Statistical and ML Baselines</p>
            <p class="meta">Report Generated: {timestamp_str} | Target Variable: OT | Dataset: ETTh1.csv</p>
        </header>

        <section class="section">
            <h2>Model Comparison Summary</h2>
            <table>
                <thead>
                    <tr>
                        <th>Model Class</th>
                        <th>Identifier</th>
                        <th>MAE</th>
                        <th>RMSE</th>
                        <th>sMAPE</th>
                        <th>Inference Time</th>
                    </tr>
                </thead>
                <tbody>
                    {table_rows}
                </tbody>
            </table>
        </section>

        <section class="section">
            <h2>Visualization (Static Plot)</h2>
            <div class="plot-container">
                <img src="forecast_plot.png" alt="LLMTIME Forecast Plot">
            </div>
            <p style="margin-top: 15px; font-size: 0.9rem; color: var(--text-secondary); text-align: center;">
                Note: Interactive HTML plot is also saved separately in the outputs directory as <code>forecast_plot_interactive.html</code>.
            </p>
        </section>

        <section class="section">
            <h2>Individual Experiment Runs</h2>
            {model_details}
        </section>

        <div class="footer">
            <p>LLMTIME Research Project Implementation | Professional Internship Portfolio Asset</p>
        </div>
    </div>
</body>
</html>
"""
        
        # Write to file
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        logger.info(f"Successfully generated HTML report at {output_path}")
        
    except Exception as e:
        logger.error(f"Failed to generate HTML report: {e}")
