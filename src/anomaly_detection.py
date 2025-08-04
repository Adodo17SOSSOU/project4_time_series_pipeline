# src/anomaly_detection.py

import pandas as pd
import numpy as np
import os
import time
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt

WINDOW_SIZE = 100
THRESHOLD_STD = 3

LOG_FILE = "logs/anomaly_log.csv"
PLOTS_DIR = "results/plots/anomalies"

os.makedirs(PLOTS_DIR, exist_ok=True)

def detect_anomalies(csv_path="data/processed/synthetic_time_series_preprocessed.csv"):
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)

    # Initialize log file
    with open(LOG_FILE, "w") as f:
        f.write("timestamp,sensor,actual,predicted,residual,anomaly\n")

    for sensor in df.columns:
        print(f"[INFO] Detecting anomalies for {sensor}...")
        anomalies = []

        series_window = []
        for timestamp, value in df[sensor].items():
            series_window.append(value)
            if len(series_window) > WINDOW_SIZE:
                series_window.pop(0)

                # Fit ARIMA model
                try:
                    model = ARIMA(series_window, order=(2,0,2))
                    model_fit = model.fit()
                    predicted = model_fit.forecast()[0]
                except:
                    predicted = series_window[-1]  # fallback

                residual = value - predicted
                anomaly = abs(residual) > THRESHOLD_STD * np.std(series_window)

                # Log result
                with open(LOG_FILE, "a") as f:
                    f.write(f"{timestamp},{sensor},{value},{predicted},{residual},{int(anomaly)}\n")

                if anomaly:
                    anomalies.append((timestamp, value))

        # Plot anomalies
        plt.figure(figsize=(12,4))
        plt.plot(df.index, df[sensor], label='Value')
        if anomalies:
            times, vals = zip(*anomalies)
            plt.scatter(times, vals, color='red', label='Anomaly')
        plt.title(f"Anomalies in {sensor}")
        plt.legend()
        plt.savefig(f"{PLOTS_DIR}/{sensor}_anomalies.png")
        plt.close()

    print(f"[INFO] Anomaly detection completed. Results in {LOG_FILE}")

