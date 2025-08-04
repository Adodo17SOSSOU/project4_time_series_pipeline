# src/evaluation.py

import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.metrics import mean_squared_error
import numpy as np

LOG_FILE = "logs/anomaly_log.csv"
EVAL_PLOTS_DIR = "results/plots/evaluation"
os.makedirs(EVAL_PLOTS_DIR, exist_ok=True)

def evaluate_anomalies(log_file=LOG_FILE):
    if not os.path.exists(log_file):
        raise FileNotFoundError(f"{log_file} not found. Run anomaly detection first!")

    df = pd.read_csv(log_file, parse_dates=["timestamp"])
    print(f"[INFO] Loaded anomaly log: {df.shape[0]} entries")

    # RMSE per sensor
    rmse_per_sensor = df.groupby("sensor").apply(
        lambda x: np.sqrt(mean_squared_error(x["actual"], x["predicted"]))
    )

    # Count anomalies per sensor
    anomaly_counts = df.groupby("sensor")["anomaly"].sum()

    # Plot anomaly counts
    plt.figure(figsize=(8,5))
    anomaly_counts.plot(kind="bar", color="tomato")
    plt.title("Anomaly Counts per Sensor")
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{EVAL_PLOTS_DIR}/anomaly_counts.png")
    plt.close()

    # Plot RMSE per sensor
    plt.figure(figsize=(8,5))
    rmse_per_sensor.plot(kind="bar", color="steelblue")
    plt.title("RMSE per Sensor")
    plt.ylabel("RMSE")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{EVAL_PLOTS_DIR}/rmse_per_sensor.png")
    plt.close()

    # Timeline of anomalies
    plt.figure(figsize=(12,5))
    for sensor in df["sensor"].unique():
        sensor_data = df[df["sensor"]==sensor]
        plt.scatter(sensor_data["timestamp"], sensor_data["anomaly"], label=sensor, alpha=0.7)
    plt.title("Anomaly Timeline")
    plt.xlabel("Time")
    plt.ylabel("Anomaly (0/1)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{EVAL_PLOTS_DIR}/anomaly_timeline.png")
    plt.close()

    print("[INFO] Evaluation complete. Plots saved in results/plots/evaluation")

    return rmse_per_sensor, anomaly_counts

