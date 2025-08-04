# scripts/generate_data.py

import numpy as np
import pandas as pd
import os
from datetime import datetime, timedelta

def generate_synthetic_data(
    num_points=2000, 
    num_sensors=3, 
    anomaly_fraction=0.02, 
    output_file="data/raw/synthetic_time_series.csv"
):
    """
    Generate synthetic multi-sensor time series data with anomalies.
    """
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    start_time = datetime(2025, 1, 1)
    timestamps = [start_time + timedelta(minutes=i) for i in range(num_points)]

    data = {}
    for sensor_id in range(1, num_sensors + 1):
        # Base components
        trend = np.linspace(0, 10, num_points) * 0.05 * sensor_id
        seasonality = np.sin(np.linspace(0, 20*np.pi, num_points)) * 0.5
        noise = np.random.normal(0, 0.2, num_points)

        # Combine components
        series = trend + seasonality + noise

        # Inject anomalies
        anomalies = np.random.choice([0, 1], size=num_points, p=[1-anomaly_fraction, anomaly_fraction])
        series[anomalies == 1] += np.random.normal(3, 1, anomalies.sum())  # spike anomalies

        data[f"sensor_{sensor_id}"] = series

    df = pd.DataFrame(data, index=pd.to_datetime(timestamps))
    df.index.name = "timestamp"
    df.to_csv(output_file)

    print(f"[INFO] Synthetic time series saved to {output_file}")

if __name__ == "__main__":
    generate_synthetic_data()

