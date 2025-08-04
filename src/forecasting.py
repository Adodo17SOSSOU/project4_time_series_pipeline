# src/forecasting.py

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from src.preprocessing import load_and_preprocess, train_test_split

RESULTS_DIR = "results"
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots", "forecasting")
MODELS_DIR = os.path.join(RESULTS_DIR, "models", "arima")

os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

def forecast_arima(csv_path="data/processed/synthetic_time_series_preprocessed.csv", steps_ahead=50):
    """
    Forecast each sensor with ARIMA and save plots + predictions.
    """
    # Load preprocessed data
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)

    # Train/test split
    train_df, test_df = train_test_split(df, train_ratio=0.8)

    for sensor in df.columns:
        print(f"[INFO] Forecasting {sensor} with ARIMA...")

        # Fit ARIMA model
        series = train_df[sensor]
        try:
            model = ARIMA(series, order=(2, 0, 2))
            model_fit = model.fit()
        except Exception as e:
            print(f"[WARNING] ARIMA failed for {sensor}: {e}")
            continue

        # Forecast
        forecast = model_fit.forecast(steps=len(test_df))
        forecast.index = test_df.index

        # Save model summary
        with open(os.path.join(MODELS_DIR, f"{sensor}_arima_summary.txt"), "w") as f:
            f.write(str(model_fit.summary()))

        # Plot forecast
        plt.figure(figsize=(12, 5))
        plt.plot(train_df[sensor], label="Train")
        plt.plot(test_df[sensor], label="Test", color="orange")
        plt.plot(forecast, label="Forecast", color="green")
        plt.title(f"ARIMA Forecast for {sensor}")
        plt.xlabel("Time")
        plt.ylabel("Value")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, f"{sensor}_forecast.png"))
        plt.close()

        print(f"[INFO] Saved forecast plot for {sensor}")

    print(f"[INFO] Forecasting complete. Plots in {PLOTS_DIR}, models in {MODELS_DIR}")

if __name__ == "__main__":
    forecast_arima()

