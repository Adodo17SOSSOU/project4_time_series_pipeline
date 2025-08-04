import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from statsmodels.tsa.seasonal import seasonal_decompose

# Paths
PROCESSED_PATH = "data/processed/synthetic_time_series_preprocessed.csv"
EDA_PLOTS_DIR = "results/plots/eda"

# Create directory if not exists
os.makedirs(EDA_PLOTS_DIR, exist_ok=True)

def plot_time_series(df):
    """Plot all sensor time series."""
    plt.figure(figsize=(12, 6))
    for col in df.columns:
        plt.plot(df.index, df[col], label=col)
    plt.legend()
    plt.title("Time Series Overview")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.tight_layout()
    plt.savefig(os.path.join(EDA_PLOTS_DIR, "time_series_overview.png"))
    plt.close()
    print(f"[INFO] Time series plot saved to {EDA_PLOTS_DIR}/time_series_overview.png")

def plot_correlation(df):
    """Plot correlation heatmap."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Sensor Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(os.path.join(EDA_PLOTS_DIR, "correlation_heatmap.png"))
    plt.close()
    print(f"[INFO] Correlation heatmap saved to {EDA_PLOTS_DIR}/correlation_heatmap.png")

def plot_rolling_stats(df, window=50):
    """Plot rolling mean and std for first sensor."""
    sensor = df.columns[0]
    rolling_mean = df[sensor].rolling(window=window).mean()
    rolling_std = df[sensor].rolling(window=window).std()

    plt.figure(figsize=(12, 5))
    plt.plot(df.index, df[sensor], label="Original", alpha=0.5)
    plt.plot(df.index, rolling_mean, label="Rolling Mean", color="red")
    plt.plot(df.index, rolling_std, label="Rolling Std", color="green")
    plt.title(f"Rolling Statistics ({sensor})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(EDA_PLOTS_DIR, f"rolling_stats_{sensor}.png"))
    plt.close()
    print(f"[INFO] Rolling stats saved to {EDA_PLOTS_DIR}/rolling_stats_{sensor}.png")

def seasonal_decomposition_plot(df, sensor=None, freq=50):
    """Perform seasonal decomposition for a selected sensor."""
    if sensor is None:
        sensor = df.columns[0]
    decomposition = seasonal_decompose(df[sensor], period=freq, model='additive')
    fig = decomposition.plot()
    fig.set_size_inches(12, 8)
    plt.suptitle(f"Seasonal Decomposition ({sensor})", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(EDA_PLOTS_DIR, f"seasonal_decomposition_{sensor}.png"))
    plt.close()
    print(f"[INFO] Seasonal decomposition saved to {EDA_PLOTS_DIR}/seasonal_decomposition_{sensor}.png")

def main():
    # Load processed CSV
    if not os.path.exists(PROCESSED_PATH):
        raise FileNotFoundError(f"{PROCESSED_PATH} not found! Run preprocessing first.")

    df = pd.read_csv(PROCESSED_PATH, index_col=0, parse_dates=True)
    print(f"[INFO] Loaded preprocessed data with shape {df.shape}")

    # Generate EDA plots
    plot_time_series(df)
    plot_correlation(df)
    plot_rolling_stats(df)
    seasonal_decomposition_plot(df)
    print("[INFO] EDA completed successfully!")

if __name__ == "__main__":
    main()


