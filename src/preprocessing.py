# src/preprocessing.py

import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler

RAW_PATH = "data/raw/synthetic_time_series.csv"
PROCESSED_DIR = "data/processed"
PROCESSED_FILE = os.path.join(PROCESSED_DIR, "synthetic_time_series_preprocessed.csv")

def load_and_preprocess(csv_path=RAW_PATH, normalize=True, save=True):
    """
    Load and preprocess time series data.
    1. Load CSV
    2. Handle missing values
    3. Normalize features (optional)
    4. Save to processed folder if save=True
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"{csv_path} does not exist. Please generate data first!")

    # Load CSV and set timestamp as index
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    
    # Handle missing values
    df = df.ffill().bfill()

    scaler = None
    if normalize:
        scaler = StandardScaler()
        df[:] = scaler.fit_transform(df)

    # Save preprocessed data
    if save:
        os.makedirs(PROCESSED_DIR, exist_ok=True)
        df.to_csv(PROCESSED_FILE)
        print(f"[INFO] Preprocessed data saved to {PROCESSED_FILE}")

    print(f"[INFO] Data loaded and preprocessed. Shape: {df.shape}")
    return df, scaler

def train_test_split(df, train_ratio=0.8):
    """
    Split time series into train and test sets.
    """
    split_idx = int(len(df) * train_ratio)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    return train_df, test_df

if __name__ == "__main__":
    df, scaler = load_and_preprocess()
    train_df, test_df = train_test_split(df)
    print(f"[INFO] Train shape: {train_df.shape}, Test shape: {test_df.shape}")

