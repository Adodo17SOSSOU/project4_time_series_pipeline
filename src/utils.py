# src/utils.py

import os
import matplotlib.pyplot as plt
from datetime import datetime

def save_plot(fig, filename, folder="results/plots"):
    """
    Save a matplotlib figure with a timestamp to avoid overwriting.
    """
    os.makedirs(folder, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(folder, f"{timestamp}_{filename}")
    fig.savefig(path)
    plt.close(fig)
    print(f"[INFO] Plot saved to {path}")

def log_message(msg, log_file="logs/project.log"):
    """
    Append a log message to logs/project.log with a timestamp.
    """
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    with open(log_file, "a") as f:
        f.write(f"{datetime.now()} - {msg}\n")
    print(f"[LOG] {msg}")

