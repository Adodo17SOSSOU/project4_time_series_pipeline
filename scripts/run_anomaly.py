# scripts/run_anomaly.py
import os
import sys

# Ensure project root is in Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.anomaly_detection import detect_anomalies

if __name__ == "__main__":
    detect_anomalies()
