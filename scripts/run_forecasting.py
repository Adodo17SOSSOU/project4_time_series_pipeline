import os
import sys

# Ensure project root is in Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.forecasting import forecast_arima

if __name__ == "__main__":
    forecast_arima()

