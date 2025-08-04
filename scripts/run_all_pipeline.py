# scripts/run_all_pipeline.py

from src.preprocessing import load_and_preprocess, train_test_split
from src.eda import run_eda
from src.forecasting import run_forecasting
from src.anomaly_detection import detect_anomalies
from src.evaluation import evaluate_anomalies

if __name__ == "__main__":
    # 1. Preprocessing
    df, scaler = load_and_preprocess()
    train_df, test_df = train_test_split(df)

    # 2. EDA
    run_eda(df)

    # 3. Forecasting
    run_forecasting(df)

    # 4. Anomaly Detection
    detect_anomalies()

    # 5. Evaluation
    evaluate_anomalies()

