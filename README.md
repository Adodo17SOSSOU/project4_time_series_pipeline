#  Time Series Project

This project demonstrates **end-to-end time series analysis**, covering:

1. **Data generation & preprocessing**
2. **Exploratory Data Analysis (EDA)**
3. **Time Series Forecasting (ARIMA)**
4. **Anomaly Detection**
5. **Evaluation & Visualization**


---

## **Project Structure**

```
final_time_series_project/
│
├── data/
│   ├── raw/                 # Original synthetic data
│   ├── processed/           # Preprocessed CSV files
│   └── logs/                # Streaming logs or anomaly logs
│
├── results/
│   ├── models/              # Saved forecasting models
│   └── plots/
│       ├── eda/             # Exploratory plots
│       ├── forecasting/     # Forecast plots per sensor
│       ├── anomaly/         # Anomaly detection plots
│       └── evaluation/      # RMSE, anomaly counts, timeline plots
│
├── scripts/                 # Runner scripts for each stage
│   ├── generate_data.py
│   ├── run_eda.py
│   ├── run_forecasting.py
│   ├── run_anomaly.py
│   └── run_all_pipeline.py  # Full project pipeline
│
└── src/                     # Core Python modules
    ├── preprocessing.py
    ├── eda.py
    ├── forecasting.py
    ├── anomaly_detection.py
    ├── evaluation.py
    ├── utils.py
    └── __init__.py
```

---

## **1. Data Generation & Preprocessing**

* **`scripts/generate_data.py`** generates synthetic time series data with multiple sensors and anomalies.
* **`src/preprocessing.py`** loads, fills missing values, normalizes features, and saves preprocessed CSVs to `data/processed/`.

Run preprocessing:

```bash
python scripts/generate_data.py
```

---

## **2. Exploratory Data Analysis (EDA)**

* **`src/eda.py`** generates visual insights:

  * Line plots of all sensors
  * Correlation heatmap
  * Rolling statistics
  * Seasonal decomposition per sensor

Run EDA:

```bash
python scripts/run_eda.py
```

Generated plots are saved in `results/plots/eda/`.

---

## **3. Time Series Forecasting**

* **`src/forecasting.py`** applies **ARIMA** models per sensor.
* Forecast vs. actual vs. test split plots are saved in `results/plots/forecasting/`.
* Models can be optionally saved to `results/models/`.

Run forecasting:

```bash
python scripts/run_forecasting.py
```

---

## **4. Anomaly Detection**

* **`src/anomaly_detection.py`** flags anomalies using **forecast residuals**.
* Logs each timestamp, sensor, actual vs. predicted values, and anomaly flag to `logs/anomaly_log.csv`.
* Plots anomalies over time per sensor.

Run anomaly detection:

```bash
python scripts/run_anomaly.py
```

---

## **5. Evaluation & Summary**

* **`src/evaluation.py`** computes:

  * RMSE per sensor
  * Anomaly counts per sensor
  * Timeline of detected anomalies

Results saved to `results/plots/evaluation/`.

---

## **Run the Full Pipeline**

To generate data, preprocess, run EDA, forecasting, anomaly detection, and evaluation in one go:

```bash
python scripts/run_all_pipeline.py
```

---

## **Requirements**

Install dependencies:

```bash
pip install -r requirements.txt
```

Typical dependencies:

* pandas
* numpy
* matplotlib
* seaborn
* scikit-learn
* statsmodels

---

## **Example Output**

* **EDA plots:** Seasonal decomposition & correlations
* **Forecasting plots:** ARIMA predictions vs. ground truth
* **Anomaly plots:** Points flagged as anomalies
* **Evaluation:** RMSE per sensor & anomaly counts

This structure demonstrates mastery of:

* Time series preprocessing
* Exploratory analysis
* Forecasting
* Anomaly detection
* Automated evaluation & reporting

---

**License:** MIT
