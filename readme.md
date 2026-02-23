# CO₂ Factor Forecasting (Netherlands)

Production-grade pipeline for hourly CO₂ emission factor forecasting
using energy mix data (NED API) and weather covariates (Open-Meteo).

Models: - Temporal Fusion Transformer (TFT) - N-HiTS - Seasonal Naive
baseline

### LightGBM baselines (notebooks/01_lgb_singlr.ipynb & 02_lgb_168.ipynb)

- **LGB (1 model)**: single model predicting the 168h horizon using lag/time/weather features. Fast and interpretable baseline.
- **LGB (168 models)**: one model per horizon step (h=1..168). Often improves accuracy at the cost of training/deployment complexity.

Forecast horizon: 168 hours (7 days)\
Resolution: 1 hour

------------------------------------------------------------------------

## Project Overview

This project implements an end-to-end, reproducible forecasting pipeline:

1. Collect hourly energy production data from the NED API  
2. Collect historical weather data and 7-day forecasts from Open-Meteo  
3. Construct a unified hourly UTC master index  
4. Align all sources to the master index  
5. Apply split-before-fill logic to prevent temporal leakage  
6. Perform controlled interpolation and missing-value diagnostics  
7. Engineer cyclical time and deterministic future features  
8. Train and benchmark multiple forecasting models (TFT, N-HiTS, LightGBM, Seasonal Naive)  
9. Evaluate strictly out-of-sample performance  
10. Generate operational 168-hour forecasts with prediction intervals  

------------------------------------------------------------------------

## Project Structure

co2-forecasting/
│
├── src/
│   ├── forecasting/
│   │   ├── collect_data.py        # Full data pipeline (NED + weather)
│   │   ├── __init__.py
│   │   └── ...
│   │
│   └── co2_forecasting.egg-info/
│
├── notebooks/
│   ├── 02_lgb_168.ipynb           # LightGBM multi-model experiments
│   ├── 03_tft.ipynb               # TFT & N-HiTS training and evaluation
│   ├── test.ipynb
│   └── ...
│
├── data/                          # Generated datasets (not versioned)
│   ├── master_dataset.csv
│   ├── ned_hourly_filled.csv
│   ├── openmeteo_historical.csv
│   └── openmeteo_forecast_7days.csv
│
├── pyproject.toml
├── .gitignore
└── README.md
------------------------------------------------------------------------

## Installation

git clone https://github.com/your-username/co2-forecasting.git cd
co2-forecasting

python -m venv .venv .venv`\Scripts`{=tex}`\activate`{=tex}

pip install -e .

------------------------------------------------------------------------

## API Key Setup

Create `.env` in project root:

NED_API_KEY=your_api_key_here

The pipeline loads it automatically via python-dotenv.

------------------------------------------------------------------------

## Data Pipeline

Run:

python src/forecasting/collect_data.py

Outputs: - data/ned_hourly_filled.csv - data/openmeteo_historical.csv -
data/openmeteo_forecast_7days.csv - data/master_dataset.csv

------------------------------------------------------------------------

## Training

From notebook:

from forecasting.train_model import main main(train=True)

Artifacts saved to:

artifacts/ ├── checkpoints/ ├── logs/ └── predictions/

------------------------------------------------------------------------

## Evaluation Metrics

-   MAE
-   RMSE
-   MAPE
-   Per-horizon MAE
-   Worst forecast windows

------------------------------------------------------------------------

## Inference

Produces: - predictions/co2_forecast_168h_noweather.csv -
predictions/co2_forecast_7days_noweather.png

------------------------------------------------------------------------

## License

Private / Research Use Only

