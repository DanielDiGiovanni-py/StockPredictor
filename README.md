# Leveraged ETF Trading Strategy with IBKR API and Machine Learning

This project implements a machine learning-driven trading strategy for leveraged ETFs (SPXL and SPXS) based on predictions of the second derivative (acceleration) of SPY's price movement. It integrates data from both the IBKR API and Yahoo Finance, utilizes a transformer-based model for time-series forecasting, and performs backtesting to evaluate the strategy's performance.

## Features

- **Data Fetching**:
  - Retrieves SPXL and SPXS data from the IBKR API.
  - Fetches financial data for Crude Oil, Gold, USD Index, SPY, TNX (Treasury Yield), and VIX (Volatility Index) from Yahoo Finance.
  - Combines data into a unified dataset for analysis.

- **Feature Engineering**:
  - Calculates SPY's first and second derivatives to represent price velocity and acceleration.
  - Standardizes all features for use in the machine learning model.

- **Transformer-Based Prediction Model**:
  - A PyTorch-based transformer model is trained to predict SPY's second derivative one day ahead.
  - Walk-forward validation ensures robust performance evaluation.

- **Leveraged ETF Trading Strategy**:
  - Trades SPXL (3x leveraged long S&P 500) and SPXS (3x leveraged short S&P 500) based on predictions.
  - Buys SPXL for positive acceleration and SPXS for negative acceleration.
  - Calculates cumulative returns from the strategy.

- **Backtesting**:
  - Simulates the strategy on historical data.
  - Provides performance metrics, including daily and cumulative returns.

- **Visualization**:
  - Plots cumulative returns of the strategy for performance evaluation.

## Requirements

- Python 3.7+
- Libraries:
  - `yfinance`
  - `pandas`
  - `numpy`
  - `torch`
  - `matplotlib`
  - `scikit-learn`
  - `ib_insync`
  - `psutil`
  - `nest_asyncio`

Install the required libraries using pip:
```bash
pip install yfinance pandas numpy torch matplotlib scikit-learn ib_insync psutil nest_asyncio
