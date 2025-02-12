# Crypto Price Prediction Using RandomForestRegressor

## Overview
This project fetches historical cryptocurrency price data, preprocesses it, and uses a `RandomForestRegressor` model to predict future prices. It also assesses the risk level based on predicted price changes.

## Installation
Before running the script, install the required dependencies:

```bash
pip install numpy pandas requests scikit-learn matplotlib
```

## Usage
Run the script and enter a valid cryptocurrency token key (e.g., `bitcoin`, `ethereum`).

```bash
python script.py
```

## Code Segments Explained

### 1. Fetching Historical Data
```python
def fetch_historical_data(token_key):
```
Fetches the last 365 days of price data from the CoinGecko API. If the API request fails, an error is raised.

### 2. Preprocessing Data
```python
def load_data(df, sequence_length=50):
```
- Converts price data into a scaled format using `MinMaxScaler`.
- Creates input sequences (`X`) and target values (`y`).

### 3. Building the Model
```python
def build_model():
```
Initializes a `RandomForestRegressor` with 100 estimators and a fixed random seed.

### 4. Making Predictions
```python
def predict(model, X, scaler):
```
Uses the trained model to make predictions and inverse transforms them back to the original price scale.

### 5. Risk Assessment
```python
def assess_risk(predictions):
```
Calculates the percentage change in predicted prices and categorizes risk as:
- **High Risk** (>10%)
- **Moderate Risk** (5-10%)
- **Low Risk** (<5%)

### 6. Plotting Predictions
```python
def plot_predictions(df, predictions):
```
Plots actual vs. predicted prices using Matplotlib.

## Example Output
```
Enter token key: bitcoin
Risk Level: Moderate Risk
(Price prediction graph displayed)
```

## Notes
- Ensure a valid token key is used (e.g., `bitcoin`, `ethereum`).
- The model is basic and does not consider external market factors.
- Improve predictions by using more advanced models like LSTMs or XGBoost.

