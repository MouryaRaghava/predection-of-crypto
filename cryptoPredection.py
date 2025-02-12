import numpy as np
import pandas as pd
import requests
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

def fetch_historical_data(token_key):
    url = f'https://api.coingecko.com/api/v3/coins/{token_key}/market_chart?vs_currency=usd&days=365'
    response = requests.get(url)
    if response.status_code != 200:
        raise ValueError("Failed to fetch data. Check token key and try again.")
    data = response.json()
    if 'prices' not in data:
        raise ValueError("Invalid response format from API.")
    prices = [item[1] for item in data['prices']]
    return pd.DataFrame({'Close': prices})

def load_data(df, sequence_length=50):
    if df.empty:
        raise ValueError("Empty data frame received. Check API response.")
    data = df['Close'].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)
    
    X, y = [], []
    for i in range(sequence_length, len(data_scaled)):
        X.append(data_scaled[i-sequence_length:i, 0])
        y.append(data_scaled[i, 0])
    
    X, y = np.array(X), np.array(y)
    return X, y, scaler, df

def build_model():
    return RandomForestRegressor(n_estimators=100, random_state=42)

def predict(model, X, scaler):
    if X.size == 0:
        raise ValueError("Prediction data is empty. Check preprocessing steps.")
    predictions = model.predict(X).reshape(-1, 1)
    return scaler.inverse_transform(predictions)

def assess_risk(predictions):
    if predictions.size == 0:
        return "Risk assessment unavailable due to missing predictions."
    change = (predictions[-1] - predictions[0]) / predictions[0] * 100
    if change > 10:
        return "High Risk"
    elif change > 5:
        return "Moderate Risk"
    else:
        return "Low Risk"

def plot_predictions(df, predictions):
    plt.figure(figsize=(12, 6))
    plt.plot(df.index[-len(predictions):], df['Close'].values[-len(predictions):], label='Actual Prices', color='blue')
    plt.plot(df.index[-len(predictions):], predictions, label='Predicted Prices', color='red')
    plt.xlabel('Time')
    plt.ylabel('Price (USD)')
    plt.title('Crypto Price Prediction')
    plt.legend()
    plt.show()

# Example usage:
try:
    token_key = input("Enter token key: ")
    df = fetch_historical_data(token_key)
    X_train, y_train, scaler, df = load_data(df)
    model = build_model()
    model.fit(X_train, y_train)
    predictions = predict(model, X_train, scaler)
    risk_level = assess_risk(predictions)
    print(f"Risk Level: {risk_level}")
    plot_predictions(df, predictions)
except Exception as e:
    print(f"Error: {e}")
