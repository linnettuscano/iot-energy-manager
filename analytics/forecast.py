#!/usr/bin/env python3
"""
Forecasting Sensor Data using Enhanced ARIMA and LSTM

This script reads historical sensor data from 'data/sensor_data.csv',
performs forecasting using an enhanced ARIMA model (with log transformation and an expanded grid search)
and an enhanced LSTM model (with increased capacity and StandardScaler) on each sensor variable.
Key visualization (forecast comparison plot) and a single CSV file with forecast results are generated.
If no ARIMA model can be fitted, the error is caught and processing continues using only the LSTM forecast.
Usage:
    python analytics/forecast.py --steps 15 --look_back 24 --epochs 100 --batch_size 32

Note: --steps specifies the forecast horizon (number of future points).
"""

import os
import argparse
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Disable warnings globally
warnings.filterwarnings("ignore")

# ---------------------------
# Enhanced ARIMA with Log Transformation and Expanded Grid Search
# ---------------------------
def grid_search_arima(ts: pd.Series, forecast_steps: int, freq: str,
                      p_values=[0, 1, 2, 3], d_values=[1], q_values=[0, 1, 2, 3]):
    # Apply log transformation to stabilize variance (assumes all values > 0)
    ts_log = np.log(ts)
    
    best_aic = np.inf
    best_order = None
    best_model = None

    total_combinations = len(p_values) * len(d_values) * len(q_values)
    current_combo = 0
    print(f"Starting grid search for ARIMA parameters (Total combinations: {total_combinations})...")

    for p in p_values:
        for d in d_values:
            for q in q_values:
                current_combo += 1
                print(f"Evaluating combination {current_combo}/{total_combinations}: order=({p},{d},{q})...")
                try:
                    model = ARIMA(ts_log, order=(p, d, q))
                    model_fit = model.fit()
                    if model_fit.aic < best_aic:
                        best_aic = model_fit.aic
                        best_order = (p, d, q)
                        best_model = model_fit
                except Exception as e:
                    continue
                    
    if best_model is None:
        raise ValueError("No ARIMA model could be fitted with the given parameter ranges.")
    
    # Forecast on log scale then revert back with exponential
    forecast_log = best_model.forecast(steps=forecast_steps)
    forecast = np.exp(forecast_log)
    
    last_timestamp = ts.index[-1]
    freq_offset = pd.tseries.frequencies.to_offset(freq)
    forecast_index = pd.date_range(start=last_timestamp + freq_offset, periods=forecast_steps, freq=freq)
    forecast_series = pd.Series(forecast, index=forecast_index)
    
    print(f"Grid search complete. Best ARIMA order: {best_order}")
    return forecast_series, best_model, best_order

# ---------------------------
# Enhanced LSTM Forecasting with StandardScaler and Increased Capacity
# ---------------------------
def create_dataset(dataset, look_back=1):
    X, y = [], []
    for i in range(len(dataset) - look_back):
        X.append(dataset[i:(i + look_back)])
        y.append(dataset[i + look_back])
    return np.array(X), np.array(y)

def forecast_lstm(train_series, forecast_steps, look_back=24, epochs=100, batch_size=32):
    scaler = StandardScaler()
    train_values = train_series.values.reshape(-1, 1)
    train_scaled = scaler.fit_transform(train_values)
    
    X, y = create_dataset(train_scaled, look_back)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    
    # Updated LSTM architecture with increased capacity
    model = Sequential([
        Input(shape=(look_back, 1)),
        LSTM(100, return_sequences=True),
        Dropout(0.3),
        LSTM(50),
        Dense(1)
    ])
    model.compile(loss='mean_squared_error', optimizer='adam')
    
    callbacks = [
        EarlyStopping(monitor='loss', patience=10, restore_best_weights=True),
        ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, min_lr=1e-6)
    ]
    
    print(f"Training LSTM model for {epochs} epochs with batch size {batch_size}...")
    model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=1, callbacks=callbacks)
    print("LSTM training complete.")
    
    last_sequence = train_scaled[-look_back:].reshape(1, look_back, 1)
    lstm_forecast_scaled = []
    for _ in range(forecast_steps):
        next_value = model.predict(last_sequence, verbose=0)
        lstm_forecast_scaled.append(next_value[0, 0])
        last_sequence = np.append(last_sequence[:, 1:, :], next_value.reshape(1, 1, 1), axis=1)
    
    lstm_forecast = scaler.inverse_transform(np.array(lstm_forecast_scaled).reshape(-1, 1)).flatten()
    
    last_timestamp = train_series.index[-1]
    inferred_freq = pd.infer_freq(train_series.index)
    if not inferred_freq:
        inferred_freq = "H"
    freq_offset = pd.tseries.frequencies.to_offset(inferred_freq)
    forecast_index = pd.date_range(start=last_timestamp + freq_offset, periods=forecast_steps, freq=inferred_freq)
    lstm_forecast_series = pd.Series(lstm_forecast, index=forecast_index)
    return lstm_forecast_series, model

# ---------------------------
# Key Visualization and CSV Output for Model Comparison
# ---------------------------
def compare_forecasts(ts, forecast_steps, freq, look_back, epochs, batch_size):
    train = ts.iloc[:-forecast_steps]
    test = ts.iloc[-forecast_steps:]
    
    # Try fitting ARIMA; if it fails, catch the error and log it
    try:
        arima_forecast, arima_model, best_order = grid_search_arima(train, forecast_steps, freq)
    except Exception as e:
        print(f"ARIMA model fitting error: {e}")
        arima_forecast = None
        best_order = "ARIMA model not fitted"
    
    lstm_forecast, _ = forecast_lstm(train, forecast_steps, look_back, epochs, batch_size)
    
    plt.figure(figsize=(12, 6))
    plt.plot(train.index, train, label="Training Data", color="blue")
    plt.plot(test.index, test, label="Actual Test Data", color="black", marker="o")
    
    # Plot ARIMA forecast only if available
    if arima_forecast is not None:
        plt.plot(arima_forecast.index, arima_forecast, label="ARIMA Forecast", color="red", linestyle="--", marker="x")
    else:
        print("Skipping ARIMA forecast plot as it is not available.")
    
    plt.plot(lstm_forecast.index, lstm_forecast, label="LSTM Forecast", color="green", linestyle="--", marker="^")
    plt.title(f"Forecast Comparison for {ts.name}")
    plt.xlabel("Timestamp")
    plt.ylabel("Value")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    
    comparison_plot_filename = os.path.join("data", f"comparison_forecast_{ts.name}.png")
    plt.savefig(comparison_plot_filename)
    plt.close()
    
    # Create CSV output, including ARIMA forecast if available
    data_dict = {
        "timestamp": test.index,
        "actual": test.values,
        "lstm_forecast": lstm_forecast.values
    }
    if arima_forecast is not None:
        data_dict["arima_forecast"] = arima_forecast.values
    else:
        data_dict["arima_forecast"] = [None] * len(test.index)
    
    comparison_df = pd.DataFrame(data_dict)
    comparison_csv_filename = os.path.join("data", f"comparison_forecast_{ts.name}.csv")
    comparison_df.to_csv(comparison_csv_filename, index=False)
    
    # Compute error metrics only for LSTM if ARIMA is not fitted
    if arima_forecast is not None:
        arima_mse = mean_squared_error(test, arima_forecast)
        arima_mae = mean_absolute_error(test, arima_forecast)
    else:
        arima_mse = None
        arima_mae = None
    lstm_mse = mean_squared_error(test, lstm_forecast)
    lstm_mae = mean_absolute_error(test, lstm_forecast)
    
    if arima_mse is not None and arima_mse < lstm_mse:
        better_model = "ARIMA"
        explanation = ("ARIMA performed better with lower MSE and MAE. The log transformation helped stabilize variance.")
    else:
        better_model = "LSTM"
        explanation = ("LSTM performed better with lower MSE and MAE, capturing non-linear patterns effectively.")
    
    result_text = (f"ARIMA MSE: {arima_mse if arima_mse is not None else 'N/A'}, ARIMA MAE: {arima_mae if arima_mae is not None else 'N/A'}\n"
                   f"LSTM MSE: {lstm_mse:.2f}, LSTM MAE: {lstm_mae:.2f}\n"
                   f"Better Model: {better_model}\n"
                   f"Explanation: {explanation}")
    print(result_text)
    return result_text

# ---------------------------
# Forecasting for Each Sensor Variable (Key Outputs)
# ---------------------------
def forecast_models_for_variable(ts, variable, forecast_steps, freq, look_back, epochs, batch_size):
    print(f"\nProcessing forecasts for sensor variable: {variable}")
    result_text = compare_forecasts(ts, forecast_steps, freq, look_back, epochs, batch_size)
    
    metrics_file = os.path.join("data", f"forecast_metrics_{variable}.txt")
    try:
        with open(metrics_file, "w") as f:
             f.write(result_text)
    except Exception as e:
        print(f"Error writing metrics file for {variable}: {e}")
    
    return result_text

# ---------------------------
# Main Function
# ---------------------------
def main(forecast_steps: int, look_back: int, epochs: int, batch_size: int):
    data_path = os.path.join("data", "sensor_data.csv")
    if not os.path.exists(data_path):
        print(f"Error: '{data_path}' not found. Please ensure sensor_data.csv exists in the 'data' folder.")
        return
    
    df = pd.read_csv(data_path)
    if "timestamp" not in df.columns:
        print("Error: 'timestamp' column not found in sensor_data.csv. Please include a timestamp column.")
        return
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df.set_index("timestamp", inplace=True)
    df.sort_index(inplace=True)
    
    freq = pd.infer_freq(df.index)
    if freq is None:
        print("Warning: Could not infer frequency. Defaulting to hourly frequency.")
        freq = "H"
    else:
        print(f"Inferred frequency: {freq}")
    
    sensor_vars = df.columns.tolist()
    print("Sensor variables found:", sensor_vars)
    
    for variable in sensor_vars:
        ts = df[variable]
        print(f"\nForecasting for sensor variable: {variable}")
        result_text = forecast_models_for_variable(ts, variable, forecast_steps, freq, look_back, epochs, batch_size)
        print(result_text)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Forecasting using Enhanced ARIMA and LSTM for sensor variables")
    parser.add_argument("--steps", type=int, default=15, help="Number of forecast steps (default: 15)")
    parser.add_argument("--look_back", type=int, default=24, help="Look-back period for LSTM (default: 24)")
    parser.add_argument("--epochs", type=int, default=100, help="Epochs for LSTM training (default: 100)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for LSTM training (default: 32)")
    args = parser.parse_args()
    main(args.steps, args.look_back, args.epochs, args.batch_size)