#!/usr/bin/env python3
"""
Forecasting Sensor Data using Enhanced ARIMA and LSTM

This script reads historical sensor data from 'data/sensor_data.csv',
performs forecasting using both an enhanced ARIMA model (with grid search parameter tuning)
and an enhanced LSTM model (stacked with dropout) on each sensor variable.
The data is split into training and test sets, and forecasts are compared using error metrics.
Detailed visualizations (forecast comparisons, residual analyses, error histograms) are saved as CSV and PNG files.

Usage:
    python forecast_arima.py --steps 15 --look_back 24 --epochs 50 --batch_size 32

Note: --steps specifies the forecast horizon (number of future points).
"""

import os
import argparse
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input


# ---------------------------
# Enhanced ARIMA with Grid Search
# ---------------------------

def grid_search_arima(ts: pd.Series, forecast_steps: int, freq: str,
                      p_values=[0, 1, 2], d_values=[1], q_values=[0, 1, 2]):
    best_aic = np.inf
    best_order = None
    best_model = None
    # Try different combinations and select the one with the lowest AIC
    for p in p_values:
        for d in d_values:
            for q in q_values:
                try:
                    model = ARIMA(ts, order=(p, d, q))
                    model_fit = model.fit()
                    if model_fit.aic < best_aic:
                        best_aic = model_fit.aic
                        best_order = (p, d, q)
                        best_model = model_fit
                except Exception as e:
                    # Skip models that don't converge
                    continue
    if best_model is None:
        raise ValueError("No ARIMA model could be fitted with the given parameter ranges.")
    
    forecast = best_model.forecast(steps=forecast_steps)
    last_timestamp = ts.index[-1]
    freq_offset = pd.tseries.frequencies.to_offset(freq)
    forecast_index = pd.date_range(start=last_timestamp + freq_offset, periods=forecast_steps, freq=freq)
    forecast_series = pd.Series(forecast, index=forecast_index)
    return forecast_series, best_model, best_order


def plot_arima_residuals(model_fit, variable):
    # Plot the residuals to check if they are white noise
    residuals = model_fit.resid
    plt.figure(figsize=(10, 4))
    plt.plot(residuals, label="Residuals", marker="o", linestyle="-")
    plt.title(f"ARIMA Residuals for {variable}")
    plt.xlabel("Timestamp")
    plt.ylabel("Residuals")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    res_plot_filename = os.path.join("data", f"residuals_{variable}.png")
    plt.savefig(res_plot_filename)
    plt.close()
    
    # Also plot histogram of residuals
    plt.figure(figsize=(8, 4))
    plt.hist(residuals, bins=20, color="skyblue", edgecolor="black")
    plt.title(f"Residual Histogram for {variable}")
    plt.xlabel("Residual")
    plt.ylabel("Frequency")
    plt.tight_layout()
    res_hist_filename = os.path.join("data", f"residuals_hist_{variable}.png")
    plt.savefig(res_hist_filename)
    plt.close()


# ---------------------------
# Enhanced LSTM Forecasting Function with Dropout and Stacked Layers
# ---------------------------

def create_dataset(dataset, look_back=1):
    X, y = [], []
    for i in range(len(dataset) - look_back):
        X.append(dataset[i:(i + look_back)])
        y.append(dataset[i + look_back])
    return np.array(X), np.array(y)


def forecast_lstm(train_series, forecast_steps, look_back=24, epochs=50, batch_size=32):
    # Scale the training data
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_values = train_series.values.reshape(-1, 1)
    train_scaled = scaler.fit_transform(train_values)
    
    # Create dataset for LSTM using the look-back window
    X, y = create_dataset(train_scaled, look_back)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    
    # Build a stacked LSTM model with dropout using an Input layer
    model = Sequential([
        Input(shape=(look_back, 1)),
        LSTM(50, return_sequences=True),
        Dropout(0.2),
        LSTM(25),
        Dense(1)
    ])
    model.compile(loss='mean_squared_error', optimizer='adam')
    
    # Train the model
    model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=0)
    
    # Iterative forecasting using the last look_back values
    last_sequence = train_scaled[-look_back:].reshape(1, look_back, 1)
    lstm_forecast_scaled = []
    for _ in range(forecast_steps):
        next_value = model.predict(last_sequence, verbose=0)
        lstm_forecast_scaled.append(next_value[0, 0])
        # Ensure the prediction has the right shape before appending
        last_sequence = np.append(last_sequence[:, 1:, :], next_value.reshape(1, 1, 1), axis=1)
    
    # Inverse scale the forecasted values
    lstm_forecast = scaler.inverse_transform(np.array(lstm_forecast_scaled).reshape(-1, 1)).flatten()
    
    # Generate forecast index based on the training series frequency
    last_timestamp = train_series.index[-1]
    inferred_freq = pd.infer_freq(train_series.index)
    if not inferred_freq:
        inferred_freq = "H"
    freq_offset = pd.tseries.frequencies.to_offset(inferred_freq)
    forecast_index = pd.date_range(start=last_timestamp + freq_offset, periods=forecast_steps, freq=inferred_freq)
    lstm_forecast_series = pd.Series(lstm_forecast, index=forecast_index)
    return lstm_forecast_series, model


# ---------------------------
# Model Comparison and Diagnostic Plots
# ---------------------------

def compare_forecasts(ts, forecast_steps, freq, look_back, epochs, batch_size):
    # Split the time series into training and test sets
    train = ts.iloc[:-forecast_steps]
    test = ts.iloc[-forecast_steps:]
    
    # ARIMA forecast with parameter tuning
    arima_forecast, arima_model, best_order = grid_search_arima(train, forecast_steps, freq)
    print(f"Best ARIMA order: {best_order}")
    
    # Plot ARIMA residuals and histogram for diagnostics
    plot_arima_residuals(arima_model, ts.name)
    
    # LSTM forecast using enhanced model
    lstm_forecast, _ = forecast_lstm(train, forecast_steps, look_back, epochs, batch_size)
    
    # Compute error metrics for both models
    arima_mse = mean_squared_error(test, arima_forecast)
    arima_mae = mean_absolute_error(test, arima_forecast)
    lstm_mse = mean_squared_error(test, lstm_forecast)
    lstm_mae = mean_absolute_error(test, lstm_forecast)
    
    # Plot forecast comparisons
    plt.figure(figsize=(12, 6))
    plt.plot(train.index, train, label="Training Data", color="blue")
    plt.plot(test.index, test, label="Actual Test Data", color="black", marker="o")
    plt.plot(arima_forecast.index, arima_forecast, label="ARIMA Forecast", color="red", linestyle="--", marker="x")
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
    
    # Save forecast comparison data as CSV
    comparison_df = pd.DataFrame({
        "timestamp": test.index,
        "actual": test.values,
        "arima_forecast": arima_forecast.values,
        "lstm_forecast": lstm_forecast.values
    })
    comparison_csv_filename = os.path.join("data", f"comparison_forecast_{ts.name}.csv")
    comparison_df.to_csv(comparison_csv_filename, index=False)
    
    # Plot error distribution histograms
    arima_errors = test.values - arima_forecast.values
    lstm_errors = test.values - lstm_forecast.values
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.hist(arima_errors, bins=20, color="red", edgecolor="black")
    plt.title("ARIMA Forecast Errors")
    plt.xlabel("Error")
    plt.ylabel("Frequency")
    plt.subplot(1, 2, 2)
    plt.hist(lstm_errors, bins=20, color="green", edgecolor="black")
    plt.title("LSTM Forecast Errors")
    plt.xlabel("Error")
    plt.ylabel("Frequency")
    plt.tight_layout()
    error_hist_filename = os.path.join("data", f"forecast_error_hist_{ts.name}.png")
    plt.savefig(error_hist_filename)
    plt.close()
    
    # Determine which model performs better (based on MSE)
    if arima_mse < lstm_mse:
        better_model = "ARIMA"
        explanation = ("ARIMA performed better due to lower MSE and MAE. This suggests that the linear model is capturing "
                       "the underlying trend effectively in this dataset, perhaps because the data is less complex or less noisy.")
    else:
        better_model = "LSTM"
        explanation = ("LSTM performed better with lower MSE and MAE, indicating that it captured complex non-linear patterns "
                       "and longer-term dependencies in the sensor data more successfully than ARIMA.")
    
    result_text = (f"ARIMA MSE: {arima_mse:.2f}, ARIMA MAE: {arima_mae:.2f}\n"
                   f"LSTM MSE: {lstm_mse:.2f}, LSTM MAE: {lstm_mae:.2f}\n"
                   f"Better Model: {better_model}\n"
                   f"Explanation: {explanation}")
    print(result_text)
    return result_text


# ---------------------------
# Forecasting for Each Variable with Ensemble Diagnostics
# ---------------------------

def forecast_models_for_variable(ts, variable, forecast_steps, freq, look_back, epochs, batch_size):
    print(f"\nProcessing forecasts for variable: {variable}")
    
    # Compare forecasts on a held-out test set and generate diagnostics
    result_text = compare_forecasts(ts, forecast_steps, freq, look_back, epochs, batch_size)
    
    # Store the comparison text and metrics into a text file for this sensor variable
    metrics_file = os.path.join("data", f"forecast_metrics_{variable}.txt")
    with open(metrics_file, "w") as f:
         f.write(result_text)
    
    # Additionally, train on the full dataset to generate future forecasts using both models
    full_arima_forecast, arima_model, best_order = grid_search_arima(ts, forecast_steps, freq)
    full_lstm_forecast, _ = forecast_lstm(ts, forecast_steps, look_back, epochs, batch_size)
    
    # Save ARIMA forecast visualization and data
    plt.figure(figsize=(12, 6))
    plt.plot(ts, label="Historical Data", marker="o")
    plt.plot(full_arima_forecast.index, full_arima_forecast, label="ARIMA Forecast", marker="x", linestyle="--", color="red")
    plt.title(f"ARIMA Forecast for {variable} (Full Data)")
    plt.xlabel("Timestamp")
    plt.ylabel(variable)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    arima_plot_filename = os.path.join("data", f"forecast_{variable}_arima.png")
    plt.savefig(arima_plot_filename)
    plt.close()
    
    arima_forecast_df = full_arima_forecast.reset_index()
    arima_forecast_df.columns = ["timestamp", f"forecast_{variable}_arima"]
    arima_csv_filename = os.path.join("data", f"forecast_{variable}_arima.csv")
    arima_forecast_df.to_csv(arima_csv_filename, index=False)
    
    # Save LSTM forecast visualization and data
    plt.figure(figsize=(12, 6))
    plt.plot(ts, label="Historical Data", marker="o")
    plt.plot(full_lstm_forecast.index, full_lstm_forecast, label="LSTM Forecast", marker="^", linestyle="--", color="green")
    plt.title(f"LSTM Forecast for {variable} (Full Data)")
    plt.xlabel("Timestamp")
    plt.ylabel(variable)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    lstm_plot_filename = os.path.join("data", f"forecast_{variable}_lstm.png")
    plt.savefig(lstm_plot_filename)
    plt.close()
    
    lstm_forecast_df = full_lstm_forecast.reset_index()
    lstm_forecast_df.columns = ["timestamp", f"forecast_{variable}_lstm"]
    lstm_csv_filename = os.path.join("data", f"forecast_{variable}_lstm.csv")
    lstm_forecast_df.to_csv(lstm_csv_filename, index=False)
    
    return result_text


# ---------------------------
# Main Function
# ---------------------------

def main(forecast_steps: int, look_back: int, epochs: int, batch_size: int):
    data_path = os.path.join("data", "sensor_data.csv")
    if not os.path.exists(data_path):
        print(f"Error: '{data_path}' not found. Please ensure sensor_data.csv exists in the 'data' folder.")
        return
    
    # Load and preprocess sensor data
    df = pd.read_csv(data_path)
    if "timestamp" not in df.columns:
        print("Error: 'timestamp' column not found in sensor_data.csv. Please include a timestamp column.")
        return
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df.set_index("timestamp", inplace=True)
    df.sort_index(inplace=True)
    
    # Infer time series frequency
    freq = pd.infer_freq(df.index)
    if freq is None:
        print("Warning: Could not infer frequency. Defaulting to hourly frequency.")
        freq = "H"
    else:
        print(f"Inferred frequency: {freq}")
    
    # Identify sensor variables (all remaining columns)
    sensor_vars = df.columns.tolist()
    print("Sensor variables found:", sensor_vars)
    
    # Apply forecasting and diagnostics to each sensor variable
    for variable in sensor_vars:
        ts = df[variable]
        print(f"\nForecasting for sensor variable: {variable}")
        compare_text = forecast_models_for_variable(ts, variable, forecast_steps, freq, look_back, epochs, batch_size)
        print(compare_text)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Forecasting using Enhanced ARIMA and LSTM for all sensor variables")
    parser.add_argument("--steps", type=int, default=15, help="Number of forecast steps (default: 15)")
    parser.add_argument("--look_back", type=int, default=24, help="Look-back period for LSTM (default: 24)")
    parser.add_argument("--epochs", type=int, default=50, help="Epochs for LSTM training (default: 50)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for LSTM training (default: 32)")
    args = parser.parse_args()
    main(args.steps, args.look_back, args.epochs, args.batch_size)