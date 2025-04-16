#!/usr/bin/env python3
"""
Simulate Sensor Data for Smart Energy Management System

This script simulates sensor readings (smart plug, temperature, humidity, occupancy)
from March 1st to April 10th at an hourly frequency.
The generated data is saved as data/sensor_data.csv.
"""

import pandas as pd
import numpy as np

def simulate_data(start_date: str, end_date: str, freq: str = 'H') -> pd.DataFrame:
    # Create a date range with the specified frequency
    timestamps = pd.date_range(start=start_date, end=end_date, freq=freq)
    n = len(timestamps)
    
    # For reproducibility
    np.random.seed(42)
    
    # Simulate sensor data:
    # - smart_plug: random power consumption between 50 and 300 units
    # - temperature: random temperature between 18 and 30 °C
    # - humidity: random humidity between 30 and 70%
    # - occupancy: random True/False
    smart_plug = np.random.uniform(low=50.0, high=300.0, size=n).round(2)
    temperature = np.random.uniform(low=18, high=30, size=n).round(2)
    humidity = np.random.uniform(low=30, high=70, size=n).round(2)
    occupancy = np.random.choice([True, False], size=n)
    
    # Compile the data into a DataFrame
    df = pd.DataFrame({
        "timestamp": timestamps,
        "smart_plug": smart_plug,
        "temperature": temperature,
        "humidity": humidity,
        "occupancy": occupancy
    })
    return df

def main():
    # Define the simulation period
    start_date = "2023-01-01"
    end_date = "2025-04-14"
    
    # Simulate the data
    df = simulate_data(start_date, end_date, freq='H')
    
    # Ensure the "data" folder exists
    output_folder = "data"
    import os
    os.makedirs(output_folder, exist_ok=True)
    
    # Save to CSV (this file will later be used for forecasting)
    output_file = os.path.join(output_folder, "sensor_data.csv")
    df.to_csv(output_file, index=False)
    print(f"Simulated sensor data saved to {output_file}")

if __name__ == "__main__":
    main()