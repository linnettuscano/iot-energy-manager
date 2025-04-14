import os
import pandas as pd
from datetime import datetime

def load_forecast(sensor_variable, model_type="lstm"):
    """
    Loads the forecast CSV for the specified sensor variable and model type.
    """
    forecast_filename = f"forecast_{sensor_variable}_{model_type}.csv"
    forecast_path = os.path.join("data", forecast_filename)
    if os.path.exists(forecast_path):
        df = pd.read_csv(forecast_path)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df.set_index("timestamp", inplace=True)
        return df
    else:
        print(f"Forecast file '{forecast_filename}' not found in 'data' directory.")
        return None

def simulate_energy_distribution(forecast_df, threshold=250):
    """
    Simulates energy distribution decisions based on forecasted demand.
    
    If the forecasted demand exceeds the threshold, non-critical loads are reduced;
    otherwise, standard energy distribution is maintained.
    
    Args:
      forecast_df (DataFrame): The forecast data (indexed by timestamp) with demand values.
      threshold (float): Threshold value to trigger energy optimization actions.
    
    Returns:
      A list of tuples containing (timestamp, forecasted demand, control action).
    """
    actions = []
    # Assume the forecasted sensor reading is in the first column
    forecast_col = forecast_df.columns[0]
    for timestamp, row in forecast_df.iterrows():
        demand = row[forecast_col]
        if demand > threshold:
            action = "High demand: Increase supply and curtail non-critical loads."
        else:
            action = "Normal demand: Standard distribution maintained."
        actions.append((timestamp, demand, action))
    return actions

def store_results(actions, output_csv="energy_distribution_results.csv", output_excel="energy_distribution_results.xlsx"):
    """
    Stores the energy distribution decisions to both a CSV and an Excel file.
    
    Args:
      actions (list): List of tuples (timestamp, demand, action).
      output_csv (str): Filename for the CSV output.
      output_excel (str): Filename for the Excel output.
    """
    df = pd.DataFrame(actions, columns=["timestamp", "forecast_demand", "control_action"])
    # Save to CSV
    df.to_csv(output_csv, index=False)
    # Save to Excel
    df.to_excel(output_excel, index=False)
    print(f"Results have been saved to {output_csv} and {output_excel}")

def main():
    # Example sensor variable for energy usage (e.g., smart_plug represents energy consumption data)
    sensor_variable = "smart_plug"
    # Load the forecast data from the LSTM model (could also try "arima" for comparison)
    forecast_df = load_forecast(sensor_variable, model_type="lstm")
    if forecast_df is None:
        return

    # Simulate energy distribution decisions based on forecasted demand
    threshold = 250  # Example threshold value; adjust based on typical demand patterns
    actions = simulate_energy_distribution(forecast_df, threshold=threshold)
    
    # Output simulated decisions (this is the "dashboard" in our simulation)
    print("Simulated Energy Distribution Decisions:")
    for timestamp, demand, action in actions:
        print(f"{timestamp}: Forecast Demand = {demand:.2f} -> {action}")
    
    # Store results to CSV and Excel files in the data folder (ensure the folder exists)
    os.makedirs("data", exist_ok=True)
    output_csv_path = os.path.join("data", "energy_distribution_results.csv")
    output_excel_path = os.path.join("data", "energy_distribution_results.xlsx")
    store_results(actions, output_csv=output_csv_path, output_excel=output_excel_path)

if __name__ == "__main__":
    main()