import os
import time
import json
import random
from dotenv import load_dotenv
import pandas as pd
from datetime import datetime
from azure.iot.device import IoTHubDeviceClient, Message

load_dotenv()

# Get the connection string from the environment variable, or define it directly here.
IOTHUB_DEVICE_CONNECTION_STRING = os.getenv("IOTHUB_DEVICE_CONNECTION_STRING", "Your_IoT_Hub_Device_Connection_String")

# File path to pre-computed energy distribution results (if available)
CSV_FILE_PATH = os.path.join("data", "energy_distribution_results.csv")

def simulate_energy_distribution(threshold=250):
    """
    Simulates an energy distribution decision.
    Generates a random forecast demand and determines the control action
    based on a threshold value.
    """
    forecast_demand = round(random.uniform(200, 300), 2)
    if forecast_demand > threshold:
        control_action = "High demand: Increase supply and curtail non-critical loads."
    else:
        control_action = "Normal demand: Standard distribution maintained."
    return forecast_demand, control_action

def load_forecast_data(csv_path):
    """
    Loads forecast data from a CSV file if available.
    Expects the CSV to have columns: 'timestamp', 'forecast_demand', 'control_action'.
    """
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"])
            return df
        except Exception as e:
            print(f"Error loading CSV file: {e}")
            return None
    else:
        return None

def push_message(client, payload):
    """
    Sends a JSON-formatted message payload to Azure IoT Hub.
    """
    message = Message(json.dumps(payload))
    print(f"Sending message: {payload}")
    client.send_message(message)

def main():
    # Initialize IoT Hub device client
    client = IoTHubDeviceClient.create_from_connection_string(IOTHUB_DEVICE_CONNECTION_STRING)
    client.connect()
    
    # Attempt to load forecast data from CSV file
    forecast_df = load_forecast_data(CSV_FILE_PATH)
    index = 0
    if forecast_df is not None and not forecast_df.empty:
        forecast_df = forecast_df.sort_values(by="timestamp")
        total_records = len(forecast_df)
        print(f"Loaded {total_records} records from CSV. Cycling through forecast data.")
    else:
        print("No valid forecast CSV found. Simulating energy distribution data.")
    
    try:
        while True:
            # If CSV data is available, use it; otherwise, simulate new data.
            if forecast_df is not None and not forecast_df.empty:
                if index >= total_records:
                    index = 0  # Reset to the beginning if we've reached the end
                row = forecast_df.iloc[index]
                index += 1
                # Use the current timestamp to represent a "real-time" message
                timestamp = datetime.utcnow().isoformat() + "Z"
                # Retrieve forecast values from the row
                forecast_demand = float(row["forecast_demand"])
                control_action = row["control_action"]
            else:
                # Simulate a new energy distribution decision
                timestamp = datetime.utcnow().isoformat() + "Z"
                forecast_demand, control_action = simulate_energy_distribution()
            
            # Construct the payload to push to Azure IoT Hub
            payload = {
                "timestamp": timestamp,
                "forecast_demand": forecast_demand,
                "control_action": control_action
            }
            
            # Send payload as a message
            push_message(client, payload)
            
            # Wait for 15 seconds before sending the next message
            time.sleep(15)
    except KeyboardInterrupt:
        print("Script interrupted by user. Stopping transmission.")
    finally:
        client.disconnect()

if __name__ == "__main__":
    main()