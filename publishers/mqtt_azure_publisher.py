import os
import ssl
import time
import json
from dotenv import load_dotenv
from sensors.smart_plug import read_smart_plug
from sensors.temperature_sensor import read_temperature_humidity
from sensors.occupancy_sensor import read_occupancy
from utils.mqtt_client import MQTTClient
from utils.sas_token import get_sas_token
from utils.logger import setup_logger

# Load environment variables
load_dotenv()
logger = setup_logger("AzurePublisher")

# Read Azure configuration from env
AZURE_IOTHUB_HOSTNAME = os.getenv("AZURE_IOTHUB_HOSTNAME")
AZURE_DEVICE_ID = os.getenv("AZURE_DEVICE_ID")
AZURE_PRIMARY_KEY = os.getenv("AZURE_PRIMARY_KEY")
AZURE_MQTT_PORT = int(os.getenv("AZURE_MQTT_PORT", 8883))

# Generate SAS token using the utility. The SAS token is valid for the specified period (e.g., 3600 seconds)
sas_token = get_sas_token(AZURE_IOTHUB_HOSTNAME, AZURE_DEVICE_ID, AZURE_PRIMARY_KEY, validity_period=3600)

# Construct the MQTT username and topic for Azure IoT Hub
AZURE_USERNAME = f"{AZURE_IOTHUB_HOSTNAME}/{AZURE_DEVICE_ID}/?api-version=2018-06-30"
AZURE_CLIENT_ID = AZURE_DEVICE_ID
AZURE_TOPIC = f"devices/{AZURE_DEVICE_ID}/messages/events/"

PUBLISH_INTERVAL = int(os.getenv("PUBLISH_INTERVAL", 5))

def run():
    client = MQTTClient(
        host=AZURE_IOTHUB_HOSTNAME,
        port=AZURE_MQTT_PORT,
        client_id=AZURE_CLIENT_ID,
        username=AZURE_USERNAME,
        password=sas_token
    )
    # For secure connection with TLS, uncomment the following line and configure as needed:
    client.client.tls_set(
        cert_reqs=ssl.CERT_REQUIRED,
        tls_version=ssl.PROTOCOL_TLSv1_2
    )
    
    client.connect()
    try:
        while True:
            # Read sensor data
            smart_plug_value = read_smart_plug()
            temp_humidity = read_temperature_humidity()
            occupancy = read_occupancy()

            # Create payload as JSON
            payload = {
                "smart_plug": smart_plug_value,
                "temperature": temp_humidity["temperature"],
                "humidity": temp_humidity["humidity"],
                "occupancy": occupancy
            }
            payload_str = json.dumps(payload)
            logger.info("Publishing to Azure IoT Hub: %s", payload_str)
            client.publish(AZURE_TOPIC, payload_str)
            time.sleep(PUBLISH_INTERVAL)
    except KeyboardInterrupt:
        logger.info("Azure publisher interrupted by user.")
    except Exception as e:
        logger.error("Error in Azure Publisher: %s", str(e))
    finally:
        client.disconnect()

if __name__ == "__main__":
    run()