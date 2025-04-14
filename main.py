from publishers import mqtt_azure_publisher
from utils.logger import setup_logger

logger = setup_logger("Main")

def main():
    try:
        logger.info("Starting Azure IoT Hub MQTT Publisher...")
        mqtt_azure_publisher.run()
    except Exception as e:
        logger.error("Error in main: %s", str(e))

if __name__ == "__main__":
    main()