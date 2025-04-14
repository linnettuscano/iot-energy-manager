import paho.mqtt.client as mqtt
from utils.logger import setup_logger

logger = setup_logger("MQTTClient")

class MQTTClient:
    def __init__(self, host: str, port: int, client_id: str, username: str = None, password: str = None, keepalive: int = 60):
        self.host = host
        self.port = port
        self.client_id = client_id
        self.username = username
        self.password = password
        self.keepalive = keepalive

        self.client = mqtt.Client(client_id=self.client_id)
        if self.username and self.password:
            self.client.username_pw_set(self.username, self.password)
        # Set up callbacks for logging/debugging
        self.client.on_connect = self.on_connect
        self.client.on_disconnect = self.on_disconnect
        self.client.on_publish = self.on_publish
        self.client.on_log = self.on_log

    def on_log(self, client, userdata, level, buf):
        logger.debug("MQTT Log: %s", buf)

    def on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            logger.info("Connected successfully to MQTT broker at %s:%s", self.host, self.port)
        else:
            logger.error("Connection failed with return code: %s", rc)

    def on_disconnect(self, client, userdata, rc):
        if rc != 0:
            logger.warning("Unexpected disconnection from MQTT broker.")
        else:
            logger.info("Disconnected from MQTT broker.")

    def on_publish(self, client, userdata, mid):
        logger.info("Message %s published successfully.", mid)

    def connect(self):
        try:
            self.client.connect(self.host, self.port, self.keepalive)
            self.client.loop_start()
        except Exception as e:
            logger.error("Error connecting to MQTT broker: %s", str(e))

    def disconnect(self):
        try:
            self.client.loop_stop()
            self.client.disconnect()
        except Exception as e:
            logger.error("Error disconnecting from MQTT broker: %s", str(e))

    def publish(self, topic: str, payload: str, qos: int = 1):
        try:
            result = self.client.publish(topic, payload, qos=qos)
            result.wait_for_publish(timeout=10)
            if result.rc != mqtt.MQTT_ERR_SUCCESS:
                logger.error("Failed to publish message. Error code: %s", result.rc)
            else:
                logger.info("Published payload to topic '%s'.", topic)
        except Exception as e:
            logger.error("Error publishing message: %s", str(e))