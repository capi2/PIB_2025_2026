import logging
import time
import paho.mqtt.client as mqtt
from datetime import datetime

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('op_log.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ThingSpeak MQTT Configuration
CHANNEL_ID = "2784071"
WRITE_API_KEY = "Q205G8B1XJO4CDG5"
MQTT_HOST = "mqtt3.thingspeak.com"
MQTT_USERNAME = "GA8aFRgdDiQsKCArKBgBOC8"
MQTT_PASSWORD = "EkxDZ+q0YxbxmlRgGzSmTyKZ"
CLIENT_ID = "GA8aFRgdDiQsKCArKBgBOC8"

# Build payload for ThingSpeak
def build_payload(temp, sensor_value):
    return f"field1={temp}&field2={sensor_value}"

# Send data to ThingSpeak using MQTT
def envia_thingspeak(payload):
    topic = f"channels/{CHANNEL_ID}/publish"

    try:
        # Configure MQTT client for WebSocket transport
        mqtt_client = mqtt.Client(client_id=CLIENT_ID, transport="websockets")

        # Authenticate with MQTT broker
        mqtt_client.username_pw_set(username=MQTT_USERNAME, password=MQTT_PASSWORD)

        # Connect to ThingSpeak MQTT broker
        mqtt_client.connect(host=MQTT_HOST, port=80, keepalive=60)

        # Enable logger for MQTT client
        mqtt_client.enable_logger(logger)

        # Publish payload to the channel
        result, _ = mqtt_client.publish(topic, payload, qos=1)

        if result == mqtt.MQTT_ERR_SUCCESS:
            mqtt_client.loop(timeout=5.0)
            logger.info(f"Data sent successfully: {payload}")
            return True
        else:
            logger.error(f"Failed to send data. Error code: {result}")
            return False

    except Exception as e:
        logger.error(f"MQTT Error: {e}")
        return False
    finally:
        if 'mqtt_client' in locals():
            mqtt_client.disconnect()

# Read dataset and send valid data
def process_dataset(file_path):
    logger.info("Starting dataset processing")
    
    try:
        with open("readings.dat", 'r') as file:
            for line in file:
                # Split and clean the line
                data = line.strip().split(',')

                if len(data) < 5 or not data[1] or not data[-1]:
                    logger.warning(f"Invalid or incomplete data: {line.strip()}")
                    continue

                try:
                    # Extract water temperature and sensor value
                    water_temp = float(data[1])
                    sensor_value = int(data[-1])

                    # Build payload and send to ThingSpeak
                    payload = build_payload(water_temp, sensor_value)
                    success = envia_thingspeak(payload)

                    if not success:
                        logger.warning(f"Failed to send data: {payload}")

                except ValueError as e:
                    logger.warning(f"Data conversion error in line: {line.strip()} - {e}")

                time.sleep(5)  # Avoid exceeding rate limit

    except FileNotFoundError:
        logger.error(f"Dataset file not found: {file_path}")
    except Exception as e:
        logger.error(f"Error processing dataset: {e}")

# Main function
def main():
    dataset_file = "dataset.txt"  # Replace with your dataset file name
    process_dataset(dataset_file)

if __name__ == "__main__":
    main()
