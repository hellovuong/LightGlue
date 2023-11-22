from pathlib import Path

from lightglue import SuperPoint
from lightglue.utils import load_image, numpy_image_to_torch, rbd
from lightglue import viz2d
import torch

import paho.mqtt.client as mqtt

import json 

MQTT_HOST = "localhost"
MQTT_PORT = 1883
FEATURE_TOPIC = "extracted_pair_image"

def on_publish(client, userdata, result):
    print("mqqt_extractor: Data published!")

# Convert tensor values in feats to lists for proper JSON serialization
def convert_to_lists(obj):
    if isinstance(obj, torch.Tensor):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_lists(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_lists(item) for item in obj]
    else:
        return obj

def on_connect(client, userdata, flags, rc):
    print(f"Connected to MQTT broker with result code {rc}")


if __name__ == "__main__":
    client = mqtt.Client()
    client.on_connect = on_connect
    client.connect(MQTT_HOST, MQTT_PORT, 60)

    torch.set_grad_enabled(False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 'mps', 'cpu'
    extractor = SuperPoint(max_num_keypoints=1024).eval().to(device)  # load the extractor
    
    images = Path("assets")
    image0 = load_image(images / "sacre_coeur1.jpg")
    image1 = load_image(images / "sacre_coeur2.jpg")
    
    feats0 = extractor.extract(image0.to(device))
    feats1 = extractor.extract(image1.to(device))
    # Convert the feats dictionaries to a list
    feats_list = [feats0, feats1]
    
    # Convert the list of feats dictionaries to JSON
    json_data = json.dumps(convert_to_lists(feats_list), indent=2)
    # Publish the JSON data to MQTT
    client.on_publish = on_publish
    ret, _ = client.publish(FEATURE_TOPIC, json_data, qos=1)

    client.loop_start()
    try:
        while True:
            pass
    except KeyboardInterrupt:
        client.disconnect()
        client.loop_stop()
