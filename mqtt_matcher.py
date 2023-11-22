from lightglue import LightGlue
import torch

import paho.mqtt.client as mqtt

import json

from mqtt_extractor import FEATURE_TOPIC, MQTT_PORT, MQTT_HOST

MATCHED_TOPIC = "matches"


# Convert JSON data back to feats
def convert_to_tensors(obj):
    if isinstance(obj, list):
        return [convert_to_tensors(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: convert_to_tensors(value) for key, value in obj.items()}
    elif isinstance(obj, (int, float)):
        return torch.tensor(obj)
    else:
        return obj


def on_connect(client, userdata, flags, rc):
    print("Connected with result code "+str(rc))
    client.subscribe(FEATURE_TOPIC)


def on_message(client, userdata, msg):
    print("Received features. Start to decode msg...")
    # Get the JSON data from the received message
    json_data = msg.payload.decode("utf-8")
    # Parse the JSON data and convert it back to feats
    received_feats_list = json.loads(json_data)
    print("Decoded msg. Start to matched")

    feats0 = {
        "keypoints":   torch.tensor(received_feats_list[0]["keypoints"]).to(device),
        "descriptors": torch.tensor(received_feats_list[0]["descriptors"]).to(device),
        "image_size":  torch.tensor(received_feats_list[0]["image_size"]).to(device),
    }
    feats1 = {
        "keypoints":   torch.tensor(received_feats_list[1]["keypoints"]).to(device),
        "descriptors": torch.tensor(received_feats_list[1]["descriptors"]).to(device),
        "image_size":  torch.tensor(received_feats_list[1]["image_size"]).to(device),
    }
    matches01 = matcher({"image0": feats0, "image1": feats1})

    print("Matched. Start to publish")
    json_data = json.dumps(matches01["matches"][0].tolist(), indent=2)
    # Publish the JSON data to MQTT
    ret, _ = client.publish(MATCHED_TOPIC, json_data, qos=1)
    print("Done matching and publishing")

torch.set_grad_enabled(False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 'mps', 'cpu'
matcher = LightGlue(features="superpoint").eval().to(device)

client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message

client.connect(MQTT_HOST, MQTT_PORT, 60)

client.loop_start()
try:
    while True:
        pass
except KeyboardInterrupt:
    client.disconnect()
    client.loop_stop()
