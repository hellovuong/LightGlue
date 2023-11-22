from lightglue import LightGlue
from pathlib import Path
from lightglue.utils import load_image, rbd
from lightglue import viz2d
import torch

import paho.mqtt.client as mqtt

import json

import mqtt_extractor


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
    client.subscribe(mqtt_extractor.FEATURE_TOPIC)

def on_message(client, userdata, msg):
    print("Received features. Start to decode msg...")
    # Get the JSON data from the received message
    json_data = msg.payload.decode("utf-8")
    # Parse the JSON data and convert it back to feats
    received_feats_list = json.loads(json_data, object_hook=convert_to_tensors)
    print("Decoded msg. Start to matched")
    matches01 = matcher({"image0": received_feats_list[0], "image1": received_feats_list[1]})

    print("Decoded msg. Visualizing result")
    feats0, feats1, matches01 = [
        rbd(x) for x in [received_feats_list[0], received_feats_list[1], matches01]
    ]  # remove batch dimension
    
    kpts0, kpts1, matches = feats0["keypoints"], feats1["keypoints"], matches01["matches"]
    m_kpts0, m_kpts1 = kpts0[matches[..., 0]], kpts1[matches[..., 1]]
    

    images = Path("assets")
    image0 = load_image(images / "sacre_coeur1.jpg")
    image1 = load_image(images / "sacre_coeur2.jpg")
    axes = viz2d.plot_images([image0, image1])
    viz2d.plot_matches(m_kpts0, m_kpts1, color="lime", lw=0.2)
    viz2d.add_text(0, f'Stop after {matches01["stop"]} layers')
    viz2d.save_plot("./test.png")
    print("Done matching")


torch.set_grad_enabled(False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 'mps', 'cpu'
matcher = LightGlue(features="superpoint").eval().to(device)

client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message

client.connect(mqtt_extractor.MQTT_HOST, 1883, 60)

client.loop_start()
try:
    while True:
        pass
except KeyboardInterrupt:
    client.disconnect()
    client.loop_stop()
