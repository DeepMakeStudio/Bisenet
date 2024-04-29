from fastapi import FastAPI, HTTPException
from argparse import Namespace
from PIL import Image
import os
import base64
import io
from io import BytesIO
import numpy as np
import threading
import time
import psutil
import sys
import pickle

import torch
import torchvision.transforms as transforms
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from plugin import Plugin, fetch_image, store_image
from .config import plugin, config, endpoints

sys.path.append(os.path.join(os.path.dirname(__file__), "Bisenet"))
from .Bisenet.model import BiSeNet

app = FastAPI()

@app.get("/set_model")
def set_model():
    global net 
    net = BiSeNet(n_classes=19)
    save_pth = config["model_name"]
    net.load_state_dict(torch.load(save_pth, map_location=torch.device('cpu')))
    net.eval()
    if torch.cuda.is_available():
        net.cuda()
    elif torch.backends.mps.is_available():
        net.to("mps")

    return {"status": "Success", "detail": f"Model set successfully to config['model_name']"}

@app.on_event("startup")
async def startup_event():
    print("Starting up")
    # A slight delay to ensure the app has started up.
    try:
        set_model()
        print("Successfully started up")
        bisenet_plugin.notify_main_system_of_startup("True")
    except:
        bisenet_plugin.notify_main_system_of_startup("False")

@app.get("/get_info/")
def plugin_info():
    return {"plugin": plugin,"config": config, "endpoints": endpoints}

@app.get("/get_config/")
def get_config():
    return config

@app.put("/set_config/")
def set_config(update: dict):
    global config
    config.update(update) # TODO: Validate config dict are all valid keys
    if "model_name" in update:
        response = set_model()
        if response["status"] == "Failed":
            return response
    return config 

def self_terminate():
    time.sleep(3)
    parent = psutil.Process(psutil.Process(os.getpid()).ppid())
    print(f"Killing parent process {parent.pid}")
    # os.kill(parent.pid, 1)
    # parent.kill()

@app.get("/shutdown/") #Shutdown the plugin
def shutdown():
    # sys.exit()
    threading.Thread(target=self_terminate, daemon=True).start()
    return {"Success": True}

@app.get("/execute/{img_id}")
async def generate_output(img_id: str, skin: bool = True, l_brow: bool = False, r_brow: bool = False, l_eye: bool = False, 
                          r_eye: bool = False, eye_g: bool = False, l_ear: bool = False, r_ear: bool = False, 
                          ear_r: bool = False, nose: bool = False, mouth: bool = False, u_lip: bool = False, 
                          l_lip: bool = False, neck: bool = False, neck_l: bool = False, cloth: bool = False, 
                          hair: bool = False, hat: bool = False):
    img_data = fetch_image(img_id)
    image = Image.open(io.BytesIO(img_data))
    img_size = image.size

    image = image.resize((512, 512), Image.BILINEAR)

    #remove alpha if it exists
    if image.mode == 'RGBA':
        image = image.convert('RGB')

    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    img = to_tensor(image)
    img = torch.unsqueeze(img, 0)

    if torch.cuda.is_available():
        img = img.cuda()
    elif torch.backends.mps.is_available():
        img = img.to("mps")

    atts = ['skin', 'l_brow', 'r_brow', 'l_eye', 'r_eye', 'eye_g', 'l_ear', 'r_ear', 'ear_r',
            'nose', 'mouth', 'u_lip', 'l_lip', 'neck', 'neck_l', 'cloth', 'hair', 'hat']
    selected_atts = []
    for attr in atts:
        if eval(attr):
            selected_atts.append(attr)
    print(selected_atts)
    seg_maps = {att: None for att in atts}

    with torch.no_grad():
        out = net(img)[0]
        parsing = out.squeeze(0).cpu().numpy().argmax(0)
        for i, att in enumerate(atts, 1):
            seg_map_att = np.where(parsing == i, 1, 0)
            seg_maps[att] = seg_map_att

    output = BytesIO()
    #Create imabe from mask using all the selected attributes

    outimage = np.zeros((512, 512), dtype=np.int32)

    for att in selected_atts:
        outimage += seg_maps[att]

    output_image = Image.fromarray((outimage*255).astype(np.uint8))
    output_image.resize(img_size, Image.BILINEAR).save(output, format="PNG")

    img_id = store_image(output.getvalue())

    return {"status": "Success", "output_mask": img_id}

args = {"plugin": plugin, "config": config, "endpoints": endpoints}
class BisenetPlugin(Plugin):
    """
    Prediction inference.
    """
    def __init__(self, arguments: "Namespace") -> None:
        super().__init__(arguments)
        self.plugin_name = "Bisenet"

bisenet_plugin = BisenetPlugin(Namespace(**args))
set_model()