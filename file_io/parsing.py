#parsing.py
import cv2
import torch
import numpy as np

def preprocess(image):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (512, 512))
    img = img.astype(np.float32) / 255.0
    return torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0)

def postprocess(out):
    return out.squeeze(0).cpu().numpy().argmax(0)

def get_parsing(img, model):
    with torch.no_grad():
        inp = preprocess(img)
        out = model(inp)[0]
        return postprocess(out)