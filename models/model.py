#model.py
import os
import torch
from face_parsing.model import BiSeNet
_model = None

BASE_DIR = os.path.join(os.path.dirname(__file__), "..", "face_parsing")

def load_model():
    model = BiSeNet(n_classes=19)

    model.load_state_dict(torch.load(
        os.path.join(BASE_DIR, "res/cp/79999_iter.pth"),
        map_location="cpu"
    ))

    model.eval()
    return model

def get_model():
    global _model
    if _model is None:
        _model = load_model()
    return _model



