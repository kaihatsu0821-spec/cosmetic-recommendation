#recommend.py
import numpy as np

class Foundation:
    def __init__(self, name, lab):
        self.name = name
        self.lab = np.array(lab)

def lab_distance(a, b):
    d = np.array(a) - np.array(b)
    return np.sqrt(1.5 * d[0]**2 + d[1]**2 + d[2]**2)

def recommend(skin, db):
    scored = []
    for f in db:
        dist = lab_distance(skin, f.lab)
        scored.append((f.name, f.lab, dist))
    
    return sorted(scored, key=lambda x: x[2])