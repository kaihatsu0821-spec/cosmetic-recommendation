#vision.py
import numpy as np
import cv2


def bgr_to_lab(bgr):
    lab = cv2.cvtColor(np.uint8([[bgr]]), cv2.COLOR_BGR2LAB)[0][0]
    L, a, b = lab_opencv_to_true(lab)

    return np.array([L,a,b])


def lab_opencv_to_true(lab):
    L, a, b = lab
    return np.array([
        L * 100 / 255,
        a - 128,
        b - 128
    ])

def delta_e(lab1, lab2):
    return np.linalg.norm(lab1 - lab2)


def compute_beard_features(pixels,base_lab, skin_std):
    
    diff=pixels-base_lab

    z_scores=diff/skin_std

    return z_scores

def compute_lab_stats(img, mask):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    pixels = lab[mask == 255]
    
    if len(pixels) == 0:
        return None, None,None
        
    pixels[:, 0] *= 100/255
    pixels[:, 1:] -= 128
    pixels[:, 2:] -= 128
    
    mean = np.median(pixels, axis=0) 
    std = np.std(pixels, axis=0) 
    std = np.maximum(std, 1.0)
    
    return mean, std,pixels

def lab_opencv_to_true(lab):
    L, a, b = lab
    return np.array([
        L * 100 / 255,
        a - 128,
        b - 128
    ])

def delta_e(lab1, lab2):
    return np.linalg.norm(lab1 - lab2)

def white_float_score(skin_lab, face_lab):
    w_L = 1.5
    w_C = 1.0

    dL = face_lab[0] - skin_lab[0]
    dC = np.linalg.norm(face_lab[1:] - skin_lab[1:])

    return w_L * abs(dL) + w_C * dC

def median_skin_color(pixels):
    return np.median(pixels, axis=0) if pixels is not None else None
    
def unpack_optional_pair(data):
    return data if data else (None, None)