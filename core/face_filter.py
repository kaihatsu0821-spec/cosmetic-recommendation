#face_filter.py
import cv2
import numpy as np
def apply_filter(img, mask, target_L, target_A, target_B, strength=0.5):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    L, A, B = cv2.split(lab)

    L = L * 100 / 255
    A = A - 128
    B = B - 128

    mask = cv2.GaussianBlur(mask, (21, 21), 0) / 255.0
    α = strength * mask

    A += (target_A - A) * α
    B += (target_B - B) * α
    L += (target_L - L) * α * 0.2

    L = np.clip(L, 0, 100)
    A = np.clip(A, -128, 128)
    B = np.clip(B, -128, 128)

    L = L * 255 / 100
    A = A + 128
    B = B + 128

    lab = cv2.merge([
        L.astype(np.uint8),
        A.astype(np.uint8),
        B.astype(np.uint8)
    ])

    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)