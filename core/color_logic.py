#color_logic.py
import numpy as np
def extract_beard_from_features(beard_mask, zL, zB, params, img):
    blue_mask = (
        (zL < -params.zL) &
        (zB < -params.zB)
    )

    mask = (beard_mask == 255) & blue_mask

    pixels = img[mask]

    if len(pixels) < params.min_beard_pixels:
        return None, None

    return pixels, mask.astype(np.uint8) * 255