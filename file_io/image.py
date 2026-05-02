#image.py
import cv2
def resize_with_scale(img, target_width):
    h, w = img.shape[:2]
    scale = target_width / w
    new_h = int(h * scale)
    resized = cv2.resize(img, (target_width, new_h))
    return resized, scale

def scale_roi(roi, scale):
    x1, y1, x2, y2 = roi
    return tuple(int(p * scale) for p in (x1, y1, x2, y2))