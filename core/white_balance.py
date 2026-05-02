#white_balance.py
import numpy as np
import cv2

def white_balance_by_paper(img, region):
    x1, y1, x2, y2 = region

    paper = img[y1:y2, x1:x2]
    pixels = paper.reshape(-1, 3)

    if len(pixels) < 50:
        print("紙のエリアが適切ではありません")
        return img, np.zeros(img.shape[:2], dtype=np.uint8)

    brightness = np.sum(pixels, axis=1)
    thresh = np.percentile(brightness, 70)
    
    condition = brightness > thresh
    valid = pixels[condition]
    wb = np.median(valid, axis=0)

    target = np.array([240, 240, 240], dtype=np.float32)
    scale = target / np.maximum(wb, 1)

    balanced = img.astype(np.float32)
    for i in range(3):
        balanced[:, :, i] *= scale[i]

    balanced = np.clip(balanced, 0, 255).astype(np.uint8)

    roi_h, roi_w = y2 - y1, x2 - x1

    mask_roi = condition.reshape(roi_h, roi_w).astype(np.uint8) * 255

    full_mask = np.zeros(img.shape[:2], dtype=np.uint8)
    full_mask[y1:y2, x1:x2] = mask_roi

    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)

    return balanced, full_mask