#segmentation.py
import numpy as np
import cv2
def create_cheek_mask(landmarks, w, h):
    LEFT = [50, 101, 118, 117, 123, 147, 187, 207]
    RIGHT = [280, 330, 347, 346, 352, 376, 411, 427]

    mask = np.zeros((h, w), dtype=np.uint8)

    for cheek in [LEFT, RIGHT]:
        pts = np.array([
            [int(landmarks.landmark[i].x * w),
             int(landmarks.landmark[i].y * h)]
            for i in cheek
        ])
        cv2.fillPoly(mask, [pts], 255)

    return mask

def has_enough_pixels(img, mask, min_pixels=50):
    pixels = img[mask == 255]
    return len(pixels) >= min_pixels

def remove_dark_pixels(img, mask, percentile=20):
    pixels = img[mask == 255]

    if len(pixels) < 50:
        return mask

    brightness = np.sum(pixels, axis=1)
    thresh = np.percentile(brightness, percentile)

    full_brightness = np.sum(img, axis=2)

    mask[(mask == 255) & (full_brightness <= thresh)] = 0

    return mask

def remove_red_tint(img, mask):
    diff_rg = img[:, :, 2].astype(np.int16) - img[:, :, 1].astype(np.int16)
    mask[(mask == 255) & (diff_rg >= 60)] = 0
    return mask

def extract_skin(img, landmarks, w, h):
    mask = create_cheek_mask(landmarks, w, h)

    mask = remove_dark_pixels(img, mask)
    if mask is None:
        return None, None

    mask = remove_red_tint(img, mask)

    final_pixels = img[mask == 255]

    if len(final_pixels) < 30:
        return None, None

    return final_pixels, mask



def analyze_skin_before_after(img, balanced_img, landmarks):
    h, w = img.shape[:2]

    skin_before = extract_skin(img, landmarks, w, h)
    skin_after = extract_skin(balanced_img, landmarks, w, h)

    pixels_before, mask_before = skin_before if skin_before else (None, None)
    pixels_after, mask_after = skin_after if skin_after else (None, None)

    return pixels_before, mask_before, pixels_after, mask_after

def create_beard_mask(landmarks, w, h):
    BEARD = [97,206,216,186,39,37,0,267,269,270,409,410,436,426,326,2]

    mask = np.zeros((h, w), dtype=np.uint8)

    pts = np.array([
        [int(landmarks.landmark[i].x * w),
         int(landmarks.landmark[i].y * h)]
        for i in BEARD
    ])
    cv2.fillPoly(mask, [pts], 255)
    return mask




def create_blue_mask(img,lab_img,z_scores,params,beard_mask):
    
    is_dark = z_scores[:, 0] < params.l_threshold
    is_blue = z_scores[:, 2] < params.b_threshold 
    
    blue_mask_indices = is_dark & is_blue

    full_blue_mask = np.zeros(beard_mask.shape, dtype=bool)
    full_blue_mask[beard_mask == 255] = blue_mask_indices
    
    bgr_pixels = img[full_blue_mask]
    lab_pixels=lab_img[full_blue_mask]

    if len(bgr_pixels) < params.min_beard_pixels:
        return None, None,None
    return bgr_pixels,lab_pixels, full_blue_mask.astype(np.uint8) * 255

def create_mask(image,parsing,class_id):
    h, w = image.shape[:2]

    mask = (parsing == class_id).astype(np.uint8) * 255

    mask = cv2.resize(mask, (w, h))

    return mask

def create_nose_mask(image, parsing,class_id):
    mask = create_mask(image, parsing,class_id)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    kernel_dilate = np.ones((7, 7), np.uint8) 
    mask = cv2.dilate(mask, kernel_dilate, iterations=1)

    return mask