from file_io.image import resize_with_scale,scale_roi
from file_io.landmarks import get_landmarks
from core.segmentation import extract_skin
from core.vision import median_skin_color,delta_e
from core.white_balance import white_balance_by_paper
from core.vision import unpack_optional_pair
from datatypes import SkinResult
from core.vision import bgr_to_lab
import numpy as np

def Srun_skin_pipeline(img, paper_region):
    img,scale=resize_with_scale(img,800)
    x1,y1,x2,y2=scale_roi(paper_region,scale)
    h,w=img.shape[:2]
    landmarks=get_landmarks(img)
    skin_before_data=extract_skin(img,landmarks,w,h)

    paper_region_fixed = (min(x1,x2), min(y1,y2), max(x1,x2), max(y1,y2))

    balanced_img, paper_mask = white_balance_by_paper(img, paper_region_fixed)
    skin_after_data = extract_skin(balanced_img, landmarks, w, h)

    skin_pixels_bgr_before, mask_before = unpack_optional_pair(skin_before_data)
    skin_pixels_bgr_after, mask_after = unpack_optional_pair(skin_after_data)

    skin_bgr_before=median_skin_color(skin_pixels_bgr_before)
    skin_bgr_after=median_skin_color(skin_pixels_bgr_after)

    skin_Lab_before=bgr_to_lab(skin_bgr_before)
    skin_Lab_after=bgr_to_lab(skin_bgr_after)

    delta=delta_e(skin_Lab_before,skin_Lab_after)
    return SkinResult(
        balanced_img=balanced_img,
        lab=skin_Lab_after,
        landmarks=landmarks
    )