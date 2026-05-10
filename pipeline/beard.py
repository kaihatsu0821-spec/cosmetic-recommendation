import numpy as np
from pipeline.skin import Srun_skin_pipeline
from datatypes import SkinResult
from core.segmentation import create_beard_mask,create_blue_mask
from core.vision import compute_lab_stats,compute_beard_features,bgr_lab
from datatypes import Params,BeardResult

def Srun_beard_pipeline(skin_result:SkinResult):
    img=skin_result.balanced_img
    lab_img=bgr_lab(img)
    landmarks=skin_result.landmarks
    base_lab=skin_result.lab
    params=Params()


    h,w=img.shape[:2]
    beard_mask=create_beard_mask(landmarks,w,h)
    beard_mean,beard_std,beard_pixels_lab=compute_lab_stats(img,beard_mask)
    z_scores=compute_beard_features(beard_pixels_lab,base_lab,beard_std)
    bgr_pixels,lab_pixels,blue_mask=create_blue_mask(img,lab_img,z_scores,params,beard_mask)
    bgr_color=np.median(bgr_pixels,axis=0)
    lab_color=np.median(lab_pixels,axis=0)
    
    
    return BeardResult(
        color=bgr_color,
        lab=lab_color,
        mask=blue_mask
    )