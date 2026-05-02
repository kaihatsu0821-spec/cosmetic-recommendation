import numpy as np
from pipeline.skin import run_skin_pipeline
from datatypes import SkinResult
from core.segmentation import create_beard_mask,create_blue_mask
from core.vision import compute_lab_stats,compute_beard_features
from datatypes import Params,BeardResult

def Srun_beard_pipeline(skin_result:SkinResult):
    img=skin_result.balanced_img
    landmarks=skin_result.landmarks
    base_Lab=skin_result.Lab
    params=Params

    h,w=img.shape[:2]
    beard_mask=create_beard_mask(landmarks,w,h)
    beard_mean,beard_std,beard_pixels_lab=compute_lab_stats(img,beard_mask)
    z_scores=compute_beard_features(beard_pixels_lab,base_Lab,beard_std)
    beard_pixels_after,blue_mask=create_blue_mask(img,z_scores,params,beard_mask)
    beard_color=np.median(beard_pixels_after,axis=0)
    
    return BeardResult(
        color=beard_color,
        lab=beard_pixels_lab,
        mask=blue_mask
    )
