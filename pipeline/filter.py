import cv2
from models.model import get_model
from file_io.parsing import get_parsing
from core.segmentation import create_mask,create_nose_mask
from core.face_filter import apply_filter

SKIN=1
NOSE=10
def Srun_filter_pipeline(img,Lab,strength):
    target_L,target_a,target_b=Lab

    model=get_model()
    parsing=get_parsing(img,model)
    skin_mask=create_mask(img,parsing,SKIN)
    nose_mask=create_nose_mask(img,parsing,NOSE)
    mask=cv2.bitwise_or(skin_mask,nose_mask)
    result=apply_filter(img,mask,target_L,target_a,target_b,strength)
    return result,mask