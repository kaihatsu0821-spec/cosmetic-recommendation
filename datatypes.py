import numpy as np
from dataclasses import dataclass
@dataclass
class Params:
    blue_L_max: float = 5
    blue_A_max: float = 2
    blue_B_max: float = 2
    zL: float = 1.0
    zB: float = 0.5
    min_beard_pixels: int = 30

    warm_threshold: float = 15
    cool_threshold: float = 0

    min_skin_pixels: int = 30

@dataclass
class SkinResult:
    balanced_img: np.ndarray
    lab: np.ndarray
    landmarks:any

@dataclass
class SkinDebug:
    before:np.array
    delta:float
    skin_mask:np.array
    paper_mask:np.array


    

@dataclass
class BeardResult:
    color: np.ndarray     
    lab: np.ndarray        
    mask: np.ndarray    


@dataclass
class Foundation:
    name: str
    lab: np.ndarray
FOUNDATION_DB = [
    Foundation("Light Ivory", [92.5, 1.2, 10.5]),
    Foundation("Soft Beige", [85.0, 3.5, 14.2]),
    Foundation("Natural Ochre", [78.2, 5.1, 18.8]),
    Foundation("Warm Nude", [74.5, 6.8, 20.3]),
    Foundation("Golden Sand", [68.0, 7.2, 22.5]),
    Foundation("Rosy Beige", [82.0, 8.5, 12.0]),
    Foundation("Deep Honey", [58.5, 9.4, 24.1]),
    Foundation("Pale Pink", [89.0, 6.2, 8.5]),
    Foundation("Classic Tan", [62.3, 8.1, 19.4]),
    Foundation("Rich Mocha", [45.0, 10.5, 18.0])
]   