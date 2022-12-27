import datetime
from tqdm import tqdm
from glob import glob

import numpy as np
import cv2

from psd_tools import PSDImage

def psd_layesrs_to_npy(path):
    psd   = PSDImage.open(path)
    LAYERS= []
    for i in tqdm(psd):
        image = np.asarray(i.composite(psd.viewbox))
        LAYERS.append(image)
    return np.asarray(LAYERS)
