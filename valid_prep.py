import datetime
from tqdm import tqdm
from glob import glob

import numpy as np
import cv2

import pycocotools
from pycocotools.coco import COCO
import torch, torchvision
import torchvision.datasets as dset
from torchvision import transforms

from skimage import data, color, img_as_ubyte, measure, filters, io
from skimage import segmentation, morphology, transform, util, exposure
from skimage.feature import canny, peak_local_max
from skimage.transform import hough_ellipse, hough_circle, hough_circle_peaks
from skimage.draw import ellipse_perimeter
from skimage import draw

import mysegmentation as myseg
import dataprep as dp

size = 512
lw   = 2

im_path = "/mnt/Local_SSD/stefan/cell_data_sets/LIVECell_dataset_2021/images/livecell_train_val_images/SHSY5Y/"
im_list = sorted(glob(im_path+"*.tif"))
an_path = "/mnt/Local_SSD/stefan/cell_data_sets/LIVECell_dataset_2021/annotations/LIVECell_single_cells/shsy5y/train.json"

for i in list(range(1,100)):
    t_image, t_masks = dp.data_from_coco(im_path, an_path, idn=i)
    
    ic = t_image[:,:size].astype("uint8")
    im = (dp.outline_mask(t_masks[:,:,:size],lw)*100).astype("uint8")
    
    io.imsave(out_path+"validation/"+str(10000000+i*100)[1:]+"raw.png" ,ic)
    io.imsave(out_path+"validation/"+str(10000000+i*100)[1:]+"mask.png" ,im)
    
    ic = t_image[:,-size:].astype("uint8")
    im = (dp.outline_mask(t_masks[:,:,-size:],lw)*100).astype("uint8")
    
    io.imsave(out_path+"validation/"+str(10000000+i*101)[1:]+"raw.png" ,ic)
    io.imsave(out_path+"validation/"+str(10000000+i*101)[1:]+"mask.png" ,im)