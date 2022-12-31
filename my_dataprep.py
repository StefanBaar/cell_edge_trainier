from glob import glob

import numpy as np
import cv2

from tqdm import tqdm

from scipy import spatial

import torch
from torch.utils.data import Dataset, DataLoader, sampler

import torchvision.datasets as dset
from torchvision import transforms

from skimage import draw, io, exposure, morphology, transform

from PIL import Image


from joblib import Parallel, delayed
import multiprocessing

from mysegmentation import mCPU
import dataprep
import psdio

cpus = multiprocessing.cpu_count()

def process_image(image, masks, out_path, lims):

    dimages, dmasks = [], []
    for i in range(10):
        di, dm = dataprep.distort_all(image, masks, 10, 2., 20, 0.1*i)
        dimages.append(di)
        dmasks.append(dm)

    image_crops, mask_crops = np.stack(dimages), np.stack(dmasks)

    def process_all(ind):

        size = 512

        lw = 2

        for n, i in enumerate(lims):
            ic = image_crops[ind]
            ic = dataprep.image_crop_resize(ic, c=i, size=size)
            ic = ic.astype("uint8")

            im = mask_crops[ind][1:]
            co = mask_crops[ind][0]
            co = dataprep.image_crop_resize(co, c=i, size=size)*2.55
            im = dataprep.mask_crop_resize(im, c=i, size=size)
            im = dataprep.outline_mask(im, lw)
            co[im==1] = 1
            co[im==2] = 2
            im = (co*100).astype("uint8")

            io.imsave(out_path+str(10000000+ind*100+n)[1:]+"raw.png", ic)
            io.imsave(out_path+str(10000000+ind*100+n)[1:]+"mask.png", im)

            for j in list(range(1, 4)):
                io.imsave(out_path+str(10000000+ind*100+n+10*j)
                          [1:]+"raw.png", np.rot90(ic, j))
                io.imsave(out_path+str(10000000+ind*100+n+10*j)
                          [1:]+"mask.png", np.rot90(im, j))

                io.imsave(out_path+str(10000000+ind*100+n+10*j+30)
                          [1:]+"raw.png", np.rot90(ic.T, j))
                io.imsave(out_path+str(10000000+ind*100+n+10*j+30)
                          [1:]+"mask.png", np.rot90(im.T, j))

                io.imsave(out_path+str(10000000+ind*100+n+10*j+60)
                          [1:]+"raw.png", np.rot90(ic[::-1].T, j))
                io.imsave(out_path+str(10000000+ind*100+n+10*j+60)
                          [1:]+"mask.png", np.rot90(im[::-1].T, j))

    #_ = mCPU(process_all, range(len(image_crops)), cpus)
    _ = [process_all(i) for i in tqdm(range(len(image_crops)))]

if __name__ == '__main__':
    
    psd_path  = "../annotations/contaminants/raw_layerpercell/V2/"
    psd_paths = sorted(glob(psd_path+"*.psd"))
     
    out_path = "../annotations/contaminants/raw_layerpercell/trainingdata/V2/"
    lims     = (np.arange(1,10)*50).astype(int)

    ind = 0
    
    psd  = psdio.psd_layesrs_to_npy(psd_paths[ind])
    name = psd_paths[ind].split("/")[-1].split(".")[0]

    image, masks = psdio.get_masks_from_layer(psd)

    process_image(image[:,:,0], masks, out_path+name+"_", lims)
