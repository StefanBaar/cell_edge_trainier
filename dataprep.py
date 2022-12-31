from glob import glob

import numpy as np
import cv2

from tqdm import tqdm

from scipy import spatial

import torch
from torch.utils.data import Dataset, DataLoader, sampler

import torchvision.datasets as dset
from torchvision import transforms

from skimage import draw, io, exposure, morphology,transform

from PIL import Image


from joblib import Parallel, delayed
import multiprocessing

import mysegmentation as myseg


class MyDataset(Dataset):
    def __init__(self, path, pytorch=True):
        super().__init__()
        self.raw_files  = sorted(glob(path+"*raw.png"))
        self.anno_files = sorted(glob(path+"*mask.png"))
        self.pytorch=True
        #self.colors      = colors

    def check_pad(self,im):
        dim = im.shape[0]
        if dim%32!=0:
            res  = int(dim/32)
            diff = int(((res+1)*32-dim)/2)
            im   = np.pad(im,diff,"reflect")
        return im

    def load_image(self,idx, invert=True):
        raw_rgb = io.imread(self.raw_files[idx])
        #if invert:
            #raw_rgb = raw_rgb.transpose((2,0,1))
        norm = (raw_rgb / np.iinfo(raw_rgb.dtype).max)
        norm = self.check_pad(norm)
        return torch.unsqueeze(torch.tensor(norm, dtype=torch.float32),0)
        
    def load_mask(self, idx):
        """load mask from path"""
        mask = io.imread(self.anno_files[idx])/100
        mask = np.round(mask,0)
        mask = self.check_pad(mask)
        return torch.unsqueeze(torch.tensor(mask, dtype=torch.int64),0)

    def __getitem__(self, idx):
        image = self.load_image(idx)
        mask  = self.load_mask(idx)
        return image, mask

    def __len__(self):
        return len(self.raw_files)

    def __repr__(self):
        s = 'Dataset class with {} files'.format(self.__len__())
        return s

def load_train_val(train_SET,validation_SET,batch_size=12):
    DLEN  = int(train_SET.__len__())
    VALEN = int(validation_SET.__len__())
    ratio = (DLEN-VALEN,VALEN)

    DATA_TRAIN = DataLoader(train_SET     , batch_size=batch_size, shuffle=True)
    DATA_VALID = DataLoader(validation_SET, batch_size=batch_size, shuffle=True)

    return DATA_TRAIN, DATA_VALID

def mCPU(func, var, n_jobs=40,verbose=10):
    return Parallel(n_jobs=n_jobs, verbose=verbose)(delayed(func)(i) for i in var)

def get_ds(M,S):
    """returns closest points: of main island M and slave island s: 
    returns: closest points of M and S and their distance D"""
    T   = spatial.cKDTree(M)
    d,p = T.query(S)
    m   = np.argmin(d)
    return S[m], M[p[m]], d[m]

def connect_nearest_dots(mask):
    """I forgot, what this does"""
    unis   = np.unique(mask)[1:]
    dlists = [np.argwhere(mask==i) for i in unis]
    l1     = dlists[0]
    lS     = dlists[1:]
    
    M,S,D  = [],[],[]
    for i in lS:
        m,s,d = get_ds(l1,i)
        M.append(m)
        S.append(s)
        D.append(d)
    
    MIN = np.argmin(D)
    MP  = M[MIN]
    SP  = S[MIN]
    X,Y = [MP[1],SP[1]],[MP[0],SP[0]]
    return Y,X

def connect_nearest(MA):
    """I forgot, what this does"""
    Y, X = connect_nearest_dots(MA)
    MA2        = MA.copy()
    MA2[MA2>0] = 1
    line = draw.line(Y[0],X[0],Y[1],X[1])
    MA2[line] = 1 
    return myseg.get_segments(MA2)[0]
    
def fill_gaps(IM,max_iter=100):
    """I forgot, what this does"""
    MA = IM.copy()
    for i in range(100):
        unilen = len(np.unique(MA))
        if unilen < 3:
            break
        MA = connect_nearest(MA)
    return MA

def data_from_coco(im_path, an_path, idn=100):
    coco    = dset.CocoDetection(root    = im_path,
                                 annFile = an_path)

    im_ids  = list(sorted(coco.coco.imgs.keys()))
    imidn   = im_ids[idn]
    im_info = coco.coco.loadImgs([imidn])

    image   = cv2.imread(im_path+im_info[0]["file_name"])[:,:,0].astype("int")
    annos   = coco.coco.loadAnns(coco.coco.getAnnIds(imidn))
    
    masks    = []#np.zeros_like(image)
    for n,i in tqdm(enumerate(annos)):
        ma = coco.coco.annToMask(i)
        ma = myseg.get_segments(ma)[0]
        ma = fill_gaps(ma)
        masks.append(ma)
        #mask[ma==1] = n+1
    
    #### the following will sort cells by area. Removes overlap info
    areas =  [len(np.argwhere(i==1)) for i in masks]
    asort = np.argsort(areas) #not finnished
    return image, np.asarray(masks)[asort]

def flatten_mask(masks):
    mask = np.zeros(masks.shape[1:])
    for n,i in enumerate(masks):
        mask[i==1] = n+1
    return mask 


################################################################################
 ################################ Augmentations ################################
################################################################################

###### extract crops

def get_crop_pars(mask, ind=1, pad = 1.2):
    """get tight fitted crops (if pad = 1)
       input: n x m numpy array
       output: [1,1,1,1] <- [top,left,height,width] coordinates of rectangle """
    anps       = np.argwhere(mask == ind)
    ymin, xmin = anps.min(0)
    ymax, xmax = anps.max(0)
    yc  , xc   = anps.mean(0)
    dx         = xmax-xmin
    dy         = ymax-ymin
    d          = np.max([dy,dx])*pad       
    top    = int(yc-d/2)
    left   = int(xc-d/2)
    height = int(d)
    width  = int(d)
    return top,left,height,width

def create_crops(mask,pad=1.2):
    croplist = []
    cells    = np.unique(mask)[1:]
    for i in cells:
        croplist.append(get_crop_pars(mask,i,pad=pad))
    
    return np.asarray(croplist)

def to_PIL(im):
        return Image.fromarray(np.uint8(im))

def crop_resize(IM,CROPPAR,s=512):
    im         = to_PIL(IM)
    resize     = transforms.Resize(size=(s, s),interpolation=transforms.InterpolationMode.NEAREST_EXACT)
    i, j, h, w = CROPPAR
    im         = TF.crop(im, i, j, h, w)
    return np.asarray(resize(im))

def im_resize(IM,CROPPAR,s=512):
    im         = to_PIL(IM)
    resize     = transforms.Resize(size=(s, s),interpolation=transforms.InterpolationMode.BILINEAR)
    i, j, h, w = CROPPAR
    im         = TF.crop(im, i, j, h, w)
    return np.asarray(resize(im))

def image_crop_resize(im,c=200,size=512):
    im0 = transform.resize(im[c:-c,c:-c],(size,size),preserve_range=True,order=3)
    return im0

def mask_crop_resize(im,c=200,size=512):
    def cr(ma,c,size):
        return transform.resize(ma[c:-c,c:-c],(size,size),preserve_range=True,anti_aliasing=True)
    im0 = np.stack([cr(i,c,size) for i in tqdm(im)])
    return im0.astype(int)

######## Distortions 

def create_dist_maps(size,amp=10):
    ysize, xsize = size
    """creates distortion map to apply to image or mask"""
    def rd(dxy):
        return (np.random.rand())

    src_cols           = np.linspace(0, xsize, 20)
    src_rows           = np.linspace(0, ysize, 20)
    dx                 = src_cols[1]
    dy                 = src_rows[1]
    src_rows, src_cols = np.meshgrid(src_rows, src_cols)
        
    src = np.dstack([src_cols.flat, src_rows.flat])[0]

    dst_rows = src[:, 1] - np.sin(np.linspace(0,   2*rd(dy)*np.pi, src.shape[0])) * amp
    dst_cols = src[:, 0] - np.cos(np.linspace(0,   2*rd(dx)*np.pi, src.shape[0])) * amp
    #dst_rows *= 1.5
    #dst_rows -= 1.5 * 50
    dst = np.vstack([dst_cols, dst_rows]).T

    mins0 = np.argwhere(src[:,0]==0)
    mins1 = np.argwhere(src[:,1]==0)
    maxs0 = np.argwhere(src[:,0]==src.max())
    maxs1 = np.argwhere(src[:,1]==src.max())

    dst = np.asarray(dst)
    dst[:,0][mins0] = 0
    dst[:,1][mins1] = 0
    dst[:,0][maxs0] = src.max()
    dst[:,1][maxs1] = src.max()

    return [src, dst]

def distortions(image, maps,radius=None,shift=20,strength=0.1):

    def shift_left(xy):
        xy[:, 0] += shift
        return xy

    if radius == None:
        radius = image.shape[0]*2

    src, dst = maps
    tform    = transform.PiecewiseAffineTransform()
    tform.estimate(src, dst)

    dist = transform.warp(image, tform, output_shape=image.shape,
                          preserve_range=True,mode='reflect')
    dist = transform.swirl(dist, rotation=0, strength=strength,
                           radius=radius,preserve_range=True,mode='reflect')

    return dist

def distort_all(image,masks,amp=10,radius=2.,shift=20,strength=0.1):

    dmap   = create_dist_maps(image.shape,amp=10)
    dimage = distortions(image, dmap,radius=radius,shift=shift,strength=strength)
    
    def distort(mask):
        dtemp = distortions(mask, dmap,radius=radius,shift=shift,strength=strength)
        return np.round(dtemp,0).astype(int)

    cpus   = multiprocessing.cpu_count()
    dmasks = np.stack(mCPU(distort,masks,cpus)).astype(int)

    #dmask  = flatten_mask(np.asarray(dmasks))

    return dimage, dmasks


def outline_mask(mask,line_width=2):
    outlines = np.zeros_like(mask[0]) 
    for i in mask:
        dill = morphology.binary_dilation(i,morphology.disk(line_width))
        outlines[dill==1] = 1
        #mind = morphology.binary_erosion(i,morphology.disk(line_width))
        outlines[i==1] = 2
    return outlines


if __name__ == '__main__':

    im_path  = "/mnt/Local_SSD/stefan/cell_data_sets/LIVECell_dataset_2021/images/livecell_train_val_images/SHSY5Y/"
    an_path  = "/mnt/Local_SSD/stefan/cell_data_sets/LIVECell_dataset_2021/annotations/LIVECell_single_cells/shsy5y/train.json"
    out_path = "/mnt/Local_SSD/stefan/cell_data_sets/training/221219_livecell_1/pngs/" 

    image_id = 100

    lims     = (np.arange(1,10)*20).astype(int)
    im_list  = sorted(glob(im_path+"*.tif"))

    image, masks = data_from_coco(im_path, an_path, idn=image_id)

    dimages, dmasks = [],[]
    for i in range(10):
        di,dm = distort_all(image,masks,amp=10,radius=2.,shift=20,strength=0.1*i)
        dimages.append(di)
        dmasks.append(dm)
    
    dimages, dmasks = np.stack(dimages), np.stack(dmasks)

    dimages_left  = dimages[:,:, :dimages.shape[1] ]
    dimages_right = dimages[:,:, -dimages.shape[1]:]

    dmasks_left  = dmasks[:,:,:, :dimages.shape[1] ]
    dmasks_right = dmasks[:,:,:, -dimages.shape[1]:]

    image_crops = np.vstack([dimages_left,dimages_right])
    mask_crops  = np.vstack([dmasks_left ,dmasks_right ])

    
    def process_all(ind):
        
        size = dimages.shape[1]
        lw   = 2
        
        ic = image_crops[ind].astype("uint8")
        im = (outline_mask(mask_crops[ind],lw)*100).astype("uint8")
        
        io.imsave(out_path+str(10000000+ind*100)[1:]+"raw.png" ,ic)
        io.imsave(out_path+str(10000000+ind*100)[1:]+"mask.png" ,im)
        
        for n,i in enumerate(lims):
            ic = image_crops[ind]
            ic = image_crop_resize(ic,c=i,size=size)    
            ic = ic.astype("uint8")
            
            im = mask_crops[ind]
            im = mask_crop_resize(im,c=i,size=size)
            im = outline_mask(im,lw*size/(size-i))*100
            im = im.astype(int)

            io.imsave(out_path+str(10000000+ind*100+n)[1:]+"raw.png" ,ic)
            io.imsave(out_path+str(10000000+ind*100+n)[1:]+"mask.png" ,im)
            
            for j in list(range(1,4)):
                io.imsave(out_path+str(10000000+ind*100+n+10*j)[1:]+"raw.png" ,np.rot90(ic,j))
                io.imsave(out_path+str(10000000+ind*100+n+10*j)[1:]+"mask.png" ,np.rot90(im,j))
            
                io.imsave(out_path+str(10000000+ind*100+n+10*j+30)[1:]+"raw.png" ,np.rot90(ic.T,j))
                io.imsave(out_path+str(10000000+ind*100+n+10*j+30)[1:]+"mask.png" ,np.rot90(im.T,j))

                io.imsave(out_path+str(10000000+ind*100+n+10*j+60)[1:]+"raw.png" ,np.rot90(ic[::-1].T,j))
                io.imsave(out_path+str(10000000+ind*100+n+10*j+60)[1:]+"mask.png" ,np.rot90(im[::-1].T,j))
                
    _ = mCPU(process_all,range(len(image_crops)),40) 