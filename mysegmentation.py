import numpy as np
from skimage.morphology import disk
from skimage import data, color, img_as_ubyte, measure, filters, io, morphology, segmentation

from scipy import ndimage

from pathlib import Path
from PIL import Image, ImageOps

from joblib import Parallel, delayed

import warnings
warnings.filterwarnings("ignore")


def mCPU(func, var, n_jobs=20,verbose=10):
    return Parallel(n_jobs=n_jobs, verbose=verbose)(delayed(func)(i) for i in var)

def get_image(path):
    im = Image.open(path)
    im = ImageOps.autocontrast(im,cutoff=0.1)
    return np.asarray(im)

def get_object(DATA, index=1):
    OBS = DATA.copy()
    OBS[OBS!=index] = 0
    return OBS

def normalize(IMAGE, NORM = 1):
    IMAGE = IMAGE-IMAGE.min()
    return IMAGE/IMAGE.max()*NORM

def rem_holes(DATA, th=250,con=1):
    return morphology.remove_small_holes(DATA.astype("bool"),
                                         area_threshold=th,
                                         connectivity=con)
def closing(DATA, DISK = 2):
    return morphology.closing(DATA.astype("bool"),
                              disk(DISK))
def rem_obj(DATA):
    return morphology.remove_small_objects(DATA.astype("bool"))

def get_segments(IMAGE):
    marker = measure.label(IMAGE)
    LABELS = range(marker.max()+1)
    SIZE = []
    for i in LABELS:
        SIZE.append(np.where(marker == i)[0].size)
    sortmask = np.argsort(SIZE)[::-1]
    AREA     = np.asarray(SIZE)[sortmask]
    REGIONS  = np.zeros_like(marker)
    for i in range(marker.max()+1):
        REGIONS[marker == sortmask[i]] = i
    return REGIONS, AREA

def get_segmentation_mask(DATA, THRESH = 0.07, cl_disk = 4,di_disk=2, median=3):

    #Median  = filters.median(C1, selem=disk(median))
    #Foregrnd= np.abs(DATA-Median)
    Foregrnd= DATA.copy()
    Sobel   = normalize(filters.sobel(Foregrnd))
    Mask    = np.zeros_like(DATA).astype(int)

    Mask[Sobel > THRESH]  = 1
    Mask    = rem_obj(Mask)
    Mask    = morphology.dilation(Mask,disk(di_disk), out=None)
    Mask    = closing(Mask,DISK=cl_disk)
    Mask    = rem_holes(Mask)
    return rem_obj(Mask)


def get_segmentation(DATA, THRESH = 0.07, cl_disk = 2,di_disk=2, median=3):
    Mask = get_segmentation_mask(DATA, THRESH, cl_disk,di_disk, median)
    return get_segments(Mask)

def get_all_segs(DATA, THRESH = 0.07, cl_disk = 2,di_disk=2, median=3):
    SEGMENTS, AREAS,S_L,A_L = [],[],[],[]
    for i in DATA:
        SS,AA = get_segmentation(i, THRESH, cl_disk, di_disk, median)
        SEGMENTS.append(list(SS.astype(float)))
        AREAS.append(AA)
        A_L.append(AA[0])
        S = SS.copy()
        S[S>1] =0
        S_L.append(S)

    return np.array(SEGMENTS), np.asarray(S_L) ,AREAS, np.asarray(A_L)


def new_size_filter(LABELS,fsize=400):
    """remove objects with an area less than fsize"""
    UNIS,AREA  = np.unique(LABELS,return_counts=True)
    NEW_LABELS = LABELS.copy()
    for i in range(len(UNIS)):
        if AREA[i] < fsize:
            NEW_LABELS[LABELS==UNIS[i]] = 0
    return NEW_LABELS


def size_filter_SEGS(SEGS,AREA, FRAME, FILTER=400):
    if np.min(AREA[FRAME]) < FILTER:
        AMASK   = list(range(len(AREA[FRAME]))[np.min(np.argwhere(AREA[FRAME] < FILTER)):])
        SMALLS  = np.isin(SEGS[FRAME].copy(), AMASK)
        NEW_SEG = SEGS[FRAME].copy()
        NEW_SEG[SMALLS] = 0
        return NEW_SEG
    else:
        return SEGS[FRAME]



def watershed_seg(cell_map,nukes,gausnukes):
    dmap  = ndimage.distance_transform_edt(cell_map)
    ma   = np.zeros_like(dmap, dtype=bool)
    ma[tuple(nukes.astype(int).T)] = True

    gausdmap   = ndimage.distance_transform_edt(gausnukes.astype(bool))
    gmarks ,_  = ndimage.label(ma)
    gaus_segs  = segmentation.watershed(-gausdmap,
                                        gmarks,
                                        mask=gausnukes.astype(bool),
                                        #connectivity=4,
                                        compactness=0,
                                        watershed_line=True)

    markers,_  = ndimage.label(gausnukes)
    seg        = segmentation.watershed(-dmap,
                                        gaus_segs,
                                        mask=cell_map,
                                        #connectivity=4,
                                        compactness=0 )
    seg        = seg * (seg.max()*2)
    #for p,i in tqdm(nukes):
    #    seg = segmentation.flood_fill(seg, (int(i[0]),int(i[1])), p+1)
    return get_segments(seg)[0]




# if __name__ == '__main__':
#     PATH     = "data/images/"
#     IN_FILES = [str(i) for i in sorted(Path(PATH).rglob("*.png"))]
#     OUT_FILES= [i.replace("images/","npys/").replace(".png","") for i in IN_FILES]
#     OUT_PATHS= np.unique([i[:-len(i.split("/")[-1])] for i in OUT_FILES])
#     ### create dics
#     for i in OUT_PATHS:
#         Path(i).mkdir(parents=True, exist_ok=True)
#
#
#     def process(frame ):
#         image     = get_image(IN_FILES[frame])[:,:,0]
#         mask      = get_segmentation_mask(image,THRESH = 0.06, cl_disk = 1,di_disk=1, median=1)
#         segs,area = get_segments(mask)
#         segs      = size_filter_SEGS([segs],[area],0)
#         np.save(OUT_FILES[frame],segs)
#
#     mCPU(process, range(len(IN_FILES)))
