from tqdm import tqdm

import numpy as np
from skimage import measure,morphology

from psd_tools import PSDImage

from matplotlib import cm

from pytoshop.user import nested_layers
from pytoshop import enums
from pytoshop.image_data import ImageData


from mysegmentation import get_segments, mCPU

def layer_to_RGB(layer):
    return (layer[:, :, :-1].astype(int)//41*41).astype("uint8")

def get_colors(arr):
    return np.unique(arr.reshape(-1, 3), axis=0)[:-1]

def large_psd_to_npy(path,CPUS=16):
    psd    = PSDImage.open(path)
    lnr    = len(psd)

    if lnr>=CPUS:
        n = CPUS
    else:
        n = lnr

    def process(i):
        return np.asarray(i.composite(psd.viewbox))

    return np.stack(mCPU(process,psd,n,0))


def psd_layesrs_to_npy(path):
    psd   = PSDImage.open(path)
    LAYERS= []

    for n,i in enumerate(tqdm(psd)):
        #print(n,i)
        image = np.asarray(i.composite(psd.viewbox))
        LAYERS.append(image)

    return np.asarray(LAYERS)

def get_cells(image):
    RGB  = layer_to_RGB(image)
    US   = get_colors(RGB)
    UM   = np.zeros_like(RGB[:,:,0])

    for n,i in enumerate(tqdm(US)):
        UM[np.all(RGB == i, axis=-1)] = n+1

    return UM.astype(int)

def get_masks_from_layer(psd, alpha_th=1):
    """get image, and binary masks from psd layers
       first layer:      image
       second layer:     contaminants 
       remaining layers: cells (one cell per image)"""
    
    def get_binary_mask(images, alpha_th):
        mask = np.zeros_like(images[:,:,:,0])
        mask[images[:,:,:,3]>alpha_th] = 1
        return mask
    
    image   = psd[0]
    blayers = get_binary_mask(psd[1:], alpha_th)

    return image, blayers

def bin_stack_to_ind_mask(cells):
    return np.sum(cells.T*np.arange(1, cells.shape[0]+1), 2).T

def bin_stack_to_ind_stack(cells):
    return (cells.T*np.arange(1, cells.shape[0]+1)).T

def load_psd(path):
    Layers    = psd_layesrs_to_npy(path)
    image     = Layers[0]
    cells     = get_cells(Layers[1].astype(int)//4)
    cells     = get_segments(cells)[0]
    conts     = get_cells(Layers[1].astype(int)//4)

    return image,cells,conts

def cell_segments(layers):
    cells = layers[4:, :, :, -1]
    masks = np.zeros_like(cells, dtype=int)
    masks[cells > 0] = 1
    return get_segments(np.sum(masks, 0))[0]


def get_psd_data(image, conts, mask_fl, VMAX=None):

    mask = mask_fl.astype("int")

    IM = np.asarray(image)
    IM = np.asarray([IM, IM, IM, np.ones_like(IM)*255]).astype("uint8")
    PSD_IM = ImageData(channels=IM)

    mask3d = [nested_layers.Image(channels=PSD_IM.channels,
                                  name="raw image")]

    inds   = np.unique(mask)[1:]
    sample = np.zeros_like(mask).astype("uint8")

    if VMAX == None:
        VMAX = mask.max()

    color = cm.CMRmap(inds.astype(float)/VMAX*2, bytes=True)

    # print(mask.shape)
    CONTS     = {}
    CONTS[-1] = conts[:,:,3]
    CONTS[ 0] = conts[:,:,0]
    CONTS[ 1] = conts[:,:,1]
    CONTS[ 2] = conts[:,:,2]

    clayer = nested_layers.Image(channels    = CONTS,
                                 name        = "contaminants",
                                 layer_color = 0,
                                 color_mode  = 3)
    mask3d.append(clayer)
    
    n=0
    for _, i in enumerate(inds):
        alpha_mask = sample.copy()
        alpha_mask[mask == i] = 1
        alpha_mask = morphology.remove_small_holes(alpha_mask,256)*255
        alpha_mask = alpha_mask.astype("uint8")

        area = len(np.argwhere(alpha_mask>1))

        if area > 400:
            r = alpha_mask.copy()
            g = alpha_mask.copy()
            b = alpha_mask.copy()

            r[alpha_mask == 255] = color[n][0]
            g[alpha_mask == 255] = color[n][1]
            b[alpha_mask == 255] = color[n][2]

            # RGBA     = np.asarray([r,g,b,alpha_mask])
            RGBA = {}
            RGBA[-1] = alpha_mask
            # RGBA[-1] = alpha_mask
            RGBA[0] = r
            RGBA[1] = g
            RGBA[2] = b

            newLayer = nested_layers.Image(channels=RGBA,

                                        # opacity    = 0,
                                        # blend_mode = enums.,
                                        name=str(int(i)),
                                        layer_color=0,
                                        color_mode=3)

            # print(RGBA[:,0,0])
            mask3d.append(newLayer)
            n+=1

    return nested_layers.nested_layers_to_psd(mask3d[::-1],
                                              color_mode=enums.ColorMode.rgb,
                                              size=image.shape)


def save_as_psd(name, image, mask_fl, VMAX=None):

    output = get_psd_data(image, mask_fl, VMAX)

    with open(name, 'wb') as fd:
        output.write(fd)
    fd.close()


