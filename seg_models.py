import openslide
from glob import glob
import numpy as np
from PIL import Image
import segmentation_models_pytorch as smp
import torch
import matplotlib.pyplot as plt
from pathlib import Path
import sys
from pathlib import Path
from tqdm import tqdm
from random import shuffle
import cv2

device = sys.argv[2]
wsi_path = sys.argv[1]
capacity = 22429696

folds_model = smp.UnetPlusPlus(
    encoder_name="resnet18",
    encoder_weights="imagenet",
    in_channels=3,
    classes=2,
)
folds_model.load_state_dict(torch.load('folds.pt', map_location=device))
folds_model = folds_model.eval()
folds_model = folds_model.to(device)


tissue_model = smp.UnetPlusPlus(
    encoder_name="resnet18",
    encoder_weights="imagenet",
    in_channels=3,
    classes=3,
)
tissue_model.load_state_dict(torch.load('tissue.pt', map_location=device))
tissue_model = tissue_model.eval()
tissue_model = tissue_model.to(device)

focus_model = smp.UnetPlusPlus(
    encoder_name="resnet18",
    encoder_weights="imagenet",
    in_channels=3,
    classes=8,
)

focus_model.load_state_dict(torch.load('blur.pt', map_location=device))
focus_model = focus_model.eval()
focus_model = focus_model.to(device)

pen_model = smp.UnetPlusPlus(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    in_channels=3,
    classes=2,
)
pen_model.load_state_dict(torch.load('pen.pt'))
pen_model = pen_model.to(device)
pen_model = pen_model.eval()


def pred(image, model):
    with torch.no_grad():
        w_, h_ = image.size
        wa, ha = w_%32, h_%32
        image_ = Image.new('RGB', (w_+(32-wa)+128, h_+(32-ha)+128))
        image_.paste(image, (64, 64, w_+64, h_+64))
        image = image_
        image = np.moveaxis(np.array(image), -1, 0)
        image = (torch.Tensor(image)/255)-0.5
        image = image.unsqueeze(0)
        image = image.to(device)
        return (torch.argmax(model(image)[0], 0).cpu().numpy()[64:64+h_, 64:64+w_]).astype('uint8')

    
def pred_wsi(wsi_path, save_np=False):
    wsi = openslide.OpenSlide(wsi_path)

    w, h = wsi.dimensions
    magni = int(float(wsi.properties['aperio.AppMag']))
    magni_mpp = round(10/float(wsi.properties['openslide.mpp-x']))

    ds = int(magni/5)
    thumb = wsi.get_thumbnail((w//ds, h//ds))
    pen_thumb = wsi.get_thumbnail((w//(ds*8), h//(ds*8)))
    tw, th = thumb.size
    num_pixels = tw*th
    num_patches = int(np.ceil(num_pixels/capacity))
    height = int(np.ceil(th/num_patches))
    crops = [height*(i+1) for i in range(num_patches)]

    blur_pred = np.zeros((crops[-1], tw))
    folds_pred = np.zeros((crops[-1], tw))
    tissue_pred = np.zeros((crops[-1]//2, tw//2))

    with torch.no_grad():
        for c in crops:
            patch = thumb.crop((0, c-height, tw, c))
            patch_rs = patch.resize((patch.size[0]//2, patch.size[1]//2))
            blur_pred[c-height:c, :] = pred(patch, focus_model)
            tissue_pred[(c-height)//2:((c-height)//2)+patch_rs.size[1], :] = pred(patch_rs, tissue_model)
            folds_pred[c-height:c, :] = pred(patch, folds_model)

        pen_pred = pred(pen_thumb, pen_model)
    if save_np: np.savez_compressed(f'{Path(wsi_path).stem}', tissue=tissue_pred, blur=blur_pred, folds=folds_pred, pen=pen_pred)
    return {
        'tissue': tissue_pred, 
        'blur': blur_pred, 
        'folds': folds_pred, 
        'pen': pen_pred 
    }



arrays = pred_wsi(wsi_path)

folds = arrays['folds']
tissue = arrays['tissue']
rh, rw = tissue.shape
blur = arrays['blur']
blur = cv2.resize(blur, (rw, rh), interpolation=cv2.INTER_NEAREST)
folds = cv2.resize(folds, (rw, rh), interpolation=cv2.INTER_NEAREST)
viz_image = np.ones((rh, rw, 3), dtype='uint8')*255
viz_image[tissue==1] = [0, 127, 127]
viz_image[np.logical_and(tissue==2, blur==0)] = [0, 0, 0]
viz_image[np.logical_and(tissue==2, blur==1)] = [22, 22, 22]
viz_image[np.logical_and(tissue==2, blur==2)] = [0, (3*255//6), 0]
viz_image[np.logical_and(tissue==2, blur==3)] = [0, (4*255//6), 0]
viz_image[np.logical_and(tissue==2, blur==4)] = [0, (5*255//6), 0]
viz_image[np.logical_and(tissue==2, blur>=5)] = [255, 255, 255]
viz_image[np.logical_and(tissue==2, folds==1)] = [255, 0, 0]
viz_image = cv2.resize(viz_image, (rw//4, rh//4), interpolation=cv2.INTER_NEAREST)
cv2.imwrite(f'{Path(wsi_path).stem}_prediction.png', viz_image[:, :, ::-1])
