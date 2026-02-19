'''
Utils for prostate segmentation tutorial, including distance transform function and accuracy measures.
'''

import numpy as np
from scipy.ndimage import distance_transform_edt
import torch

def distance_transform(mask):
    """
    DISTANCE TRANSFORM SEPARATELY FOR INSIDE AND OUTSIDE OF THE MASK, WHICH CAN BE USED TO CONSTRUCT THE SIGNED DISTANCE MAP

    mask_np: (H, W) binary {0,1}
    returns: (H, W) tuple of (distance outside, distance inside)
    """

    fg_dt = distance_transform_edt(mask == 0)
    bg_dt = distance_transform_edt(mask == 1)
    return fg_dt, bg_dt  # outside DT, inside DT


def signed_distance_map(mask):
    """
    MAPS EACH PIXEL TO THE NEAREST BOUNDARY OF THE MASK, WITH SIGN INDICATING INSIDE/OUTSIDE
    
    mask_np: (H, W) binary {0,1}
    returns: (H, W) signed distance map (positive outside, negative inside)
    """
    fg_dt, bg_dt = distance_transform(mask)
    sdf = fg_dt - bg_dt
    return sdf.astype(np.float32)


def accuracy(outputs, masks):
    '''CALCULATE THE ACCURACY OF THE PREDICTIONS, WHICH IS THE FRACTION OF PIXELS CORRECTLY CLASSIFIED '''

    # The outputs are logits, apply sigmoid and threshold at 0.5
    predicted = torch.sigmoid(outputs).float()
    predicted = (predicted > 0.5).squeeze(1).float()  # Binarize predictions

    correct = (predicted == masks.squeeze(1)).sum().item()
    total = masks.numel()
    accuracy = correct / total
    
    return accuracy

def dice_coefficient(outputs, masks, eps=1e-7):
    ''' CALCULATE THE DICE COEFFICIENT, WHICH IS A MEASURE OF OVERLAP BETWEEN THE PREDICTION AND THE GROUND TRUTH MASK. IT RANGES FROM 0 TO 1, WHERE 1 MEANS PERFECT OVERLAP AND 0 MEANS NO OVERLAP. THIS IS A MORE MEANINGFUL METRIC THAN ACCURACY FOR IMBALANCED DATASETS '''

    # The outputs are logits, apply sigmoid and threshold at 0.5
    predicted = torch.sigmoid(outputs).float()
    predicted = (predicted > 0.5).squeeze(1).float()
    masks = masks.squeeze(1).float()
    
    intersection = (predicted * masks).sum().item()
    union = predicted.sum().item() + masks.sum().item()
    dice = (2 * intersection + eps) / (union + eps)
    
    return dice

