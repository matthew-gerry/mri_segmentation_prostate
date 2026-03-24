# src/image_seg/core/utils.py
'''
Utils for prostate segmentation tutorial, including distance transform function and accuracy measures.
'''

import numpy as np
from scipy.ndimage import distance_transform_edt
import torch


def _get_logits(output):
    """Support both torchvision dict outputs and raw tensor outputs, based on the behaviour of the specific model."""
    return output["out"] if isinstance(output, dict) else output


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


# def accuracy(logits, masks):
#     '''CALCULATE THE ACCURACY OF THE PREDICTIONS, WHICH IS THE FRACTION OF PIXELS CORRECTLY CLASSIFIED '''

#     # The outputs are logits, apply sigmoid and threshold at 0.5
#     predictions = torch.sigmoid(logits) > 0.5  # Binarize predictions

#     # predictions = (predictions > 0.5).squeeze(1).float()  # Binarize predictions

#     batch_size = predictions.shape[0]
#     predictions = predictions.view(batch_size, -1)
#     masks = masks.view(batch_size, -1)

#     # Compute accuracy for each sample in batch
#     # correct = torch.sum(predictions == masks, dim=1).sum().item()
#     # Convert to float for division, and compute total number of pixels
#     correct = torch.sum(predictions == masks, dim=1).float()
#     total = predictions.shape[1]
#     sample_accuracy = correct / total
    
#     return torch.mean(sample_accuracy)


def dice_coefficient(logits, masks, threshold=0.5, eps=1e-7):
    ''' CALCULATE THE DICE COEFFICIENT, WHICH IS A MEASURE OF OVERLAP BETWEEN THE PREDICTION AND THE GROUND TRUTH MASK. IT RANGES FROM 0 TO 1, WHERE 1 MEANS PERFECT OVERLAP AND 0 MEANS NO OVERLAP. THIS IS A MORE MEANINGFUL METRIC THAN ACCURACY FOR IMBALANCED DATASETS '''

    # The outputs are logits, apply sigmoid and threshold at 0.5
    predictions = torch.sigmoid(logits) > threshold  # Binarize predictions

    # Flatten spatial dimensions while preserving batch
    batch_size = predictions.shape[0]
    predictions = predictions.view(batch_size, -1)
    masks = masks.view(batch_size, -1)
    
    # Compute Dice for each sample in batch
    intersection = torch.sum(predictions * masks, dim=1)
    union = torch.sum(predictions, dim=1) + torch.sum(masks, dim=1)
    # Avoid division by zero
    dice = (2.0 * intersection) / (union + eps)
    
    return torch.mean(dice)



# def bland_altman_areas(logits, masks, threshold=0.5):
#     ''' CALCULATE THE BLAND-ALTMAN MEASURES FOR THE PREDICTED VS GROUND TRUTH AREAS FOR A BATCH OF PREDICTIONS AND MASKS. RETURNS A DICT WITH ALL QUANTITIES RELEVANT TO CREATING THE BLAND-ALTMAN PLOT '''

#     # The outputs are logits, apply sigmoid and threshold at 0.5
#     predictions = torch.sigmoid(logits) > threshold  # Binarize predictions

#     # Flatten spatial dimensions while preserving batch
#     batch_size = predictions.shape[0]
#     predictions = predictions.view(batch_size, -1)
#     masks = masks.view(batch_size, -1)

#     gt_areas = masks.sum(dim=1).numpy()  # (B,)
#     pred_areas = predictions.sum(dim=1).numpy()  # (B,)

#     # means = 0.5 * (pred_areas + gt_areas)
#     # diffs = pred_areas - gt_areas
#     # bias = float(diffs.mean())
#     # sd = float(diffs.std(ddof=1)) if diffs.size > 1 else 0.0
#     # loa_low = bias - 1.96 * sd
#     # loa_high = bias + 1.96 * sd

#     return gt_areas, pred_areas