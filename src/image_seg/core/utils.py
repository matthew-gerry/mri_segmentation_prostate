# src/image_seg/core/utils.py
'''
Utils for prostate segmentation tutorial, including distance transform function and accuracy measures.
'''

import numpy as np
from scipy.ndimage import distance_transform_edt
import torch


def _get_logits(output):
    """INTERNAL HELPER FUNCTION TO SUPPORT BOTH TORCHVISION DICT OUTPUTS AND RAW TENSOR OUTPUTS, BASED ON THE BEHAVIOUR OF THE SPECIFIC MODEL """
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


def confusion_matrix(logits, masks, threshold=0.5):
    ''' CALCULATE THE CONFUSION MATRIX COMPONENTS (TP, FP, TN, FN) FOR A BATCH OF PREDICTIONS AND MASKS. RETURNS A DICT WITH THE COUNTS OF EACH COMPONENT '''

    # The outputs are logits, apply sigmoid and threshold at 0.5
    predictions = torch.sigmoid(logits) > threshold  # Binarize predictions

    # Flatten spatial dimensions while preserving batch
    batch_size = predictions.shape[0]
    predictions = predictions.view(batch_size, -1)
    masks = masks.view(batch_size, -1)

    TP = torch.sum((predictions == 1) & (masks == 1)).item()
    FP = torch.sum((predictions == 1) & (masks == 0)).item()
    TN = torch.sum((predictions == 0) & (masks == 0)).item()
    FN = torch.sum((predictions == 0) & (masks == 1)).item()

    return {"TP": TP, "FP": FP, "TN": TN, "FN": FN}


def precision_recall(conf):
    ''' CALCULATE THE PRECISION AND RECALL FROM THE CONFUSION MATRIX COMPONENTS '''

    # Grab the confusion matrix components from the input dict
    TP = conf["TP"]
    FP = conf["FP"]
    FN = conf["FN"]

    # Calculate precision and recall, handling division by zero cases
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0

    return precision, recall


def bland_altman_areas(logits, masks, threshold=0.5):
    ''' CALCULATE THE BLAND-ALTMAN MEASURES FOR THE PREDICTED VS GROUND TRUTH AREAS FOR A BATCH OF PREDICTIONS AND MASKS. RETURNS A DICT WITH ALL QUANTITIES RELEVANT TO CREATING THE BLAND-ALTMAN PLOT '''

    # The outputs are logits, apply sigmoid and threshold at 0.5
    predictions = torch.sigmoid(logits) > threshold  # Binarize predictions

    # Flatten spatial dimensions while preserving batch
    batch_size = predictions.shape[0]
    predictions = predictions.view(batch_size, -1)
    masks = masks.view(batch_size, -1)

    gt_areas = masks.sum(dim=1).numpy()  # (B,)
    pred_areas = predictions.sum(dim=1).numpy()  # (B,)

    return gt_areas, pred_areas