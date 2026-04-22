'''
Definitions for various loss functions used in different attempts at training the model.
'''

import torch
import torch.nn as nn
from image_seg.core.utils import distance_transform

def bce_loss(logits, targets, pos_weight=1.0):
    '''BINARY CROSS-ENTROPY LOSS WITH LOGITS '''
    # pos_weight > 1 gives more weight to positive class (prostate), helping with class imbalance
    bce_loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))
    return bce_loss(logits, targets)

def dice_loss(logits, targets, eps=1e-7):
    """
    MEASURE OF OVERLAP BETWEEN TWO SAMPLES. IT RANGES FROM 0 TO 1, WHERE 1 MEANS PERFECT OVERLAP AND 0 MEANS NO OVERLAP
     
    logits: (B, 1, H, W) raw outputs
    targets: (B, 1, H, W) binary masks
    eps: small constant to avoid division by zero
    """
    prob = logits.sigmoid()
    prob   = prob.view(prob.size(0), -1)
    targets = targets.view(targets.size(0), -1)
    intersection = (prob * targets).sum(dim=1) # Implement dice loss per-sample (sum over specific dimension)
    union = prob.sum(dim=1) + targets.sum(dim=1)
    dice = (2 * intersection + eps) / (union + eps)
    return 1 - dice.mean() # Subtract from 1 to convert to a loss (we want to maximize dice)


### BOUNDARY-SPECIFIC LOSS FUNCTION CONTRIBUTIONS ###


def boundary_loss_from_logits(logits, sdf, max_dist=20.0):
    """
    BOUNDARY LOSS GIVEN LOGITS (WHICH ARE THE RAW MODEL OUTPUTS) AND THE SIGNED DISTANCE MAP
    
    logits: (B, 1, H, W) raw outputs
    sdf:    (B, 1, H, W) signed distance map (float)
    max_dist: maximum distance from boundary to consider for normalization (so pixel far from the boundary do not dominate the loss)
    """
    probs = torch.sigmoid(logits)
    sdf = torch.clamp(sdf, -max_dist, max_dist) / max_dist  # now in [-1, 1]
    # boundary loss is mean of p(x) * sdf(x)
    return torch.mean(probs * sdf)


def boundary_band_loss(logits, targets, sdf, max_dist=20.0):
    """
    BOUNDARY THAT EMPHASIZES ERRORS NEAR THE BOUNDARY, WHICH IS WHERE THE ISSUES TEND TO BE
    
    logits: (B, 1, H, W) raw outputs
    sdf:    (B, 1, H, W) signed distance map (float)
    max_dist: maximum distance from boundary to consider for normalization (so pixel far from the boundary do not dominate the loss)
    """
    probs = torch.sigmoid(logits)
    # weights high near boundary (distance ~0), low far away
    w = 1.0 - torch.clamp(torch.abs(sdf), 0, max_dist) / max_dist  # in [0,1]
    # focus on mistakes near boundary
    return torch.mean(w * torch.abs(probs - targets))



def hausdorff_dt_loss(logits, targets, fg_dt, bg_dt):
    """
    HAUSDORFF DISTANCE-BASED LOSS THAT PENALIZES FALSE POSITIVES AND FALSE NEGATIVES BASED ON THEIR DISTANCE TO THE TRUE BOUNDARY.
    GOOD AT ADDRESSING WHEN "CHUNKS" OF THE SEGMENTATION ARE MISSING OR EXTRA.

    logits: (B,1,H,W)
    targets:   (B,1,H,W) binary mask
    fg_dt:  (B,1,H,W) distance-transform outside the foreground (FN penalty field)
    bg_dt:  (B,1,H,W) distance-transform inside the foreground (FP penalty field)
    """
    probs = torch.sigmoid(logits)

    # Convert targets to numpy for distance transform calculations
    # targets = targets.detach().cpu().numpy()

    # false positives: predict 1 where targets = 0
    fp = probs * (1 - targets)
    # false negatives: predict 0 where targets = 1
    fn = (1 - probs) * targets

    # Convert distance map numpy arrays to tensors
    fg_dt = torch.tensor(fg_dt, dtype=torch.float32)
    bg_dt = torch.tensor(bg_dt, dtype=torch.float32)
    
    loss_fp = (fp * fg_dt).mean()
    loss_fn = (fn * bg_dt).mean()

    return loss_fp + loss_fn


# Combine BCE, Dice, and Boundary losses
def combined_loss(logits, targets, bce_weight=1.0, dice_weight=1.0, boundary_weight=1.0):
    """ COMBINATION OF DIFFERENT CONTRIBUTIONS TO THE LOSS FUNCTION WITH WEIGHTS SPECIFIED """
    bce = bce_loss(logits, targets)
    dice = dice_loss(logits, targets)

    fg_dt, bg_dt = distance_transform(targets.detach().cpu().numpy().squeeze(1))  # (B,H,W) numpy arrays
    bndry_loss = hausdorff_dt_loss(logits, targets, fg_dt, bg_dt)

    # sdf = fg_dt - bg_dt  # (B,H,W) signed distance map
    # bndry_loss = boundary_band_loss(logits, targets, sdf)

    return bce_weight * bce + dice_weight * dice + boundary_weight * bndry_loss
