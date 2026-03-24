# src/image_seg/commands/evaluate.py

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Tuple, Optional

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np

# Import functions from custom modules
from image_seg.core.data import MRIDataset
from image_seg.core.losses import combined_loss
from image_seg.core.models import SimpleUNet, TLDeepLabV3MobileNet
from image_seg.core.utils import dice_coefficient, _get_logits


# -------- Config container for evaluation --------
@dataclass
class EvalConfig:
    dataset: str                # 'promise12'
    split: str                  # usually 'val'
    arch: str                   # 'unet' or 'deeplabv3-mnv3'
    checkpoint: str             # path to .pt file
    threshold: float            # prob threshold
    device: str                 # cpu/cuda
    num_workers: int            # dataloader workers
    # Optional: keep consistent with train()
    resize: Optional[int] = None
    bce_weight: float = 1.0
    dice_weight: float = 1.0
    boundary_weight: float = 0.0


def build_dataset(cfg: EvalConfig):
    """
    BUILD TRAINING AND VALIDATION DATASETS, SET UP TO USE THE Promise12MSBench DATASET AS AN EXAMPLE
    """

    if cfg.dataset == "promise12":
        try:
            from medsegbench import Promise12MSBench
        except Exception as e:
            raise RuntimeError(
                "Dataset 'promise12' requires the optional dependency medsegbench. "
                "Install it via: pip install image-seg[promise12]."
            ) from e
    

        base_ds = Promise12MSBench(
                split=cfg.split,
                download=True,
                size=cfg.resize if cfg.resize else 128,  # fallback
            )
        return MRIDataset(base_ds)

    elif cfg.dataset != "promise12":
        # Return an error saying the project must be modified to load data from other sources
        raise RuntimeError("The tool is currently designed to perform segmentation on the Promise12 dataset available through medsegbench(https://medsegbench.github.io/). Modifications may be made to incorporate other datasets. Additional arguments to the tool may be necessary.")
    

def load_model(cfg: EvalConfig) -> torch.nn.Module:
    """
    Build a model of the appropriate architecture, then load checkpoint weights.
    For now we support:
        - UNet (scratch)
        - SimpleDeepLabSeg (MobileNetV3 TL)
    """

    if cfg.arch == "unet":
        model = SimpleUNet()
    elif cfg.arch == "deeplabv3-mnv3":
        model = TLDeepLabV3MobileNet()
    else:
        raise ValueError(f"Unsupported architecture '{cfg.arch}' specified in config.")
    
    # Load checkpoint
    ckpt = torch.load(cfg.checkpoint, map_location="cpu")
    state = ckpt["model_state"]
    model.load_state_dict(state)

    return model


# -------- Evaluation routine --------

@torch.no_grad()
def evaluate(cfg: EvalConfig) -> dict:
    """
    Runs one full evaluation on the specified split and returns a dict with:
        - mean_loss
        - mean_dice
        - per-sample dice list
    """

    # Set up model and configure DataLoader
    device = torch.device(cfg.device)
    ds = build_dataset(cfg)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=cfg.num_workers)

    model = load_model(cfg).to(device)
    model.eval()

    # Evaluation loop
    dice_scores = []
    losses = []

    for images, masks in loader:
        images = images.to(device)   # (1,1,H,W)
        masks = masks.to(device)     # (1,1,H,W)

        out = model(images)
        logits = _get_logits(out)
        H, W = masks.shape[-2:]

        # Upsample if model output smaller
        if logits.shape[-2:] != (H, W):
            logits = F.interpolate(logits, size=(H, W), mode="bilinear", align_corners=False)

        # Compute loss
        loss = combined_loss(
            logits, masks,
            bce_weight=cfg.bce_weight,
            dice_weight=cfg.dice_weight,
            boundary_weight=cfg.boundary_weight
        )
        losses.append(float(loss.item()))

        # Convert to binary mask but keep as tensors for dice computation
        probs = torch.sigmoid(logits)[0, 0]  # (H,W
        pred = (probs > cfg.threshold).float()
        gt = masks[0, 0].float()

        # Compute dice score
        dice = dice_coefficient(pred, gt)
        dice_scores.append(dice)

    return {
        "mean_loss": float(np.mean(losses)),
        "mean_dice": float(np.mean(dice_scores)),
        "dice_scores": dice_scores
        # "losses": losses,
    }


# -------- CLI entry point --------

def run(args) -> int:
    """
    args: Namespace from argparse after config merging
    """
    cfg = EvalConfig(
        dataset=args.dataset,
        split=args.split,
        arch=args.arch,
        checkpoint=args.checkpoint,
        threshold=args.threshold,
        device=args.device,
        num_workers=args.num_workers,
        resize=args.resize,
        bce_weight=getattr(args, "bce_weight", 1.0),
        dice_weight=getattr(args, "dice_weight", 1.0),
        boundary_weight=getattr(args, "boundary_weight", 0.0),
    )

    print("[evaluate] config:")
    for k, v in cfg.__dict__.items():
        print(f"  - {k}: {v}")

    results = evaluate(cfg)

    print("\n[evaluate] results:")
    print(f"  Mean Loss: {results['mean_loss']:.4f}")
    print(f"  Mean Dice: {results['mean_dice']:.4f}")

    return 0
