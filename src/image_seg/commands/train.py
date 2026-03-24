# src/image_seg/commands/train.py
from __future__ import annotations

import os
import time
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional
from xml.parsers.expat import model

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

# Import functions from custom modules
from image_seg.core.data import MRIDataset
from image_seg.core.models import SimpleUNet, TLDeepLabV3MobileNet
from image_seg.core.losses import combined_loss
from image_seg.core.utils import dice_coefficient, _get_logits


def _ensure_dir(path: str):
    ''' Helper function to create a directory if it doesn't exist '''
    os.makedirs(path, exist_ok=True)
    return path

# -------- Config container --------

@dataclass
class TrainConfig:
    dataset: str               # 'promise12' | 'folders'
    train_split: str
    val_split: str
    download: bool             # whether to download the dataset if not present (only applies to online datasets like 'promise12')
    arch: str                  # 'unet' | 'deeplabv3-mnv3'
    pretrained: bool
    resize: int
    epochs: int
    batch_size: int
    lr: float                  # 'cpu' or 'cuda'
    weight_decay: float
    device: str
    save_dir: str              # where to write checkpoints/logs
    track_val_dice: bool
    bce_weight: float = 1.0
    dice_weight: float = 1.0
    boundary_weight: float = 0.0
    num_workers: int = 2
    threshold: float = 0.5


# -------- Dataset factory --------

def build_datasets(cfg: TrainConfig):
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
    
        base_train = Promise12MSBench(split=cfg.train_split, download=cfg.download, size=cfg.resize)
        base_val   = Promise12MSBench(split=cfg.val_split,   download=cfg.download, size=cfg.resize)

        train_ds = MRIDataset(base_train)
        val_ds   = MRIDataset(base_val)
        return train_ds, val_ds

    elif cfg.dataset != "promise12":
        # Return an error saying the project must be modified to load data from other sources
        RuntimeError("The tool is currently designed to perform segmentation on the Promise12 dataset available through medsegbench(https://medsegbench.github.io/). Modifications may be made to incorporate other datasets. Additional arguments to the tool may be necessary.")

# -------- Model factory --------

def build_model(cfg: TrainConfig):
    """
    Construct the model based on CLI flags.
    - 'unet': Simple U-Net from scratch model builder
    - 'deeplabv3-mnv3': transfer-learning class
    """
    if cfg.arch == "unet":
        model = SimpleUNet()
    elif cfg.arch == "deeplabv3-mnv3":
        # Transfer learning based model that adapts first conv 3->1, replaces final head, and partially unfreezes
        model = TLDeepLabV3MobileNet()
    else:
        raise ValueError(f"Unsupported arch: {cfg.arch}")

    return model


# -------- Training/validation steps ------------------


def train_one_epoch(model, loader, optimizer, device, cfg: TrainConfig):
    model.train()
    running = 0.0

    for images, masks in loader:
        images = images.to(device)   # (B,1,H,W)
        masks  = masks.to(device)    # (B,1,H,W)

        optimizer.zero_grad(set_to_none=True)
        out = model(images)
        logits = _get_logits(out)

        # Make sure logits size matches masks (most torchvision heads already upsample)
        # if logits.shape[-2:] != masks.shape[-2:]:
        #     logits = F.interpolate(logits, size=masks.shape[-2:], mode="bilinear", align_corners=False)

        loss = combined_loss(
            logits, masks,
            bce_weight=cfg.bce_weight,
            dice_weight=cfg.dice_weight,
            boundary_weight=cfg.boundary_weight  # 0 by default
        )
        loss.backward()
        optimizer.step()

        running += float(loss.item())

    return running / max(1, len(loader))

@torch.no_grad() # Decorator to tell PyTorch not to calculate gradients during this function, since it's for evaluation only
def validate_one_epoch(model, loader, device, cfg: TrainConfig):
    model.eval()
    running_loss = 0.0
    running_dice = 0.0

    for images, masks in loader:
        images = images.to(device)
        masks  = masks.to(device)

        out = model(images)
        logits = _get_logits(out)
        # if logits.shape[-2:] != masks.shape[-2:]:
        #     logits = F.interpolate(logits, size=masks.shape[-2:], mode="bilinear", align_corners=False)

        loss = combined_loss(
            logits, masks,
            bce_weight=cfg.bce_weight,
            dice_weight=cfg.dice_weight,
            boundary_weight=cfg.boundary_weight
        )

        dice = dice_coefficient(logits, masks, threshold=cfg.threshold)

        running_loss += float(loss.item())
        running_dice += dice

    return running_loss / max(1, len(loader)), running_dice / max(1, len(loader))


def run(args) -> int:
    """
    Entry point for `image-seg train ...`.
    For now, this echoes our arguments to confirm wiring.
    Later, we'll import your dataset/model/loss and call your training loop here.
    """
    # Resolve config
    save_dir = _ensure_dir(getattr(args, "save_dir", "./runs/image-seg"))

    cfg = TrainConfig(
        dataset=args.dataset,
        train_split=args.train_split,
        val_split=args.val_split,
        download=args.download,
        arch=args.arch,
        pretrained=args.pretrained,
        resize=args.resize,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        device=args.device,
        save_dir=save_dir,
        track_val_dice=args.track_val_dice,

        # Optional training hyperparameters with defaults specified in case they are not provided via CLI
        bce_weight=getattr(args, "bce_weight", 1.0),
        dice_weight=getattr(args, "dice_weight", 1.0),
        boundary_weight=getattr(args, "boundary_weight", 0.0),
        num_workers=getattr(args, "num_workers", 2),
        threshold=getattr(args, "threshold", 0.5),
    )

    # Load data and set up DataLoaders
    train_ds, val_ds = build_datasets(cfg)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,  num_workers=cfg.num_workers)
    val_loader   = DataLoader(val_ds,   batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)

    # Initialize model and optimizer
    device = torch.device(cfg.device)
    model = build_model(cfg).to(device)

    optimizer = torch.optim.Adam(
        (p for p in model.parameters() if p.requires_grad),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay
    )

    # Training loop with best checkpoint saving
    best_val = float("inf")
    best_path = os.path.join(cfg.save_dir, "best.pth")

    val_dice_over_epochs = []

    t0 = time.time()
    for epoch in range(1, cfg.epochs + 1):
        # Run training procedure packaged inside train_one_epoch function
        train_loss = train_one_epoch(model, train_loader, optimizer, device, cfg)
        # Evaluate on the validation data
        val_loss, val_dice = validate_one_epoch(model, val_loader, device, cfg)

        if cfg.track_val_dice:
            val_dice_over_epochs.append(val_dice)

        total_time = time.time() - t0
        print(f"Epoch {epoch}/{cfg.epochs}  "
              f"Train loss: {train_loss:.4f}  Val loss: {val_loss:.4f}  Val DICE: {val_dice:.4f}\n"
              f"Total elapsed time: [{int(total_time) // 60}:{(int(total_time) % 60):02d}]")

        # Save best model encountered through training, based on validation loss
        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "epoch": epoch,
                "val_loss": val_loss,
                "config": cfg.__dict__,
            }, best_path)
            print(f"  ↳ Saved best checkpoint to: {best_path}")

    if cfg.track_val_dice:
        np.save(os.path.join(cfg.save_dir, "val_dice_over_epochs.npy"), np.array(val_dice_over_epochs))

    print(f"[train] complete. Best val loss: {best_val:.4f} @ {best_path}")

    return 0