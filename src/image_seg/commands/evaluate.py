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
from image_seg.core.utils import _get_logits, dice_coefficient, confusion_matrix, precision_recall


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
def evaluate(cfg: EvalConfig, metrics: list[str]):
    """
    Evaluate the model and compute the metrics requested by the user.
    Supported:
      - 'dice'
      - 'pr'
      - 'confusion'
    Returns a dict suitable for printing or saving.
    """

    # Configure model and dataloader
    device = torch.device(cfg.device)
    ds = build_dataset(cfg)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=cfg.num_workers)
    model = load_model(cfg).to(device)
    model.eval()

    # Initialize accumulators for metrics
    dice_scores = []
    losses = []
    all_confusions = []   # list of TP/FP/TN/FN dicts
    all_precisions = []
    all_recalls = []

    # Evaluation loop
    for images, masks in loader:
        images = images.to(device)
        masks  = masks.to(device)  # (1,1,H,W)

        out = model(images)
        logits = _get_logits(out)

        H, W = masks.shape[-2:]

        # Upsample if model output smaller
        if logits.shape[-2:] != (H, W):
            logits = F.interpolate(logits, size=(H, W), mode="bilinear", align_corners=False)

        # Loss always computed
        loss = combined_loss(
            logits, masks,
            bce_weight=cfg.bce_weight,
            dice_weight=cfg.dice_weight,
            boundary_weight=cfg.boundary_weight
        )
        losses.append(float(loss.item()))

        # Convert model outputs to predicted mask
        probs = torch.sigmoid(logits)[0, 0]
        pred  = (probs > cfg.threshold).float()
        gt    = masks[0, 0].float()

        # Metrics
        if "dice" in metrics:
            # Append the dice coefficient converted from a 1x1 tensor to a scalar float
            dice_scores.append(float(dice_coefficient(pred, gt, threshold=cfg.threshold)))
            # dice_scores.append(dice_coefficient(pred, gt))

        if "confusion" in metrics or "pr" in metrics:
            conf = confusion_matrix(pred, gt, threshold=cfg.threshold)
            all_confusions.append(conf)

            if "pr" in metrics:
                precision, recall = precision_recall(conf)
                all_precisions.append(precision)
                all_recalls.append(recall)

    # Aggregate results from all samples
    results = {
        "mean_loss": float(np.mean(losses)),
        "losses": losses
    }

    if "dice" in metrics:
        results["mean_dice"] = float(np.mean(dice_scores))
        results["dice_scores"] = dice_scores

    if "confusion" in metrics:
        # Sum across all confusion matrices
        total_conf = {"TP":0,"FP":0,"TN":0,"FN":0}
        for c in all_confusions:
            for k in total_conf:
                total_conf[k] += c[k]
        results["confusion_matrix"] = total_conf
        results["per_confusion"] = all_confusions

    if "pr" in metrics:
        results["mean_precision"] = float(np.mean(all_precisions))
        results["mean_recall"]    = float(np.mean(all_recalls))
        results["precisions"] = all_precisions
        results["recalls"]    = all_recalls

    return results


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

    print("[evaluate] resolved config:")
    for k, v in cfg.__dict__.items():
        print(f"  - {k}: {v}")

    metrics = args.metrics if args.metrics is not None else ["dice"]

    # Compute metrics
    print(f"\n[evaluate] computing metrics: {metrics}\n")
    results = evaluate(cfg, metrics)

    # Print aggregate metrics to shell
    print("[evaluate] results:")
    print(f"  - Mean loss: {results['mean_loss']:.4f}")

    if "mean_dice" in results:
        print(f"  - Mean dice: {results['mean_dice']:.4f}")

    if "mean_precision" in results:
        print(f"  - Mean precision: {results['mean_precision']:.4f}")

    if "mean_recall" in results:
        print(f"  - Mean recall: {results['mean_recall']:.4f}")

    if "confusion_matrix" in results:
        cm = results["confusion_matrix"]
        print("  - Total confusion matrix:")
        print(f"      TP: {cm['TP']}")
        print(f"      FP: {cm['FP']}")
        print(f"      TN: {cm['TN']}")
        print(f"      FN: {cm['FN']}")


    # If path specified in config file, save per-sample metrics
    if args.save_metrics:
        import json
        save_path = args.save_metrics

        # Build a consolidated per-sample list
        per_sample = []

        # Determine number of samples
        n = len(results.get("losses", []))

        for i in range(n):
            entry = {
                "index": i,
                "loss": results["losses"][i],
            }
            if "dice_scores" in results:
                entry["dice"] = results["dice_scores"][i]
            if "precisions" in results:
                entry["precision"] = results["precisions"][i]
            if "recalls" in results:
                entry["recall"] = results["recalls"][i]
            if "confusion_matrix" in results and "per_confusion" in results:
                entry["confusion"] = results["per_confusion"][i]
            elif "confusion_matrix" not in results and "per_confusion" in results:
                entry["confusion"] = results["per_confusion"][i]

            per_sample.append(entry)

        # Wrap into a JSON object
        out_dict = {
            "aggregate": {
                k: v for k, v in results.items()
                if not isinstance(v, list)
            },
            "per_sample": per_sample,
        }

        with open(save_path, "w") as f:
            json.dump(out_dict, f, indent=2)

        print(f"\nSaved per-sample metrics to: {save_path}")
        
    return 0