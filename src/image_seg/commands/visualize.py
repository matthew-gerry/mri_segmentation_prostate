# src/image_seg/commands/visualize.py

from __future__ import annotations

import os
from typing import Optional

from dataclasses import dataclass
import numpy as np
from matplotlib import pyplot as plt

import torch
from torch.utils.data import DataLoader

# Import functions from custom modules
from image_seg.core.data import MRIDataset
# from image_seg.core.losses import combined_loss
from image_seg.core.models import SimpleUNet, TLDeepLabV3MobileNet
from image_seg.core.utils import _get_logits#, dice_coefficient, confusion_matrix, precision_recall

# -------- Config container for visualization --------
@dataclass
class VisualizeConfig:
    dataset: str                        # 'promise12'
    split: str                          # usually 'val'
    arch: str                           # 'unet' or 'deeplabv3-mnv3'
    checkpoint: str                     # path to .pt file
    threshold: float                    # prob threshold for binary mask
    device: str                         # cpu/cuda
    num_workers: int                    # dataloader workers
    fig_save_dir: str                   # where to save the visualization figure
    val_dice_path: Optional[str] = None # optional path to val dice history .npy file for plotting val dice vs. epoch
    resize: Optional[int] = None        # square resize (e.g. 128)
    num_samples: int = 4                # number of samples to visualize from the validation set

# -------- Dataset building function ---------
def build_dataset(cfg: VisualizeConfig):
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
    
# -------- Model loading function ---------
def load_model(cfg: VisualizeConfig) -> torch.nn.Module:
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

# -------- Visualization functions ---------

def overlay_contours(ax, img_gray, gt_mask, pred_mask, gt_color='lime', pred_color='magenta', legend_flag=False):
    """ DRAWS OUTLINE OF THE GROUND TRUTH (GREEN) AND PREDICTED (MAGENTA) CONTOURS ON TOP OF THE GRAYSCALE IMAGE """
    
    # Display grayscale image as background
    ax.imshow(img_gray, cmap='gray', vmin=0, vmax=255)

    # Plot contours of the ground truth and predicted masks
    # Smoothing not needed; ensure masks are 0/1 floats for contouring at level=0.5
    gt_cs = ax.contour(gt_mask.astype(float), levels=[0.5], colors=gt_color, linewidths=2.0)
    pred_cs = ax.contour(pred_mask.astype(float), levels=[0.5], colors=pred_color, linewidths=2.0)

    if legend_flag:
        # Get legend handles from the contour sets
        gt_handles, _   = gt_cs.legend_elements()
        pred_handles, _ = pred_cs.legend_elements()

        # Specify handles and labels for the legend
        handles = [gt_handles[0], pred_handles[0]]
        labels  = ['Ground Truth', 'Predicted']

        # Add legend with line samples visible
        ax.legend(handles, labels, loc='upper left', frameon=True, fontsize=12)

    ax.set_axis_off()

def visualize_predictions(cfg):
    """
    MAIN FUNCTION TO VISUALIZE PREDICTIONS ON THE VALIDATION SET
    """

    # Build dataset and dataloader
    dataset = build_dataset(cfg)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=cfg.num_workers)

    # Load model and set to eval mode
    model = load_model(cfg)
    model.to(cfg.device)
    model.eval()

    # Create figure
    fig, axes = plt.subplots(2, cfg.num_samples, figsize=(4 * cfg.num_samples, 9))

    # Loop through a few samples from the dataloader
    with torch.no_grad():
        for idx in range(cfg.num_samples):
            images, _ = dataset[idx] # Index directly to the dataset
            image_batch = images.unsqueeze(0).to(cfg.device)  # Add batch dimension and move to device
            outputs = model(image_batch)

            logits = _get_logits(outputs)
            
            # Prob -> binary
            probs = torch.sigmoid(logits).squeeze(0).squeeze(0).cpu().numpy()  # [H, W]
            pred_mask = (probs > cfg.threshold).astype(np.uint8)  # Binary mask

            # Raw image and ground truth mask from base dataset (PIL images)
            raw_pil, mask_pil = dataset.base_dataset[idx]
            raw_np = np.array(raw_pil.convert("L"))  # Grayscale image as numpy array
            gt_mask_np = (np.array(mask_pil) > 0).astype(np.uint8)  # Ground truth mask as numpy array

            # Plot with contours
            overlay_contours(axes[0, idx], raw_np, gt_mask_np, pred_mask, legend_flag=(idx==0))  # Only add legend for the first sample

            # Probability heatmap
            im = axes[1, idx].imshow(probs, cmap='magma', vmin=0, vmax=1)
            axes[1, idx].axis('off')
            fig.colorbar(im, ax=axes[1, idx], fraction=0.046, pad=0.04)
            if idx == 0:
                axes[1, idx].text(0.05 , 0.9, 'Predicted Probability Heatmap', transform=axes[1, idx].transAxes, fontsize=12, color='white', weight='bold')

    plt.suptitle("Visualization of model behaviour: transfer learning", fontsize=16, weight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.fig_save_dir, "prediction_visualization.png"), dpi=300)

    
def plot_val_dice_vs_epoch(cfg):
    """
    FUNCTION TO PLOT VALIDATION DICE COEFFICIENT VS. TRAINING EPOCH, IF VAL DICE WAS TRACKED AND SAVED DURING TRAINING
    """

    # Load validation dice history directory
    val_dice_history_path = cfg.val_dice_path

    val_dice_history = np.load(val_dice_history_path)

    epochs = np.arange(1, len(val_dice_history) + 1)
    # Plot val dice vs. epoch
    plt.figure(figsize=(6, 4))
    plt.plot(epochs, val_dice_history, marker='o')

    plt.xlim(1, len(val_dice_history))
    plt.ylim(min(val_dice_history), 1)
    plt.xticks(epochs)

    plt.title("Validation Dice Coefficient vs. Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Validation Dice Coefficient")
    plt.grid()
    plt.savefig(os.path.join(cfg.fig_save_dir, "val_dice_vs_epoch.png"), dpi=300)
    plt.close()

# -------- CLI entry point ---------

def run(args) -> int:
    """
    args: Namespace object from argparse with fields corresponding to VisualizeConfig
    """

    cfg = VisualizeConfig(
        dataset=args.dataset,
        split=args.split,
        arch=args.arch,
        checkpoint=args.checkpoint,
        threshold=args.threshold,
        device=args.device,
        num_workers=args.num_workers,
        resize=args.resize,
        num_samples=args.num_samples,
        fig_save_dir=args.fig_save_dir,
        val_dice_path=args.val_dice_path
    )

    # Create specified figure save directory if it doesn't exist
    os.makedirs(cfg.fig_save_dir, exist_ok=True)

    print(f"\n[visualize] running visualization with config:")
    for k, v in cfg.__dict__.items():
        print(f"  - {k}: {v}")

    visualizations = args.visualizations if args.visualizations is not None else ["preds"]

    # Generate visualizations
    print(f"\n[visualize] generating visualizations: {visualizations}\n")

    if "preds" in visualizations:
        visualize_predictions(cfg)

    if "val_dice_vs_epoch" in visualizations:
        plot_val_dice_vs_epoch(cfg)

    return 0