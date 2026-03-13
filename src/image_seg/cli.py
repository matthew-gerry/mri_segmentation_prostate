# src/image_seg/cli.py

import argparse
from image_seg.commands.train import run as run_train

def build_parser():
    parser = argparse.ArgumentParser(
        prog="image_seg",
        description="Image segmentation CLI (minimal skeleton)"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # First tiny subcommand: 'hello' just to verify wiring works
    # p_hello = sub.add_parser("hello", help="Sanity check command")
    # p_hello.add_argument("--name", default="world", help="Name to greet")
    # p_hello.set_defaults(func=cmd_hello)

    # --- Arguments for train subcommand ---
    p_train = sub.add_parser("train", help="Train a model")
    
    # Data import settings
    p_train.add_argument("--dataset", required=True,
        choices=["promise12"],
        help="Current data source: 'promise12' via MedSegBench. Support for additional online or local datasets may be added through modifications to the source code."
    )
    p_train.add_argument("--train-split", default="train", help="Name of training split")
    p_train.add_argument("--val-split", default="val", help="Name of validation split")
    p_train.add_argument("--download", action="store_true", help="Whether to download the dataset if not present (only applies to online datasets like 'promise12')")
    
    # # Filesystem options (only used for folders)
    # p_train.add_argument("--data-root", help="Root folder containing images/masks for local datasets")
    # p_train.add_argument("--images-dir", default="images", help="[folders] subdir with images")
    # p_train.add_argument("--masks-dir",  default="masks",  help="[folders] subdir with masks")
    # p_train.add_argument("--img-ext",    default=".png",   help="[folders] image file extension")
    # p_train.add_argument("--mask-ext",   default=".png",   help="[folders] mask file extension")

    # Model choice
    p_train.add_argument(
        "--arch",
        choices=["unet", "deeplabv3-mnv3"],
        default="unet",
        help="Model architecture",
    )
    p_train.add_argument(
        "--pretrained",
        action="store_true",
        help="Use pretrained weights (ignored for UNet)",
    )

    # Basic training hyperparams
    p_train.add_argument("--resize", type=int, default=128, help="Square resize (e.g. 128)")
    p_train.add_argument("--epochs", type=int, default=10)
    p_train.add_argument("--batch-size", type=int, default=16)
    p_train.add_argument("--lr", type=float, default=1e-3)
    p_train.add_argument("--weight-decay", type=float, default=0.0)

    # Device
    p_train.add_argument("--device", choices=["cpu", "cuda"], default="cpu")

    # Data loading and training hyperparameters
    p_train.add_argument("--num-workers", type=int, default=2)
    p_train.add_argument("--bce-weight", type=float, default=1.0)
    p_train.add_argument("--dice-weight", type=float, default=1.0)
    p_train.add_argument("--boundary-weight", type=float, default=0.0)
    p_train.add_argument("--threshold", type=float, default=0.5)
    p_train.add_argument("--save-dir", default="./runs/image-seg")

    # Bind to function
    p_train.set_defaults(func=run_train)

    return parser

# def cmd_hello(args: argparse.Namespace) -> int:
#     # For testing that CLI wiring works
#     print(f"Hello, {args.name}! CLI wiring works.")
#     return 0

def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)

if __name__ == "__main__":
    raise SystemExit(main())
