# src/image_seg/cli.py

import argparse
import yaml
from image_seg.commands.train import run as run_train
from image_seg.commands.evaluate import run as run_evaluate

def build_parser():
    parser = argparse.ArgumentParser(
        prog="image_seg",
        description="Image segmentation CLI (minimal skeleton)"
    )
    subs = parser.add_subparsers(dest="command", required=True)

    # -------- Arguments for train subcommand --------
    p_train = subs.add_parser("train", help="Train a model")
    
    # Path to YAML config file (optional, can also specify options via CLI args)
    p_train.add_argument("--config", help="Path to YAML config file")

    # Data import settings
    p_train.add_argument("--dataset",
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
    p_train.add_argument("--track-val-dice", action="store_true",
        help="Whether to compute and track validation Dice coefficient during training (adds overhead, so optional)")

    # Bind to function
    p_train.set_defaults(func=run_train)


    # -------- Arguments for evaluate subcommand --------
    p_eval = subs.add_parser("evaluate", help="Evaluate a model")
    
    # Path to YAML config file
    p_eval.add_argument("--config", help="Path to YAML config file")

    # Data and model import settings
    p_eval.add_argument("--dataset", choices=["promise12"], default=None, help="Dataset to evaluate on (from YAML or CLI)")
    p_eval.add_argument("--split", default=None)
    p_eval.add_argument("--arch", choices=["unet", "deeplabv3-mnv3"], default=None, help="Model architecture")
    p_eval.add_argument("--checkpoint", default=None, help="Path to the trained model checkpoint to evaluate")
    
    # Device
    p_eval.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    
    # Data loading and evaluation hyperparams
    p_eval.add_argument("--resize", type=int, default=128, help="Square resize (e.g. 128)")
    p_eval.add_argument("--threshold", type=float, default=0.5)
    p_eval.add_argument("--num-workers", type=int, default=None, help="Number of DataLoader workers.")
    p_eval.add_argument("--metrics", nargs="+", default=None, help="Which metrics to compute (e.g. dice, pr, confusion).")
    p_eval.add_argument("--save-metrics", default=None, help="Optional: Path to save computed metrics as JSON file.")

    # Bind to function
    p_eval.set_defaults(func=run_evaluate)

    return parser


def load_yaml(path: str) -> dict:
    ''' Load YAML config file and return as dict '''
    with open(path, "r") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError("Config YAML must contain a top-level mapping (dict).")
    return data


def get_subparser(parser: argparse.ArgumentParser, name: str) -> argparse.ArgumentParser:
    ''' Utility to access a subparser by name (e.g. "train") for setting defaults from YAML '''
    subparsers_action = next(a for a in parser._actions if isinstance(a, argparse._SubParsersAction))
    return subparsers_action.choices[name]


def main() -> int:
    # Parse minimally to see command and config path
    parser = build_parser()
    args1, _unknown = parser.parse_known_args()

    # If train and config file provided, set YAML values as defaults on the train subparser
    if getattr(args1, "config", None):
        cfg = load_yaml(args1.config) # Load YAML config as dict
        defaults = cfg.get("defaults", {}) or {}
        section = cfg.get(args1.command, {}) or {}
        if not isinstance(defaults, dict) or not isinstance(section, dict):
            raise ValueError("In YAML, 'defaults' and each command section must be mappings (dicts).")
        
        # Merged configuration for the chosen command
        cfg_defaults = {**defaults, **section}

        # Rebuild to access subparser cleanly
        parser = build_parser()
        if args1.command and cfg_defaults:
            subs = get_subparser(parser, args1.command)

            # Filter to valid keys for this subparser
            valid_keys = {a.dest for a in subs._actions if a.dest not in (argparse.SUPPRESS, None)}
            defaults_for_sub = {k: v for k, v in cfg_defaults.items() if k in valid_keys}

            # Set YAML as defaults for relevant command; CLI flags will still override these
            subs.set_defaults(**defaults_for_sub)

        # Real parse with defaults from YAML applied
        args = parser.parse_args()
    else:
        # No config — do a normal parse
        args = parser.parse_args()

    # Validate required after defaults applied
    if getattr(args, "command", None) == "train":
        if not getattr(args, "dataset", None):
            parser.error("Missing required option: --dataset (provide it in --config YAML or via CLI)")

    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
