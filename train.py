"""Train myVLA: uv run train.py --config configs/v2.yaml.

Relative paths are interpreted from the project root, including when this script
is launched from another directory (for example, uv run ../../train.py ...).
"""

import argparse
import os
from pathlib import Path

from myvla.cli import add_train_arguments, run_training

PROJECT_ROOT = Path(__file__).resolve().parent


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Train V1 or V2 on LIBERO or MetaWorld MT50.",
        epilog="Relative config, data, model and output paths use the project root.",
    )
    add_train_arguments(parser)
    args = parser.parse_args(argv)
    os.chdir(PROJECT_ROOT)
    return run_training(args, parser)


if __name__ == "__main__":
    main()
