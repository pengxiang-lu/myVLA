"""Run the myVLA convert-metaworld command directly."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from myvla.cli import main

if __name__ == "__main__":
    main(["convert-metaworld", *sys.argv[1:]])
