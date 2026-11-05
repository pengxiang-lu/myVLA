"""Run the myVLA inspect command directly."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from myvla.cli import main

if __name__ == "__main__":
    main(["inspect", *sys.argv[1:]])
