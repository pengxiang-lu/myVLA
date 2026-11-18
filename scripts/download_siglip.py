"""Download the pinned SigLIP Base checkpoint used by V4 (about 813 MB)."""

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download

REPO = "google/siglip-base-patch16-224"
REVISION = "7fd15f0689c79d79e38b1c2e2e2370a7bf2761ed"
FILES = [
    "config.json", "model.safetensors", "preprocessor_config.json",
    "tokenizer_config.json", "special_tokens_map.json", "spiece.model", "tokenizer.json",
]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", default="models/siglip-base-patch16-224")
    parser.add_argument("--endpoint", help="Optional Hugging Face mirror URL")
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error("--workers must be positive")
    return snapshot_download(
        repo_id=REPO, revision=REVISION, local_dir=str(Path(args.directory).resolve()),
        allow_patterns=FILES, endpoint=args.endpoint, max_workers=args.workers,
    )


if __name__ == "__main__":
    main()
