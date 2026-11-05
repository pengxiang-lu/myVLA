"""Download pinned CLIP via curl, with reusable chunks and official hash checks."""

import argparse
import hashlib
import json
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO = "openai/clip-vit-base-patch16"
REVISION = "57c216476eefef5ab752ec549e440a49ae4ae5f3"
WEIGHT_SIZE = 598641023
WEIGHT_SHA256 = "ec89c7b09c749a60aae3c9cd910516f24b58214a7df060b48962d14c469cfbf0"
# Git blob hashes from the official API at REVISION.
SMALL_FILES = {
    "config.json": "9f7102db4ae77c02982bfec1c16a63039fbc78db",
    "merges.txt": "76e821f1b6f0a9709293c3b6b51ed90980b3166b",
    "preprocessor_config.json": "5a12a1eb250987a4eee0e3e7d7338c4b22724be1",
    "special_tokens_map.json": "9bfb42aa97dcd61e89f279ccaee988bccb4fabae",
    "tokenizer.json": "580c79c6862f31d1f9bd08dd1a415ba0d0502cd9",
    "tokenizer_config.json": "a104224724ffb0721da7bee21ab59081b10fccef",
    "vocab.json": "4297ea6a8d2bae1fea8f48b45e257814dcb11f69",
}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_hash(path):
    data = path.read_bytes()
    return hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()


def fetch(url, output, extra=()):
    subprocess.run(
        [
            "curl", "--fail", "--location", "--silent", "--show-error",
            "--connect-timeout", "15", "--max-time", "240", "--retry", "3",
            "--retry-delay", "2", "--output", str(output), *extra, url,
        ],
        check=True,
    )


def download_chunk(directory, base, index, start, end):
    path = directory / f"{index:04d}.chunk"
    size = end - start + 1
    if path.exists() and path.stat().st_size == size:
        return size
    temp = path.with_suffix(".tmp")
    headers = path.with_suffix(".headers")
    # Older attempts may have received a complete body just before curl reported
    # an interrupted connection. Retain only bodies with the matching range header.
    expected = f"content-range: bytes {start}-{end}/{WEIGHT_SIZE}"
    if temp.exists() and (
        temp.stat().st_size > size or not headers.exists()
        or expected not in headers.read_text().lower().splitlines()
    ):
        raise RuntimeError(f"Invalid cached chunk {index}; move {temp} aside before retrying")
    for attempt in range(30):
        offset = temp.stat().st_size if temp.exists() else 0
        if offset == size:
            temp.replace(path)
            print(f"Chunk {index} complete", flush=True)
            return size
        segment = path.with_suffix(".segment")
        segment_headers = path.with_suffix(".segment-headers")
        segment.write_bytes(b"")
        segment_headers.write_text("")
        result = subprocess.run([
            "curl", "--http1.1", "--fail", "--location", "--silent", "--show-error",
            "--connect-timeout", "10", "--max-time", "45",
            "--range", f"{start + offset}-{end}",
            "--dump-header", str(segment_headers), "--output", str(segment),
            f"{base}/pytorch_model.bin?part={index}&offset={offset}",
        ])
        wanted = f"content-range: bytes {start + offset}-{end}/{WEIGHT_SIZE}"
        valid = (
            segment.exists() and segment_headers.exists()
            and wanted in segment_headers.read_text().lower().splitlines()
            and 0 < segment.stat().st_size <= size - offset
        )
        if valid:
            with temp.open("ab") as dest, segment.open("rb") as source:
                shutil.copyfileobj(source, dest)
            headers.write_text(expected + "\n")
            print(f"Chunk {index}: {temp.stat().st_size}/{size} bytes", flush=True)
        elif result.returncode == 0:
            raise RuntimeError(f"Server ignored requested range for chunk {index}")
        time.sleep(min(attempt + 1, 5))
    raise RuntimeError(f"Chunk {index} still incomplete; rerun to resume")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", default="models/clip-vit-base-patch16")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    if not shutil.which("curl"):
        parser.error("curl is required")
    out = Path(args.directory).resolve()
    out.mkdir(parents=True, exist_ok=True)
    base = f"https://huggingface.co/{REPO}/resolve/{REVISION}"
    for name, expected in SMALL_FILES.items():
        path = out / name
        if not path.exists() or git_hash(path) != expected:
            temp = path.with_suffix(path.suffix + ".tmp")
            fetch(f"{base}/{name}", temp)
            if git_hash(temp) != expected:
                raise RuntimeError(f"Official Git blob hash mismatch: {name}")
            temp.replace(path)
        print(f"Verified {name}", flush=True)
    weight = out / "pytorch_model.bin"
    if not weight.exists() or sha256(weight) != WEIGHT_SHA256:
        chunks = out / ".chunks"
        chunks.mkdir(exist_ok=True)
        chunk_size = 16 * 1024 * 1024
        ranges = list(enumerate(range(0, WEIGHT_SIZE, chunk_size)))
        completed = 0
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [
                pool.submit(
                    download_chunk, chunks, base, i, start,
                    min(start + chunk_size, WEIGHT_SIZE) - 1,
                )
                for i, start in ranges
            ]
            for future in as_completed(futures):
                completed += future.result()
                print(f"Weights: {completed / WEIGHT_SIZE:.1%} verified-size chunks", flush=True)
        assembled = out / "pytorch_model.bin.assembled"
        with assembled.open("wb") as stream:
            for i, _ in ranges:
                with (chunks / f"{i:04d}.chunk").open("rb") as source:
                    shutil.copyfileobj(source, stream)
        if sha256(assembled) != WEIGHT_SHA256:
            raise RuntimeError("Weight SHA256 mismatch; final model was not replaced")
        assembled.replace(weight)
    (out / "download_verified.json").write_text(json.dumps({
        "repo": REPO, "revision": REVISION, "sha256": WEIGHT_SHA256,
    }, indent=2))
    print(f"CLIP ready, SHA256 verified: {out}", flush=True)


if __name__ == "__main__":
    main()
