import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def downloader():
    path = Path(__file__).resolve().parents[1] / "scripts/download_clip.py"
    spec = importlib.util.spec_from_file_location("download_clip", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_resume_partial_range_after_connection_reset(downloader, tmp_path, monkeypatch):
    monkeypatch.setattr(downloader, "WEIGHT_SIZE", 8)
    monkeypatch.setattr(downloader.time, "sleep", lambda _: None)
    (tmp_path / "0000.tmp").write_bytes(b"ab")
    (tmp_path / "0000.headers").write_text("content-range: bytes 0-7/8\n")
    requested = []

    def fake_curl(args):
        selected = args[args.index("--range") + 1]
        requested.append(selected)
        start, end = map(int, selected.split("-"))
        Path(args[args.index("--dump-header") + 1]).write_text(
            f"HTTP/1.1 206 Partial Content\nContent-Range: bytes {start}-{end}/8\n"
        )
        Path(args[args.index("--output") + 1]).write_bytes(b"abcdefgh"[start:min(start + 3, 8)])
        return SimpleNamespace(returncode=18)

    monkeypatch.setattr(downloader.subprocess, "run", fake_curl)
    assert downloader.download_chunk(tmp_path, "https://example.com", 0, 0, 7) == 8
    assert requested == ["2-7", "5-7"]
    assert (tmp_path / "0000.chunk").read_bytes() == b"abcdefgh"


def test_reject_server_ignoring_range(downloader, tmp_path, monkeypatch):
    def fake_curl(args):
        Path(args[args.index("--dump-header") + 1]).write_text("HTTP/1.1 200 OK\n")
        Path(args[args.index("--output") + 1]).write_bytes(b"wrong body")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(downloader.subprocess, "run", fake_curl)
    with pytest.raises(RuntimeError, match="ignored requested range"):
        downloader.download_chunk(tmp_path, "https://example.com", 0, 2, 7)
    assert not (tmp_path / "0000.chunk").exists()
