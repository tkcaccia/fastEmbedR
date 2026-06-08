from __future__ import annotations

import contextlib
import gzip
import os
import resource
import shutil
import tarfile
import threading
import time
import urllib.request
from pathlib import Path
from typing import Iterator

import numpy as np


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def default_cache_dir() -> Path:
    root = os.environ.get("DIMRED_BENCH_CACHE")
    if root:
        return ensure_dir(Path(root).expanduser())
    return ensure_dir(Path.home() / ".cache" / "dimred_bench")


def download(url: str, dest: Path, *, timeout: int = 120) -> Path:
    ensure_dir(dest.parent)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    with urllib.request.urlopen(url, timeout=timeout) as response, tmp.open("wb") as handle:
        shutil.copyfileobj(response, handle)
    tmp.replace(dest)
    return dest


def gunzip_file(src: Path, dest: Path) -> Path:
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    ensure_dir(dest.parent)
    with gzip.open(src, "rb") as fin, dest.open("wb") as fout:
        shutil.copyfileobj(fin, fout)
    return dest


def extract_tar(src: Path, dest_dir: Path) -> Path:
    marker = dest_dir / ".extracted"
    if marker.exists():
        return dest_dir
    ensure_dir(dest_dir)
    with tarfile.open(src, "r:*") as tar:
        tar.extractall(dest_dir, filter="data")
    marker.write_text("ok\n", encoding="utf-8")
    return dest_dir


def subset_arrays(
    X: np.ndarray,
    y: np.ndarray | None,
    subset: int | None,
    seed: int,
    stratify: bool = True,
) -> tuple[np.ndarray, np.ndarray | None]:
    if subset is None or subset >= X.shape[0]:
        return X, y
    rng = np.random.default_rng(seed)
    if y is None or not stratify:
        keep = np.sort(rng.choice(X.shape[0], size=subset, replace=False))
        return X[keep], None if y is None else y[keep]

    labels = np.asarray(y)
    selected: list[np.ndarray] = []
    classes, counts = np.unique(labels, return_counts=True)
    quotas = np.floor(counts / counts.sum() * subset).astype(int)
    remainder = subset - int(quotas.sum())
    if remainder > 0:
        order = np.argsort(-(counts / counts.sum() * subset - quotas))
        quotas[order[:remainder]] += 1
    for cls, quota in zip(classes, quotas):
        idx = np.flatnonzero(labels == cls)
        if quota > 0:
            selected.append(rng.choice(idx, size=min(quota, idx.size), replace=False))
    keep = np.sort(np.concatenate(selected))
    if keep.size > subset:
        keep = np.sort(rng.choice(keep, size=subset, replace=False))
    return X[keep], labels[keep]


def max_rss_mb() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if os.uname().sysname == "Darwin":
        return usage / (1024.0 * 1024.0)
    return usage / 1024.0


class MemorySampler:
    def __init__(self, interval_sec: float = 0.05) -> None:
        self.interval_sec = interval_sec
        self._running = False
        self._thread: threading.Thread | None = None
        self.peak_mb: float | None = None

    def _sample(self) -> None:
        while self._running:
            current = max_rss_mb()
            self.peak_mb = current if self.peak_mb is None else max(self.peak_mb, current)
            time.sleep(self.interval_sec)

    def __enter__(self) -> "MemorySampler":
        self.peak_mb = max_rss_mb()
        self._running = True
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *args: object) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=self.interval_sec * 4)
        self.peak_mb = max(self.peak_mb or 0.0, max_rss_mb())


@contextlib.contextmanager
def timed_memory() -> Iterator[tuple[float, MemorySampler]]:
    start = time.perf_counter()
    with MemorySampler() as sampler:
        yield start, sampler


def import_optional(module: str):
    try:
        return __import__(module, fromlist=["*"])
    except ImportError as exc:
        raise RuntimeError(f"Optional dependency `{module}` is not installed.") from exc
