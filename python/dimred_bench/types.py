from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


ArrayLike = np.ndarray


@dataclass(frozen=True)
class Dataset:
    name: str
    X: ArrayLike
    y: ArrayLike | None
    source: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EmbeddingResult:
    implementation: str
    embedding: ArrayLike | None
    status: str
    elapsed_sec: float
    peak_rss_mb: float | None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BenchmarkRow:
    dataset: str
    subset: int | None
    implementation: str
    run_id: int
    status: str
    elapsed_sec: float
    peak_rss_mb: float | None
    trustworthiness: float | None
    continuity: float | None
    knn_preservation: float | None
    silhouette: float | None
    stability_procrustes: float | None
    error: str | None


def as_path(path: str | Path) -> Path:
    return path if isinstance(path, Path) else Path(path)
