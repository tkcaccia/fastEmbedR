from __future__ import annotations

import inspect
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from .types import EmbeddingResult
from .utils import MemorySampler


RunnerFn = Callable[[np.ndarray, int, int], np.ndarray]


@dataclass(frozen=True)
class Implementation:
    name: str
    family: str
    description: str
    runner: RunnerFn


def _tsne_kwargs(
    *,
    n_components: int,
    perplexity: float,
    seed: int,
    method: str,
    n_iter: int,
) -> dict[str, object]:
    sig = inspect.signature(TSNE)
    kwargs: dict[str, object] = {
        "n_components": n_components,
        "perplexity": perplexity,
        "method": method,
        "init": "pca",
        "learning_rate": "auto",
        "random_state": seed,
    }
    if "max_iter" in sig.parameters:
        kwargs["max_iter"] = n_iter
    else:
        kwargs["n_iter"] = n_iter
    return kwargs


def run_sklearn_tsne_barnes_hut(X: np.ndarray, seed: int, n_components: int) -> np.ndarray:
    perplexity = min(30.0, max(5.0, (X.shape[0] - 1) / 3.0))
    model = TSNE(**_tsne_kwargs(
        n_components=n_components,
        perplexity=perplexity,
        seed=seed,
        method="barnes_hut",
        n_iter=750,
    ))
    return model.fit_transform(X)


def run_sklearn_tsne_exact(X: np.ndarray, seed: int, n_components: int) -> np.ndarray:
    if X.shape[0] > 5000:
        raise RuntimeError("sklearn exact t-SNE is disabled above 5,000 samples.")
    perplexity = min(30.0, max(5.0, (X.shape[0] - 1) / 3.0))
    model = TSNE(**_tsne_kwargs(
        n_components=n_components,
        perplexity=perplexity,
        seed=seed,
        method="exact",
        n_iter=750,
    ))
    return model.fit_transform(X)


def run_umap_learn(X: np.ndarray, seed: int, n_components: int) -> np.ndarray:
    try:
        import umap
    except ImportError as exc:
        raise RuntimeError("Optional dependency `umap-learn` is not installed.") from exc
    model = umap.UMAP(
        n_neighbors=15,
        n_components=n_components,
        min_dist=0.1,
        metric="euclidean",
        random_state=seed,
        n_epochs=None,
    )
    return model.fit_transform(X)


def run_opentsne_fft(X: np.ndarray, seed: int, n_components: int) -> np.ndarray:
    try:
        from openTSNE import TSNE as OpenTSNE
    except ImportError as exc:
        raise RuntimeError("Optional dependency `openTSNE` is not installed.") from exc
    perplexity = min(30.0, max(5.0, (X.shape[0] - 1) / 3.0))
    model = OpenTSNE(
        n_components=n_components,
        perplexity=perplexity,
        initialization="pca",
        negative_gradient_method="fft",
        random_state=seed,
        n_jobs=1,
        verbose=False,
    )
    return np.asarray(model.fit(X))


def run_pca_baseline(X: np.ndarray, seed: int, n_components: int) -> np.ndarray:
    model = PCA(n_components=n_components, random_state=seed)
    return model.fit_transform(X)


IMPLEMENTATIONS: dict[str, Implementation] = {
    "umap_learn": Implementation("umap_learn", "umap", "Python umap-learn", run_umap_learn),
    "sklearn_tsne_bh": Implementation("sklearn_tsne_bh", "tsne", "scikit-learn Barnes-Hut t-SNE", run_sklearn_tsne_barnes_hut),
    "sklearn_tsne_exact": Implementation("sklearn_tsne_exact", "tsne", "scikit-learn exact t-SNE, small datasets only", run_sklearn_tsne_exact),
    "opentsne_fft": Implementation("opentsne_fft", "tsne", "openTSNE FFT interpolation t-SNE", run_opentsne_fft),
    "pca": Implementation("pca", "baseline", "PCA baseline for metric calibration", run_pca_baseline),
}


def list_implementations() -> list[Implementation]:
    return list(IMPLEMENTATIONS.values())


def run_implementation(
    name: str,
    X: np.ndarray,
    *,
    seed: int,
    n_components: int = 2,
) -> EmbeddingResult:
    if name not in IMPLEMENTATIONS:
        choices = ", ".join(sorted(IMPLEMENTATIONS))
        raise KeyError(f"Unknown implementation `{name}`. Available implementations: {choices}")
    impl = IMPLEMENTATIONS[name]
    start = time.perf_counter()
    with MemorySampler() as memory:
        try:
            embedding = impl.runner(X, seed, n_components)
            status = "ok"
            error = None
        except Exception as exc:  # noqa: BLE001 - benchmark rows should capture failures
            embedding = None
            status = "skipped" if "not installed" in str(exc) or "disabled" in str(exc) else "error"
            error = str(exc)
    elapsed = time.perf_counter() - start
    return EmbeddingResult(
        implementation=name,
        embedding=embedding,
        status=status,
        elapsed_sec=elapsed,
        peak_rss_mb=memory.peak_mb,
        error=error,
        metadata={"family": impl.family, "description": impl.description},
    )
