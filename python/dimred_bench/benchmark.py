from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from .datasets import load_dataset
from .metrics import embedding_metrics, procrustes_stability, sample_for_metrics
from .runners import run_implementation
from .types import BenchmarkRow, Dataset
from .utils import ensure_dir


@dataclass(frozen=True)
class BenchmarkConfig:
    datasets: tuple[str, ...] = ("iris", "digits")
    implementations: tuple[str, ...] = ("umap_learn", "sklearn_tsne_bh", "pca")
    subsets: dict[str, int | None] = field(default_factory=dict)
    repeats: int = 2
    seed: int = 42
    n_components: int = 2
    metric_sample: int = 3000
    metric_neighbors: int = 15
    pca_input_dims: int | None = 50
    cache_dir: str | None = None
    output_dir: str = "benchmark/python_results"
    save_embeddings: bool = False


def _prepare_input(dataset: Dataset, config: BenchmarkConfig) -> tuple[np.ndarray, dict[str, object]]:
    X = np.asarray(dataset.X, dtype=np.float32)
    meta: dict[str, object] = {"input_n": int(X.shape[0]), "input_dim": int(X.shape[1])}
    if config.pca_input_dims is not None and X.shape[1] > config.pca_input_dims:
        n_components = min(config.pca_input_dims, min(X.shape) - 1)
        X = PCA(n_components=n_components, random_state=config.seed).fit_transform(X).astype(np.float32)
        X = StandardScaler().fit_transform(X).astype(np.float32)
        meta["shared_pca_dims"] = int(n_components)
    return X, meta


def _write_manifest(config: BenchmarkConfig, output_dir: Path) -> None:
    manifest = asdict(config)
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")


def _save_embedding(output_dir: Path, dataset: str, implementation: str, run_id: int, embedding: np.ndarray) -> None:
    path = output_dir / "embeddings" / dataset
    ensure_dir(path)
    np.save(path / f"{implementation}_run{run_id}.npy", embedding.astype(np.float32, copy=False))


def _row_from_result(
    *,
    dataset_name: str,
    subset: int | None,
    implementation: str,
    run_id: int,
    result,
    metrics: dict[str, float | None] | None,
    stability: float | None,
) -> BenchmarkRow:
    metrics = metrics or {}
    return BenchmarkRow(
        dataset=dataset_name,
        subset=subset,
        implementation=implementation,
        run_id=run_id,
        status=result.status,
        elapsed_sec=result.elapsed_sec,
        peak_rss_mb=result.peak_rss_mb,
        trustworthiness=metrics.get("trustworthiness"),
        continuity=metrics.get("continuity"),
        knn_preservation=metrics.get("knn_preservation"),
        silhouette=metrics.get("silhouette"),
        stability_procrustes=stability,
        error=result.error,
    )


def run_benchmark(config: BenchmarkConfig) -> pd.DataFrame:
    output_dir = ensure_dir(Path(config.output_dir))
    _write_manifest(config, output_dir)
    rows: list[BenchmarkRow] = []

    for dataset_name in config.datasets:
        subset = config.subsets.get(dataset_name)
        dataset = load_dataset(dataset_name, subset=subset, seed=config.seed, cache_dir=config.cache_dir)
        X, prep_meta = _prepare_input(dataset, config)
        y = dataset.y
        X_metric, y_metric, metric_keep = sample_for_metrics(X, y, config.metric_sample, config.seed)
        references: dict[str, np.ndarray] = {}

        for implementation in config.implementations:
            for run_id in range(config.repeats):
                run_seed = config.seed + run_id
                result = run_implementation(
                    implementation,
                    X,
                    seed=run_seed,
                    n_components=config.n_components,
                )
                metrics = None
                stability = None
                if result.embedding is not None:
                    emb = np.asarray(result.embedding, dtype=np.float32)
                    emb_metric = emb[metric_keep]
                    metrics = embedding_metrics(
                        X_metric,
                        emb_metric,
                        y_metric,
                        n_neighbors=config.metric_neighbors,
                    )
                    if run_id == 0:
                        references[implementation] = emb_metric
                    else:
                        stability = procrustes_stability(references[implementation], emb_metric)
                    if config.save_embeddings:
                        _save_embedding(output_dir, dataset_name, implementation, run_id, emb)
                row = _row_from_result(
                    dataset_name=dataset_name,
                    subset=subset,
                    implementation=implementation,
                    run_id=run_id,
                    result=result,
                    metrics=metrics,
                    stability=stability,
                )
                rows.append(row)

        dataset_meta = {
            "name": dataset.name,
            "source": dataset.source,
            "metadata": dataset.metadata,
            "preprocessing": prep_meta,
        }
        (output_dir / f"{dataset_name}_metadata.json").write_text(
            json.dumps(dataset_meta, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    df = pd.DataFrame([asdict(row) for row in rows])
    df.to_csv(output_dir / "results.csv", index=False)
    summary_cols = [
        "elapsed_sec",
        "peak_rss_mb",
        "trustworthiness",
        "continuity",
        "knn_preservation",
        "silhouette",
        "stability_procrustes",
    ]
    summary = (
        df[df["status"] == "ok"]
        .groupby(["dataset", "subset", "implementation"], dropna=False)[summary_cols]
        .agg(["mean", "std"])
    )
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary = summary.reset_index()
    summary.to_csv(output_dir / "summary.csv", index=False)
    return df


def config_from_names(
    *,
    datasets: Iterable[str],
    implementations: Iterable[str],
    subsets: dict[str, int | None] | None = None,
    **kwargs: object,
) -> BenchmarkConfig:
    return BenchmarkConfig(
        datasets=tuple(datasets),
        implementations=tuple(implementations),
        subsets=subsets or {},
        **kwargs,
    )
