from __future__ import annotations

import gzip
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from scipy import io, sparse
from sklearn.datasets import fetch_openml, load_digits, load_iris
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.preprocessing import StandardScaler

from .types import Dataset
from .utils import default_cache_dir, download, extract_tar, gunzip_file, subset_arrays


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    description: str
    default_subset: int | None
    allowed_subsets: tuple[int | None, ...]
    loader: Callable[..., Dataset]


FASHION_MNIST_URLS = {
    "train_images": "https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/train-images-idx3-ubyte.gz",
    "train_labels": "https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/train-labels-idx1-ubyte.gz",
    "test_images": "https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/t10k-images-idx3-ubyte.gz",
    "test_labels": "https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/t10k-labels-idx1-ubyte.gz",
}

PBMC3K_URL = (
    "https://cf.10xgenomics.com/samples/cell-exp/1.1.0/"
    "pbmc3k/pbmc3k_filtered_gene_bc_matrices.tar.gz"
)


def _as_float32(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X)
    if X.dtype != np.float32:
        X = X.astype(np.float32, copy=False)
    return X


def _standardize_dense(X: np.ndarray) -> np.ndarray:
    return StandardScaler().fit_transform(_as_float32(X)).astype(np.float32, copy=False)


def _read_idx(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as handle:
        magic = struct.unpack(">I", handle.read(4))[0]
        dtype_code = (magic >> 8) & 0xFF
        ndim = magic & 0xFF
        if dtype_code != 0x08:
            raise ValueError(f"Unsupported IDX dtype in {path}: {dtype_code}")
        shape = tuple(struct.unpack(">I", handle.read(4))[0] for _ in range(ndim))
        data = np.frombuffer(handle.read(), dtype=np.uint8)
    return data.reshape(shape)


def load_iris_dataset(*, subset: int | None = None, seed: int = 42, **_: object) -> Dataset:
    raw = load_iris()
    X = _standardize_dense(raw.data)
    y = raw.target.astype(np.int64)
    X, y = subset_arrays(X, y, subset, seed)
    return Dataset("iris", X, y, "sklearn.load_iris", {"features": raw.feature_names})


def load_digits_dataset(*, subset: int | None = None, seed: int = 42, **_: object) -> Dataset:
    raw = load_digits()
    X = _standardize_dense(raw.data)
    y = raw.target.astype(np.int64)
    X, y = subset_arrays(X, y, subset, seed)
    return Dataset("digits", X, y, "sklearn.load_digits")


def load_mnist_dataset(
    *,
    subset: int | None = None,
    seed: int = 42,
    cache_dir: str | Path | None = None,
    **_: object,
) -> Dataset:
    cache = Path(cache_dir) if cache_dir else default_cache_dir()
    raw = fetch_openml("mnist_784", version=1, as_frame=False, data_home=str(cache / "openml"))
    X = _as_float32(raw.data) / 255.0
    y = raw.target.astype(np.int64)
    X, y = subset_arrays(X, y, subset, seed)
    return Dataset("mnist", X, y, "OpenML mnist_784 version 1")


def load_fashion_mnist_dataset(
    *,
    subset: int | None = None,
    seed: int = 42,
    cache_dir: str | Path | None = None,
    **_: object,
) -> Dataset:
    cache = (Path(cache_dir) if cache_dir else default_cache_dir()) / "fashion_mnist"
    paths = {key: download(url, cache / Path(url).name) for key, url in FASHION_MNIST_URLS.items()}
    X_train = _read_idx(paths["train_images"]).reshape(60000, -1)
    y_train = _read_idx(paths["train_labels"])
    X_test = _read_idx(paths["test_images"]).reshape(10000, -1)
    y_test = _read_idx(paths["test_labels"])
    X = np.vstack([X_train, X_test]).astype(np.float32) / 255.0
    y = np.concatenate([y_train, y_test]).astype(np.int64)
    X, y = subset_arrays(X, y, subset, seed)
    return Dataset("fashion_mnist", X, y, "Zalando Research Fashion-MNIST GitHub files")


def load_openml_tabular(
    openml_name: str,
    dataset_name: str,
    *,
    subset: int | None = None,
    seed: int = 42,
    cache_dir: str | Path | None = None,
    **_: object,
) -> Dataset:
    cache = Path(cache_dir) if cache_dir else default_cache_dir()
    raw = fetch_openml(openml_name, version="active", as_frame=False, data_home=str(cache / "openml"))
    X = _standardize_dense(raw.data)
    y = raw.target
    if y is not None:
        _, y = np.unique(y, return_inverse=True)
        y = y.astype(np.int64)
    X, y = subset_arrays(X, y, subset, seed)
    return Dataset(dataset_name, X, y, f"OpenML {openml_name}")


def load_pendigits_dataset(**kwargs: object) -> Dataset:
    return load_openml_tabular("pendigits", "pendigits", **kwargs)


def load_shuttle_dataset(**kwargs: object) -> Dataset:
    return load_openml_tabular("shuttle", "shuttle", **kwargs)


def _find_10x_dir(root: Path) -> Path:
    candidates = list(root.rglob("matrix.mtx")) + list(root.rglob("matrix.mtx.gz"))
    if not candidates:
        raise FileNotFoundError(f"No 10x matrix.mtx found under {root}")
    return candidates[0].parent


def _load_10x_matrix(matrix_dir: Path) -> sparse.csr_matrix:
    matrix_path = matrix_dir / "matrix.mtx"
    if not matrix_path.exists():
        matrix_path = gunzip_file(matrix_dir / "matrix.mtx.gz", matrix_dir / "matrix.mtx")
    mat = io.mmread(matrix_path).tocsr()
    return mat.T.tocsr().astype(np.float32)


def preprocess_single_cell(
    counts: sparse.csr_matrix,
    *,
    n_hvg: int = 2000,
    n_pcs: int = 50,
) -> np.ndarray:
    counts = counts.tocsr().astype(np.float32)
    cell_sums = np.asarray(counts.sum(axis=1)).ravel()
    cell_sums[cell_sums == 0] = 1.0
    normalized = counts.multiply(10000.0 / cell_sums[:, None])
    normalized.data = np.log1p(normalized.data)

    mean = np.asarray(normalized.mean(axis=0)).ravel()
    mean_sq = np.asarray(normalized.power(2).mean(axis=0)).ravel()
    var = np.maximum(mean_sq - mean * mean, 0.0)
    keep = np.argsort(var)[-min(n_hvg, normalized.shape[1]) :]
    hvg = normalized[:, keep]

    n_components = min(n_pcs, min(hvg.shape) - 1)
    if n_components < 2:
        raise ValueError("PBMC matrix is too small for PCA preprocessing.")
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    pcs = svd.fit_transform(hvg)
    return _standardize_dense(pcs)


def load_pbmc3k_dataset(
    *,
    subset: int | None = None,
    seed: int = 42,
    cache_dir: str | Path | None = None,
    n_hvg: int = 2000,
    n_pcs: int = 50,
    **_: object,
) -> Dataset:
    cache = (Path(cache_dir) if cache_dir else default_cache_dir()) / "pbmc3k"
    archive = download(PBMC3K_URL, cache / "pbmc3k_filtered_gene_bc_matrices.tar.gz")
    extract_tar(archive, cache / "extracted")
    matrix_dir = _find_10x_dir(cache / "extracted")
    X = preprocess_single_cell(_load_10x_matrix(matrix_dir), n_hvg=n_hvg, n_pcs=n_pcs)
    X, _ = subset_arrays(X, None, subset, seed, stratify=False)
    return Dataset(
        "pbmc3k",
        X,
        None,
        "10x Genomics PBMC 3k filtered gene-barcode matrix",
        {"n_hvg": n_hvg, "n_pcs": n_pcs},
    )


def load_medmnist_dataset(
    *,
    subset: int | None = None,
    seed: int = 42,
    medmnist_name: str = "pathmnist",
    cache_dir: str | Path | None = None,
    **_: object,
) -> Dataset:
    import medmnist
    from medmnist import INFO

    if medmnist_name not in INFO:
        raise ValueError(f"Unknown MedMNIST dataset `{medmnist_name}`.")
    data_class = getattr(medmnist, INFO[medmnist_name]["python_class"])
    root = str((Path(cache_dir) if cache_dir else default_cache_dir()) / "medmnist")
    parts = [data_class(split=split, root=root, download=True) for split in ("train", "val", "test")]
    X = np.concatenate([np.asarray(part.imgs) for part in parts], axis=0)
    y = np.concatenate([np.asarray(part.labels).reshape(-1) for part in parts], axis=0).astype(np.int64)
    X = X.reshape(X.shape[0], -1).astype(np.float32) / 255.0
    n_components = min(50, min(X.shape) - 1)
    if X.shape[1] > 100:
        X = PCA(n_components=n_components, random_state=seed).fit_transform(X).astype(np.float32)
        X = _standardize_dense(X)
    X, y = subset_arrays(X, y, subset, seed)
    return Dataset(f"medmnist_{medmnist_name}", X, y, "medmnist package")


DATASETS: dict[str, DatasetSpec] = {
    "iris": DatasetSpec("iris", "Iris sanity-check dataset", None, (None,), load_iris_dataset),
    "digits": DatasetSpec("digits", "scikit-learn handwritten digits", None, (None,), load_digits_dataset),
    "mnist": DatasetSpec("mnist", "OpenML MNIST 70k", 10000, (10000, 30000, 70000, None), load_mnist_dataset),
    "fashion_mnist": DatasetSpec(
        "fashion_mnist",
        "Fashion-MNIST 70k direct public files",
        10000,
        (10000, 30000, 70000, None),
        load_fashion_mnist_dataset,
    ),
    "pendigits": DatasetSpec("pendigits", "UCI/OpenML PenDigits", None, (None,), load_pendigits_dataset),
    "shuttle": DatasetSpec("shuttle", "UCI/OpenML Shuttle", 10000, (10000, 30000, None), load_shuttle_dataset),
    "pbmc3k": DatasetSpec("pbmc3k", "10x Genomics PBMC 3k with single-cell preprocessing", None, (None,), load_pbmc3k_dataset),
    "medmnist_pathmnist": DatasetSpec(
        "medmnist_pathmnist",
        "Optional MedMNIST PathMNIST",
        10000,
        (10000, None),
        lambda **kwargs: load_medmnist_dataset(medmnist_name="pathmnist", **kwargs),
    ),
    "medmnist_organcmnist": DatasetSpec(
        "medmnist_organcmnist",
        "Optional MedMNIST OrganCMNIST",
        10000,
        (10000, None),
        lambda **kwargs: load_medmnist_dataset(medmnist_name="organcmnist", **kwargs),
    ),
}


def list_datasets() -> list[DatasetSpec]:
    return list(DATASETS.values())


def load_dataset(
    name: str,
    *,
    subset: int | None = None,
    seed: int = 42,
    cache_dir: str | Path | None = None,
    **kwargs: object,
) -> Dataset:
    if name not in DATASETS:
        choices = ", ".join(sorted(DATASETS))
        raise KeyError(f"Unknown dataset `{name}`. Available datasets: {choices}")
    spec = DATASETS[name]
    actual_subset = spec.default_subset if subset == -1 else subset
    return spec.loader(subset=actual_subset, seed=seed, cache_dir=cache_dir, **kwargs)
