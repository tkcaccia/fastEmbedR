#!/usr/bin/env python3
"""Download and preprocess Kaggle-only datasets used in the TriMap paper.

The default fastEmbedR benchmark deliberately avoids authenticated sources.
This helper is for optional paper-replication runs where Kaggle credentials are
available. It never prints credentials and writes compact NumPy `.npz` files
that the R benchmark scripts can load through reticulate.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Iterable


DATASETS = {
    "usps": {
        "slug": "bistaumanga/usps-dataset",
        "processed": "kaggle_usps.npz",
        "paper_name": "USPS",
    },
    "lyrics": {
        "slug": "gyani95/380000-lyrics-from-metrolyrics",
        "processed": "kaggle_metrolyrics_svd100.npz",
        "paper_name": "360K+ Lyrics",
    },
}


def info(message: str) -> None:
    print(message, flush=True)


def die(message: str) -> None:
    raise SystemExit(message)


def have_kaggle_credentials() -> bool:
    home = Path.home()
    return any(
        [
            bool(os.environ.get("KAGGLE_API_TOKEN")),
            (home / ".kaggle" / "access_token").exists(),
            (home / ".kaggle" / "kaggle.json").exists(),
        ]
    )


def require_python_modules(names: Iterable[str]) -> None:
    missing: list[str] = []
    for name in names:
        try:
            __import__(name)
        except Exception:
            missing.append(name)
    if missing:
        joined = " ".join(missing)
        die(
            "Missing Python package(s): "
            + ", ".join(missing)
            + f"\nInstall them with: {sys.executable} -m pip install --user {joined}"
        )


def run_kaggle_download(slug: str, raw_dir: Path, force: bool = False) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    marker = raw_dir / ".download_complete"
    if marker.exists() and not force:
        return
    if not have_kaggle_credentials():
        die(
            "Kaggle credentials not found. Put the current API token in "
            "~/.kaggle/access_token, set KAGGLE_API_TOKEN, or use ~/.kaggle/kaggle.json."
        )
    cmd = [
        sys.executable,
        "-m",
        "kaggle",
        "datasets",
        "download",
        "-d",
        slug,
        "-p",
        str(raw_dir),
        "--unzip",
    ]
    info(f"Downloading Kaggle dataset {slug} ...")
    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError as exc:
        raise SystemExit(
            f"Kaggle CLI is not installed. Install it with: {sys.executable} -m pip install --user kaggle"
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise SystemExit(
            "Kaggle download failed. Check that the dataset terms are accepted "
            "on Kaggle and that your token is valid."
        ) from exc
    marker.write_text(slug + "\n", encoding="utf-8")


def write_npz(path: Path, x, labels, metadata: dict) -> None:
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        x=x.astype("float32", copy=False),
        labels=np.asarray(labels).astype(str),
        metadata=json.dumps(metadata, sort_keys=True),
    )
    (path.with_suffix(path.suffix + ".json")).write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    info(f"Wrote {path}")


def h5_get(group, names: Iterable[str]):
    for name in names:
        if name in group:
            return group[name][()]
    return None


def prepare_usps(raw_dir: Path, out_path: Path, force: bool = False) -> None:
    if out_path.exists() and not force:
        info(f"Already prepared: {out_path}")
        return
    require_python_modules(["numpy", "h5py"])
    import h5py
    import numpy as np
    import pandas as pd

    h5_files = sorted(raw_dir.rglob("*.h5")) + sorted(raw_dir.rglob("*.hdf5"))
    if h5_files:
        with h5py.File(h5_files[0], "r") as handle:
            if "train" in handle and "test" in handle:
                train = handle["train"]
                test = handle["test"]
                x_train = h5_get(train, ["data", "x", "X", "images"])
                y_train = h5_get(train, ["target", "targets", "label", "labels", "y"])
                x_test = h5_get(test, ["data", "x", "X", "images"])
                y_test = h5_get(test, ["target", "targets", "label", "labels", "y"])
                if x_train is not None and y_train is not None and x_test is not None and y_test is not None:
                    x = np.vstack([x_train.reshape((x_train.shape[0], -1)), x_test.reshape((x_test.shape[0], -1))])
                    labels = np.concatenate([np.asarray(y_train).ravel(), np.asarray(y_test).ravel()])
                    write_npz(
                        out_path,
                        x,
                        labels,
                        {
                            "dataset": "USPS",
                            "source": DATASETS["usps"]["slug"],
                            "paper": "TriMap; LocalMAP",
                            "n": int(x.shape[0]),
                            "p": int(x.shape[1]),
                            "format": "hdf5_train_test",
                        },
                    )
                    return
            x = h5_get(handle, ["data", "x", "X", "images"])
            labels = h5_get(handle, ["target", "targets", "label", "labels", "y"])
            if x is not None and labels is not None:
                x = x.reshape((x.shape[0], -1))
                write_npz(
                    out_path,
                    x,
                    np.asarray(labels).ravel(),
                    {
                        "dataset": "USPS",
                        "source": DATASETS["usps"]["slug"],
                        "paper": "TriMap; LocalMAP",
                        "n": int(x.shape[0]),
                        "p": int(x.shape[1]),
                        "format": "hdf5_flat",
                    },
                )
                return

    csv_files = sorted(raw_dir.rglob("*.csv"))
    if csv_files:
        frames = []
        for csv in csv_files:
            try:
                frames.append(pd.read_csv(csv))
            except Exception:
                pass
        if frames:
            dat = max(frames, key=len)
            label_col = next((c for c in dat.columns if c.lower() in {"label", "labels", "target", "digit", "y"}), dat.columns[-1])
            labels = dat[label_col].to_numpy()
            x = dat.drop(columns=[label_col]).select_dtypes(include=["number"]).to_numpy()
            if x.size > 0:
                write_npz(
                    out_path,
                    x,
                    labels,
                    {
                        "dataset": "USPS",
                        "source": DATASETS["usps"]["slug"],
                        "paper": "TriMap; LocalMAP",
                        "n": int(x.shape[0]),
                        "p": int(x.shape[1]),
                        "format": "csv",
                    },
                )
                return

    die("Could not find a usable USPS HDF5 or CSV file in " + str(raw_dir))


def grouped_genre(value: object) -> str | None:
    text = str(value).strip().lower()
    if not text or text in {"nan", "none", "not available", "other"}:
        return None
    if "rock" in text or "metal" in text:
        return "rock_metal"
    if "pop" in text or "r&b" in text or "rnb" in text:
        return "pop_rnb"
    if "hip" in text or "rap" in text:
        return "hiphop_rap"
    if "country" in text or "folk" in text:
        return "country_folk"
    if "elect" in text or "dance" in text:
        return "electronic"
    if "jazz" in text or "blues" in text:
        return "jazz_blues"
    if "indie" in text or "alternative" in text:
        return "indie_alt"
    return None


def prepare_lyrics(
    raw_dir: Path,
    out_path: Path,
    svd_dims: int,
    max_features: int,
    max_rows: int,
    seed: int,
    force: bool = False,
) -> None:
    if out_path.exists() and not force:
        info(f"Already prepared: {out_path}")
        return
    require_python_modules(["numpy", "pandas", "sklearn"])
    import numpy as np
    import pandas as pd
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer

    csv_files = sorted(raw_dir.rglob("*.csv"))
    if not csv_files:
        die("Could not find a lyrics CSV file in " + str(raw_dir))
    dat = None
    for csv in csv_files:
        try:
            candidate = pd.read_csv(csv, low_memory=False, on_bad_lines="skip")
        except TypeError:
            candidate = pd.read_csv(csv, low_memory=False)
        except Exception:
            continue
        lower = {c.lower(): c for c in candidate.columns}
        if any(name in lower for name in ["lyrics", "lyric", "text"]) and "genre" in lower:
            dat = candidate
            break
    if dat is None:
        die("Could not identify lyrics and genre columns in " + str(raw_dir))

    lower = {c.lower(): c for c in dat.columns}
    text_col = next(lower[name] for name in ["lyrics", "lyric", "text"] if name in lower)
    genre_col = lower["genre"]
    work = dat[[text_col, genre_col]].dropna()
    work.columns = ["lyrics", "genre"]
    work["label"] = work["genre"].map(grouped_genre)
    work = work.dropna(subset=["label"])
    work = work[work["lyrics"].astype(str).str.len() > 20]
    if max_rows and max_rows > 0 and len(work) > max_rows:
        work = work.sample(n=max_rows, random_state=seed)

    vectorizer = TfidfVectorizer(
        max_features=max_features,
        min_df=5,
        max_df=0.80,
        stop_words="english",
        dtype=np.float32,
        sublinear_tf=True,
    )
    info(f"Vectorizing {len(work)} lyrics ...")
    x_sparse = vectorizer.fit_transform(work["lyrics"].astype(str).to_numpy())
    dims = min(svd_dims, max(2, x_sparse.shape[1] - 1))
    info(f"Running TruncatedSVD to {dims} dimensions ...")
    svd = TruncatedSVD(n_components=dims, random_state=seed)
    x = svd.fit_transform(x_sparse).astype("float32", copy=False)
    write_npz(
        out_path,
        x,
        work["label"].to_numpy(),
        {
            "dataset": "360K+ Lyrics",
            "source": DATASETS["lyrics"]["slug"],
            "paper": "TriMap",
            "n": int(x.shape[0]),
            "p": int(x.shape[1]),
            "svd_dims": int(dims),
            "tfidf_max_features": int(max_features),
            "max_rows": int(max_rows),
            "genre_grouping": "7_group_paper_approximation",
            "explained_variance_ratio_sum": float(np.sum(svd.explained_variance_ratio_)),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["usps", "lyrics", "all"], default="all")
    parser.add_argument("--cache-dir", default="results/rjournal_benchmark/cache/kaggle")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--svd-dims", type=int, default=100)
    parser.add_argument("--lyrics-max-features", type=int, default=200_000)
    parser.add_argument("--lyrics-max-rows", type=int, default=0, help="0 means use all rows after cleaning.")
    parser.add_argument("--seed", type=int, default=4)
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    raw_root = cache_dir / "raw"
    processed_dir = cache_dir / "processed"
    selected = ["usps", "lyrics"] if args.dataset == "all" else [args.dataset]

    require_python_modules(["numpy", "pandas"])
    for key in selected:
        slug = DATASETS[key]["slug"]
        raw_dir = raw_root / key
        run_kaggle_download(slug, raw_dir, force=args.force)
        if key == "usps":
            prepare_usps(raw_dir, processed_dir / DATASETS[key]["processed"], force=args.force)
        else:
            processed_name = DATASETS[key]["processed"]
            if args.lyrics_max_rows and args.lyrics_max_rows > 0:
                processed_name = processed_name.replace(".npz", f"_n{args.lyrics_max_rows}.npz")
            prepare_lyrics(
                raw_dir,
                processed_dir / processed_name,
                svd_dims=args.svd_dims,
                max_features=args.lyrics_max_features,
                max_rows=args.lyrics_max_rows,
                seed=args.seed,
                force=args.force,
            )


if __name__ == "__main__":
    main()
