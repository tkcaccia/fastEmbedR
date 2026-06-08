from __future__ import annotations

import argparse
from pathlib import Path

from .benchmark import BenchmarkConfig, run_benchmark
from .datasets import DATASETS, list_datasets
from .runners import IMPLEMENTATIONS, list_implementations


def _parse_subset(values: list[str]) -> dict[str, int | None]:
    subsets: dict[str, int | None] = {}
    for value in values:
        if "=" not in value:
            raise argparse.ArgumentTypeError("Subsets must be written as dataset=size, e.g. mnist=10000")
        name, raw = value.split("=", 1)
        subsets[name] = None if raw.lower() in {"all", "none", "full"} else int(raw)
    return subsets


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark UMAP and t-SNE implementations.")
    parser.add_argument("--list-datasets", action="store_true", help="List available public datasets and exit.")
    parser.add_argument("--list-implementations", action="store_true", help="List benchmarkable implementations and exit.")
    parser.add_argument("--datasets", nargs="+", default=["iris", "digits"], choices=sorted(DATASETS))
    parser.add_argument("--implementations", nargs="+", default=["umap_learn", "sklearn_tsne_bh", "pca"], choices=sorted(IMPLEMENTATIONS))
    parser.add_argument("--subset", action="append", default=[], help="Override subset as dataset=size. Use full/all for all rows.")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--metric-sample", type=int, default=3000)
    parser.add_argument("--metric-neighbors", type=int, default=15)
    parser.add_argument("--pca-input-dims", type=int, default=50, help="Shared PCA input dimension for high-dimensional datasets. Use 0 to disable.")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--output-dir", default="benchmark/python_results")
    parser.add_argument("--save-embeddings", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.list_datasets:
        for spec in list_datasets():
            subsets = ", ".join("full" if s is None else str(s) for s in spec.allowed_subsets)
            print(f"{spec.name}\t{subsets}\t{spec.description}")
        return 0
    if args.list_implementations:
        for impl in list_implementations():
            print(f"{impl.name}\t{impl.family}\t{impl.description}")
        return 0

    output_dir = Path(args.output_dir)
    config = BenchmarkConfig(
        datasets=tuple(args.datasets),
        implementations=tuple(args.implementations),
        subsets=_parse_subset(args.subset),
        repeats=args.repeats,
        seed=args.seed,
        metric_sample=args.metric_sample,
        metric_neighbors=args.metric_neighbors,
        pca_input_dims=None if args.pca_input_dims == 0 else args.pca_input_dims,
        cache_dir=args.cache_dir,
        output_dir=str(output_dir),
        save_embeddings=args.save_embeddings,
    )
    df = run_benchmark(config)
    print(df)
    print(f"\nWrote results to {output_dir / 'results.csv'}")
    print(f"Wrote summary to {output_dir / 'summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
