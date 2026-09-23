# Benchmarks

[Home](../README.md) |
[Installation](installation.md) |
[Bioconductor](bioconductor.md) |
[Implementation](implementation.md) |
[Examples](examples.md) |
**Benchmarks** |
[API](usage-api.md) |
[Reproducibility](reproducibility.md) |
[References](references.md) |
[Benchmark repository](https://github.com/tkcaccia/fastEmbedR-extra)

The complete publication benchmark is maintained in the separate
[`fastEmbedR-extra`](https://github.com/tkcaccia/fastEmbedR-extra)
repository. It contains scripts, manifests, preprocessing instructions,
environment recipes, Slurm launchers, and validation code, but no raw
benchmark data. Provider-controlled or restricted datasets must be obtained
independently under their original terms; missing data are recorded as
unavailable rather than silently replaced.

The GitHub documentation keeps one small sanity-check dataset (`iris`) and one
large benchmark dataset (`MNIST70k`). This avoids mixing current package
results with historical exploratory benchmarks.

## Current Public Benchmark

The current public benchmark is MNIST70k from flattened 28 x 28 images. The
results, machine specification, runtime bar plot, embedding plot, and source
CSV are shown on the [Examples](examples.md) page.

The example R code compares:

- `fastEmbedR::tsne()`;
- `Rtsne::Rtsne()`;
- `fastEmbedR::umap(..., graph_mode = "fuzzy")`;
- `uwot::umap(..., fast_sgd = TRUE)`.

## Publication Quality Outputs

The publication benchmark scripts report total elapsed runtime and embedding
quality in the same row. Runtime alone is not used to support the main
performance claim. Each successful benchmark row includes:

- runtime in seconds;
- peak RSS memory when `/usr/bin/time -v` is available;
- trustworthiness;
- nearest-neighbor preservation at k = 15, 30, and 50;
- silhouette score when labels are available;
- embedding-space KNN label accuracy when labels are available.

The HPC driver writes the manuscript-facing files:

- `embedding_quality_table.csv`;
- `embedding_quality_table.md`;
- `embedding_parameter_table.csv`;
- `embedding_parameter_table.md`;
- `embedding_runtime_quality_pareto.csv`;
- `embedding_runtime_quality_pareto.png`;
- `benchmark_command_lines.txt`;
- `sessionInfo.txt`;
- `reproducibility_manifest.txt`;
- `reproducibility_manifest.json` when `jsonlite` is installed.

The table and Pareto plot are generated for all completed rows and prioritize
the reviewer-requested datasets plus the explicit metabolomics benchmark:
`MNIST`, `FashionMNIST`, `flow18`, `mass41`, `imagenet`,
`FlowRepository_FR-FCM-ZYRM_files`, and `MetRef`. These outputs are intended to
support the Results section claim that speed improvements do not come from a
material loss of embedding quality.

`MetRef` is the metabolomics dataset in the publication embedding benchmark.
Simulated matrices are reserved for nearest-neighbor stress testing and are
not used as evidence for UMAP/t-SNE embedding quality in the manuscript.

The parameter table records the settings needed to interpret benchmark
fairness: `n_neighbors`/`k`, perplexity, iterations or epochs, early
exaggeration, learning-rate policy, initialization, distance metric, thread
count, random seed, approximate versus exact or package-internal KNN, and
whether KNN was precomputed or computed internally. The manuscript compares
total elapsed runtime because Rtsne, FIt-SNE, umap, uwot, and fastEmbedR expose
different computational boundaries.

See [Reproducibility](reproducibility.md) for the exact commit, release-tag
policy, archival DOI field, hardware/session metadata, CUDA/FAISS/cuVS probes,
and benchmark command lines recorded with each run.

## Reference Implementation Validation

The validation script
[`tools/validate_reference_implementations.R`](https://github.com/tkcaccia/fastEmbedR-extra/blob/main/benchmarks/legacy/fastEmbedR-benchmark/tools/validate_reference_implementations.R)
checks the native implementation on a small deterministic problem. It is a
correctness-oriented validation, not a performance benchmark.

The default run uses iris with exact Euclidean KNN, a fixed seed, and a fixed
PCA initialization for t-SNE:

```bash
git clone https://github.com/tkcaccia/fastEmbedR-extra.git
cd fastEmbedR-extra/benchmarks/legacy/fastEmbedR-benchmark
Rscript tools/validate_reference_implementations.R \
  --out-dir=results/reference_validation_current \
  --threads=2 \
  --seed=4 \
  --perplexity=10 \
  --k=31
```

The script writes:

- `reference_validation_results.csv`;
- `reference_validation_results.md`;
- `reference_validation_plots.png`;
- `reference_validation_manifest.txt`.

The validation compares:

- `fastEmbedR::tsne_knn()` against `Rtsne::Rtsne_neighbors()` using the
  same exact KNN matrix and fixed PCA initialization. Because iris is a small
  correctness check, the fastEmbedR run uses the exact negative-gradient path
  rather than the large-data FFT-grid approximation;
- `fastEmbedR::umap_knn(..., graph_mode = "fuzzy")` against `uwot::umap()`;
- the fastEmbedR UMAP sparse graph from `prepare_umap_knn()` against
  `uwot::similarity_graph()`.

The reported checks include trustworthiness, nearest-neighbor preservation,
embedding-space KNN label accuracy, t-SNE KL/cost traces where available,
Procrustes-aligned embedding similarity, UMAP graph edge overlap, and UMAP
common-edge weight correlation. The goal is not bitwise identity. Parallel
floating-point reductions, stochastic edge sampling, and implementation-level
optimizer differences are expected to change the exact coordinates. The
validation asks whether the native implementation preserves the same local
structure, reaches a comparable t-SNE objective value, and builds a UMAP graph
consistent with the reference implementation.

For the current iris validation run, fastEmbedR interpolation-based t-SNE and
`Rtsne_neighbors()` reached nearly identical final KL values (`0.30910` versus
`0.30887`), the same embedding-space KNN label accuracy (`0.967`), and a
Procrustes-aligned coordinate correlation of `0.848` under the fixed PCA
initialization. The fastEmbedR fuzzy UMAP graph had high common-edge weight
agreement with `uwot::similarity_graph()` (Spearman correlation `0.970` on
common edges).

## Iris Smoke Test

The iris examples in [Examples](examples.md) and the reference manual are kept
as fast smoke tests. They are not used to claim large-data performance.
