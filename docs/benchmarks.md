# Benchmarks

[Home](../README.md) |
[Installation](installation.md) |
[Implementation](implementation.md) |
[Examples](examples.md) |
**Benchmarks** |
[API](usage-api.md) |
[Provenance](algorithm-provenance.md)

This page points to the reusable benchmark scripts. The MNIST 70k example
results, machine specification, runtime bar plot, embedding plot, and source
CSV are kept on the [Examples](examples.md) page to avoid duplicating benchmark
information across pages.

## Larger Benchmark Suite

The larger paper-oriented benchmark scripts are:

- `tools/benchmark1_nn_speed.R`: neighbour-search benchmark.
- `tools/benchmark2_tsne_speed_accuracy.R`: t-SNE/openTSNE benchmark.
- `tools/benchmark3_umap_speed_accuracy.R`: UMAP benchmark.
- `tools/benchmark_embeddings.sh`: Singularity/HPC wrapper that runs method
  workers independently and records timeout/OOM failures instead of stopping
  the whole benchmark.

See [extended-benchmark-suite.md](extended-benchmark-suite.md) for the
multi-dataset plan and [benchmark-gallery.md](benchmark-gallery.md) for
publication-style figure guidance.
