# Benchmarks

[Home](../README.md) |
[Installation](installation.md) |
[Implementation](implementation.md) |
[Examples](examples.md) |
**Benchmarks** |
[API](usage-api.md) |
[References](references.md)

The GitHub documentation keeps one small sanity-check dataset (`iris`) and one
large benchmark dataset (`MNIST70k`). This avoids mixing current package
results with historical exploratory benchmarks.

## Current Public Benchmark

The current public benchmark is MNIST70k from flattened 28 x 28 images. The
results, machine specification, runtime bar plot, embedding plot, and source
CSV are shown on the [Examples](examples.md) page.

The benchmark command is:

```sh
Rscript tools/benchmark_github_mnist70k.R --n=70000 --k=15 --perplexity=15
```

The script can compare:

- `fastEmbedR::opentsne()`;
- `Rtsne::Rtsne()`;
- `fastEmbedR::umap(..., graph_mode = "fuzzy")`;
- `uwot::umap(..., fast_sgd = TRUE)`.

## Iris Smoke Test

The iris examples in [Examples](examples.md) and the reference manual are kept
as fast smoke tests. They are not used to claim large-data performance.
