# Benchmarks

[Home](../README.md) |
[Installation](installation.md) |
[Implementation](implementation.md) |
[Examples](examples.md) |
**Benchmarks** |
[API](usage-api.md) |
[Provenance](algorithm-provenance.md)

This page summarizes the GitHub-facing benchmark setup. The publication
benchmark scripts under `tools/` contain the larger multi-dataset benchmark
suite.

## MNIST 70k Benchmark

The GitHub benchmark uses the full 70,000 MNIST observations from the public
IDX files, represented as flattened 28x28 raw images.

Run locally on Apple Silicon:

```sh
Rscript tools/benchmark_github_mnist70k.R \
  --n=70000 \
  --k=15 \
  --perplexity=15 \
  --threads=4 \
  --run-metal=true \
  --run-cuda=false \
  --run-references=true \
  --out-dir=results/github_mnist70k_local
```

Run on CUDA:

```sh
Rscript tools/benchmark_github_mnist70k.R \
  --n=70000 \
  --k=15 \
  --perplexity=15 \
  --threads=4 \
  --run-metal=false \
  --run-cuda=true \
  --run-references=true \
  --out-dir=results/github_mnist70k_cuda
```

Compared methods:

| Family | fastEmbedR method | Reference method |
| --- | --- | --- |
| openTSNE/t-SNE | `fastEmbedR::opentsne()` on CPU, Metal, CUDA | `Rtsne::Rtsne()` full call with its own internal KNN |
| UMAP | `fastEmbedR::umap(..., graph_mode = "fuzzy")` on CPU, Metal, CUDA | `uwot::umap(..., fast_sgd = TRUE)` full call with its own internal KNN |

`graph_mode = "binary"` is not displayed in this GitHub benchmark.

## openTSNE Result

The figure below shows a representative MNIST 70k openTSNE comparison.

![MNIST 70k openTSNE CPU Metal CUDA](assets/mnist70k-opentsne-pca-embeddings-cpu-metal-cuda.png)

Timing from the saved run:

| method | backend | machine | NN sec | embedding sec | total sec | trust | label KNN acc |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| fastEmbedR openTSNE | CPU | Mac | 61.642 | 3.896 | 65.538 | 0.324 | 0.958 |
| fastEmbedR openTSNE | Metal | Mac | 40.904 | 3.250 | 44.154 | 0.312 | 0.966 |
| fastEmbedR openTSNE | CUDA | chiamaka | 2.046 | 0.410 | 2.456 | 0.327 | 0.972 |

The source CSV is
[assets/mnist70k-opentsne-pca-timing.csv](assets/mnist70k-opentsne-pca-timing.csv).

For full t-SNE comparison, use the benchmark script above. It runs
`Rtsne::Rtsne()` as a complete function call, including its own neighbour
calculation.

## UMAP Result: Fuzzy Graph Only

The UMAP GitHub examples show only the standard fuzzy graph mode.

| fastEmbedR CPU fuzzy | fastEmbedR Metal fuzzy |
| --- | --- |
| ![fastEmbedR CPU fuzzy](assets/mnist70k-umap-fastembedr-cpu-fuzzy.png) | ![fastEmbedR Metal fuzzy](assets/mnist70k-umap-fastembedr-metal-fuzzy.png) |

| fastEmbedR CUDA fuzzy | uwot fast_sgd full baseline |
| --- | --- |
| ![fastEmbedR CUDA fuzzy](assets/mnist70k-umap-fastembedr-cuda-fuzzy.png) | ![uwot fast_sgd](assets/mnist70k-umap-uwot-fast-sgd.png) |

Representative timing from saved MNIST runs:

| method | backend | k | NN sec | embedding sec | total sec |
| --- | --- | ---: | ---: | ---: | ---: |
| fastEmbedR UMAP fuzzy | CPU | 15 | 20.897 | 5.817 | 26.714 |
| fastEmbedR UMAP fuzzy | Metal | 15 | 20.897 | 0.937 | 21.834 |
| fastEmbedR UMAP fuzzy | CUDA | 15 | 12.594 | 3.207 | 15.801 |
| `uwot::umap(..., fast_sgd = TRUE)` | CPU | 15 | internal | 6.832 | full-call benchmark |

The `uwot` row in the saved local table reports the exposed embedding call time
from the test harness; for the GitHub benchmark script, `uwot::umap()` is run
as a full function call with its own internal neighbour calculation.

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
