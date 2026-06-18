# Examples

[Home](../README.md) |
[Installation](installation.md) |
[Implementation](implementation.md) |
**Examples** |
[Benchmarks](benchmarks.md) |
[API](usage-api.md) |
[Provenance](algorithm-provenance.md)

## KNN-First Workflow

```r
library(fastEmbedR)

x <- scale(as.matrix(iris[, 1:4]))
labels <- iris$Species

knn <- fastEmbedR::nn(x, k = 15, backend = "auto", n_threads = 4)

y_tsne <- fastEmbedR::opentsne_knn(knn, init_data = x, backend = "cpu", seed = 1)
y_umap <- fastEmbedR::umap_knn(knn, backend = "cpu", graph_mode = "fuzzy", seed = 1)

plot(y_tsne, pch = 21, bg = labels)
plot(y_umap, pch = 21, bg = labels)
```

## One-Call openTSNE

```r
fit <- fastEmbedR::opentsne(
  x,
  perplexity = 30,
  backend = "cpu",
  n_threads = 4,
  seed = 1
)

plot(fit)
fit$metrics
```

Use `backend = "metal"` on Apple Silicon or `backend = "cuda"` on a CUDA build.
Explicit GPU requests fail clearly if the backend is unavailable.
For matrix input, the KNN search is fixed inside `opentsne()`: CPU and Metal
use FAISS CPU IVF-Flat, while CUDA uses FAISS GPU IVF-Flat. The internal
non-self KNN width is `ceiling(perplexity)`. Use
`opentsne_knn()` with an explicit `nn()` result when benchmarking alternative
KNN algorithms.

## One-Call UMAP

For standard UMAP comparison use the fuzzy graph:

```r
fit <- fastEmbedR::umap(
  x,
  n_neighbors = 30,
  backend = "cpu",
  graph_mode = "fuzzy",
  n_threads = 4,
  seed = 1
)

plot(fit)
fit$metrics
```

For matrix input, `umap()` uses the same fixed KNN policy as `opentsne()`.
Use `umap_knn()` when you want to reuse or benchmark a separately computed KNN
object.

## MNIST 70k Benchmark Example

The benchmark script uses the full 70,000 MNIST observations as flattened
28x28 images. It can either download the public IDX files or load a prepared
`.RData` object with `data` and `labels` fields.

The CUDA run below was executed on `chiamaka` on 2026-06-18 using the prepared
dataset at `/mnt/sata_ssd/fastEmbedR/Data/MNIST/MNIST.RData`:

```sh
export LD_PRELOAD=/home/chiamaka/.fastEmbedR/micromamba/envs/fastembedr-faissgpu-cuvs/lib/libstdc++.so.6

Rscript tools/benchmark_github_mnist70k.R \
  --mnist-rdata=/mnt/sata_ssd/fastEmbedR/Data/MNIST/MNIST.RData \
  --n=70000 \
  --k=15 \
  --perplexity=15 \
  --threads=4 \
  --run-metal=false \
  --run-cuda=true \
  --run-references=true \
  --out-dir=/mnt/sata_ssd/fastEmbedR/results/github_mnist70k_cuda_codex_20260618
```

The script compares:

- `fastEmbedR::opentsne()` on CPU, Metal, and/or CUDA;
- `Rtsne::Rtsne()` as the full Rtsne baseline with its own internal KNN;
- `fastEmbedR::umap(..., graph_mode = "fuzzy")` on CPU, Metal, and/or CUDA;
- `uwot::umap(..., fast_sgd = TRUE)` as the full uwot baseline with its own
  internal KNN.

This run used:

- Machine: `icgeb-bioinformatics-unit`
- System: Linux 6.8.0-124-generic, x86_64
- CPU: 13th Gen Intel(R) Core(TM) i7-13700
- GPU: NVIDIA GeForce RTX 5060 Ti, driver 595.71.05, 16311 MiB
- RAM: 31.02 GB
- R: 4.6.0
- fastEmbedR: 0.1.0
- faissR: 0.1.0
- uwot: 0.2.4
- Rtsne: 0.17
- Requested benchmark threads: 4

The benchmark intentionally does not show `graph_mode = "binary"`.

### MNIST 70k Results

![MNIST 70k computational time](assets/mnist70k_cuda_codex_20260618/mnist70k_github_benchmark_time_barplot.png)

| method | backend | NN sec | embedding sec | total sec | trust | label KNN acc |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| fastEmbedR openTSNE CPU | CPU | 659.379 | 29.381 | 693.347 | 0.255 | 0.921 |
| fastEmbedR openTSNE CUDA | CUDA | 5.087 | 0.805 | 9.548 | 0.255 | 0.920 |
| Rtsne full | CPU | internal | 349.551 | 349.551 | 0.324 | 0.973 |
| fastEmbedR UMAP CPU fuzzy | CPU | 659.677 | 5.768 | 666.550 | 0.217 | 0.920 |
| fastEmbedR UMAP CUDA fuzzy | CUDA | 2.915 | 0.531 | 4.405 | 0.214 | 0.923 |
| uwot UMAP fast_sgd full | CPU | internal | 37.071 | 37.071 | 0.277 | 0.971 |

![MNIST 70k embeddings](assets/mnist70k_cuda_codex_20260618/mnist70k_github_benchmark.png)

Source files:

- [mnist70k_github_benchmark.csv](assets/mnist70k_cuda_codex_20260618/mnist70k_github_benchmark.csv)
- [machine-specs.md](assets/mnist70k_cuda_codex_20260618/machine-specs.md)
