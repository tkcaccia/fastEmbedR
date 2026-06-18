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
  labels = labels,
  n_neighbors = 30,
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

## One-Call UMAP

For standard UMAP comparison use the fuzzy graph:

```r
fit <- fastEmbedR::umap(
  x,
  labels = labels,
  n_neighbors = 30,
  backend = "cpu",
  graph_mode = "fuzzy",
  n_threads = 4,
  seed = 1
)

plot(fit)
fit$metrics
```

## MNIST 70k Benchmark Example

The GitHub benchmark script downloads MNIST automatically from the public IDX
files and uses flattened 28x28 images.

CPU and Metal on a Mac:

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

CUDA on a Linux/NVIDIA machine:

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

The script compares:

- `fastEmbedR::opentsne()` on CPU, Metal, and/or CUDA;
- `Rtsne::Rtsne()` as the full Rtsne baseline with its own internal KNN;
- `fastEmbedR::umap(..., graph_mode = "fuzzy")` on CPU, Metal, and/or CUDA;
- `uwot::umap(..., fast_sgd = TRUE)` as the full uwot baseline with its own
  internal KNN.

The output files are:

- `mnist70k_github_benchmark.csv`
- `mnist70k_github_benchmark.png`

The benchmark intentionally does not show `graph_mode = "binary"`.

## Example Figures

openTSNE CPU / Metal / CUDA:

![MNIST 70k openTSNE CPU Metal CUDA](assets/mnist70k-opentsne-pca-embeddings-cpu-metal-cuda.png)

UMAP fuzzy graph only:

| fastEmbedR CPU fuzzy | fastEmbedR Metal fuzzy | fastEmbedR CUDA fuzzy | uwot fast_sgd |
| --- | --- | --- | --- |
| ![fastEmbedR CPU fuzzy](assets/mnist70k-umap-fastembedr-cpu-fuzzy.png) | ![fastEmbedR Metal fuzzy](assets/mnist70k-umap-fastembedr-metal-fuzzy.png) | ![fastEmbedR CUDA fuzzy](assets/mnist70k-umap-fastembedr-cuda-fuzzy.png) | ![uwot fast_sgd](assets/mnist70k-umap-uwot-fast-sgd.png) |
