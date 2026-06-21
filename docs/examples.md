# Examples

[Home](../README.md) |
[Installation](installation.md) |
[Implementation](implementation.md) |
**Examples** |
[Benchmarks](benchmarks.md) |
[API](usage-api.md) |
[References](references.md)

## Iris KNN-First Workflow

```r
library(fastEmbedR)

x <- scale(as.matrix(iris[, 1:4]))
labels <- iris$Species

knn <- faissR::nn(x, k = 15, backend = "auto", n_threads = 4)

y_tsne <- fastEmbedR::opentsne_knn(knn, init_data = x, backend = "cpu", seed = 1)
y_umap <- fastEmbedR::umap_knn(knn, backend = "cpu", graph_mode = "fuzzy", seed = 1)

plot(y_tsne, pch = 21, bg = labels)
plot(y_umap, pch = 21, bg = labels)
```

## Iris One-Call openTSNE

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
For matrix input, the KNN search is delegated to `faissR::nn_without_self()`:
CPU and Metal request the faissR CPU backend, while CUDA requests the faissR
CUDA backend. faissR then selects the concrete KNN method and tuning
automatically. The internal non-self KNN width is `ceiling(perplexity)`. Use
`opentsne_knn()` with an explicit `faissR::nn()` result when benchmarking alternative
KNN algorithms.

## Iris One-Call UMAP

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

The CUDA result below was produced on 2026-06-21 using a prepared MNIST
`.RData` file. CPU paths were requested with 4 threads for KNN search and
embedding. Replace `MNIST.RData` with the path to your own prepared MNIST file,
or omit `--mnist-rdata` to let the script download the public IDX files:

```sh
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
export RCPP_PARALLEL_NUM_THREADS=4

Rscript tools/benchmark_github_mnist70k.R \
  --mnist-rdata=/path/to/MNIST.RData \
  --n=70000 \
  --k=15 \
  --perplexity=15 \
  --threads=4 \
  --run-metal=false \
  --run-cuda=true \
  --run-references=true \
  --out-dir=results/github_mnist70k_cuda_current
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
- R: 4.5.3
- fastEmbedR: 0.1.0
- faissR: 0.1.0
- uwot: 0.2.4
- Rtsne: 0.17
- Requested benchmark threads: 4

The benchmark intentionally does not show `graph_mode = "binary"`.

### MNIST 70k Results

![MNIST 70k computational time](assets/mnist70k_cuda_codex_20260621/mnist70k_github_benchmark_time_barplot.png)

| method | backend | KNN backend | NN sec | embedding sec | total sec | trust | label KNN acc |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| fastEmbedR openTSNE CPU | CPU | faiss_ivf | 29.612 | 24.995 | 55.419 | 0.332 | 0.968 |
| fastEmbedR openTSNE CUDA | CUDA | faiss_gpu_ivf_flat | 6.772 | 1.605 | 9.134 | 0.332 | 0.970 |
| Rtsne full | CPU | internal | internal | 97.427 | 97.427 | 0.324 | 0.973 |
| fastEmbedR UMAP CPU fuzzy | CPU | faiss_ivf | 29.731 | 5.706 | 36.102 | 0.279 | 0.972 |
| fastEmbedR UMAP CUDA fuzzy | CUDA | faiss_gpu_ivf_flat | 6.459 | 0.565 | 7.705 | 0.274 | 0.973 |
| uwot UMAP fast_sgd full | CPU | internal | internal | 37.211 | 37.211 | 0.277 | 0.971 |

The `KNN backend` column records the concrete faissR backend selected during
that run. The reference methods use their own internal neighbour search, so the
table reports their full runtime as embedding/runtime.

![MNIST 70k embeddings](assets/mnist70k_cuda_codex_20260621/mnist70k_github_benchmark.png)

Source files:

- [mnist70k_github_benchmark.csv](assets/mnist70k_cuda_codex_20260621/mnist70k_github_benchmark.csv)
- [machine-specs.md](assets/mnist70k_cuda_codex_20260621/machine-specs.md)
