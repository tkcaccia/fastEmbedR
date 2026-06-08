# fastknnumap

`fastknnumap` is a small R package for fast UMAP embeddings from precomputed
KNN index and distance matrices. The fuzzy graph construction and layout
optimization are implemented in C++ via Rcpp.

For a reproducible Python-first benchmark suite comparing UMAP/t-SNE
implementations on public datasets, see `BENCHMARK_PYTHON.md`.

Install the development version from GitHub:

```r
remotes::install_github("tkcaccia/fastEmbedR")
```

```r
load("/Users/stefano/Documents/GPUPLS/Data/metref_remote_task.RData")

data <- scale(out$Xtrain)
nn <- fastknnumap::nn(data, data, 30, parallel = TRUE)

layout <- fastknnumap::fast_knn_umap(
  nn$indices[, -1],
  nn$distances[, -1],
  n_epochs = 500,
  init_sdev = "range",
  seed = 42
)

plot(layout, pch = 21, bg = out$Ytrain)
```

On macOS, exact Euclidean KNN can use the native Metal GPU backend:

```r
fastknnumap::metal_available()
nn <- fastknnumap::nn(data, data, 30, backend = "gpu")
```

The SGD optimizer exposes UMAP-style controls such as `a`, `b`,
`repulsion_strength`, `negative_sample_rate`, `init_sdev`, and epoch-based edge
pruning. Use `init_sdev = "range"` for Python UMAP-compatible spectral scaling,
or leave it as `NULL` for native spectral scaling.

For larger datasets, the package also supports a paper-inspired spectral path
based on the normalized fuzzy KNN graph:

```r
layout <- fastknnumap::fast_knn_umap(
  nn$indices[, -1],
  nn$distances[, -1],
  mode = "hybrid",
  n_epochs = 100,
  spectral_n_iter = 25,
  seed = 4
)
```

Run the included benchmark:

```r
bench <- fastknnumap::benchmark_metref(
  "/Users/stefano/Documents/GPUPLS/Data/metref_remote_task.RData"
)
bench$timings
bench$silhouette
```

Compare multiple implementations from one shared KNN output:

```r
bench <- fastknnumap::benchmark_knn_umap(
  data,
  labels,
  k = 30,
  implementations = c(
    "fastknnumap_hybrid",
    "fastknnumap_sgd",
    "fastknnumap_spectral",
    "knn_tsne",
    "knn_pacmap"
  )
)

bench$knn_time
bench$metrics
```

Compare the native implementations across public datasets:

```r
suite <- fastknnumap::benchmark_embed(
  output_csv = "benchmark/native_suite.csv"
)
suite$metrics
```

The compact method names are `"fast"`, `"tsne"`, `"pacmap"`, `"trimap"`,
`"localmap"`, and `"all"`. Use `preset = "quick"`, `"balanced"`, or
`"accuracy"` to trade speed for repeat-based stability estimates. When
`output_csv` is set, benchmark plots are written next to the CSV with base R
graphics. Advanced controls are available through
`benchmark_embedding_datasets()` and `benchmark_knn_umap()`.

The benchmark also supports a landmark approximation inspired by bipartite
landmark UMAP methods. It selects hub-like landmarks from the KNN graph, keeps
a small landmark graph plus a few original local KNN edges, then runs the same
C++ optimizer:

```r
layout <- fastknnumap::landmark_knn_umap(
  nn$indices[, -1],
  nn$distances[, -1],
  landmark_ratio = 0.1,
  landmark_k = 10,
  local_k = 5,
  n_epochs = 100
)
```

Additional KNN-driven objectives are available for benchmarking:

```r
layout <- fastknnumap::knn_tsne(idx, dst, n_epochs = 100, n_threads = 4)
layout <- fastknnumap::knn_pacmap(idx, dst, n_epochs = 100, n_threads = 4)
layout <- fastknnumap::knn_trimap(idx, dst, n_epochs = 100, n_threads = 4)
layout <- fastknnumap::knn_localmap(idx, dst, n_epochs = 100, n_threads = 4)
```

Fashion-MNIST benchmark:

```r
bench <- fastknnumap::benchmark_fashion_mnist(
  n_train = 2000,
  pca_dims = 30,
  implementations = c(
    "fastknnumap_hybrid",
    "knn_tsne",
    "knn_pacmap",
    "knn_trimap",
    "knn_localmap"
  ),
  n_threads = 4
)
bench$metrics
```

The `tools/run_metal_smoke_test.sh` script checks whether local Metal command
line compilation is available.
