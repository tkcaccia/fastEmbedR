# Benchmarks

The full current benchmark summary is kept in
[../BENCHMARK_SUMMARY.md](../BENCHMARK_SUMMARY.md). This page records the main
MNIST 70k benchmark command and the latest local raw-image snapshot.

## Methods To Compare

After the cleanup, benchmarks should compare:

- `fastEmbedR::umap_knn()` from a supplied KNN graph.
- `uwot::umap()` / `uwot::umap(..., fast_sgd = TRUE)` as the R reference UMAP
  path.
- `fastEmbedR::opentsne_knn()` from a supplied KNN graph.
- `Rtsne::Rtsne_neighbors()` as the R reference t-SNE-from-KNN path.
- `ReductionWrappers::openTSNE()` as the Python openTSNE wrapper when the
  configured `reticulate` Python can import `openTSNE`.

## MNIST 70k Raw-Image Benchmark

The command below runs the local MNIST 70k benchmark on raw flattened 28x28
images, not on PCA features:

```sh
Rscript tools/benchmark_mnist70k_current_backends.R \
  --feature-source=raw \
  --n=70000 \
  --k=50 \
  --seed=6 \
  --threads=4 \
  --backends=cpu,metal \
  --run-uwot=true \
  --run-umap=true \
  --run-opentsne=true \
  --run-rtsne=true \
  --run-landmark=true
```

Latest local raw-MNIST run, using flattened 784-column images and a
5,000-point quality sample:

| method | backend | NN sec | embed sec | proj+transform sec | trust |
| --- | --- | ---: | ---: | ---: | ---: |
| openTSNE | CPU | 64.752 | 4.263 | NA | 0.319 |
| openTSNE | Metal | 40.787 | 6.696 | NA | 0.305 |
| Rtsne_neighbors | CPU | 64.752 | 34.871 | NA | 0.198 |
| openTSNE landmark50 | CPU | 192.299 | 2.535 | 143.509 | 0.269 |
| openTSNE landmark50 | Metal | 50.390 | 3.709 | 49.807 | 0.270 |
| UMAP | CPU | 64.752 | 6.914 | NA | 0.281 |
| UMAP | Metal | 40.787 | 2.159 | NA | 0.283 |
| UMAP landmark50 | CPU | 197.833 | 3.581 | 159.008 | 0.267 |
| UMAP landmark50 | Metal | 51.452 | 1.236 | 53.215 | 0.262 |
| uwot::umap fast_sgd | CPU | NA | 67.581 | NA | 0.283 |

For `uwot::umap`, `embed sec` is the total exposed call time because `uwot`
does not report KNN, graph construction, and SGD timing separately. For
fastEmbedR rows, KNN and embedding/projection timing are recorded separately.

## MPSGraph Diagnostic Result

The MPSGraph FFT diagnostic is not the default path. On flattened MNIST 70k
with the same cached KNN and same PCA initialization, it was slightly faster
but shifted the plot and quality slightly:

| backend | NN sec | PCA init sec | embed sec | trust | label acc |
| --- | ---: | ---: | ---: | ---: | ---: |
| current Metal FFT-grid | 40.904 | 3.734 | 3.608 | 0.303 | 0.969 |
| MPSGraph FFT diagnostic | 40.904 | 3.734 | 3.271 | 0.300 | 0.967 |

The conclusion for now is to keep package-native Metal FFT-grid as the default
and keep MPSGraph diagnostic-only.

