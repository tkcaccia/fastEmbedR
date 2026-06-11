# fastEmbedR Benchmark Summary

Date: 2026-06-11

This summary records the current local benchmark evidence used for the
fastEmbedR restart. The focus is MNIST, not iris, because the package is meant
for reusable KNN graphs and larger embeddings.

## Scope

- Hardware: local Mac, CPU thread cap 4 for comparable multicore runs.
- Main dataset: MNIST IDX cache, PCA-50, labels retained.
- Main settings: `k = 50`, `seed = 6`, Euclidean PCA space.
- Quality metrics shown here: trustworthiness / KNN preservation at 30,
  silhouette in the embedding, and embedding-space KNN label accuracy.
- The full 70k UMAP comparison excludes PCA preprocessing and separates KNN
  from embedding where possible.
- External R-package comparison uses MNIST 2.5k so slower packages and Python
  wrappers can complete locally. Packages that accept precomputed KNN receive
  the same KNN graph. Packages that do not accept KNN are timed as package
  calls after PCA.

## Available R Tools Checked Locally

| Package | Status | Notes |
|---|---:|---|
| fastEmbedR 0.1.0 | installed | Native C++ UMAP, t-SNE, openTSNE-style paths from KNN. |
| uwot 0.2.4 | installed | Strong GPL UMAP reference with `fast_sgd`. |
| umap 0.2.10.0 | installed | CRAN UMAP, accepts `umap.knn`. |
| Rtsne 0.17 | installed | Reference `Rtsne_neighbors()` t-SNE-from-KNN path. |
| tsne 0.2-0 | installed | Slower reference t-SNE package. |
| Rdimtools 1.1.4 | installed | Exposes `do.tsne()` locally; no UMAP/PaCMAP/TriMap adapters were available in this installation. |
| ReductionWrappers 2.5.4 | installed | Python openTSNE wrapper, works with explicit framework Python. |
| m3addon 0.2 | installed | Not included in this UMAP/t-SNE MNIST table. |
| fftRtsne | not installed | Not available in this local R library. |
| dim.reduction.wrappers | not installed | The installed wrapper package is `ReductionWrappers`. |

## MNIST 70k: fastEmbedR UMAP/t-SNE/openTSNE-style Paths

The KNN stage used `fastEmbedR::nn(..., backend = "cpu_nndescent")`. Timings
are split so the shared KNN cost is visible.

| Method | Shared KNN sec | Embedding sec | KNN + embedding sec | Trust / preserve@30 | Silhouette | Label KNN acc |
|---|---:|---:|---:|---:|---:|---:|
| fastEmbedR UMAP | 14.762 | 7.334 | 22.096 | 0.168 | 0.214 | 0.870 |
| fastEmbedR t-SNE | 14.762 | 25.229 | 39.991 | 0.132 | 0.066 | 0.800 |
| fastEmbedR openTSNE-style | 14.762 | 18.363 | 33.125 | 0.304 | 0.168 | 0.884 |

Current conclusion: the openTSNE-style path is the strongest quality result
among the three fastEmbedR methods on full MNIST 70k, while UMAP is the fastest
embedding stage. The NN-descent KNN cost is now a major part of the total.

Output files:

- `results/mnist_umap_tsne_opentsne/mnist_70000_umap_tsne_opentsne_seed6.csv`
- `results/mnist_umap_tsne_opentsne/mnist_70000_umap_tsne_opentsne_seed6.png`

## MNIST 2.5k: Available R Package And Python-Wrapper Comparison

Shared exact KNN cost before methods that accept KNN: 0.302 seconds.
`ReductionWrappers_openTSNE` recomputes exact neighbours internally through the
Python wrapper because that interface does not consume the package KNN object.

| Method | Package | Status | Wall sec | Trust / preserve@30 | Silhouette | Label KNN acc |
|---|---|---:|---:|---:|---:|---:|
| fastEmbedR UMAP | fastEmbedR | success | 0.494 | 0.380 | 0.129 | 0.735 |
| fastEmbedR t-SNE | fastEmbedR | success | 0.526 | 0.415 | 0.094 | 0.774 |
| fastEmbedR openTSNE-style | fastEmbedR | success | 0.805 | 0.406 | 0.100 | 0.769 |
| uwot fast_sgd | uwot | success | 0.823 | 0.367 | 0.130 | 0.734 |
| umap package | umap | success | 9.284 | 0.367 | 0.115 | 0.720 |
| Rtsne_neighbors | Rtsne | success | 0.857 | 0.419 | 0.090 | 0.781 |
| tsne package | tsne | success | 63.516 | 0.389 | 0.051 | 0.750 |
| Rdimtools do.tsne | Rdimtools | success | 2.294 | 0.036 | -0.132 | 0.190 |
| openTSNE wrapper | ReductionWrappers/Python openTSNE | success | 1.863 | 0.416 | 0.086 | 0.776 |

Current conclusion: fastEmbedR is the fastest group on this MNIST 2.5k
comparison. For t-SNE quality, `Rtsne_neighbors()` and the Python openTSNE
wrapper remain slightly ahead in trustworthiness, but fastEmbedR t-SNE is close
and faster in this reduced run.

Output files:

- `results/mnist_available_r_tools/mnist_2500_available_r_tools_seed6.csv`
- `results/mnist_available_r_tools/mnist_2500_available_r_tools_seed6.png`

## Prior MNIST 70k UMAP-Only Comparison Against uwot

This older full-size UMAP-only comparison is important because it used the same
PCA cache and separated preprocessing from KNN/embedding.

| Method | Package | KNN sec | Embedding sec | Timed no-preprocess sec | Trust | Silhouette | Label KNN acc |
|---|---|---:|---:|---:|---:|---:|---:|
| uwot fast_sgd from fastEmbedR KNN | uwot | 10.556 | 11.271 | 21.827 | 0.284 | 0.245 | 0.845 |
| uwot fast_sgd end-to-end | uwot | NA | 21.868 | 21.868 | 0.284 | 0.250 | 0.852 |
| fastEmbedR UMAP from fastEmbedR KNN | fastEmbedR | 10.556 | 15.491 | 26.047 | 0.284 | 0.242 | 0.849 |
| fastEmbedR UMAP landmark50 | fastEmbedR | NA | 111.755 | 111.755 | 0.279 | 0.248 | 0.845 |

Current conclusion: on this full MNIST 70k UMAP-only benchmark, uwot is still
faster than the current fastEmbedR CPU UMAP at essentially tied quality. This
is the main remaining multicore UMAP target: preserve the uwot-equivalent
mathematics while reducing CSR graph and SGD overhead.

## Licensing And Acknowledgements

fastEmbedR is licensed as `GPL (>= 3)`. This is the right posture because UMAP
behaviour and fast-SGD scheduling were intentionally compared against and
partly adapted from GPL-compatible R implementations such as `uwot`.

Detailed provenance is maintained in `inst/NOTICE` and
`inst/ALGORITHMIC_REFERENCES.md`. Current acknowledgements include `uwot`,
`Rtsne`, FAISS, RAPIDS cuML/cuVS, `mlx-vis`, `annembed`, KeOps, TorchDR,
openTSNE, and t-SNE-CUDA. Optional external backends are not silently relabelled
as package-native GPU work; unavailable GPU libraries must fail clearly.
