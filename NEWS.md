# fastEmbedR 0.1.0

- Focused the package on fast nearest-neighbour search and native
  openTSNE-style embeddings from precomputed KNN matrices.
- Kept the public KNN-first API: `nn()`, `opentsne_knn()`, `opentsne()`,
  `embed_knn()`, `transform_tsne()`, and `landmark_tsne()`.
- Removed the experimental package UMAP implementation, legacy native
  `tsne()`, and experimental `infotsne()` reducers because they were slower
  and/or lower quality than the reference packages in benchmark runs.
- Kept explicit backend detection with `backend_info()`, `metal_available()`,
  `cuda_available()`, `cuvs_available()`, and `faiss_available()`.
- Kept optional external FAISS/cuVS KNN link paths when available at build
  time, while keeping the openTSNE optimizer CPU-labelled unless a true native
  optimizer backend exists.
- Added a compact getting-started vignette that shows the recommended
  workflow: compute KNN once, run `opentsne_knn()`, score with
  `evaluate_embedding()`, and use landmarking/transform when needed.
