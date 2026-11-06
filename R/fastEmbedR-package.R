#' fastEmbedR: native embedding, PCA, and clustering
#'
#' fastEmbedR provides an opinionated fixed-policy float32 UMAP implementation
#' and a configurable native interpolation-based t-SNE implementation from
#' data or
#' precomputed nearest-neighbor matrices. Use [umap()], [tsne()], or [pca()]
#' for everyday analysis. [precompute_knn()] can run CPU, Metal, or CUDA
#' search once; pass its result to [umap()] or [tsne()] for reuse.
#' Request
#' `tsne_init = TRUE` when a t-SNE-ready initialization is needed.
#' Use [graph_cluster()] for resident or file-backed clustering;
#' set `landmarks` for approximate all-row assignment.
#' Score embeddings with [evaluate_embedding()]. Precomputed KNN results can
#' be passed directly to [umap()] or [tsne()]; full file-backed fits also
#' expose graph preparation helpers.
#' Session defaults can be set with
#' `options(backend = "cpu", n.cores = 4L)`; explicit function arguments take
#' precedence over these options.
#'
#' UMAP exposes neighborhood, metric, graph mode, preprocessing, backend, seed,
#' output dimension, and CPU thread count. Epochs, minimum distance, spread,
#' learning rate, repulsion strength, negative sampling, KNN index tuning, and
#' optimizer mode follow the recorded package policy. Users requiring arbitrary
#' sweeps of those controls should use a general-purpose UMAP implementation.
#'
#' The package intentionally does not export the earlier legacy t-SNE,
#' InfoTSNE, PaCMAP, TriMap, or LocalMAP reducers.
#'
#' @keywords internal
"_PACKAGE"

utils::globalVariables(c(
    "landmark_refinement_epoch_count",
    "landmark_tsne_transform_resident_cuda_cpp",
    "landmark_tsne_transform_resident_metal_cpp",
    "run_native_knn_optimizer",
    "transform_tsne_cuda_cpp"
))
