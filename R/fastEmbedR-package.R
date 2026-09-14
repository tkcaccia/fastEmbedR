#' fastEmbedR: native UMAP and interpolation-based t-SNE embeddings
#'
#' fastEmbedR provides an opinionated fixed-policy float32 UMAP implementation
#' and a configurable native interpolation-based t-SNE implementation from
#' data or
#' precomputed nearest-neighbor matrices. Use [umap()] or [tsne()]
#' for one-call workflows, [precompute_knn()] to run the package-native CPU,
#' Metal, or CUDA search separately, or pass reusable KNN `indices` and
#' `distances` to [umap_knn()] or [tsne_knn()].
#' Use [pca()] for reusable backend-native PCA and request
#' `tsne_init = TRUE` when a t-SNE-ready initialization is needed.
#' Score embeddings with [evaluate_embedding()]. Optional downstream graph
#' utilities are available through [knn_graph()] and [graph_cluster()].
#' Use [fastEmbedR_api()] for the complete canonical, advanced, diagnostic,
#' secondary, and compatibility API map.
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
