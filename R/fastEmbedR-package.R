#' fastEmbedR: fast KNN, UMAP, and native openTSNE from KNN
#'
#' fastEmbedR focuses on fast nearest-neighbour search and native
#' UMAP/openTSNE-style embeddings from precomputed KNN matrices. The
#' recommended workflow is to compute KNN once with [nn()], embed with
#' [umap_knn()], [opentsne_knn()], [umap()], or [opentsne()], and score the
#' result with [evaluate_embedding()].
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
