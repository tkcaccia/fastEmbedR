normalize_nn_threads <- function(n_threads) {
  n_threads <- suppressWarnings(as.integer(n_threads))
  if (length(n_threads) != 1L || is.na(n_threads) || n_threads < 1L) {
    n_threads <- 1L
  }
  n_threads
}

#' Nearest-neighbour and graph utilities from faissR
#'
#' These functions are re-exported as thin wrappers around the companion
#' `faissR` package so users can call the complete fastEmbedR workflow from one
#' namespace. The implementation, optional FAISS/cuVS backends, and detailed
#' backend selection logic remain in `faissR`.
#'
#' @param ... Arguments passed directly to the matching `faissR` function.
#'
#' @return The value returned by the matching `faissR` function.
#'
#' @name faissR_wrappers
NULL

#' @rdname faissR_wrappers
#' @export
nn <- function(...) {
  faissR::nn(...)
}

#' @rdname faissR_wrappers
#' @export
nn_without_self <- function(...) {
  faissR::nn_without_self(...)
}

#' @rdname faissR_wrappers
#' @export
candidate_knn <- function(...) {
  faissR::candidate_knn(...)
}

#' @rdname faissR_wrappers
#' @export
knn_graph <- function(...) {
  faissR::knn_graph(...)
}

#' @rdname faissR_wrappers
#' @export
fast_kmeans <- function(...) {
  faissR::fast_kmeans(...)
}

#' @rdname faissR_wrappers
#' @export
knn_fit <- function(...) {
  faissR::knn_fit(...)
}

#' @rdname faissR_wrappers
#' @export
predict_proba <- function(...) {
  faissR::predict_proba(...)
}

#' @rdname faissR_wrappers
#' @export
knn_recall <- function(...) {
  faissR::knn_recall(...)
}

#' Check whether the native CUDA embedding backend is available
#'
#' @return `TRUE` when fastEmbedR was built with CUDA embedding support and the
#'   CUDA runtime reports an available device.
#' @export
cuda_available <- function() {
  isTRUE(embedding_cuda_available_cpp())
}

#' Check whether the native Metal embedding backend is available
#'
#' @return `TRUE` when a Metal device is available to the fastEmbedR embedding
#'   kernels.
#' @export
metal_available <- function() {
  isTRUE(embedding_metal_available_cpp())
}

#' @rdname faissR_wrappers
#' @export
faiss_available <- function() {
  isTRUE(faissR::faiss_available())
}

#' @rdname faissR_wrappers
#' @export
cuvs_available <- function() {
  isTRUE(faissR::cuvs_available())
}
