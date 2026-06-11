#' Embed from precomputed KNN
#'
#' @param indices A KNN object returned by `nn()`, or an integer KNN index
#'   matrix. If a self-neighbor first column is present it is removed
#'   automatically.
#' @param distances Numeric KNN distance matrix matching `indices`. Leave as
#'   `NULL` when `indices` is an `nn()` result.
#' @param method Embedding method. Use `"umap"` for the package's focused UMAP
#'   path, `"tsne"` for an `Rtsne_neighbors()`-style exact t-SNE optimizer, or
#'   `"infotsne"` for a TorchDR-inspired negative-sampling t-SNE objective from
#'   precomputed neighbors.
#' @param n_components Output dimensionality.
#' @param seed Random seed.
#' @param verbose Print native optimizer progress.
#' @param backend Execution backend. `"gpu"` explicitly requests a real native
#'   GPU backend and errors if CUDA or Metal is unavailable.
#' @param ... Additional method-specific parameters. For `method = "tsne"` or
#'   `"infotsne"`, supported parameters include `perplexity`, `max_iter`,
#'   `Y_init`, learning-rate controls, exaggeration controls, and `n_threads`.
#' @return A numeric embedding matrix with resolved settings stored in
#'   `attr(layout, "fastEmbedR_config")`.
#' @export
embed_knn <- function(indices,
                      distances = NULL,
                      method = "umap",
                      n_components = 2L,
                      seed = 4L,
                      verbose = FALSE,
                      backend = c("auto", "cpu", "gpu", "metal", "cuda"),
                      ...) {
  method <- match.arg(method, c("umap", "tsne", "infotsne"))
  backend <- match.arg(backend)
  n_components <- validate_n_components(n_components)

  if (identical(method, "umap")) {
    return(fast_knn_umap(
      indices,
      distances,
      n_components = n_components,
      seed = seed,
      verbose = verbose,
      backend = backend
    ))
  }
  if (identical(method, "tsne")) {
    return(fast_knn_tsne_core(
      indices,
      distances,
      n_components = n_components,
      seed = seed,
      verbose = verbose,
      backend = backend,
      ...
    ))
  }
  fast_knn_infotsne_core(
    indices,
    distances,
    n_components = n_components,
    seed = seed,
    verbose = verbose,
    backend = backend,
    ...
  )
}
