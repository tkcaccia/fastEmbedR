#' Embed from precomputed KNN
#'
#' @param indices A KNN object returned by `nn()`, or an integer KNN index
#'   matrix. If a self-neighbor first column is present it is removed
#'   automatically.
#' @param distances Numeric KNN distance matrix matching `indices`. Leave as
#'   `NULL` when `indices` is an `nn()` result.
#' @param method Embedding method: `"opentsne"` or `"umap"`.
#' @param n_components Output dimensionality.
#' @param seed Random seed.
#' @param verbose Print native optimizer progress.
#' @param backend Execution backend. `"gpu"` explicitly requests a real native
#'   GPU backend and errors if CUDA or Metal is unavailable.
#' @param n_threads Number of CPU worker threads used by the CPU optimizer.
#'   Native GPU optimizers ignore this argument.
#' @param ... Additional openTSNE-specific parameters such as `perplexity`,
#'   iteration controls, `Y_init`, learning-rate controls, exaggeration
#'   controls.
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
                      n_threads = NULL,
                      ...) {
  method <- match.arg(method, c("umap", "opentsne"))
  backend <- match.arg(backend)
  n_components <- validate_n_components(n_components)

  if (identical(method, "umap")) {
    return(fast_knn_umap(
      indices,
      distances,
      n_components = n_components,
      seed = seed,
      verbose = verbose,
      backend = backend,
      n_threads = n_threads
    ))
  }

  fast_knn_opentsne_core(
    indices,
    distances,
    n_components = n_components,
    seed = seed,
    verbose = verbose,
    backend = backend,
    n_threads = n_threads,
    ...
  )
}
