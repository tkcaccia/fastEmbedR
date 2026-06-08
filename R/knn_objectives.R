#' Neighbor-embedding objectives from precomputed KNN
#'
#' @param indices Integer KNN index matrix without the self-neighbor column.
#' @param distances Numeric KNN distance matrix matching `indices`.
#' @param objective One of `"tsne"`, `"pacmap"`, `"trimap"`, or `"localmap"`.
#' @param n_components Output dimensionality.
#' @param n_epochs Number of optimization epochs.
#' @param negative_sample_rate Number of negative samples per positive edge.
#' @param learning_rate Initial learning rate.
#' @param n_threads Number of CPU threads for batch optimization.
#' @param seed Random seed.
#' @param verbose Print progress from C++.
#' @return A numeric embedding matrix.
#' @export
knn_embed <- function(indices,
                      distances,
                      objective = c("tsne", "pacmap", "trimap", "localmap"),
                      n_components = 2L,
                      n_epochs = 200L,
                      negative_sample_rate = 5L,
                      learning_rate = 0.1,
                      n_threads = 1L,
                      init = c("spectral", "random"),
                      spectral_n_iter = 25L,
                      seed = 4L,
                      verbose = FALSE) {
  objective <- match.arg(objective)
  init <- match.arg(init)
  indices <- as.matrix(indices)
  distances <- as.matrix(distances)
  if (!is.integer(indices)) storage.mode(indices) <- "integer"
  storage.mode(distances) <- "double"
  if (!identical(dim(indices), dim(distances))) {
    stop("`indices` and `distances` must have the same dimensions.", call. = FALSE)
  }
  init_embedding <- matrix(0, nrow(indices), as.integer(n_components))
  use_init <- init == "spectral"
  if (use_init) {
    init_embedding <- fast_knn_umap(
      indices,
      distances,
      n_components = n_components,
      mode = "spectral",
      spectral_n_iter = spectral_n_iter,
      seed = seed
    )
  }
  knn_objective_embed_cpp(
    indices,
    distances,
    objective,
    init_embedding,
    use_init,
    as.integer(n_components),
    as.integer(n_epochs),
    as.integer(negative_sample_rate),
    learning_rate,
    as.integer(n_threads),
    as.integer(seed),
    isTRUE(verbose)
  )
}

#' @rdname knn_embed
#' @export
knn_tsne <- function(indices, distances, ...) {
  knn_embed(indices, distances, objective = "tsne", ...)
}

#' @rdname knn_embed
#' @export
knn_pacmap <- function(indices, distances, ...) {
  knn_embed(indices, distances, objective = "pacmap", ...)
}

#' @rdname knn_embed
#' @export
knn_trimap <- function(indices, distances, ...) {
  knn_embed(indices, distances, objective = "trimap", ...)
}

#' @rdname knn_embed
#' @export
knn_localmap <- function(indices, distances, ...) {
  knn_embed(indices, distances, objective = "localmap", ...)
}
