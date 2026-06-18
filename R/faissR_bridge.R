normalize_nn_threads <- function(n_threads) {
  n_threads <- suppressWarnings(as.integer(n_threads))
  if (length(n_threads) != 1L || is.na(n_threads) || n_threads < 1L) {
    n_threads <- 1L
  }
  n_threads
}
