normalize_nn_threads <- function(n_threads) {
  n_threads <- suppressWarnings(as.integer(n_threads))
  if (length(n_threads) != 1L || is.na(n_threads) || n_threads < 1L) {
    n_threads <- 1L
  }
  n_threads
}

.fastembedr_faissr_cache <- new.env(parent = emptyenv())

fastembedr_faissr_function <- function(name) {
  fn <- .fastembedr_faissr_cache[[name]]
  if (!is.null(fn)) return(fn)
  if (!requireNamespace("faissR", quietly = TRUE)) {
    stop(
      "Package `faissR` is required for one-call fastEmbedR embeddings. ",
      "Install faissR or pass a precomputed KNN object to `opentsne_knn()` ",
      "or `umap_knn()`.",
      call. = FALSE
    )
  }
  fn <- getExportedValue("faissR", name)
  .fastembedr_faissr_cache[[name]] <- fn
  fn
}

fastembedr_call_supported <- function(fn, args) {
  formal_names <- names(formals(fn))
  if (!("..." %in% formal_names)) {
    args <- args[names(args) %in% formal_names]
  }
  do.call(fn, args)
}

fastembedr_supports_formal <- function(fn, name) {
  name %in% names(formals(fn))
}

fastembedr_convert_knn_distances <- function(knn, output) {
  if (!identical(output, "float")) return(knn)
  if (!is.list(knn) || !("distances" %in% names(knn))) return(knn)
  if (is_float32_matrix(knn$distances)) return(knn)
  if (requireNamespace("float", quietly = TRUE)) {
    knn$distances <- float::fl(knn$distances)
    attr(knn, "distance_type") <- "float32"
  }
  knn
}

fastembedr_embedding_nn_policy <- function(embedding_backend) {
  embedding_backend <- resolve_embedding_backend(embedding_backend)
  if (identical(embedding_backend, "cuda")) {
    return(list(
      backend = "cuda",
      method = "auto",
      tuning = "auto",
      target_recall = 0.99
    ))
  }
  list(
    backend = "cpu",
    method = "hnsw",
    tuning = "auto",
    target_recall = 0.99
  )
}

fastembedr_nn_without_self <- function(data,
                                       k,
                                       backend,
                                       method = "auto",
                                       metric = "euclidean",
                                       output = "double",
                                       n_threads = NULL,
                                       tuning = "auto",
                                       target_recall = NULL) {
  fn <- fastembedr_faissr_function("nn")
  k <- as.integer(k)
  use_exclude_self <- fastembedr_supports_formal(fn, "exclude_self")
  args <- list(
    data = data,
    k = if (use_exclude_self) k else k + 1L,
    exclude_self = TRUE,
    backend = backend,
    method = method,
    metric = metric,
    tuning = tuning,
    output = output,
    n_threads = n_threads
  )
  if (!is.null(target_recall)) {
    args$target_recall <- target_recall
  }
  out <- fastembedr_call_supported(fn, args)
  if (!use_exclude_self && is.list(out) && !is.null(out$indices) && !is.null(out$distances)) {
    idx <- out$indices
    dst <- out$distances
    n <- nrow(idx)
    if (n > 0L && ncol(idx) > k) {
      new_idx <- matrix(NA_integer_, n, k)
      new_dst <- matrix(NA_real_, n, k)
      for (i in seq_len(n)) {
        keep <- which(idx[i, ] != i)
        if (length(keep) < k) keep <- seq_len(ncol(idx))
        keep <- keep[seq_len(min(k, length(keep)))]
        new_idx[i, seq_along(keep)] <- as.integer(idx[i, keep, drop = TRUE])
        new_dst[i, seq_along(keep)] <- as.numeric(dst[i, keep, drop = TRUE])
      }
      out$indices <- new_idx
      out$distances <- new_dst
    } else if (ncol(idx) > k) {
      out$indices <- idx[, seq_len(k), drop = FALSE]
      out$distances <- dst[, seq_len(k), drop = FALSE]
    }
  }
  fastembedr_convert_knn_distances(out, output)
}
