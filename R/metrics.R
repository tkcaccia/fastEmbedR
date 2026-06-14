silhouette_score <- function(labels, layout) {
  layout <- as.matrix(layout)
  labels <- as.factor(labels)
  if (length(labels) != nrow(layout)) {
    stop("`labels` must have one entry per row of `layout`.", call. = FALSE)
  }
  silhouette_score_cpp(layout, as.integer(labels))
}

cuda_metric_requested <- function(backend) {
  identical(backend, "cuda") || identical(backend, "gpu")
}

cuda_metric_available <- function() {
  isTRUE(embedding_cuda_available_cpp())
}

metal_metric_requested <- function(backend) {
  identical(backend, "metal")
}

metal_metric_available <- function() {
  isTRUE(embedding_metal_available_cpp())
}

resolve_metric_backend <- function(backend) {
  backend <- as.character(backend)[1L]
  if (length(backend) != 1L || is.na(backend) || !nzchar(backend)) {
    backend <- "auto"
  }
  backend <- tolower(backend)
  if (backend %in% c("auto", "cpu")) {
    return(list(backend = "cpu", reason = NA_character_))
  }
  if (identical(backend, "gpu")) {
    selected <- available_native_gpu_backend(need_embedding = TRUE)
    if (identical(selected, "cuda") && cuda_metric_available()) {
      return(list(backend = selected, reason = NA_character_))
    }
    if (identical(selected, "metal")) {
      return(list(backend = "cpu", reason = "metal_knn_metric_backend_unavailable"))
    }
    return(list(backend = "cpu", reason = "gpu_metric_backend_unavailable"))
  }
  if (identical(backend, "metal")) {
    return(list(backend = "cpu", reason = "metal_knn_metric_backend_unavailable"))
  }
  if (identical(backend, "cuda")) {
    if (cuda_metric_available()) {
      return(list(backend = "cuda", reason = NA_character_))
    }
    return(list(backend = "cpu", reason = "cuda_metric_backend_unavailable"))
  }
  list(backend = "cpu", reason = paste0(backend, "_metric_backend_not_supported"))
}

structure_score_with_backend <- function(layout,
                                         indices,
                                         keep,
                                         preserve_k,
                                         labels_int,
                                         n_label_levels,
                                         backend = "cpu") {
  if (!is.matrix(layout)) layout <- as.matrix(layout)
  if (!is.matrix(indices)) indices <- as.matrix(indices)
  if (!is.integer(indices)) storage.mode(indices) <- "integer"
  keep <- as.integer(keep)
  labels_int <- if (length(labels_int) == 0L) integer(0L) else as.integer(labels_int)
  preserve_k <- as.integer(preserve_k)

  reason <- NA_character_
  if (cuda_metric_requested(backend)) {
    if (!cuda_metric_available()) {
      reason <- "cuda_scoring_unavailable"
    } else if (ncol(layout) != 2L) {
      reason <- "cuda_scoring_requires_2d_layout"
    } else if (preserve_k > 64L) {
      reason <- "cuda_scoring_supports_at_most_64_neighbors"
    } else {
      out <- tryCatch(
        knn_structure_score_cuda_cpp(
          layout,
          indices,
          keep,
          preserve_k,
          labels_int,
          as.integer(n_label_levels)
        ),
        error = function(e) {
          reason <<- conditionMessage(e)
          NULL
        }
      )
      if (!is.null(out)) {
        return(list(values = out, backend = "cuda", reason = NA_character_))
      }
    }
  }
  if (metal_metric_requested(backend)) {
    if (!metal_metric_available()) {
      reason <- append_metric_backend_reason(reason, "metal_scoring_unavailable")
    } else if (ncol(layout) != 2L) {
      reason <- append_metric_backend_reason(reason, "metal_scoring_requires_2d_layout")
    } else if (preserve_k > 64L) {
      reason <- append_metric_backend_reason(reason, "metal_scoring_supports_at_most_64_neighbors")
    } else {
      out <- tryCatch(
        knn_structure_score_metal_cpp(
          layout,
          indices,
          keep,
          preserve_k,
          labels_int,
          as.integer(n_label_levels)
        ),
        error = function(e) {
          reason <<- append_metric_backend_reason(reason, conditionMessage(e))
          NULL
        }
      )
      if (!is.null(out)) {
        return(list(values = out, backend = "metal", reason = NA_character_))
      }
    }
  }

  out <- knn_structure_score_cpp(
    layout,
    indices,
    keep,
    preserve_k,
    labels_int,
    as.integer(n_label_levels)
  )
  list(values = out, backend = "cpu", reason = reason)
}

silhouette_score_with_backend <- function(labels_int,
                                          layout,
                                          n_label_levels,
                                          backend = "cpu") {
  if (length(labels_int) == 0L || nrow(layout) < 2L || n_label_levels < 2L) {
    return(list(value = NA_real_, backend = "none", reason = NA_character_))
  }
  layout <- as.matrix(layout)
  labels_int <- as.integer(labels_int)
  reason <- NA_character_
  if (cuda_metric_requested(backend)) {
    if (!cuda_metric_available()) {
      reason <- "cuda_scoring_unavailable"
    } else if (ncol(layout) != 2L) {
      reason <- "cuda_silhouette_requires_2d_layout"
    } else if (n_label_levels > 128L) {
      reason <- "cuda_silhouette_supports_at_most_128_label_levels"
    } else {
      out <- tryCatch(
        silhouette_score_cuda_cpp(
          layout,
          labels_int,
          as.integer(n_label_levels)
        ),
        error = function(e) {
          reason <<- conditionMessage(e)
          NULL
        }
      )
      if (!is.null(out)) {
        return(list(value = out, backend = "cuda", reason = NA_character_))
      }
    }
  }
  if (metal_metric_requested(backend)) {
    if (!metal_metric_available()) {
      reason <- append_metric_backend_reason(reason, "metal_scoring_unavailable")
    } else if (ncol(layout) != 2L) {
      reason <- append_metric_backend_reason(reason, "metal_silhouette_requires_2d_layout")
    } else if (n_label_levels > 128L) {
      reason <- append_metric_backend_reason(reason, "metal_silhouette_supports_at_most_128_label_levels")
    } else {
      out <- tryCatch(
        silhouette_score_metal_cpp(
          layout,
          labels_int,
          as.integer(n_label_levels)
        ),
        error = function(e) {
          reason <<- append_metric_backend_reason(reason, conditionMessage(e))
          NULL
        }
      )
      if (!is.null(out)) {
        return(list(value = out, backend = "metal", reason = NA_character_))
      }
    }
  }

  list(
    value = silhouette_score_cpp(layout, labels_int),
    backend = "cpu",
    reason = reason
  )
}

sample_indices <- function(n, sample_size = NULL, seed = 4L) {
  if (is.null(sample_size)) {
    return(integer(0L))
  }
  sample_size <- as.integer(sample_size)
  if (length(sample_size) != 1L || is.na(sample_size) || !is.finite(sample_size) || sample_size < 1L) {
    return(integer(0L))
  }
  if (sample_size >= n) {
    return(seq_len(n))
  }
  set.seed(seed)
  sort(sample.int(n, sample_size))
}
