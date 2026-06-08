#' Benchmark fastknnumap on the metref example data
#'
#' @param data_path Path to `metref_remote_task.RData`.
#' @param k Number of neighbors requested from `nn`, including self.
#' @param n_epochs Number of UMAP optimization epochs.
#' @param seed Random seed.
#' @return A list containing timings, silhouettes, and embeddings.
#' @export
benchmark_metref <- function(data_path = "/Users/stefano/Documents/GPUPLS/Data/metref_remote_task.RData",
                             k = 30L,
                             n_epochs = 200L,
                             min_dist = 0.01,
                             negative_sample_rate = 10L,
                             learning_rate = 1.5,
                             seed = 4L) {
  if (!requireNamespace("cluster", quietly = TRUE)) {
    stop("Package `cluster` is required for silhouette scoring.", call. = FALSE)
  }

  env <- new.env(parent = emptyenv())
  load(data_path, envir = env)
  out <- env$out
  data <- scale(out$Xtrain)
  labels <- as.integer(as.factor(out$Ytrain))

  t_knn <- system.time({
    nn <- fastknnumap::nn(data, data, k, parallel = TRUE)
  })

  indices <- nn$indices[, -1L, drop = FALSE]
  distances <- nn$distances[, -1L, drop = FALSE]

  t_fast <- system.time({
    fast <- fast_knn_umap(
      indices,
      distances,
      n_epochs = n_epochs,
      min_dist = min_dist,
      negative_sample_rate = negative_sample_rate,
      learning_rate = learning_rate,
      seed = seed
    )
  })

  fast_sil <- mean(cluster::silhouette(labels, dist(fast))[, "sil_width"])

  result <- list(
    timings = rbind(knn = t_knn, fastknnumap = t_fast),
    silhouette = c(fastknnumap = fast_sil),
    layout = fast,
    labels = out$Ytrain
  )

  if (requireNamespace("umap", quietly = TRUE)) {
    t_umap <- system.time({
      uknn <- umap::umap.knn(indices, distances)
      config <- umap::umap.defaults
      config$knn <- uknn
      reference <- umap::umap(data, knn = uknn, config = config)$layout
    })
    result$timings <- rbind(result$timings, umap_knn = t_umap)
    result$silhouette <- c(
      result$silhouette,
      umap_knn = mean(cluster::silhouette(labels, dist(reference))[, "sil_width"])
    )
    result$reference_layout <- reference
  }

  result
}

#' Benchmark fastknnumap on the single-cell example data
#'
#' @param data_path Path to `singlecell.RData`.
#' @param k Number of neighbors requested from `nn`, including self.
#' @param n_features Number of leading feature columns to use.
#' @param mode Embedding mode passed to `fast_knn_umap`.
#' @param n_epochs Number of optimization epochs for `"sgd"` or `"hybrid"` modes.
#' @param spectral_n_iter Number of spectral power iterations.
#' @param silhouette_sample Optional sample size for silhouette scoring. Use `NULL`
#'   for a full silhouette.
#' @param seed Random seed.
#' @param run_umap Compare against `umap`'s precomputed-KNN path.
#' @return A list containing timings, silhouettes, and embeddings.
#' @export
benchmark_singlecell <- function(data_path = "/Users/stefano/Documents/GPUPLS/Data/singlecell.RData",
                                 k = 30L,
                                 n_features = 20L,
                                 mode = c("hybrid", "spectral", "sgd"),
                                 n_epochs = 100L,
                                 spectral_n_iter = 25L,
                                 silhouette_sample = NULL,
                                 seed = 4L,
                                 run_umap = TRUE) {
  mode <- match.arg(mode)
  if (!requireNamespace("cluster", quietly = TRUE)) {
    stop("Package `cluster` is required for silhouette scoring.", call. = FALSE)
  }

  env <- new.env(parent = emptyenv())
  load(data_path, envir = env)
  data <- as.matrix(env$data[, seq_len(n_features), drop = FALSE])
  labels <- as.integer(as.factor(env$labels))

  t_knn <- system.time({
    nn <- fastknnumap::nn(data, data, k, parallel = TRUE)
  })
  indices <- nn$indices[, -1L, drop = FALSE]
  distances <- nn$distances[, -1L, drop = FALSE]

  t_fast <- system.time({
    fast <- fast_knn_umap(
      indices,
      distances,
      mode = mode,
      n_epochs = n_epochs,
      min_dist = 0.01,
      negative_sample_rate = 10L,
      learning_rate = 1.5,
      spectral_n_iter = spectral_n_iter,
      seed = seed
    )
  })

  score <- function(layout) {
    if (is.null(silhouette_sample) || silhouette_sample >= nrow(layout)) {
      return(mean(cluster::silhouette(labels, stats::dist(layout))[, "sil_width"]))
    }
    set.seed(seed)
    keep <- sort(sample.int(nrow(layout), silhouette_sample))
    mean(cluster::silhouette(labels[keep], stats::dist(layout[keep, , drop = FALSE]))[, "sil_width"])
  }

  result <- list(
    timings = rbind(knn = t_knn, fastknnumap = t_fast),
    silhouette = c(fastknnumap = score(fast)),
    layout = fast,
    labels = env$labels
  )

  if (isTRUE(run_umap) && requireNamespace("umap", quietly = TRUE)) {
    t_umap <- system.time({
      uknn <- umap::umap.knn(indices, distances)
      config <- umap::umap.defaults
      config$knn <- uknn
      reference <- umap::umap(data, knn = uknn, config = config)$layout
    })
    result$timings <- rbind(result$timings, umap_knn = t_umap)
    result$silhouette <- c(result$silhouette, umap_knn = score(reference))
    result$reference_layout <- reference
  }

  result
}

#' Benchmark UMAP implementations from the same KNN output
#'
#' @param data Numeric matrix used only to compute KNN once and, where required,
#'   to satisfy reference implementation APIs. The benchmarked layouts all use
#'   the same KNN output.
#' @param labels Optional labels for silhouette accuracy.
#' @param k Number of neighbors requested from `nn`, including self.
#' @param nn Optional precomputed KNN list with `indices` and `distances`,
#'   including self in the first column as returned by `nn`.
#' @param implementations Character vector of implementations to run.
#' @param n_epochs Number of optimization epochs for SGD-style methods.
#' @param hybrid_epochs Number of optimization epochs for `fastknnumap_hybrid`.
#' @param min_dist UMAP minimum distance.
#' @param negative_sample_rate Negative sample rate.
#' @param learning_rate Initial learning rate.
#' @param spectral_n_iter Number of spectral power iterations for fastknnumap.
#' @param silhouette_sample Optional sample size for silhouette scoring.
#' @param preserve_sample Optional sample size for KNN preservation scoring.
#' @param preserve_k Number of neighbors used for preservation scoring. Defaults
#'   to the KNN matrix width after dropping self.
#' @param seed Random seed.
#' @param verbose Print progress.
#' @return A list with KNN timing, per-implementation metrics, layouts, labels,
#'   and the KNN matrices used.
#' @export
benchmark_knn_umap <- function(data,
                               labels = NULL,
                               k = 30L,
                               nn = NULL,
                               implementations = c(
                                 "fastknnumap_hybrid",
                                 "fastknnumap_landmark",
                                 "fastknnumap_sgd",
                                 "fastknnumap_spectral",
                                 "umap",
                                 "uwot",
                                 "uwot_fast_sgd",
                                 "knn_tsne",
                                 "knn_pacmap",
                                 "knn_trimap",
                                 "knn_localmap"
                               ),
                               n_epochs = 200L,
                               hybrid_epochs = 100L,
                               min_dist = 0.01,
                               negative_sample_rate = 10L,
                               learning_rate = 1.5,
                               spectral_n_iter = 25L,
                               landmark_ratio = 0.1,
                               n_landmarks = NULL,
                               landmark_k = 10L,
                               landmark_local_k = 5L,
                               n_threads = 1L,
                               silhouette_sample = 5000L,
                               preserve_sample = 5000L,
                               preserve_k = NULL,
                               seed = 4L,
                               verbose = TRUE) {
  data <- as.matrix(data)
  n <- nrow(data)
  if (n < 2L) {
    stop("`data` must contain at least two rows.", call. = FALSE)
  }
  if (!is.null(labels) && length(labels) != n) {
    stop("`labels` must have one entry per row of `data`.", call. = FALSE)
  }

  t_knn <- NULL
  if (is.null(nn)) {
    if (isTRUE(verbose)) message("Computing KNN once with fastknnumap::nn()")
    t_knn <- system.time({
      nn <- fastknnumap::nn(data, data, k, parallel = TRUE, cores = n_threads)
    })
  } else {
    if (!all(c("indices", "distances") %in% names(nn))) {
      stop("`nn` must contain `indices` and `distances`.", call. = FALSE)
    }
  }

  nn_indices_self <- as.matrix(nn$indices)
  nn_distances_self <- as.matrix(nn$distances)
  if (!is.integer(nn_indices_self)) {
    storage.mode(nn_indices_self) <- "integer"
  }
  storage.mode(nn_distances_self) <- "double"
  if (!identical(dim(nn_indices_self), dim(nn_distances_self))) {
    stop("KNN `indices` and `distances` must have the same dimensions.", call. = FALSE)
  }
  if (nrow(nn_indices_self) != n) {
    stop("KNN matrix row count must match `nrow(data)`.", call. = FALSE)
  }
  if (ncol(nn_indices_self) < 2L) {
    stop("KNN matrices must include self plus at least one neighbor.", call. = FALSE)
  }

  indices <- nn_indices_self[, -1L, drop = FALSE]
  distances <- nn_distances_self[, -1L, drop = FALSE]
  preserve_k <- if (is.null(preserve_k)) ncol(indices) else min(as.integer(preserve_k), ncol(indices))

  labels_int <- if (is.null(labels)) NULL else as.integer(as.factor(labels))
  sample_indices <- function(sample_size) {
    if (is.null(sample_size) || sample_size >= n) return(seq_len(n))
    set.seed(seed)
    sort(sample.int(n, sample_size))
  }
  silhouette_keep <- sample_indices(silhouette_sample)
  preserve_keep <- sample_indices(preserve_sample)

  score_silhouette <- function(layout) {
    if (is.null(labels_int)) return(NA_real_)
    if (!requireNamespace("cluster", quietly = TRUE)) return(NA_real_)
    mean(cluster::silhouette(
      labels_int[silhouette_keep],
      stats::dist(layout[silhouette_keep, , drop = FALSE])
    )[, "sil_width"])
  }

  score_preservation <- function(layout) {
    layout <- as.matrix(layout)
    scores <- numeric(length(preserve_keep))
    for (ii in seq_along(preserve_keep)) {
      i <- preserve_keep[ii]
      dx <- rowSums((layout - matrix(layout[i, ], n, ncol(layout), byrow = TRUE))^2)
      dx[i] <- Inf
      low_nn <- order(dx)[seq_len(preserve_k)]
      high_nn <- indices[i, seq_len(preserve_k)]
      scores[ii] <- sum(low_nn %in% high_nn) / preserve_k
    }
    mean(scores)
  }

  run_one <- function(name) {
    if (isTRUE(verbose)) message("Running ", name)
    layout <- NULL
    err <- NULL
    timing <- system.time({
      layout <- tryCatch({
        switch(
          name,
          fastknnumap_spectral = fast_knn_umap(
            indices, distances,
            mode = "spectral",
            min_dist = min_dist,
            negative_sample_rate = negative_sample_rate,
            learning_rate = learning_rate,
            spectral_n_iter = spectral_n_iter,
            seed = seed
          ),
          fastknnumap_hybrid = fast_knn_umap(
            indices, distances,
            mode = "hybrid",
            n_epochs = hybrid_epochs,
            min_dist = min_dist,
            negative_sample_rate = negative_sample_rate,
            learning_rate = learning_rate,
            spectral_n_iter = spectral_n_iter,
            seed = seed
          ),
          fastknnumap_sgd = fast_knn_umap(
            indices, distances,
            mode = "sgd",
            n_epochs = n_epochs,
            min_dist = min_dist,
            negative_sample_rate = negative_sample_rate,
            learning_rate = learning_rate,
            spectral_n_iter = spectral_n_iter,
            seed = seed
          ),
          fastknnumap_landmark = landmark_knn_umap(
            indices,
            distances,
            n_landmarks = n_landmarks,
            landmark_ratio = landmark_ratio,
            landmark_k = landmark_k,
            local_k = landmark_local_k,
            mode = "hybrid",
            n_epochs = hybrid_epochs,
            min_dist = min_dist,
            negative_sample_rate = negative_sample_rate,
            learning_rate = learning_rate,
            spectral_n_iter = spectral_n_iter,
            seed = seed
          ),
          knn_tsne = knn_tsne(
            indices, distances,
            n_epochs = n_epochs,
            negative_sample_rate = negative_sample_rate,
            learning_rate = 0.05,
            n_threads = n_threads,
            seed = seed
          ),
          knn_pacmap = knn_pacmap(
            indices, distances,
            n_epochs = n_epochs,
            negative_sample_rate = negative_sample_rate,
            learning_rate = 0.1,
            n_threads = n_threads,
            seed = seed
          ),
          knn_trimap = knn_trimap(
            indices, distances,
            n_epochs = n_epochs,
            negative_sample_rate = negative_sample_rate,
            learning_rate = 0.05,
            n_threads = n_threads,
            seed = seed
          ),
          knn_localmap = knn_localmap(
            indices, distances,
            n_epochs = n_epochs,
            negative_sample_rate = negative_sample_rate,
            learning_rate = 0.05,
            n_threads = n_threads,
            seed = seed
          ),
          umap = {
            if (!requireNamespace("umap", quietly = TRUE)) {
              stop("Package `umap` is not installed.", call. = FALSE)
            }
            uknn <- umap::umap.knn(indices, distances)
            config <- umap::umap.defaults
            config$knn <- uknn
            config$n_neighbors <- ncol(indices)
            config$n_epochs <- n_epochs
            config$min_dist <- min_dist
            umap::umap(data, knn = uknn, config = config)$layout
          },
          uwot = {
            if (!requireNamespace("uwot", quietly = TRUE)) {
              stop("Package `uwot` is not installed.", call. = FALSE)
            }
            uwot::umap(
              X = NULL,
              n_neighbors = ncol(nn_indices_self),
              nn_method = list(idx = nn_indices_self, dist = nn_distances_self),
              n_epochs = n_epochs,
              learning_rate = learning_rate,
              min_dist = min_dist,
              negative_sample_rate = negative_sample_rate,
              init = "spectral",
              seed = seed,
              verbose = FALSE
            )
          },
          uwot_fast_sgd = {
            if (!requireNamespace("uwot", quietly = TRUE)) {
              stop("Package `uwot` is not installed.", call. = FALSE)
            }
            uwot::umap(
              X = NULL,
              n_neighbors = ncol(nn_indices_self),
              nn_method = list(idx = nn_indices_self, dist = nn_distances_self),
              n_epochs = n_epochs,
              learning_rate = learning_rate,
              min_dist = min_dist,
              negative_sample_rate = negative_sample_rate,
              init = "spectral",
              fast_sgd = TRUE,
              seed = seed,
              verbose = FALSE
            )
          },
          stop("Unknown implementation: ", name, call. = FALSE)
        )
      }, error = function(e) {
        err <<- conditionMessage(e)
        NULL
      })
    })

    if (is.null(layout)) {
      return(list(
        metrics = data.frame(
          implementation = name,
          elapsed = timing["elapsed"],
          user = timing["user.self"],
          system = timing["sys.self"],
          silhouette = NA_real_,
          knn_preservation = NA_real_,
          status = "error",
          error = err,
          stringsAsFactors = FALSE
        ),
        layout = NULL
      ))
    }

    list(
      metrics = data.frame(
        implementation = name,
        elapsed = timing["elapsed"],
        user = timing["user.self"],
        system = timing["sys.self"],
        silhouette = score_silhouette(layout),
        knn_preservation = score_preservation(layout),
        status = "ok",
        error = "",
        stringsAsFactors = FALSE
      ),
      layout = layout
    )
  }

  runs <- lapply(implementations, run_one)
  names(runs) <- implementations
  metrics <- do.call(rbind, lapply(runs, `[[`, "metrics"))
  rownames(metrics) <- NULL
  metrics <- metrics[order(metrics$status, metrics$elapsed), , drop = FALSE]

  list(
    knn_time = t_knn,
    metrics = metrics,
    layouts = lapply(runs, `[[`, "layout"),
    labels = labels,
    knn = list(indices = indices, distances = distances),
    knn_with_self = list(indices = nn_indices_self, distances = nn_distances_self),
    silhouette_sample = silhouette_keep,
    preservation_sample = preserve_keep
  )
}
