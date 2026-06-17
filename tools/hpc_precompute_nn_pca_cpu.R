#!/usr/bin/env Rscript

parse_args <- function(args) {
  out <- list()
  for (arg in args) {
    if (!startsWith(arg, "--")) next
    kv <- strsplit(sub("^--", "", arg), "=", fixed = TRUE)[[1L]]
    out[[kv[[1L]]]] <- if (length(kv) > 1L) paste(kv[-1L], collapse = "=") else TRUE
  }
  out
}

`%||%` <- function(x, y) {
  if (is.null(x) || length(x) == 0L || (length(x) == 1L && is.na(x))) y else x
}

args <- parse_args(commandArgs(trailingOnly = TRUE))
base_dir <- normalizePath(args$base_dir %||% "/scratch/firenze/NN", mustWork = FALSE)
data_root <- normalizePath(args$data_root %||% file.path(base_dir, "Data"), mustWork = FALSE)
k <- as.integer(args$k %||% "100")
threads <- as.integer(args$threads %||% "4")
seed <- as.integer(args$seed %||% "4")
force <- isTRUE(as.logical(args$force %||% FALSE))
nn_backend <- args$nn_backend %||% "faiss_gpu_ivf_flat"
datasets_arg <- args$datasets %||% ""

if (!dir.exists(data_root)) stop("Data root does not exist: ", data_root, call. = FALSE)
if (is.na(k) || k < 1L) stop("`k` must be a positive integer.", call. = FALSE)
if (is.na(threads) || threads < 1L) threads <- 12L
threads <- max(1L, threads)
if (is.na(seed)) seed <- 4L

Sys.setenv(
  OMP_NUM_THREADS = as.character(threads),
  OPENBLAS_NUM_THREADS = as.character(threads),
  MKL_NUM_THREADS = as.character(threads),
  VECLIB_MAXIMUM_THREADS = as.character(threads)
)
set.seed(seed)

log_msg <- function(...) {
  cat(format(Sys.time(), "%Y-%m-%d %H:%M:%S"), " ", sprintf(...), "\n", sep = "")
  flush.console()
}

need_pkg <- function(pkg) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    stop("Required package not available: ", pkg, call. = FALSE)
  }
}
need_pkg("faissR")

dataset_files <- function(data_root, datasets_arg) {
  if (nzchar(datasets_arg)) {
    datasets <- trimws(strsplit(datasets_arg, ",", fixed = TRUE)[[1L]])
    datasets <- datasets[nzchar(datasets)]
    files <- vapply(datasets, function(ds) {
      hits <- list.files(file.path(data_root, ds), pattern = "\\.[Rr][Dd]ata$", full.names = TRUE)
      hits <- hits[!grepl("(_nn|pca|manifest|summary)", basename(hits), ignore.case = TRUE)]
      if (!length(hits)) stop("No source RData file found for dataset ", ds, call. = FALSE)
      hits[[1L]]
    }, character(1L))
    return(data.frame(dataset = datasets, path = unname(files), stringsAsFactors = FALSE))
  }

  manifest <- file.path(data_root, "dataset_manifest.csv")
  if (file.exists(manifest)) {
    tab <- utils::read.csv(manifest, stringsAsFactors = FALSE)
    if (all(c("dataset", "relative_path") %in% names(tab))) {
      paths <- file.path(data_root, tab$relative_path)
      keep <- file.exists(paths)
      if (any(keep)) {
        return(data.frame(dataset = tab$dataset[keep], path = paths[keep], stringsAsFactors = FALSE))
      }
    }
  }

  dirs <- list.dirs(data_root, recursive = FALSE, full.names = TRUE)
  rows <- lapply(dirs, function(d) {
    hits <- list.files(d, pattern = "\\.[Rr][Dd]ata$", full.names = TRUE)
    hits <- hits[!grepl("(_nn|pca|manifest|summary)", basename(hits), ignore.case = TRUE)]
    if (!length(hits)) return(NULL)
    data.frame(dataset = basename(d), path = hits[[1L]], stringsAsFactors = FALSE)
  })
  out <- do.call(rbind, rows)
  if (is.null(out) || !nrow(out)) stop("No dataset RData files found under ", data_root, call. = FALSE)
  out
}

pick_dataset_object <- function(path) {
  env <- new.env(parent = emptyenv())
  names <- load(path, envir = env)
  for (nm in names) {
    obj <- get(nm, envir = env, inherits = FALSE)
    if (is.list(obj) && !is.null(obj$data)) {
      return(list(data = obj$data, labels = obj$labels %||% NULL, object_name = nm))
    }
  }
  for (nm in names) {
    obj <- get(nm, envir = env, inherits = FALSE)
    if (is.matrix(obj) || is.data.frame(obj) || inherits(obj, "Matrix")) {
      labels <- NULL
      for (candidate in c("labels", "label", "Y", "y", "classes", "class")) {
        if (exists(candidate, envir = env, inherits = FALSE)) {
          lab <- get(candidate, envir = env, inherits = FALSE)
          if (length(lab) == nrow(obj)) {
            labels <- lab
            break
          }
        }
      }
      return(list(data = obj, labels = labels, object_name = nm))
    }
  }
  stop("Could not find a dataset object in ", path, call. = FALSE)
}

as_numeric_matrix <- function(x) {
  if (inherits(x, "Matrix")) x <- as.matrix(x)
  if (is.data.frame(x)) x <- as.matrix(x)
  if (!is.matrix(x)) x <- as.matrix(x)
  storage.mode(x) <- "double"
  x
}

center_matrix <- function(x) {
  x <- as_numeric_matrix(x)
  finite_col <- vapply(seq_len(ncol(x)), function(j) all(is.finite(x[, j])), logical(1L))
  if (!all(finite_col)) x <- x[, finite_col, drop = FALSE]
  if (!ncol(x)) stop("No fully finite columns after filtering.", call. = FALSE)
  sds <- apply(x, 2L, stats::sd)
  keep <- is.finite(sds) & sds > 0
  x <- x[, keep, drop = FALSE]
  if (!ncol(x)) stop("No variable columns after filtering.", call. = FALSE)
  centers <- colMeans(x)
  x <- sweep(x, 2L, centers, "-")
  x[!is.finite(x)] <- 0
  storage.mode(x) <- "double"
  attr(x, "column_centers") <- centers
  attr(x, "kept_columns") <- which(finite_col)[keep]
  x
}

extract_scores <- function(obj) {
  candidates <- c("scores", "x", "T", "t", "Xscores", "score")
  for (nm in candidates) {
    if (!is.null(obj[[nm]]) && is.matrix(obj[[nm]]) && ncol(obj[[nm]]) >= 2L) {
      return(as.matrix(obj[[nm]][, 1:2, drop = FALSE]))
    }
  }
  if (is.matrix(obj) && ncol(obj) >= 2L) return(as.matrix(obj[, 1:2, drop = FALSE]))
  NULL
}

compute_pca2 <- function(x) {
  if (requireNamespace("fastPLS", quietly = TRUE) && "pca" %in% getNamespaceExports("fastPLS")) {
    fit <- tryCatch(fastPLS::pca(x, ncomp = 2L), error = function(e) e)
    if (!inherits(fit, "error")) {
      scores <- extract_scores(fit)
      if (!is.null(scores) && nrow(scores) == nrow(x)) {
        attr(scores, "pca_backend") <- "fastPLS::pca"
        return(scores)
      }
    }
  }

  if (nrow(x) >= ncol(x)) {
    cov_x <- crossprod(x) / max(1L, nrow(x) - 1L)
    eig <- eigen(cov_x, symmetric = TRUE)
    scores <- x %*% eig$vectors[, 1:2, drop = FALSE]
    attr(scores, "pca_backend") <- "covariance_eigen"
    return(as.matrix(scores))
  }

  pc <- stats::prcomp(x, center = FALSE, scale. = FALSE, rank. = 2L)
  scores <- pc$x[, 1:2, drop = FALSE]
  attr(scores, "pca_backend") <- "stats::prcomp"
  as.matrix(scores)
}

to_tsne_init <- function(scores) {
  scores <- as.matrix(scores[, 1:2, drop = FALSE])
  scores <- scale(scores, center = TRUE, scale = FALSE)
  sds <- apply(scores, 2L, stats::sd)
  scale_factor <- max(sds[is.finite(sds)], na.rm = TRUE)
  if (!is.finite(scale_factor) || scale_factor <= 0) scale_factor <- 1
  init <- scores * (1e-4 / scale_factor)
  storage.mode(init) <- "double"
  init
}

compute_nn <- function(x, dataset) {
  backends <- unique(trimws(strsplit(nn_backend, ",", fixed = TRUE)[[1L]]))
  backends <- backends[nzchar(backends)]
  last_error <- NULL
  for (backend in backends) {
    log_msg("%s: NN backend=%s k=%d", dataset, backend, k)
    t <- system.time({
      ans <- tryCatch(
        faissR::nn_without_self(x, k = k, backend = backend, metric = "euclidean", n_threads = threads),
        error = function(e) e
      )
    })[["elapsed"]]
    if (!inherits(ans, "error")) {
      attr(ans, "precompute_backend") <- backend
      attr(ans, "precompute_sec") <- as.numeric(t)
      return(ans)
    }
    last_error <- paste0(backend, ": ", conditionMessage(ans))
    log_msg("%s: NN failed: %s", dataset, last_error)
    gc()
  }
  stop("All NN backends failed. Last error: ", last_error, call. = FALSE)
}

datasets <- dataset_files(data_root, datasets_arg)
datasets <- datasets[order(match(datasets$dataset, c(
  "COIL20", "USPS", "FashionMNIST", "FlowRepository_FR-FCM-ZYRM_files",
  "flow18", "MNIST", "imagenet", "MetRef", "mass41", "TabulaMuris"
)), datasets$dataset), , drop = FALSE]

summary_rows <- list()
pca_rows <- list()
knn_rows <- list()

for (i in seq_len(nrow(datasets))) {
  dataset <- datasets$dataset[[i]]
  source_path <- normalizePath(datasets$path[[i]], mustWork = TRUE)
  dataset_dir <- dirname(source_path)
  pca_path <- file.path(dataset_dir, paste0(dataset, "_centered_raw_opentsne_pca2_init.RData"))
  nn_path <- file.path(dataset_dir, paste0(dataset, "_centered_raw_euclidean_k", k, "_nn.RData"))

  log_msg("==== %s ====", dataset)
  log_msg("%s: source=%s", dataset, source_path)
  status <- "success"
  error <- NA_character_
  n <- p <- NA_integer_
  pca_sec <- nn_sec <- NA_real_
  pca_backend <- nn_backend_used <- NA_character_

  tryCatch({
    obj <- pick_dataset_object(source_path)
    labels <- if (is.null(obj$labels)) NULL else as.factor(obj$labels)
    x <- center_matrix(obj$data)
    n <- nrow(x)
    p <- ncol(x)
    log_msg("%s: centered n=%d p=%d", dataset, n, p)

    if (force || !file.exists(pca_path)) {
      pca_sec <- system.time({
        pca_scores <- compute_pca2(x)
        Y_init <- to_tsne_init(pca_scores)
      })[["elapsed"]]
      pca_backend <- attr(pca_scores, "pca_backend") %||% NA_character_
      pca_metadata <- list(
        dataset = dataset,
        source_file = source_path,
        source_object = obj$object_name,
        preprocessing = "finite and variable columns kept; mean-centered only; no variance scaling",
        pca_backend = pca_backend,
        tsne_init_transform = "center PCA scores and scale so max component SD is 1e-4",
        n = n,
        p = p,
        seed = seed,
        seconds = as.numeric(pca_sec),
        created = as.character(Sys.time())
      )
      save(Y_init, labels, pca_metadata, file = pca_path, compress = "gzip")
      log_msg("%s: saved PCA init %s", dataset, pca_path)
    } else {
      log_msg("%s: PCA init exists, skipping %s", dataset, pca_path)
    }

    if (force || !file.exists(nn_path)) {
      nn_sec <- system.time({
        nn_centered_euclidean_k100 <- compute_nn(x, dataset)
      })[["elapsed"]]
      nn_backend_used <- attr(nn_centered_euclidean_k100, "precompute_backend") %||%
        attr(nn_centered_euclidean_k100, "backend") %||% NA_character_
      nn_metadata <- list(
        dataset = dataset,
        source_file = source_path,
        source_object = obj$object_name,
        preprocessing = "same centered matrix used for PCA; no variance scaling",
        metric = "euclidean",
        k = k,
        backend = nn_backend_used,
        n_threads = threads,
        n = n,
        p = p,
        seconds = as.numeric(nn_sec),
        created = as.character(Sys.time())
      )
      save(nn_centered_euclidean_k100, labels, nn_metadata, file = nn_path, compress = FALSE)
      log_msg("%s: saved NN %s", dataset, nn_path)
    } else {
      log_msg("%s: NN exists, skipping %s", dataset, nn_path)
    }

    rm(x, obj)
    gc()
  }, error = function(e) {
    status <<- "failed"
    error <<- conditionMessage(e)
    log_msg("%s: FAILED: %s", dataset, error)
  })

  summary_rows[[length(summary_rows) + 1L]] <- data.frame(
    dataset = dataset,
    status = status,
    n = n,
    p = p,
    pca_file = pca_path,
    pca_backend = pca_backend,
    pca_sec = as.numeric(pca_sec),
    nn_file = nn_path,
    nn_backend = nn_backend_used,
    nn_sec = as.numeric(nn_sec),
    error = error,
    stringsAsFactors = FALSE
  )
  pca_rows[[length(pca_rows) + 1L]] <- data.frame(
    dataset = dataset,
    pca_init_path = pca_path,
    preprocessing = "mean_center_no_scale",
    stringsAsFactors = FALSE
  )
  knn_rows[[length(knn_rows) + 1L]] <- data.frame(
    dataset = dataset,
    knn_path = nn_path,
    k = k,
    metric = "euclidean",
    preprocessing = "mean_center_no_scale",
    stringsAsFactors = FALSE
  )
}

summary <- do.call(rbind, summary_rows)
pca_manifest <- do.call(rbind, pca_rows)
knn_manifest <- do.call(rbind, knn_rows)
utils::write.csv(summary, file.path(data_root, paste0("centered_raw_precompute_summary_k", k, ".csv")), row.names = FALSE)
utils::write.csv(pca_manifest, file.path(data_root, "pca_init_manifest_centered_raw.csv"), row.names = FALSE)
utils::write.csv(knn_manifest, file.path(data_root, paste0("knn_manifest_centered_raw_k", k, ".csv")), row.names = FALSE)
print(summary)
if (any(summary$status != "success")) quit(status = 2L)
