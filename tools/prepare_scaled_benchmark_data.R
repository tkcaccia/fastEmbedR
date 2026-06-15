#!/usr/bin/env Rscript

parse_args <- function(args) {
  out <- list()
  for (arg in args) {
    if (grepl("^--", arg)) {
      kv <- strsplit(sub("^--", "", arg), "=", fixed = TRUE)[[1L]]
      key <- kv[[1L]]
      value <- if (length(kv) > 1L) paste(kv[-1L], collapse = "=") else TRUE
      out[[key]] <- value
    }
  }
  out
}

`%||%` <- function(x, y) {
  if (is.null(x) || length(x) == 0L || (length(x) == 1L && is.na(x))) y else x
}

args <- parse_args(commandArgs(trailingOnly = TRUE))
data_root <- args$data_root %||% "/mnt/sata_ssd/fastEmbedR_Data"
seed <- as.integer(args$seed %||% "42")
pca_backend <- args$pca_backend %||% "cuda"
force <- isTRUE(as.logical(args$force %||% TRUE))

requested <- strsplit(
  args$datasets %||% paste(
    c("COIL20", "USPS", "FashionMNIST", "FlowRepository_FR-FCM-ZYRM_files",
      "flow18", "MNIST", "imagenet", "MetRef", "mass41", "TabulaMuris"),
    collapse = ","
  ),
  ",",
  fixed = TRUE
)[[1L]]
requested <- trimws(requested)

log_msg <- function(...) {
  cat(format(Sys.time(), "%Y-%m-%d %H:%M:%S"), " ", sprintf(...), "\n", sep = "")
  flush.console()
}

dataset_file <- function(name) {
  direct <- file.path(data_root, name, paste0(name, ".RData"))
  if (file.exists(direct)) return(direct)
  files <- list.files(file.path(data_root, name), pattern = "\\.RData$", full.names = TRUE)
  files <- files[!grepl("pca2_init|backup", basename(files), ignore.case = TRUE)]
  if (length(files)) return(files[[1L]])
  NA_character_
}

coerce_numeric_matrix <- function(x) {
  if (inherits(x, "Matrix")) x <- as.matrix(x)
  if (is.data.frame(x)) x <- as.matrix(x)
  if (!is.matrix(x)) x <- as.matrix(x)
  if (!is.numeric(x)) storage.mode(x) <- "double"
  x
}

normalize_dataset_object <- function(path) {
  env <- new.env(parent = emptyenv())
  load(path, envir = env)
  nms <- ls(env)
  if (exists("dataset", envir = env, inherits = FALSE)) {
    obj <- get("dataset", envir = env, inherits = FALSE)
    object_name <- "dataset"
  } else if (length(nms) == 1L) {
    obj <- get(nms[[1L]], envir = env, inherits = FALSE)
    object_name <- nms[[1L]]
  } else {
    stop("Cannot identify dataset object in ", path, call. = FALSE)
  }
  if (!is.list(obj) || is.null(obj$data)) {
    stop("Dataset object in ", path, " must be a list with `$data`.", call. = FALSE)
  }
  x <- coerce_numeric_matrix(obj$data)
  labels <- obj$labels %||% attr(obj, "label_names") %||% attr(obj$data, "labels") %||% NULL
  if (!is.null(labels) && length(labels) != nrow(x)) {
    labels <- NULL
  }
  metadata <- obj$metadata %||% list()
  metadata$original_object_name <- object_name
  metadata$scaled_for_fastEmbedR_benchmark <- TRUE
  metadata$scaled_at <- as.character(Sys.time())
  metadata$scaling <- "column centered and divided by sample standard deviation; zero-variance columns centered only"
  list(data = x, labels = labels, metadata = metadata)
}

scale_matrix_in_place <- function(x) {
  x <- coerce_numeric_matrix(x)
  p <- ncol(x)
  centers <- numeric(p)
  scales <- numeric(p)
  zero_variance <- logical(p)
  for (j in seq_len(p)) {
    col <- x[, j]
    mu <- mean(col, na.rm = TRUE)
    sdv <- stats::sd(col, na.rm = TRUE)
    if (!is.finite(mu)) mu <- 0
    if (!is.finite(sdv) || sdv <= 0) {
      x[, j] <- col - mu
      sdv <- 1
      zero_variance[[j]] <- TRUE
    } else {
      x[, j] <- (col - mu) / sdv
    }
    centers[[j]] <- mu
    scales[[j]] <- sdv
  }
  attr(x, "scaled:center") <- centers
  attr(x, "scaled:scale") <- scales
  attr(x, "scaled:zero_variance") <- zero_variance
  x
}

extract_pca_layout <- function(obj, n) {
  candidates <- list(obj)
  if (is.list(obj)) {
    candidates <- c(candidates, unname(obj))
    for (nm in c("x", "scores", "layout", "u", "projection")) {
      if (!is.null(obj[[nm]])) candidates <- c(candidates, list(obj[[nm]]))
    }
  }
  for (cand in candidates) {
    if (is.null(cand)) next
    mat <- tryCatch(as.matrix(cand), error = function(e) NULL)
    if (!is.null(mat) && nrow(mat) == n && ncol(mat) >= 2L) {
      storage.mode(mat) <- "double"
      return(mat[, 1:2, drop = FALSE])
    }
  }
  NULL
}

compute_pca2 <- function(x, dataset_name) {
  set.seed(seed)
  if (requireNamespace("fastPLS", quietly = TRUE)) {
    for (backend in unique(c(pca_backend, "cpu"))) {
      ans <- tryCatch(
        fastPLS::pca(x, ncomp = 2L, center = FALSE, scale = FALSE,
                     backend = backend, method = "rsvd", seed = seed),
        error = function(e) e
      )
      if (!inherits(ans, "error")) {
        layout <- extract_pca_layout(ans, nrow(x))
        if (!is.null(layout)) {
          return(list(layout = layout, method = paste0("fastPLS::pca_rsvd_", backend)))
        }
      } else {
        log_msg("PCA fastPLS failed for %s on %s: %s", dataset_name, backend, conditionMessage(ans))
      }
    }
  }
  if (nrow(x) * ncol(x) > 2e8) {
    stop("PCA fallback skipped for large dataset after fastPLS failure.", call. = FALSE)
  }
  pc <- stats::prcomp(x, center = FALSE, scale. = FALSE, rank. = 2L)
  list(layout = pc$x[, 1:2, drop = FALSE], method = "stats::prcomp_rank2")
}

rows <- list()
pca_rows <- list()

for (dataset_name in requested) {
  path <- dataset_file(dataset_name)
  if (is.na(path) || !file.exists(path)) {
    log_msg("Skipping %s: file not found", dataset_name)
    rows[[length(rows) + 1L]] <- data.frame(dataset = dataset_name, path = NA_character_, status = "missing", n = NA_integer_, p = NA_integer_, labels = NA_integer_, file_mb = NA_real_)
    next
  }
  log_msg("Preparing %s from %s", dataset_name, path)
  backup <- sub("\\.RData$", ".raw_unscaled_backup.RData", path)
  if (!file.exists(backup)) file.copy(path, backup, overwrite = FALSE)
  status <- "success"
  err <- NA_character_
  n <- p <- label_count <- file_mb <- NA_real_
  pca_path <- file.path(dirname(path), paste0(dataset_name, "_fastPLS_pca2_init.RData"))
  tryCatch({
    ds <- tryCatch(
      normalize_dataset_object(path),
      error = function(e) {
        if (file.exists(backup)) {
          log_msg("Restoring %s from backup after load failure: %s", dataset_name, conditionMessage(e))
          file.copy(backup, path, overwrite = TRUE)
          return(normalize_dataset_object(path))
        }
        stop(e)
      }
    )
    ds$data <- scale_matrix_in_place(ds$data)
    n <- nrow(ds$data)
    p <- ncol(ds$data)
    label_count <- if (is.null(ds$labels)) NA_integer_ else length(unique(ds$labels))
    dataset <- ds
    save(dataset, file = path, compress = "gzip")
    file_mb <- round(file.info(path)$size / 1024^2, 3)
    pca <- compute_pca2(ds$data, dataset_name)
    pca_init <- list(
      layout = pca$layout,
      method = pca$method,
      seed = seed,
      source_dataset = dataset_name,
      source_path = path,
      preprocessing = "dataset$data scaled before PCA"
    )
    save(pca_init, file = pca_path, compress = "gzip")
    pca_rows[[length(pca_rows) + 1L]] <- data.frame(dataset = dataset_name, pca_init_path = pca_path, method = pca$method, stringsAsFactors = FALSE)
  }, error = function(e) {
    status <<- "failed"
    err <<- conditionMessage(e)
    log_msg("FAILED %s: %s", dataset_name, err)
  })
  rows[[length(rows) + 1L]] <- data.frame(
    dataset = dataset_name,
    path = path,
    n = as.integer(n),
    p = as.integer(p),
    labels = as.integer(label_count),
    file_mb = file_mb,
    status = status,
    error_message = err,
    stringsAsFactors = FALSE
  )
}

preprocessing_manifest <- do.call(rbind, rows)
utils::write.csv(preprocessing_manifest, file.path(data_root, "preprocessing_manifest.csv"), row.names = FALSE)
manifest <- preprocessing_manifest[preprocessing_manifest$status == "success", , drop = FALSE]
utils::write.csv(manifest[, c("dataset", "path", "n", "p", "labels", "file_mb"), drop = FALSE],
                 file.path(data_root, "dataset_manifest.csv"), row.names = FALSE)
if (length(pca_rows)) {
  pca_manifest <- do.call(rbind, pca_rows)
} else {
  pca_manifest <- data.frame(dataset = character(), pca_init_path = character(), method = character())
}
utils::write.csv(pca_manifest, file.path(data_root, "pca_init_manifest.csv"), row.names = FALSE)

knn_cache_dirs <- Sys.glob(file.path("/mnt/sata_ssd", "fastEmbedR_BENCHMARK*", "knn*"))
if (length(knn_cache_dirs)) {
  unlink(knn_cache_dirs, recursive = TRUE, force = TRUE)
  log_msg("Removed %d old benchmark KNN cache directories.", length(knn_cache_dirs))
}

log_msg("Scaled data preparation complete. Manifest: %s", file.path(data_root, "dataset_manifest.csv"))
