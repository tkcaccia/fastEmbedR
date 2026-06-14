#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)
data_root <- if (length(args) >= 1L) args[[1L]] else "/Users/stefano/Documents/fastEmbedR/Data"
manifest_path <- file.path(data_root, "dataset_manifest.csv")
seed <- 1L

if (!requireNamespace("fastPLS", quietly = TRUE)) {
  stop("The fastPLS package is required. Install tkcaccia/fastPLS before running this script.")
}

if (!file.exists(manifest_path)) {
  stop("Dataset manifest not found: ", manifest_path)
}

manifest <- read.csv(manifest_path, stringsAsFactors = FALSE)
required_cols <- c("dataset", "path")
missing_cols <- setdiff(required_cols, names(manifest))
if (length(missing_cols) > 0L) {
  stop("Dataset manifest is missing column(s): ", paste(missing_cols, collapse = ", "))
}

choose_backend <- function() {
  probe <- matrix(stats::rnorm(80), nrow = 20)
  for (backend in c("metal", "cpu")) {
    ok <- try(
      fastPLS::pca(probe, ncomp = 2L, backend = backend, method = "rsvd",
                   center = TRUE, scale = FALSE, seed = seed),
      silent = TRUE
    )
    if (!inherits(ok, "try-error")) {
      return(backend)
    }
  }
  "cpu"
}

to_numeric_matrix <- function(x) {
  if (inherits(x, "Matrix")) {
    x <- as.matrix(x)
  } else if (is.data.frame(x)) {
    x <- as.matrix(x)
  } else if (!is.matrix(x)) {
    x <- as.matrix(x)
  }
  storage.mode(x) <- "double"
  x
}

save_one_pca_init <- function(dataset_name, dataset_path, preferred_backend) {
  started <- proc.time()[["elapsed"]]
  result <- data.frame(
    dataset = dataset_name,
    dataset_path = dataset_path,
    pca_init_path = NA_character_,
    n = NA_integer_,
    p = NA_integer_,
    backend = NA_character_,
    method = "rsvd",
    center = TRUE,
    scale = FALSE,
    seed = seed,
    elapsed_sec = NA_real_,
    status = "failed",
    error = "",
    stringsAsFactors = FALSE
  )

  tryCatch({
    env <- new.env(parent = emptyenv())
    load(dataset_path, envir = env)
    if (!exists("dataset", envir = env, inherits = FALSE)) {
      stop("No object named `dataset` found in ", dataset_path)
    }

    ds <- get("dataset", envir = env, inherits = FALSE)
    if (is.null(ds$data)) {
      stop("`dataset$data` is missing in ", dataset_path)
    }

    x <- to_numeric_matrix(ds$data)
    labels <- ds$labels

    pca_fit <- try(
      fastPLS::pca(x, ncomp = 2L, backend = preferred_backend, method = "rsvd",
                   center = TRUE, scale = FALSE, seed = seed),
      silent = TRUE
    )
    used_backend <- preferred_backend

    if (inherits(pca_fit, "try-error") && preferred_backend != "cpu") {
      pca_fit <- fastPLS::pca(x, ncomp = 2L, backend = "cpu", method = "rsvd",
                              center = TRUE, scale = FALSE, seed = seed)
      used_backend <- "cpu"
    }

    scores <- as.matrix(pca_fit$scores[, 1:2, drop = FALSE])
    colnames(scores) <- c("PC1", "PC2")

    pca_init <- list(
      layout = scores,
      labels = labels,
      dataset = dataset_name,
      source_dataset = normalizePath(dataset_path, mustWork = FALSE),
      backend = used_backend,
      package = "fastPLS",
      package_version = as.character(utils::packageVersion("fastPLS")),
      method = "rsvd",
      center = TRUE,
      scale = FALSE,
      seed = seed,
      n = nrow(x),
      p = ncol(x),
      sdev = pca_fit$sdev[seq_len(min(2L, length(pca_fit$sdev)))],
      variance_explained = pca_fit$variance_explained[seq_len(min(2L, length(pca_fit$variance_explained)))],
      material_methods = ds$material_methods,
      metadata = ds$metadata,
      created_at = format(Sys.time(), "%Y-%m-%d %H:%M:%S %Z")
    )

    out_path <- file.path(dirname(dataset_path), paste0(dataset_name, "_fastPLS_pca2_init.RData"))
    save(pca_init, file = out_path, compress = "gzip")

    result$pca_init_path <- normalizePath(out_path, mustWork = FALSE)
    result$n <- nrow(scores)
    result$p <- ncol(x)
    result$backend <- used_backend
    result$elapsed_sec <- proc.time()[["elapsed"]] - started
    result$status <- "success"
    result
  }, error = function(e) {
    result$error <- conditionMessage(e)
    result$elapsed_sec <- proc.time()[["elapsed"]] - started
    result
  })
}

set.seed(seed)
preferred_backend <- choose_backend()
message("Using fastPLS backend preference: ", preferred_backend)

rows <- vector("list", nrow(manifest))
for (i in seq_len(nrow(manifest))) {
  message(sprintf("[%d/%d] PCA init for %s", i, nrow(manifest), manifest$dataset[[i]]))
  rows[[i]] <- save_one_pca_init(manifest$dataset[[i]], manifest$path[[i]], preferred_backend)
  gc(verbose = FALSE)
}

pca_manifest <- do.call(rbind, rows)
out_manifest <- file.path(data_root, "pca_init_manifest.csv")
write.csv(pca_manifest, out_manifest, row.names = FALSE)

print(pca_manifest[, c("dataset", "n", "p", "backend", "elapsed_sec", "status", "pca_init_path", "error")],
      row.names = FALSE)

if (any(pca_manifest$status != "success")) {
  quit(status = 1L)
}
