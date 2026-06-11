#!/usr/bin/env Rscript

suppressPackageStartupMessages(library(fastEmbedR))

parse_scalar <- function(name, default) {
  args <- commandArgs(trailingOnly = TRUE)
  prefix <- paste0("--", name, "=")
  value <- args[startsWith(args, prefix)]
  if (length(value) == 0L) {
    Sys.getenv(paste0("FASTEMBEDR_LARGE_", toupper(gsub("-", "_", name))), default)
  } else {
    sub(prefix, "", value[[1L]], fixed = TRUE)
  }
}

parse_csv <- function(name, default) {
  value <- parse_scalar(name, default)
  out <- trimws(strsplit(value, ",", fixed = TRUE)[[1L]])
  out[nzchar(out)]
}

parse_flag <- function(name, default = FALSE) {
  args <- commandArgs(trailingOnly = TRUE)
  env <- Sys.getenv(paste0("FASTEMBEDR_LARGE_", toupper(gsub("-", "_", name))), "")
  any(args == paste0("--", name)) ||
    identical(tolower(env), "1") ||
    identical(tolower(env), "true") ||
    identical(tolower(parse_scalar(name, if (default) "true" else "false")), "true")
}

num_arg <- function(name, default) as.numeric(parse_scalar(name, as.character(default)))
int_arg <- function(name, default) as.integer(num_arg(name, default))

script_dir <- function() {
  file_arg <- commandArgs(FALSE)
  file_arg <- file_arg[startsWith(file_arg, "--file=")]
  if (length(file_arg) > 0L) {
    dirname(normalizePath(sub("^--file=", "", file_arg[[1L]]), mustWork = FALSE))
  } else {
    file.path(getwd(), "tools")
  }
}

download_file <- function(url, path) {
  if (file.exists(path)) return(path)
  dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
  old_timeout <- getOption("timeout")
  options(timeout = max(600, as.numeric(old_timeout)))
  on.exit(options(timeout = old_timeout), add = TRUE)
  ok <- tryCatch({
    utils::download.file(url, path, quiet = TRUE, mode = "wb")
    TRUE
  }, error = function(e) FALSE)
  if (!isTRUE(ok) || !file.exists(path)) {
    curl <- Sys.which("curl")
    if (nzchar(curl)) {
      ok <- tryCatch({
        status <- system2(
          curl,
          c("-L", "--fail", "--retry", "3", "--connect-timeout", "30",
            "--max-time", "1200", "--insecure", "-o", path, url),
          stdout = FALSE,
          stderr = FALSE
        )
        identical(status, 0L) || identical(status, 0)
      }, error = function(e) FALSE)
    }
  }
  if (isTRUE(ok) && file.exists(path)) path else NA_character_
}

read_idx_images <- function(path) {
  con <- gzfile(path, "rb")
  on.exit(close(con), add = TRUE)
  header <- readBin(con, "integer", n = 4L, size = 4L, endian = "big")
  if (length(header) != 4L || header[[1L]] != 2051L) stop("Invalid IDX image file: ", path, call. = FALSE)
  n <- header[[2L]]
  rows <- header[[3L]]
  cols <- header[[4L]]
  values <- readBin(con, "integer", n = n * rows * cols, size = 1L, signed = FALSE)
  x <- matrix(as.numeric(values) / 255, nrow = n, byrow = TRUE)
  colnames(x) <- paste0("px", seq_len(ncol(x)))
  x
}

read_idx_labels <- function(path) {
  con <- gzfile(path, "rb")
  on.exit(close(con), add = TRUE)
  header <- readBin(con, "integer", n = 2L, size = 4L, endian = "big")
  if (length(header) != 2L || header[[1L]] != 2049L) stop("Invalid IDX label file: ", path, call. = FALSE)
  factor(readBin(con, "integer", n = header[[2L]], size = 1L, signed = FALSE))
}

dataset_record <- function(name, x, labels = NULL, source = "generated") {
  list(
    name = name,
    x = as.matrix(x),
    labels = if (is.null(labels)) NULL else factor(labels),
    source = source,
    raw_n = nrow(x),
    raw_p = ncol(x)
  )
}

sample_rows <- function(n, max_n, seed) {
  if (!is.finite(max_n) || max_n < 1L || n <= max_n) return(seq_len(n))
  set.seed(seed)
  sort(sample.int(n, as.integer(max_n)))
}

subsample_dataset <- function(dataset, max_n, seed) {
  keep <- sample_rows(nrow(dataset$x), max_n, seed)
  if (length(keep) == nrow(dataset$x)) return(dataset)
  dataset$x <- dataset$x[keep, , drop = FALSE]
  if (!is.null(dataset$labels)) dataset$labels <- dataset$labels[keep]
  dataset$name <- paste0(dataset$name, "_n", length(keep))
  dataset
}

make_blobs <- function(n = 60000L, p = 24L, classes = 10L, seed = 4L) {
  set.seed(seed)
  labels <- factor(rep(seq_len(classes), length.out = n))
  labels <- factor(sample(labels))
  centers <- matrix(stats::rnorm(classes * p, sd = 3), classes, p)
  x <- matrix(stats::rnorm(n * p, sd = 0.9), n, p)
  x <- x + centers[as.integer(labels), , drop = FALSE]
  dataset_record(paste0("synthetic_blobs_", n, "x", p), x, labels, "generated Gaussian blobs")
}

make_s_curve <- function(n = 60000L, seed = 4L) {
  if (!requireNamespace("reticulate", quietly = TRUE)) {
    warning("Skipping S-curve: reticulate is not installed.", call. = FALSE)
    return(NULL)
  }
  tryCatch({
    sk <- reticulate::import("sklearn.datasets")
    out <- sk$make_s_curve(n_samples = as.integer(n), noise = 0.06, random_state = as.integer(seed))
    x <- reticulate::py_to_r(out[[1L]])
    color <- as.numeric(reticulate::py_to_r(out[[2L]]))
    labels <- cut(color, breaks = stats::quantile(color, seq(0, 1, length.out = 11), na.rm = TRUE),
                  include.lowest = TRUE, labels = FALSE)
    dataset_record(paste0("sklearn_s_curve_", n), x, labels, "sklearn.datasets.make_s_curve")
  }, error = function(e) {
    warning("Skipping S-curve: ", conditionMessage(e), call. = FALSE)
    NULL
  })
}

load_mnist_idx <- function(cache_dir) {
  base <- "https://storage.googleapis.com/cvdf-datasets/mnist"
  files <- c(
    train_images = "train-images-idx3-ubyte.gz",
    train_labels = "train-labels-idx1-ubyte.gz",
    test_images = "t10k-images-idx3-ubyte.gz",
    test_labels = "t10k-labels-idx1-ubyte.gz"
  )
  paths <- vapply(files, function(file) download_file(file.path(base, file), file.path(cache_dir, "mnist", file)), character(1))
  if (any(is.na(paths))) return(NULL)
  x <- rbind(read_idx_images(paths[["train_images"]]), read_idx_images(paths[["test_images"]]))
  labels <- factor(c(as.character(read_idx_labels(paths[["train_labels"]])), as.character(read_idx_labels(paths[["test_labels"]]))))
  dataset_record("mnist_idx_70000", x, labels, "MNIST IDX public files")
}

load_fashion_mnist <- function(cache_dir) {
  base <- "https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion"
  files <- c(
    train_images = "train-images-idx3-ubyte.gz",
    train_labels = "train-labels-idx1-ubyte.gz",
    test_images = "t10k-images-idx3-ubyte.gz",
    test_labels = "t10k-labels-idx1-ubyte.gz"
  )
  paths <- vapply(files, function(file) download_file(file.path(base, file), file.path(cache_dir, "fashion_mnist", file)), character(1))
  if (any(is.na(paths))) return(NULL)
  x <- rbind(read_idx_images(paths[["train_images"]]), read_idx_images(paths[["test_images"]]))
  labels <- factor(c(as.character(read_idx_labels(paths[["train_labels"]])), as.character(read_idx_labels(paths[["test_labels"]]))))
  dataset_record("fashion_mnist_70000", x, labels, "Zalando Fashion-MNIST IDX files")
}

load_cifar_python_features <- function(cache_dir,
                                       dataset = c("cifar10", "cifar100")) {
  dataset <- match.arg(dataset)
  if (!requireNamespace("reticulate", quietly = TRUE)) {
    warning("Skipping ", dataset, ": reticulate is not installed.", call. = FALSE)
    return(NULL)
  }
  if (identical(dataset, "cifar10")) {
    url <- "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    archive <- "cifar-10-python.tar.gz"
    subdir <- "cifar-10-batches-py"
    files <- c(paste0("data_batch_", 1:5), "test_batch")
    name <- "cifar10_rgb8x8_60000"
    source <- "CIFAR-10 Toronto Python files, RGB 8x8 block-mean features"
  } else {
    url <- "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    archive <- "cifar-100-python.tar.gz"
    subdir <- "cifar-100-python"
    files <- c("train", "test")
    name <- "cifar100_rgb8x8_60000"
    source <- "CIFAR-100 Toronto Python files, RGB 8x8 block-mean features"
  }
  root <- file.path(cache_dir, dataset)
  path <- download_file(url, file.path(root, archive))
  if (is.na(path)) return(NULL)
  extracted <- file.path(root, subdir)
  if (!dir.exists(extracted)) {
    dir.create(root, recursive = TRUE, showWarnings = FALSE)
    utils::untar(path, exdir = root)
  }
  batch_paths <- file.path(extracted, files)
  if (!all(file.exists(batch_paths))) {
    warning("Skipping ", dataset, ": missing extracted batch files.", call. = FALSE)
    return(NULL)
  }
  tryCatch({
    py <- reticulate::py_run_string("
import pickle
import numpy as np

def fastembedr_load_cifar_features(files):
    xs = []
    ys = []
    for path in files:
        with open(path, 'rb') as handle:
            payload = pickle.load(handle, encoding='latin1')
        data = payload['data'].astype('float32').reshape((-1, 3, 32, 32))
        data = data.reshape((data.shape[0], 3, 8, 4, 8, 4)).mean(axis=(3, 5))
        xs.append(data.reshape((data.shape[0], -1)) / 255.0)
        labels = payload.get('labels', payload.get('fine_labels'))
        ys.extend(labels)
    return np.vstack(xs), np.asarray(ys, dtype=np.int32)
")
    out <- py$fastembedr_load_cifar_features(as.list(batch_paths))
    x <- reticulate::py_to_r(out[[1L]])
    labels <- reticulate::py_to_r(out[[2L]])
    dataset_record(name, x, labels, source)
  }, error = function(e) {
    warning("Skipping ", dataset, ": ", conditionMessage(e), call. = FALSE)
    NULL
  })
}

load_sklearn_fetch <- function(name, cache_dir, seed) {
  if (!requireNamespace("reticulate", quietly = TRUE)) return(NULL)
  tryCatch({
    sk <- reticulate::import("sklearn.datasets")
    if (identical(name, "covertype")) {
      out <- sk$fetch_covtype(data_home = cache_dir, shuffle = TRUE, random_state = as.integer(seed))
      return(dataset_record("sklearn_covtype_581012", reticulate::py_to_r(out$data), reticulate::py_to_r(out$target), "sklearn.fetch_covtype"))
    }
    if (identical(name, "shuttle")) {
      out <- sk$fetch_openml("shuttle", version = 1L, as_frame = FALSE, parser = "auto")
      return(dataset_record("openml_shuttle", reticulate::py_to_r(out$data), reticulate::py_to_r(out$target), "sklearn.fetch_openml shuttle"))
    }
    NULL
  }, error = function(e) {
    warning("Skipping ", name, ": ", conditionMessage(e), call. = FALSE)
    NULL
  })
}

load_covtype_uci <- function(cache_dir) {
  path <- download_file(
    "https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.data.gz",
    file.path(cache_dir, "covtype", "covtype.data.gz")
  )
  if (is.na(path)) return(NULL)
  tryCatch({
    dat <- utils::read.table(gzfile(path), sep = ",", header = FALSE)
    dataset_record(
      "uci_covtype_581012",
      dat[, -ncol(dat), drop = FALSE],
      dat[[ncol(dat)]],
      "UCI Covertype direct download"
    )
  }, error = function(e) {
    warning("Skipping UCI Covertype fallback: ", conditionMessage(e), call. = FALSE)
    NULL
  })
}

load_shuttle_uci <- function(cache_dir) {
  base <- "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/shuttle"
  train_z <- download_file(file.path(base, "shuttle.trn.Z"), file.path(cache_dir, "shuttle", "shuttle.trn.Z"))
  test_path <- download_file(file.path(base, "shuttle.tst"), file.path(cache_dir, "shuttle", "shuttle.tst"))
  if (is.na(train_z) || is.na(test_path)) return(NULL)
  tryCatch({
    train <- utils::read.table(pipe(paste("uncompress -c", shQuote(train_z))), header = FALSE)
    test <- utils::read.table(test_path, header = FALSE)
    dat <- rbind(train, test)
    dataset_record(
      "uci_shuttle_58000",
      dat[, -ncol(dat), drop = FALSE],
      dat[[ncol(dat)]],
      "UCI Statlog Shuttle direct download"
    )
  }, error = function(e) {
    warning("Skipping UCI Shuttle fallback: ", conditionMessage(e), call. = FALSE)
    NULL
  })
}

find_fastpls_rdata <- function(dataset_id) {
  dataset_id <- tolower(dataset_id)
  fname <- switch(
    dataset_id,
    metref = "metref.RData",
    singlecell = "singlecell.RData",
    cifar100 = "CIFAR100.RData",
    imagenet = "imagenet.RData",
    paste0(dataset_id, ".RData")
  )
  env_name <- paste0("FASTEMBEDR_", toupper(dataset_id), "_RDATA")
  home <- path.expand("~")
  candidates <- unique(c(
    Sys.getenv(env_name, ""),
    file.path(home, "Documents", "fastpls", "data", fname),
    file.path(home, "Documents", "fastPLS", "data", fname),
    file.path(home, "Documents", "Rdatasets", fname),
    file.path(home, "GPUPLS", "Data", fname)
  ))
  candidates <- candidates[nzchar(candidates)]
  for (path in candidates) {
    if (file.exists(path)) return(normalizePath(path, winslash = "/", mustWork = TRUE))
  }
  found <- suppressWarnings(list.files(
    home,
    pattern = paste0("^", gsub(".", "\\\\.", fname, fixed = TRUE), "$"),
    full.names = TRUE,
    recursive = TRUE,
    ignore.case = TRUE
  ))
  if (length(found) > 0L) normalizePath(found[[1L]], winslash = "/", mustWork = TRUE) else NA_character_
}

load_fastpls_rdata_dataset <- function(dataset_id) {
  path <- find_fastpls_rdata(dataset_id)
  if (is.na(path)) {
    warning("Skipping fastPLS ", dataset_id, ": RData file not found.", call. = FALSE)
    return(NULL)
  }
  e <- new.env(parent = emptyenv())
  objs <- load(path, envir = e)
  source <- paste0("fastPLS RData: ", path)

  make_from_xy <- function(x, y, name_suffix = NULL) {
    name <- paste0("fastpls_", dataset_id, "_", nrow(x))
    if (!is.null(name_suffix)) name <- paste0(name, "_", name_suffix)
    dataset_record(name, as.matrix(x), y, source)
  }

  if ("out" %in% objs && is.list(e$out) &&
      all(c("Xtrain", "Ytrain", "Xtest", "Ytest") %in% names(e$out))) {
    x <- rbind(as.matrix(e$out$Xtrain), as.matrix(e$out$Xtest))
    y <- factor(c(as.character(e$out$Ytrain), as.character(e$out$Ytest)))
    return(make_from_xy(x, y, "train_test"))
  }

  if (all(c("Xtrain", "Ytrain", "Xtest", "Ytest") %in% objs)) {
    x <- rbind(as.matrix(e$Xtrain), as.matrix(e$Xtest))
    y <- factor(c(as.character(e$Ytrain), as.character(e$Ytest)))
    return(make_from_xy(x, y, "train_test"))
  }

  if (all(c("Xtrain", "Ytrain") %in% objs)) {
    return(make_from_xy(e$Xtrain, e$Ytrain, "train"))
  }

  if (all(c("data", "labels") %in% objs)) {
    return(make_from_xy(e$data, e$labels))
  }

  if ("r" %in% objs && is.data.frame(e$r)) {
    r <- e$r
    feat_cols <- grep("^feat_", names(r), value = TRUE)
    if (length(feat_cols) == 0L) {
      skip <- c("image_path", "split", "label_idx", "label_name", "label")
      feat_cols <- setdiff(names(r), skip)
    }
    label_col <- intersect(c("label_idx", "label_name", "label"), names(r))
    labels <- if (length(label_col) > 0L) r[[label_col[[1L]]]] else NULL
    return(make_from_xy(r[, feat_cols, drop = FALSE], labels))
  }

  warning("Skipping fastPLS ", dataset_id, ": unsupported RData objects: ",
          paste(objs, collapse = ", "), call. = FALSE)
  NULL
}

load_named_dataset <- function(name, cache_dir, min_n, max_n, seed) {
  key <- tolower(name)
  out <- NULL
  if (grepl("^synthetic_blobs_[0-9]+x[0-9]+$", key)) {
    nums <- as.integer(strsplit(sub("^synthetic_blobs_", "", key), "x", fixed = TRUE)[[1L]])
    out <- make_blobs(nums[[1L]], nums[[2L]], seed = seed)
  } else if (grepl("^s_curve_[0-9]+$", key)) {
    out <- make_s_curve(as.integer(sub("^s_curve_", "", key)), seed)
  } else if (key %in% c("synthetic_blobs", "blobs")) {
    out <- make_blobs(max(as.integer(min_n), 60000L), 24L, seed = seed)
  } else if (key %in% c("s_curve", "sklearn_s_curve")) {
    out <- make_s_curve(max(as.integer(min_n), 60000L), seed)
  } else if (key %in% c("mnist", "mnist_70000")) {
    out <- load_mnist_idx(cache_dir)
  } else if (key %in% c("fashion_mnist", "fashion-mnist", "fashion_mnist_70000")) {
    out <- load_fashion_mnist(cache_dir)
  } else if (key %in% c("cifar", "cifar10", "cifar10_features", "cifar10_rgb8x8")) {
    out <- load_cifar_python_features(cache_dir, "cifar10")
  } else if (key %in% c("cifar100", "cifar100_features", "cifar100_rgb8x8")) {
    out <- load_cifar_python_features(cache_dir, "cifar100")
  } else if (key %in% c("covertype", "covtype")) {
    out <- load_covtype_uci(cache_dir)
    if (is.null(out)) out <- load_sklearn_fetch("covertype", cache_dir, seed)
  } else if (key %in% c("shuttle", "openml_shuttle")) {
    out <- load_shuttle_uci(cache_dir)
    if (is.null(out)) out <- load_sklearn_fetch("shuttle", cache_dir, seed)
  } else if (key %in% c("fastpls_singlecell", "singlecell_fastpls", "singlecell_rdata")) {
    out <- load_fastpls_rdata_dataset("singlecell")
  } else if (key %in% c("fastpls_metref", "metref_fastpls", "metref_rdata")) {
    out <- load_fastpls_rdata_dataset("metref")
  } else if (key %in% c("fastpls_cifar100", "cifar100_fastpls", "cifar100_rdata")) {
    out <- load_fastpls_rdata_dataset("cifar100")
  } else if (key %in% c("fastpls_imagenet", "imagenet_fastpls", "imagenet_rdata")) {
    out <- load_fastpls_rdata_dataset("imagenet")
  } else {
    stop("Unknown large dataset: ", name, call. = FALSE)
  }
  if (is.null(out)) return(NULL)
  out <- subsample_dataset(out, max_n, seed)
  if (nrow(out$x) < min_n) {
    warning("Skipping ", out$name, ": n=", nrow(out$x), " is below --min-n=", min_n, call. = FALSE)
    return(NULL)
  }
  out
}

prepare_dataset <- function(dataset, pca_dims, seed, preprocess_backend) {
  elapsed <- system.time({
    prepared <- fastEmbedR:::prepare_embedding_data(
      dataset$x,
      standardize = TRUE,
      pca_dims = if (is.finite(pca_dims) && pca_dims > 0L) as.integer(pca_dims) else NULL,
      seed = seed,
      backend = preprocess_backend
    )
  })[["elapsed"]]
  dataset$raw_x <- NULL
  dataset$x <- prepared$data
  dataset$p <- ncol(dataset$x)
  dataset$preprocess <- prepared$preprocess
  dataset$preprocess_time_sec <- as.numeric(elapsed)
  dataset
}

drop_self_knn <- function(knn, k) {
  idx <- as.matrix(knn$indices)
  dst <- as.matrix(knn$distances)
  storage.mode(idx) <- "integer"
  storage.mode(dst) <- "double"
  out_idx <- matrix(NA_integer_, nrow(idx), k)
  out_dst <- matrix(NA_real_, nrow(idx), k)
  for (i in seq_len(nrow(idx))) {
    keep <- which(idx[i, ] != i | dst[i, ] > sqrt(.Machine$double.eps))
    if (length(keep) < k) keep <- seq_len(ncol(idx))
    keep <- keep[seq_len(k)]
    out_idx[i, ] <- idx[i, keep]
    out_dst[i, ] <- dst[i, keep]
  }
  list(indices = out_idx, distances = out_dst)
}

stratified_sample <- function(labels, n, size, seed) {
  size <- min(as.integer(size), n)
  if (size >= n) return(seq_len(n))
  set.seed(seed)
  if (is.null(labels)) return(sort(sample.int(n, size)))
  labels <- factor(labels)
  levels <- levels(labels)
  per_level <- stats::setNames(
    pmax(1L, floor(size * as.numeric(table(labels)[levels]) / n)),
    levels
  )
  picked <- integer()
  for (level in levels) {
    rows <- which(labels == level)
    take <- min(length(rows), per_level[[level]])
    if (take > 0L) picked <- c(picked, sample(rows, take))
  }
  remaining <- setdiff(seq_len(n), picked)
  if (length(picked) < size && length(remaining) > 0L) {
    picked <- c(picked, sample(remaining, min(length(remaining), size - length(picked))))
  }
  sort(unique(picked))[seq_len(min(size, length(unique(picked))))]
}

current_rss_mb <- function() {
  out <- suppressWarnings(system2("ps", c("-o", "rss=", "-p", as.character(Sys.getpid())), stdout = TRUE, stderr = FALSE))
  if (length(out) == 0L) return(NA_real_)
  kb <- suppressWarnings(as.numeric(trimws(out[[1L]])))
  if (is.finite(kb)) kb / 1024 else NA_real_
}

with_timeout <- function(expr, timeout_sec) {
  timeout_sec <- as.numeric(timeout_sec)
  if (is.finite(timeout_sec) && timeout_sec > 0) {
    setTimeLimit(elapsed = timeout_sec, transient = TRUE)
    on.exit(setTimeLimit(cpu = Inf, elapsed = Inf, transient = FALSE), add = TRUE)
  }
  force(expr)
}

measure_layout <- function(expr, n, timeout_sec) {
  invisible(gc())
  rss_before <- current_rss_mb()
  elapsed <- system.time({
    layout <- with_timeout(force(expr), timeout_sec)
  })[["elapsed"]]
  rss_after <- current_rss_mb()
  fastembedr_config <- attr(layout, "fastEmbedR_config")
  layout <- as.matrix(layout)
  if (nrow(layout) != n && ncol(layout) == n) layout <- t(layout)
  if (nrow(layout) != n || ncol(layout) < 2L) stop("Method did not return an n x 2 layout.", call. = FALSE)
  layout <- layout[, 1:2, drop = FALSE]
  attr(layout, "fastEmbedR_config") <- fastembedr_config
  list(
    layout = layout,
    elapsed = as.numeric(elapsed),
    rss_before_mb = rss_before,
    rss_after_mb = rss_after,
    rss_delta_mb = if (is.finite(rss_before) && is.finite(rss_after)) rss_after - rss_before else NA_real_
  )
}

version_or_na <- function(pkg) {
  if (!requireNamespace(pkg, quietly = TRUE)) return(NA_character_)
  as.character(utils::packageVersion(pkg))
}

json_or_text <- function(x) {
  if (requireNamespace("jsonlite", quietly = TRUE)) {
    as.character(jsonlite::toJSON(x, auto_unbox = TRUE, null = "null"))
  } else {
    paste(paste(names(x), unlist(x), sep = "="), collapse = ";")
  }
}

call_rtsne_neighbors <- function(ctx) {
  args <- list(
    index = ctx$idx,
    distance = ctx$dist,
    dims = 2L,
    perplexity = ctx$perplexity,
    theta = 0.5,
    max_iter = ctx$max_iter,
    verbose = FALSE
  )
  formals_names <- names(formals(Rtsne::Rtsne_neighbors))
  args <- args[names(args) %in% formals_names]
  out <- do.call(Rtsne::Rtsne_neighbors, args)
  if (!is.null(out$Y)) out$Y else out
}

call_uwot <- function(ctx, fast_sgd = TRUE) {
  uwot::umap(
    X = ctx$x,
    n_neighbors = ctx$k,
    n_components = 2L,
    nn_method = list(idx = ctx$idx, dist = ctx$dist),
    n_epochs = ctx$n_epochs,
    init = "spectral",
    min_dist = ctx$min_dist,
    metric = "euclidean",
    learning_rate = 1,
    repulsion_strength = 1,
    negative_sample_rate = ctx$negative_sample_rate,
    fast_sgd = fast_sgd,
    n_threads = ctx$n_threads,
    n_sgd_threads = ctx$n_threads,
    ret_model = FALSE,
    verbose = FALSE,
    seed = ctx$seed
  )
}

method_table <- function() {
  list(
    fastembedr_umap = list(package = "fastEmbedR", family = "umap", function_name = "embed_knn(method='umap')",
      params = function(ctx) list(k = ctx$k, backend = ctx$embedding_backend),
      run = function(ctx) fastEmbedR::embed_knn(ctx$knn, method = "umap", seed = ctx$seed, backend = ctx$embedding_backend)),
    fastembedr_tsne = list(package = "fastEmbedR", family = "tsne", function_name = "embed_knn(method='tsne')",
      params = function(ctx) list(k = ctx$k, perplexity = ctx$perplexity, backend = ctx$embedding_backend, quality = "auto"),
      run = function(ctx) fastEmbedR::embed_knn(ctx$knn, method = "tsne", seed = ctx$seed, backend = ctx$embedding_backend)),
    fastembedr_pacmap = list(package = "fastEmbedR", family = "pacmap", function_name = "embed_knn(method='pacmap')",
      params = function(ctx) list(k = ctx$k, backend = ctx$embedding_backend),
      run = function(ctx) fastEmbedR::embed_knn(ctx$knn, method = "pacmap", seed = ctx$seed, backend = ctx$embedding_backend)),
    fastembedr_trimap = list(package = "fastEmbedR", family = "trimap", function_name = "embed_knn(method='trimap')",
      params = function(ctx) list(k = ctx$k, backend = ctx$embedding_backend),
      run = function(ctx) fastEmbedR::embed_knn(ctx$knn, method = "trimap", seed = ctx$seed, backend = ctx$embedding_backend)),
    fastembedr_localmap = list(package = "fastEmbedR", family = "localmap", function_name = "embed_knn(method='localmap')",
      params = function(ctx) list(k = ctx$k, backend = ctx$embedding_backend),
      run = function(ctx) fastEmbedR::embed_knn(ctx$knn, method = "localmap", seed = ctx$seed, backend = ctx$embedding_backend)),
    uwot_umap_fast_sgd = list(package = "uwot", family = "umap", function_name = "uwot::umap(fast_sgd=TRUE)",
      params = function(ctx) list(k = ctx$k, n_epochs = ctx$n_epochs, min_dist = ctx$min_dist, negative_sample_rate = ctx$negative_sample_rate, fast_sgd = TRUE),
      run = function(ctx) call_uwot(ctx, fast_sgd = TRUE)),
    uwot_umap_default = list(package = "uwot", family = "umap", function_name = "uwot::umap(fast_sgd=FALSE)",
      params = function(ctx) list(k = ctx$k, n_epochs = ctx$n_epochs, min_dist = ctx$min_dist, negative_sample_rate = ctx$negative_sample_rate, fast_sgd = FALSE),
      run = function(ctx) call_uwot(ctx, fast_sgd = FALSE)),
    rtsne_neighbors = list(package = "Rtsne", family = "tsne", function_name = "Rtsne::Rtsne_neighbors",
      params = function(ctx) list(k = ctx$k, perplexity = ctx$perplexity, theta = 0.5, max_iter = ctx$max_iter),
      run = call_rtsne_neighbors)
  )
}

failure_row <- function(dataset, method_id, spec, ctx, status, message) {
  data.frame(
    dataset = dataset$name,
    dataset_source = dataset$source,
    n = nrow(dataset$x),
    p = ncol(dataset$x),
    raw_p = dataset$raw_p,
    method = method_id,
    family = spec$family,
    package = spec$package,
    package_version = version_or_na(spec$package),
    function_name = spec$function_name,
    seed = ctx$seed,
    k = ctx$k,
    perplexity = ctx$perplexity,
    backend_requested = ctx$embedding_backend,
    backend_used = NA_character_,
    knn_backend = ctx$knn_backend_used,
    preprocess_time_sec = dataset$preprocess_time_sec,
    knn_time_sec = ctx$knn_time_sec,
    embedding_time_sec = NA_real_,
    total_time_sec = NA_real_,
    rss_delta_mb = NA_real_,
    status = status,
    error_message = message,
    parameters_json = json_or_text(spec$params(ctx)),
    quality_scope = "exact_subsample",
    quality_sample_n = length(ctx$quality_idx),
    trustworthiness = NA_real_,
    continuity = NA_real_,
    knn_preservation_15 = NA_real_,
    knn_preservation_30 = NA_real_,
    knn_preservation_50 = NA_real_,
    distance_spearman = NA_real_,
    distance_pearson = NA_real_,
    stress = NA_real_,
    silhouette = NA_real_,
    label_knn_accuracy = NA_real_,
    ari = NA_real_,
    nmi = NA_real_,
    rare_class_recall = NA_real_,
    procrustes_rmsd = NA_real_,
    neighbour_stability_15 = NA_real_,
    combined_score = NA_real_,
    layout_path = NA_character_,
    stringsAsFactors = FALSE
  )
}

success_row <- function(dataset, method_id, spec, ctx, measured, layout_path) {
  cfg <- attr(measured$layout, "fastEmbedR_config")
  backend_used <- if (!is.null(cfg$backend)) as.character(cfg$backend) else ctx$embedding_backend
  q <- ctx$quality_idx
  metrics <- fastEmbedR::evaluate_embedding(
    dataset$x[q, , drop = FALSE],
    measured$layout[q, , drop = FALSE],
    labels = if (is.null(dataset$labels)) NULL else dataset$labels[q],
    k = c(15L, 30L, 50L),
    sample_size_for_global_metrics = length(q),
    sample_size_for_local_metrics = length(q),
    seed = ctx$seed,
    method = method_id,
    backend = "cpu",
    dataset = paste0(dataset$name, "_quality_subsample")
  )
  data.frame(
    dataset = dataset$name,
    dataset_source = dataset$source,
    n = nrow(dataset$x),
    p = ncol(dataset$x),
    raw_p = dataset$raw_p,
    method = method_id,
    family = spec$family,
    package = spec$package,
    package_version = version_or_na(spec$package),
    function_name = spec$function_name,
    seed = ctx$seed,
    k = ctx$k,
    perplexity = ctx$perplexity,
    backend_requested = ctx$embedding_backend,
    backend_used = backend_used,
    knn_backend = ctx$knn_backend_used,
    preprocess_time_sec = dataset$preprocess_time_sec,
    knn_time_sec = ctx$knn_time_sec,
    embedding_time_sec = measured$elapsed,
    total_time_sec = dataset$preprocess_time_sec + ctx$knn_time_sec + measured$elapsed,
    rss_delta_mb = measured$rss_delta_mb,
    status = "success",
    error_message = NA_character_,
    parameters_json = json_or_text(spec$params(ctx)),
    quality_scope = "exact_subsample",
    quality_sample_n = length(q),
    trustworthiness = metrics$trustworthiness,
    continuity = metrics$continuity,
    knn_preservation_15 = metrics$knn_preservation_15,
    knn_preservation_30 = metrics$knn_preservation_30,
    knn_preservation_50 = metrics$knn_preservation_50,
    distance_spearman = metrics$distance_spearman,
    distance_pearson = metrics$distance_pearson,
    stress = metrics$stress,
    silhouette = metrics$silhouette,
    label_knn_accuracy = metrics$label_knn_accuracy,
    ari = metrics$ari,
    nmi = metrics$nmi,
    rare_class_recall = metrics$rare_class_recall,
    procrustes_rmsd = NA_real_,
    neighbour_stability_15 = NA_real_,
    combined_score = NA_real_,
    layout_path = layout_path,
    stringsAsFactors = FALSE
  )
}

save_layout <- function(layout, dataset, method_id, seed, layout_dir, save_layouts) {
  if (!isTRUE(save_layouts)) return(NA_character_)
  dir.create(layout_dir, recursive = TRUE, showWarnings = FALSE)
  path <- file.path(layout_dir, paste0(gsub("[^A-Za-z0-9_.-]+", "_", paste(dataset$name, method_id, seed, sep = "_")), ".rds"))
  saveRDS(layout, path)
  path
}

embedding_nn_indices <- function(layout, k = 15L) {
  knn <- fastEmbedR::nn(layout, layout, k + 1L, backend = "cpu")
  drop_self_knn(knn, k)$indices
}

procrustes_rmsd <- function(reference, target) {
  x <- scale(reference, center = TRUE, scale = FALSE)
  y <- scale(target, center = TRUE, scale = FALSE)
  nx <- sqrt(sum(x * x))
  ny <- sqrt(sum(y * y))
  if (!is.finite(nx) || !is.finite(ny) || nx == 0 || ny == 0) return(NA_real_)
  x <- x / nx
  y <- y / ny
  s <- svd(t(y) %*% x)
  y_aligned <- y %*% (s$u %*% t(s$v))
  sqrt(mean(rowSums((x - y_aligned)^2)))
}

neighbor_jaccard <- function(a, b) {
  mean(vapply(seq_len(nrow(a)), function(i) {
    length(intersect(a[i, ], b[i, ])) / length(union(a[i, ], b[i, ]))
  }, numeric(1)))
}

add_stability <- function(results, layouts, stability_idx_by_dataset) {
  ok <- which(results$status == "success")
  if (length(ok) == 0L) return(results)
  keys <- unique(paste(results$dataset[ok], results$method[ok], sep = "\r"))
  for (key in keys) {
    rows <- ok[paste(results$dataset[ok], results$method[ok], sep = "\r") == key]
    if (length(rows) < 2L) next
    dataset <- results$dataset[rows[[1L]]]
    keep <- stability_idx_by_dataset[[dataset]]
    ref <- layouts[[as.character(rows[[1L]])]][keep, , drop = FALSE]
    ref_nn <- embedding_nn_indices(ref, 15L)
    rmsd <- numeric(length(rows))
    neigh <- numeric(length(rows))
    for (j in seq_along(rows)) {
      y <- layouts[[as.character(rows[[j]])]][keep, , drop = FALSE]
      rmsd[[j]] <- procrustes_rmsd(ref, y)
      neigh[[j]] <- neighbor_jaccard(ref_nn, embedding_nn_indices(y, 15L))
    }
    results$procrustes_rmsd[rows] <- rmsd
    results$neighbour_stability_15[rows] <- neigh
  }
  results
}

scale01 <- function(x, inverse = FALSE) {
  out <- rep(NA_real_, length(x))
  ok <- is.finite(x)
  if (!any(ok)) return(out)
  vals <- x[ok]
  if (isTRUE(inverse)) vals <- -vals
  rng <- range(vals)
  out[ok] <- if (diff(rng) == 0) 1 else (vals - rng[[1L]]) / diff(rng)
  out
}

add_combined_score <- function(results) {
  results$combined_score <- NA_real_
  for (dataset in unique(results$dataset)) {
    rows <- which(results$dataset == dataset & results$status == "success")
    if (length(rows) == 0L) next
    trust <- scale01(results$trustworthiness[rows])
    knn <- scale01(results$knn_preservation_15[rows])
    label <- scale01(results$label_knn_accuracy[rows])
    runtime <- scale01(results$total_time_sec[rows], inverse = TRUE)
    stability <- scale01(results$neighbour_stability_15[rows])
    pieces <- cbind(trust, knn, label, stability, runtime)
    weights <- c(0.30, 0.25, 0.15, 0.15, 0.15)
    score <- apply(pieces, 1L, function(row) {
      ok <- is.finite(row)
      if (!any(ok)) return(NA_real_)
      sum(row[ok] * weights[ok]) / sum(weights[ok])
    })
    results$combined_score[rows] <- score
  }
  results
}

summarize_results <- function(results) {
  ok <- results[results$status == "success", , drop = FALSE]
  if (nrow(ok) == 0L) return(ok)
  groups <- split(ok, paste(ok$dataset, ok$method, sep = "\r"))
  out <- do.call(rbind, lapply(groups, function(x) {
    data.frame(
      dataset = x$dataset[[1L]],
      method = x$method[[1L]],
      package = x$package[[1L]],
      family = x$family[[1L]],
      n = x$n[[1L]],
      p = x$p[[1L]],
      seeds = paste(sort(unique(x$seed)), collapse = ","),
      runs = nrow(x),
      mean_total_time_sec = mean(x$total_time_sec, na.rm = TRUE),
      sd_total_time_sec = stats::sd(x$total_time_sec, na.rm = TRUE),
      mean_embedding_time_sec = mean(x$embedding_time_sec, na.rm = TRUE),
      mean_trustworthiness = mean(x$trustworthiness, na.rm = TRUE),
      mean_knn_preservation_15 = mean(x$knn_preservation_15, na.rm = TRUE),
      mean_label_knn_accuracy = mean(x$label_knn_accuracy, na.rm = TRUE),
      mean_silhouette = mean(x$silhouette, na.rm = TRUE),
      mean_distance_spearman = mean(x$distance_spearman, na.rm = TRUE),
      mean_procrustes_rmsd = mean(x$procrustes_rmsd, na.rm = TRUE),
      mean_neighbour_stability_15 = mean(x$neighbour_stability_15, na.rm = TRUE),
      mean_combined_score = mean(x$combined_score, na.rm = TRUE),
      stringsAsFactors = FALSE
    )
  }))
  out[order(out$dataset, -out$mean_combined_score, out$mean_total_time_sec), , drop = FALSE]
}

plot_results <- function(results, summary, out_dir) {
  ok <- results[results$status == "success", , drop = FALSE]
  if (nrow(ok) == 0L) return(invisible(NULL))
  png(file.path(out_dir, "speed_accuracy_scatter.png"), width = 1200, height = 850, res = 130)
  par(mar = c(5, 5, 4, 2))
  plot(ok$total_time_sec, ok$trustworthiness, log = "x", pch = 19,
       col = as.integer(factor(ok$method)), xlab = "Total runtime, seconds (log scale)",
       ylab = "Trustworthiness on exact quality subsample",
       main = "Large benchmark: speed vs accuracy")
  legend("bottomright", legend = levels(factor(ok$method)), col = seq_along(levels(factor(ok$method))),
         pch = 19, cex = 0.75)
  dev.off()
  if (nrow(summary) > 0L) {
    png(file.path(out_dir, "combined_score_by_method.png"), width = 1200, height = 850, res = 130)
    par(mar = c(9, 5, 4, 2))
    labels <- paste(summary$dataset, summary$method, sep = "\n")
    barplot(summary$mean_combined_score, names.arg = labels, las = 2,
            ylab = "Mean combined score", main = "Large benchmark combined score", col = "steelblue")
    dev.off()
  }
}

run_one <- function(dataset, method_id, spec, ctx, timeout_sec, layout_dir, save_layouts) {
  if (!requireNamespace(spec$package, quietly = TRUE)) {
    return(list(row = failure_row(dataset, method_id, spec, ctx, "not_installed", paste0("Package ", spec$package, " is not installed.")), layout = NULL))
  }
  tryCatch({
    measured <- measure_layout(spec$run(ctx), nrow(dataset$x), timeout_sec)
    path <- save_layout(measured$layout, dataset, method_id, ctx$seed, layout_dir, save_layouts)
    list(row = success_row(dataset, method_id, spec, ctx, measured, path), layout = measured$layout)
  }, error = function(e) {
    list(row = failure_row(dataset, method_id, spec, ctx, "failed", conditionMessage(e)), layout = NULL)
  })
}

datasets_arg <- parse_csv("datasets", "synthetic_blobs_60000x24,s_curve_60000")
methods_arg <- parse_csv("methods", "fastembedr_umap,fastembedr_tsne,uwot_umap_fast_sgd,rtsne_neighbors")
seeds <- as.integer(parse_csv("seeds", "4,5,6"))
seeds <- seeds[is.finite(seeds)]
if (length(seeds) == 0L) seeds <- 4L
k <- int_arg("k", 50L)
min_n <- int_arg("min-n", 50000L)
max_n <- num_arg("max-n", Inf)
pca_dims <- int_arg("pca-dims", 50L)
quality_sample_size <- int_arg("quality-sample-size", 5000L)
stability_sample_size <- int_arg("stability-sample-size", 2000L)
n_epochs <- int_arg("n-epochs", 300L)
min_dist <- num_arg("min-dist", 0.01)
negative_sample_rate <- int_arg("negative-sample-rate", 2L)
max_iter <- int_arg("max-iter", 500L)
timeout_sec <- num_arg("timeout-sec", 1800)
n_threads <- int_arg("n-threads", max(1L, parallel::detectCores(logical = FALSE)))
knn_backend <- parse_scalar("knn-backend", "auto")
embedding_backend <- parse_scalar("embedding-backend", "cpu")
results_dir <- parse_scalar("results-dir", file.path("results", "large_best_methods"))
cache_dir <- parse_scalar("cache-dir", file.path(results_dir, "cache"))
save_layouts <- !parse_flag("no-save-layouts", default = FALSE)

dir.create(results_dir, recursive = TRUE, showWarnings = FALSE)
layout_dir <- file.path(results_dir, "layouts")
all_methods <- method_table()
unknown <- setdiff(methods_arg, names(all_methods))
if (length(unknown) > 0L) stop("Unknown methods: ", paste(unknown, collapse = ", "), call. = FALSE)

datasets <- Filter(Negate(is.null), lapply(datasets_arg, load_named_dataset,
                                           cache_dir = cache_dir, min_n = min_n,
                                           max_n = max_n, seed = seeds[[1L]]))
if (length(datasets) == 0L) stop("No benchmark datasets with n >= ", min_n, " could be loaded.", call. = FALSE)

result_rows <- list()
layouts <- list()
stability_idx_by_dataset <- list()
row_id <- 0L

for (dataset in datasets) {
  message("Preparing ", dataset$name, " (n=", nrow(dataset$x), ", p=", ncol(dataset$x), ")")
  dataset <- prepare_dataset(dataset, pca_dims = pca_dims, seed = seeds[[1L]], preprocess_backend = knn_backend)
  quality_idx <- stratified_sample(dataset$labels, nrow(dataset$x), quality_sample_size, seeds[[1L]] + 100L)
  stability_idx_by_dataset[[dataset$name]] <- stratified_sample(dataset$labels, nrow(dataset$x), stability_sample_size, seeds[[1L]] + 200L)

  for (seed in seeds) {
    message("Building shared KNN for ", dataset$name, ", seed=", seed, ", k=", k)
    set.seed(seed)
    knn_time <- system.time({
      knn_self <- if (identical(knn_backend, "cpu_clustered")) {
        fastEmbedR:::nn_compute(
          dataset$x,
          dataset$x,
          k = k + 1L,
          backend = "cpu_clustered",
          points_missing = TRUE,
          exclude_self = FALSE
        )
      } else {
        fastEmbedR::nn(dataset$x, dataset$x, k = k + 1L, backend = knn_backend)
      }
    })[["elapsed"]]
    knn_no_self <- drop_self_knn(knn_self, k)
    ctx <- list(
      x = dataset$x,
      knn = knn_no_self,
      idx = knn_no_self$indices,
      dist = knn_no_self$distances,
      seed = as.integer(seed),
      k = as.integer(k),
      perplexity = max(2L, min(30L, floor(k / 3L), floor((nrow(dataset$x) - 1L) / 3L))),
      n_epochs = as.integer(n_epochs),
      min_dist = as.numeric(min_dist),
      negative_sample_rate = as.integer(negative_sample_rate),
      max_iter = as.integer(max_iter),
      n_threads = as.integer(n_threads),
      knn_backend_used = attr(knn_self, "backend"),
      knn_time_sec = as.numeric(knn_time),
      quality_idx = quality_idx,
      embedding_backend = embedding_backend
    )

    for (method_id in methods_arg) {
      message("  ", method_id)
      spec <- all_methods[[method_id]]
      row_id <- row_id + 1L
      out <- run_one(dataset, method_id, spec, ctx, timeout_sec, layout_dir, save_layouts)
      result_rows[[row_id]] <- out$row
      if (!is.null(out$layout)) layouts[[as.character(row_id)]] <- out$layout
    }
    rm(knn_self, knn_no_self)
    invisible(gc())
  }
}

results <- do.call(rbind, result_rows)
results <- add_stability(results, layouts, stability_idx_by_dataset)
results <- add_combined_score(results)
summary <- summarize_results(results)

stamp <- format(Sys.time(), "%Y%m%d_%H%M%S")
results_file <- file.path(results_dir, paste0("large_best_methods_results_", stamp, ".csv"))
summary_file <- file.path(results_dir, paste0("large_best_methods_summary_", stamp, ".csv"))
latest_results <- file.path(results_dir, "latest_large_best_methods_results.csv")
latest_summary <- file.path(results_dir, "latest_large_best_methods_summary.csv")
utils::write.csv(results, results_file, row.names = FALSE)
utils::write.csv(summary, summary_file, row.names = FALSE)
utils::write.csv(results, latest_results, row.names = FALSE)
utils::write.csv(summary, latest_summary, row.names = FALSE)
writeLines(capture.output(sessionInfo()), file.path(results_dir, "session_info.txt"))
writeLines(capture.output(fastEmbedR::backend_info()), file.path(results_dir, "backend_info.txt"))
plot_results(results, summary, results_dir)

print(summary[, intersect(c(
  "dataset", "method", "runs", "mean_total_time_sec", "mean_trustworthiness",
  "mean_knn_preservation_15", "mean_label_knn_accuracy", "mean_neighbour_stability_15",
  "mean_combined_score"
), names(summary))], row.names = FALSE)

cat("\nSaved large benchmark files:\n")
cat("  ", normalizePath(results_file, mustWork = FALSE), "\n", sep = "")
cat("  ", normalizePath(summary_file, mustWork = FALSE), "\n", sep = "")
cat("  ", normalizePath(latest_results, mustWork = FALSE), "\n", sep = "")
cat("  ", normalizePath(latest_summary, mustWork = FALSE), "\n", sep = "")
cat("  ", normalizePath(file.path(results_dir, "speed_accuracy_scatter.png"), mustWork = FALSE), "\n", sep = "")
