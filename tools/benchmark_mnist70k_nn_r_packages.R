#!/usr/bin/env Rscript

arg_value <- function(name, default = NULL) {
  prefix <- paste0("--", name, "=")
  args <- commandArgs(trailingOnly = TRUE)
  hit <- args[startsWith(args, prefix)]
  if (length(hit) == 0L) return(default)
  sub(prefix, "", hit[[length(hit)]], fixed = TRUE)
}

arg_int <- function(name, default) {
  value <- suppressWarnings(as.integer(arg_value(name, as.character(default))))
  if (length(value) != 1L || is.na(value)) default else value
}

arg_csv <- function(name, default) {
  value <- arg_value(name, default)
  value <- trimws(strsplit(value, ",", fixed = TRUE)[[1L]])
  value[nzchar(value)]
}

json_params <- function(x) {
  if (requireNamespace("jsonlite", quietly = TRUE)) {
    return(as.character(jsonlite::toJSON(x, auto_unbox = TRUE, null = "null")))
  }
  paste(paste(names(x), unlist(x, use.names = FALSE), sep = "="), collapse = ";")
}

download_file <- function(url, path) {
  dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
  if (file.exists(path) && file.info(path)$size > 0) return(path)
  utils::download.file(url, path, mode = "wb", quiet = TRUE)
  path
}

read_idx_images <- function(path) {
  con <- gzfile(path, "rb")
  on.exit(close(con), add = TRUE)
  header <- readBin(con, "integer", n = 4L, size = 4L, endian = "big")
  if (length(header) != 4L || header[[1L]] != 2051L) {
    stop("Invalid IDX image file: ", path, call. = FALSE)
  }
  n <- header[[2L]]
  rows <- header[[3L]]
  cols <- header[[4L]]
  values <- readBin(con, "integer", n = n * rows * cols, size = 1L, signed = FALSE)
  matrix(as.numeric(values) / 255, nrow = n, byrow = TRUE)
}

read_idx_labels <- function(path) {
  con <- gzfile(path, "rb")
  on.exit(close(con), add = TRUE)
  header <- readBin(con, "integer", n = 2L, size = 4L, endian = "big")
  if (length(header) != 2L || header[[1L]] != 2049L) {
    stop("Invalid IDX label file: ", path, call. = FALSE)
  }
  factor(readBin(con, "integer", n = header[[2L]], size = 1L, signed = FALSE))
}

load_mnist_flat <- function(cache_dir, n) {
  raw_cache <- file.path(cache_dir, "mnist_idx_70000_flattened.rds")
  if (file.exists(raw_cache)) {
    mnist <- readRDS(raw_cache)
  } else {
    base <- "https://storage.googleapis.com/cvdf-datasets/mnist"
    files <- c(
      train_images = "train-images-idx3-ubyte.gz",
      train_labels = "train-labels-idx1-ubyte.gz",
      test_images = "t10k-images-idx3-ubyte.gz",
      test_labels = "t10k-labels-idx1-ubyte.gz"
    )
    paths <- vapply(
      files,
      function(file) download_file(file.path(base, file), file.path(cache_dir, "mnist", file)),
      character(1L)
    )
    mnist <- list(
      x = rbind(read_idx_images(paths[["train_images"]]), read_idx_images(paths[["test_images"]])),
      labels = factor(c(
        as.character(read_idx_labels(paths[["train_labels"]])),
        as.character(read_idx_labels(paths[["test_labels"]]))
      )),
      source = "MNIST IDX public files, raw flattened 28x28 pixels"
    )
    saveRDS(mnist, raw_cache, version = 2)
  }
  n <- min(as.integer(n), nrow(mnist$x))
  list(x = mnist$x[seq_len(n), , drop = FALSE], labels = mnist$labels[seq_len(n)])
}

package_version <- function(pkg) {
  if (!requireNamespace(pkg, quietly = TRUE)) return(NA_character_)
  as.character(utils::packageVersion(pkg))
}

row_template <- function(method, package, algorithm, backend, status, error_message = NA_character_,
                         total_sec = NA_real_, build_sec = NA_real_, query_sec = NA_real_,
                         n = NA_integer_, p = NA_integer_, k = NA_integer_, threads = NA_integer_,
                         n_neighbors_returned = NA_integer_, package_version = NA_character_,
                         params = list()) {
  data.frame(
    machine = Sys.info()[["nodename"]],
    dataset = "mnist70k_flattened_idx",
    method = method,
    package = package,
    algorithm = algorithm,
    backend = backend,
    status = status,
    error_message = error_message,
    n = as.integer(n),
    p = as.integer(p),
    k = as.integer(k),
    threads = as.integer(threads),
    build_sec = as.numeric(build_sec),
    query_sec = as.numeric(query_sec),
    total_sec = as.numeric(total_sec),
    n_neighbors_returned = as.integer(n_neighbors_returned),
    package_version = package_version,
    parameters_json = json_params(params),
    timestamp = format(Sys.time(), "%Y-%m-%d %H:%M:%S %Z"),
    stringsAsFactors = FALSE
  )
}

valid_knn <- function(idx, k) {
  if (is.null(idx)) return(FALSE)
  is.matrix(idx) && nrow(idx) > 0L && ncol(idx) >= k
}

timed <- function(expr) {
  gc()
  start <- proc.time()[["elapsed"]]
  value <- force(expr)
  list(value = value, sec = proc.time()[["elapsed"]] - start)
}

quiet <- function(expr) {
  suppressWarnings(suppressMessages(force(expr)))
}

run_method <- function(method, x, k, threads) {
  n <- nrow(x)
  p <- ncol(x)

  missing_pkg <- function(pkg, algorithm, backend = "cpu") {
    row_template(
      method, pkg, algorithm, backend, "package_unavailable",
      paste0("Package ", pkg, " is not installed."),
      n = n, p = p, k = k, threads = threads, package_version = package_version(pkg)
    )
  }

  no_api <- function(pkg, algorithm, msg) {
    row_template(
      method, pkg, algorithm, "cpu", "no_nn_only_api", msg,
      n = n, p = p, k = k, threads = threads, package_version = package_version(pkg)
    )
  }

  try_row <- function(pkg, algorithm, backend, params, expr, idx_fun) {
    if (!requireNamespace(pkg, quietly = TRUE)) {
      return(missing_pkg(pkg, algorithm, backend))
    }
    result <- tryCatch({
      z <- timed(expr)
      idx <- idx_fun(z$value)
      if (!valid_knn(idx, k)) {
        stop("Returned object does not contain a valid neighbour index matrix with at least k columns.")
      }
      row_template(
        method, pkg, algorithm, backend, "success",
        total_sec = z$sec,
        n = n, p = p, k = k, threads = threads,
        n_neighbors_returned = ncol(idx),
        package_version = package_version(pkg),
        params = params
      )
    }, error = function(e) {
      row_template(
        method, pkg, algorithm, backend, "failed", conditionMessage(e),
        n = n, p = p, k = k, threads = threads,
        package_version = package_version(pkg), params = params
      )
    })
    result
  }

  switch(
    method,

    "Rnanoflann_exact" = try_row(
      "Rnanoflann", "exact_kdtree", "cpu",
      list(k_requested = k + 1L, parallel = TRUE, cores = threads, sorted = TRUE),
      quiet(Rnanoflann::nn(x, x, k = k + 1L, method = "euclidean",
                           parallel = TRUE, cores = threads, sorted = TRUE)),
      function(z) z$indices[, -1L, drop = FALSE]
    ),

    "RANN_kd_exact" = try_row(
      "RANN", "kd_tree_standard", "cpu",
      list(k_requested = k + 1L, treetype = "kd", searchtype = "standard", eps = 0),
      quiet(RANN::nn2(x, x, k = k + 1L, treetype = "kd", searchtype = "standard", eps = 0)),
      function(z) z$nn.idx[, -1L, drop = FALSE]
    ),

    "RANN_bd_exact" = try_row(
      "RANN", "bd_tree_standard", "cpu",
      list(k_requested = k + 1L, treetype = "bd", searchtype = "standard", eps = 0),
      quiet(RANN::nn2(x, x, k = k + 1L, treetype = "bd", searchtype = "standard", eps = 0)),
      function(z) z$nn.idx[, -1L, drop = FALSE]
    ),

    "RcppHNSW_hnsw" = try_row(
      "RcppHNSW", "hnsw", "cpu",
      list(k_requested = k + 1L, M = 16, ef_construction = 200, ef = 100, n_threads = threads),
      quiet(RcppHNSW::hnsw_knn(x, k = k + 1L, distance = "euclidean", M = 16,
                               ef_construction = 200, ef = 100, n_threads = threads,
                               verbose = FALSE, progress = "bar")),
      function(z) z$idx[, -1L, drop = FALSE]
    ),

    "RcppAnnoy_euclidean" = try_row(
      "RcppAnnoy", "annoy_euclidean", "cpu",
      list(k_requested = k + 1L, n_trees = 50, search_k = -1),
      {
        idx <- matrix(NA_integer_, nrow = n, ncol = k + 1L)
        ann <- new(RcppAnnoy::AnnoyEuclidean, p)
        for (i in seq_len(n)) ann$addItem(i - 1L, x[i, ])
        ann$build(50L)
        for (i in seq_len(n)) idx[i, ] <- ann$getNNsByVector(x[i, ], k + 1L) + 1L
        idx
      },
      function(z) z[, -1L, drop = FALSE]
    ),

    "BiocNeighbors_exhaustive" = try_row(
      "BiocNeighbors", "exhaustive", "cpu",
      list(k_requested = k, num.threads = threads),
      quiet(BiocNeighbors::findKNN(x, k = k, num.threads = threads,
                                   BNPARAM = BiocNeighbors::ExhaustiveParam())),
      function(z) z$index
    ),

    "BiocNeighbors_vptree" = try_row(
      "BiocNeighbors", "vptree", "cpu",
      list(k_requested = k, num.threads = threads),
      quiet(BiocNeighbors::findKNN(x, k = k, num.threads = threads,
                                   BNPARAM = BiocNeighbors::VptreeParam())),
      function(z) z$index
    ),

    "BiocNeighbors_kmknn" = try_row(
      "BiocNeighbors", "kmknn", "cpu",
      list(k_requested = k, num.threads = threads),
      quiet(BiocNeighbors::findKNN(x, k = k, num.threads = threads,
                                   BNPARAM = BiocNeighbors::KmknnParam())),
      function(z) z$index
    ),

    "BiocNeighbors_annoy" = try_row(
      "BiocNeighbors", "annoy", "cpu",
      list(k_requested = k, num.threads = threads, ntrees = 50),
      quiet(BiocNeighbors::findKNN(x, k = k, num.threads = threads,
                                   BNPARAM = BiocNeighbors::AnnoyParam(ntrees = 50))),
      function(z) z$index
    ),

    "BiocNeighbors_hnsw" = try_row(
      "BiocNeighbors", "hnsw", "cpu",
      list(k_requested = k, num.threads = threads),
      quiet(BiocNeighbors::findKNN(x, k = k, num.threads = threads,
                                   BNPARAM = BiocNeighbors::HnswParam())),
      function(z) z$index
    ),

    "uwot_fnn_internal" = try_row(
      "uwot", "fnn_internal", "cpu",
      list(k_requested = k, method = "fnn", include_self = FALSE, n_threads = threads),
      quiet(uwot:::find_nn(x, k = k, include_self = FALSE, method = "fnn",
                           metric = "euclidean", nn_args = list(),
                           n_threads = threads, verbose = FALSE)),
      function(z) z$idx
    ),

    "uwot_annoy_internal" = try_row(
      "uwot", "annoy_internal", "cpu",
      list(k_requested = k, method = "annoy", n_trees = 50, search_k = 2L * k * 50L,
           include_self = FALSE, n_threads = threads),
      quiet(uwot:::find_nn(x, k = k, include_self = FALSE, method = "annoy",
                           metric = "euclidean", n_trees = 50,
                           search_k = 2L * k * 50L, nn_args = list(),
                           n_threads = threads, verbose = FALSE)),
      function(z) z$idx
    ),

    "uwot_hnsw_internal" = try_row(
      "uwot", "hnsw_internal", "cpu",
      list(k_requested = k, method = "hnsw", M = 16, ef_construction = 200, ef = 100,
           include_self = FALSE, n_threads = threads),
      quiet(uwot:::find_nn(x, k = k, include_self = FALSE, method = "hnsw",
                           metric = "euclidean", nn_args = list(M = 16, ef_construction = 200, ef = 100),
                           n_threads = threads, verbose = FALSE)),
      function(z) z$idx
    ),

    "uwot_nndescent_internal" = try_row(
      "uwot", "nndescent_internal", "cpu",
      list(k_requested = k, method = "nndescent", include_self = FALSE, n_threads = threads),
      quiet(uwot:::find_nn(x, k = k, include_self = FALSE, method = "nndescent",
                           metric = "euclidean", nn_args = list(),
                           n_threads = threads, verbose = FALSE)),
      function(z) z$idx
    ),

    "Rtsne_neighbors_api" = no_api(
      "Rtsne", "Rtsne_neighbors",
      "Rtsne_neighbors() consumes precomputed nearest-neighbour indices/distances; it does not build an NN graph."
    ),

    "umap_knn_api" = no_api(
      "umap", "umap.knn",
      "umap.knn() wraps precomputed nearest-neighbour indices/distances; it does not build an NN graph."
    ),

    "cuda_ml_knn" = {
      if (!requireNamespace("cuda.ml", quietly = TRUE)) {
        missing_pkg("cuda.ml", "cuda_ml_knn", "cuda")
      } else if (!exists("cuda_ml_knn", asNamespace("cuda.ml"), inherits = FALSE)) {
        row_template(
          method, "cuda.ml", "cuda_ml_knn", "cuda", "no_nn_only_api",
          "Installed cuda.ml does not export cuda_ml_knn().",
          n = n, p = p, k = k, threads = threads, package_version = package_version("cuda.ml")
        )
      } else {
        try_row(
          "cuda.ml", "cuda_ml_knn", "cuda",
          list(k_requested = k, algorithm = "default"),
          quiet(cuda.ml::cuda_ml_knn(x, k = k)),
          function(z) {
            if (is.list(z) && !is.null(z$idx)) return(z$idx)
            if (is.list(z) && !is.null(z$indices)) return(z$indices)
            if (is.matrix(z)) return(z)
            NULL
          }
        )
      }
    },

    stop("Unknown method: ", method, call. = FALSE)
  )
}

default_methods <- c(
  "Rnanoflann_exact",
  "RANN_kd_exact",
  "RANN_bd_exact",
  "RcppHNSW_hnsw",
  "RcppAnnoy_euclidean",
  "BiocNeighbors_exhaustive",
  "BiocNeighbors_vptree",
  "BiocNeighbors_kmknn",
  "BiocNeighbors_annoy",
  "BiocNeighbors_hnsw",
  "uwot_fnn_internal",
  "uwot_annoy_internal",
  "uwot_hnsw_internal",
  "uwot_nndescent_internal",
  "Rtsne_neighbors_api",
  "umap_knn_api",
  "cuda_ml_knn"
)

main <- function() {
  n <- arg_int("n", 70000L)
  k <- arg_int("k", 50L)
  threads <- arg_int("threads", 4L)
  method <- arg_value("method", "all")
  methods <- if (identical(method, "all")) arg_csv("methods", paste(default_methods, collapse = ",")) else method
  out_dir <- arg_value("out-dir", file.path("results", "mnist70k_nn_r_packages"))
  cache_dir <- arg_value("cache-dir", file.path("results", "dataset_cache"))
  out_file <- arg_value("out-file", file.path(out_dir, "mnist70k_nn_package_benchmark.csv"))

  dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
  set.seed(arg_int("seed", 6L))
  Sys.setenv(OMP_NUM_THREADS = as.character(threads))
  Sys.setenv(RCPP_PARALLEL_NUM_THREADS = as.character(threads))

  message("Loading MNIST flattened IDX data: n=", n, " k=", k)
  mnist <- load_mnist_flat(cache_dir, n)
  x <- mnist$x
  storage.mode(x) <- "double"
  message("Data loaded: ", nrow(x), " x ", ncol(x))

  rows <- lapply(methods, function(m) {
    message("Running ", m, " ...")
    run_method(m, x, k, threads)
  })
  res <- do.call(rbind, rows)
  utils::write.csv(res, out_file, row.names = FALSE)
  utils::write.csv(res, file.path(out_dir, "latest_mnist70k_nn_package_benchmark.csv"), row.names = FALSE)

  ok <- res[res$status == "success", , drop = FALSE]
  if (nrow(ok) > 0L) {
    png_path <- file.path(out_dir, "mnist70k_nn_package_timing.png")
    grDevices::png(png_path, width = 1600, height = 950, res = 140)
    old <- graphics::par(mar = c(11, 5, 4, 2) + 0.1)
    on.exit(graphics::par(old), add = TRUE)
    ord <- order(ok$total_sec, decreasing = TRUE, na.last = NA)
    cols <- ifelse(ok$backend[ord] == "cuda", "#2a9d8f", "#4361ee")
    graphics::barplot(
      ok$total_sec[ord],
      names.arg = paste(ok$package[ord], ok$algorithm[ord], sep = "\n"),
      las = 2,
      col = cols,
      border = NA,
      ylab = "NN time (seconds)",
      main = sprintf("MNIST 70k flattened NN benchmark, k=%d", k)
    )
    graphics::box()
    grDevices::dev.off()
    message("Timing plot: ", normalizePath(png_path, mustWork = FALSE))
  }

  message("CSV: ", normalizePath(out_file, mustWork = FALSE))
  print(res[order(res$status != "success", res$total_sec, na.last = TRUE), c(
    "method", "package", "algorithm", "backend", "status", "total_sec", "error_message"
  )], row.names = FALSE)
}

main()
