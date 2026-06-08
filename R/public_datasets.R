download_if_missing <- function(url, dest) {
  dir.create(dirname(dest), recursive = TRUE, showWarnings = FALSE)
  if (!file.exists(dest) || file.info(dest)$size == 0) {
    utils::download.file(url, dest, mode = "wb", quiet = TRUE)
  }
  dest
}

sample_rows <- function(x, y, n = NULL, seed = 4L) {
  if (is.null(n) || n >= nrow(x)) return(list(x = x, y = y))
  set.seed(seed)
  keep <- sort(sample.int(nrow(x), as.integer(n)))
  list(x = x[keep, , drop = FALSE], y = if (is.null(y)) NULL else y[keep])
}

pca_if_requested <- function(x, pca_dims = NULL, seed = 4L) {
  x <- as.matrix(x)
  storage.mode(x) <- "double"
  if (is.null(pca_dims) || pca_dims <= 0L || pca_dims >= ncol(x)) {
    return(scale(x))
  }
  set.seed(seed)
  pc <- stats::prcomp(x, center = TRUE, scale. = FALSE, rank. = as.integer(pca_dims))
  scale(pc$x)
}

#' Download and load MNIST
#'
#' @param dir Directory for cached IDX gzip files.
#' @param n Number of rows to return after combining train and test files.
#' @param seed Seed used when subsampling.
#' @return A list with numeric matrix `x` and integer labels `y`.
#' @export
load_mnist <- function(dir = file.path(tempdir(), "mnist"),
                       n = 10000L,
                       seed = 4L) {
  base <- "https://storage.googleapis.com/cvdf-datasets/mnist"
  files <- c(
    train_images = "train-images-idx3-ubyte.gz",
    train_labels = "train-labels-idx1-ubyte.gz",
    test_images = "t10k-images-idx3-ubyte.gz",
    test_labels = "t10k-labels-idx1-ubyte.gz"
  )
  paths <- vapply(files, function(file) {
    download_if_missing(file.path(base, file), file.path(dir, file))
  }, character(1))
  x <- rbind(
    read_idx_gz(paths[["train_images"]]),
    read_idx_gz(paths[["test_images"]])
  )
  y <- c(
    read_idx_gz(paths[["train_labels"]]),
    read_idx_gz(paths[["test_labels"]])
  )
  sample_rows(x, y, n, seed)
}

#' Download and load UCI PenDigits
#'
#' @param dir Directory for cached files.
#' @param n Optional row subset.
#' @param seed Seed used when subsampling.
#' @return A list with numeric matrix `x` and integer labels `y`.
#' @export
load_pendigits <- function(dir = file.path(tempdir(), "pendigits"),
                           n = NULL,
                           seed = 4L) {
  base <- "https://archive.ics.uci.edu/ml/machine-learning-databases/pendigits"
  paths <- c(
    train = download_if_missing(file.path(base, "pendigits.tra"), file.path(dir, "pendigits.tra")),
    test = download_if_missing(file.path(base, "pendigits.tes"), file.path(dir, "pendigits.tes"))
  )
  x <- do.call(rbind, lapply(paths, utils::read.csv, header = FALSE))
  labels <- x[[ncol(x)]]
  data <- as.matrix(x[, -ncol(x), drop = FALSE])
  sample_rows(data, labels, n, seed)
}

#' Download and load UCI Shuttle
#'
#' @param dir Directory for cached files.
#' @param n Optional row subset.
#' @param seed Seed used when subsampling.
#' @return A list with numeric matrix `x` and integer labels `y`.
#' @export
load_shuttle <- function(dir = file.path(tempdir(), "shuttle"),
                         n = 10000L,
                         seed = 4L) {
  base <- "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/shuttle"
  paths <- c(
    train = download_if_missing(file.path(base, "shuttle.trn"), file.path(dir, "shuttle.trn")),
    test = download_if_missing(file.path(base, "shuttle.tst"), file.path(dir, "shuttle.tst"))
  )
  x <- do.call(rbind, lapply(paths, utils::read.table, header = FALSE))
  labels <- x[[ncol(x)]]
  data <- as.matrix(x[, -ncol(x), drop = FALSE])
  sample_rows(data, labels, n, seed)
}

#' Load a public benchmark dataset
#'
#' @param name Dataset name. One of `"iris"`, `"mnist"`, `"fashion_mnist"`,
#'   `"pendigits"`, or `"shuttle"`.
#' @param n Optional row subset.
#' @param pca_dims Optional PCA dimensionality after loading.
#' @param cache_dir Directory for cached downloaded files.
#' @param seed Seed used for subsampling and PCA.
#' @return A list with matrix `x`, labels `y`, and source metadata.
#' @export
load_embedding_dataset <- function(name,
                                   n = NULL,
                                   pca_dims = NULL,
                                   cache_dir = file.path(tempdir(), "fastembedr-data"),
                                   seed = 4L) {
  name <- match.arg(name, c("iris", "mnist", "fashion_mnist", "pendigits", "shuttle"))
  dataset <- switch(
    name,
    iris = {
      x <- as.matrix(iris[, 1:4])
      y <- iris$Species
      sample_rows(x, y, n, seed)
    },
    mnist = load_mnist(file.path(cache_dir, "mnist"), n = if (is.null(n)) 10000L else n, seed = seed),
    fashion_mnist = load_fashion_mnist(file.path(cache_dir, "fashion_mnist"), n_train = if (is.null(n)) 10000L else n, seed = seed),
    pendigits = load_pendigits(file.path(cache_dir, "pendigits"), n = n, seed = seed),
    shuttle = load_shuttle(file.path(cache_dir, "shuttle"), n = if (is.null(n)) 10000L else n, seed = seed)
  )
  list(
    x = pca_if_requested(dataset$x, pca_dims = pca_dims, seed = seed),
    y = dataset$y,
    name = name,
    n = nrow(dataset$x),
    p = ncol(dataset$x)
  )
}

#' Benchmark embeddings across multiple public datasets
#'
#' @param datasets Character vector of dataset names.
#' @param subsets Named numeric vector/list with per-dataset row subsets.
#' @param implementations Implementations passed to `benchmark_knn_umap`.
#' @param pca_dims Optional PCA dimensionality for high-dimensional datasets.
#' @param cache_dir Directory for cached downloaded files.
#' @param output_csv Optional path to save the combined metrics table.
#' @param ... Additional arguments passed to `benchmark_knn_umap`.
#' @return A list with combined `metrics` and per-dataset `results`.
#' @export
benchmark_embedding_datasets <- function(datasets = c("iris", "pendigits", "fashion_mnist"),
                                         subsets = c(iris = NA, pendigits = NA, fashion_mnist = 2000),
                                         implementations = c("fastknnumap_sgd", "umap", "rtsne"),
                                         pca_dims = 50L,
                                         cache_dir = file.path(tempdir(), "fastembedr-data"),
                                         output_csv = NULL,
                                         ...) {
  results <- list()
  metric_rows <- list()
  for (dataset_name in datasets) {
    subset <- subsets[[dataset_name]]
    if (is.null(subset) || is.na(subset)) subset <- NULL
    data <- load_embedding_dataset(
      dataset_name,
      n = subset,
      pca_dims = if (dataset_name %in% c("mnist", "fashion_mnist")) pca_dims else NULL,
      cache_dir = cache_dir
    )
    bench <- benchmark_knn_umap(
      data$x,
      data$y,
      implementations = implementations,
      ...
    )
    metrics <- bench$metrics
    metrics$dataset <- dataset_name
    metrics$n <- nrow(data$x)
    metrics$p <- ncol(data$x)
    metrics$subset <- if (is.null(subset)) NA_integer_ else as.integer(subset)
    metric_rows[[dataset_name]] <- metrics
    results[[dataset_name]] <- bench
  }
  combined <- do.call(rbind, metric_rows)
  rownames(combined) <- NULL
  combined <- combined[, c("dataset", "n", "p", "subset", setdiff(names(combined), c("dataset", "n", "p", "subset")))]
  if (!is.null(output_csv)) {
    utils::write.csv(combined, output_csv, row.names = FALSE)
  }
  list(metrics = combined, results = results)
}
