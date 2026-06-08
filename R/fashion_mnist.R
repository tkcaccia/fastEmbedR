read_idx_gz <- function(path) {
  con <- gzfile(path, "rb")
  on.exit(close(con), add = TRUE)
  magic <- readBin(con, "integer", n = 1L, size = 4L, endian = "big")
  if (magic == 2051L) {
    n <- readBin(con, "integer", n = 1L, size = 4L, endian = "big")
    rows <- readBin(con, "integer", n = 1L, size = 4L, endian = "big")
    cols <- readBin(con, "integer", n = 1L, size = 4L, endian = "big")
    x <- readBin(con, "integer", n = n * rows * cols, size = 1L, signed = FALSE)
    matrix(as.numeric(x) / 255, nrow = n, byrow = TRUE)
  } else if (magic == 2049L) {
    n <- readBin(con, "integer", n = 1L, size = 4L, endian = "big")
    readBin(con, "integer", n = n, size = 1L, signed = FALSE)
  } else {
    stop("Unsupported IDX magic number: ", magic, call. = FALSE)
  }
}

#' Download and load Fashion-MNIST
#'
#' @param dir Directory for cached IDX gzip files.
#' @param n_train Number of training images to load.
#' @param seed Seed used when subsampling.
#' @return A list with numeric matrix `x` and integer labels `y`.
#' @export
load_fashion_mnist <- function(dir = file.path(tempdir(), "fashion-mnist"),
                               n_train = 10000L,
                               seed = 4L) {
  dir.create(dir, recursive = TRUE, showWarnings = FALSE)
  base <- "https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion"
  files <- c(
    images = "train-images-idx3-ubyte.gz",
    labels = "train-labels-idx1-ubyte.gz"
  )
  for (file in files) {
    dest <- file.path(dir, file)
    if (!file.exists(dest)) {
      utils::download.file(file.path(base, file), dest, mode = "wb", quiet = TRUE)
    }
  }
  x <- read_idx_gz(file.path(dir, files[["images"]]))
  y <- read_idx_gz(file.path(dir, files[["labels"]]))
  n_train <- min(as.integer(n_train), nrow(x))
  if (n_train < nrow(x)) {
    set.seed(seed)
    keep <- sort(sample.int(nrow(x), n_train))
    x <- x[keep, , drop = FALSE]
    y <- y[keep]
  }
  list(x = x, y = y)
}

#' Benchmark KNN-based embeddings on Fashion-MNIST
#'
#' @param n_train Number of Fashion-MNIST training images to sample.
#' @param pca_dims Optional PCA dimensionality before KNN.
#' @param cache_dir Directory for cached Fashion-MNIST files.
#' @param ... Additional arguments passed to `benchmark_knn_umap`.
#' @return A benchmark result from `benchmark_knn_umap`.
#' @export
benchmark_fashion_mnist <- function(n_train = 10000L,
                                    pca_dims = 50L,
                                    cache_dir = file.path(tempdir(), "fashion-mnist"),
                                    ...) {
  fm <- load_fashion_mnist(cache_dir, n_train = n_train)
  x <- fm$x
  if (!is.null(pca_dims) && pca_dims > 0L && pca_dims < ncol(x)) {
    pc <- stats::prcomp(x, center = TRUE, scale. = FALSE, rank. = pca_dims)
    x <- pc$x
  }
  benchmark_knn_umap(x, fm$y, ...)
}

