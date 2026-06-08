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

procrustes_stability <- function(reference, candidate) {
  reference <- as.matrix(reference)
  candidate <- as.matrix(candidate)
  if (!identical(dim(reference), dim(candidate))) return(NA_real_)
  reference <- scale(reference, center = TRUE, scale = FALSE)
  candidate <- scale(candidate, center = TRUE, scale = FALSE)
  ref_norm <- sqrt(sum(reference * reference))
  cand_norm <- sqrt(sum(candidate * candidate))
  if (ref_norm == 0 || cand_norm == 0) return(NA_real_)
  reference <- reference / ref_norm
  candidate <- candidate / cand_norm
  sv <- svd(t(candidate) %*% reference)
  rotation <- sv$u %*% t(sv$v)
  aligned <- candidate %*% rotation
  1 / (1 + mean((reference - aligned)^2))
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
      x <- as.matrix(datasets::iris[, 1:4])
      y <- datasets::iris$Species
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
                                         implementations = c("fastknnumap_sgd", "knn_tsne"),
                                         pca_dims = 50L,
                                         repeats = 1L,
                                         cache_dir = file.path(tempdir(), "fastembedr-data"),
                                         output_csv = NULL,
                                         ...) {
  results <- list()
  metric_rows <- list()
  repeats <- max(1L, as.integer(repeats))
  for (dataset_name in datasets) {
    subset <- subsets[[dataset_name]]
    if (is.null(subset) || is.na(subset)) subset <- NULL
    data <- load_embedding_dataset(
      dataset_name,
      n = subset,
      pca_dims = if (dataset_name %in% c("mnist", "fashion_mnist")) pca_dims else NULL,
      cache_dir = cache_dir
    )
    dataset_results <- vector("list", repeats)
    reference_layouts <- list()
    dataset_metrics <- vector("list", repeats)
    for (run in seq_len(repeats)) {
      bench <- benchmark_knn_umap(
        data$x,
        data$y,
        implementations = implementations,
        seed = 4L + run - 1L,
        ...
      )
      metrics <- bench$metrics
      metrics$dataset <- dataset_name
      metrics$n <- nrow(data$x)
      metrics$p <- ncol(data$x)
      metrics$subset <- if (is.null(subset)) NA_integer_ else as.integer(subset)
      metrics[["repeat"]] <- run
      metrics$stability <- NA_real_
      if (run == 1L) {
        reference_layouts <- bench$layouts
      } else {
        metrics$stability <- vapply(metrics$implementation, function(implementation) {
          ref <- reference_layouts[[implementation]]
          cur <- bench$layouts[[implementation]]
          if (is.null(ref) || is.null(cur)) return(NA_real_)
          procrustes_stability(ref, cur)
        }, numeric(1))
      }
      dataset_metrics[[run]] <- metrics
      dataset_results[[run]] <- bench
    }
    metrics <- do.call(rbind, dataset_metrics)
    metric_rows[[dataset_name]] <- metrics
    results[[dataset_name]] <- if (repeats == 1L) dataset_results[[1L]] else dataset_results
  }
  combined <- do.call(rbind, metric_rows)
  rownames(combined) <- NULL
  combined <- combined[, c("dataset", "n", "p", "subset", "repeat", setdiff(names(combined), c("dataset", "n", "p", "subset", "repeat")))]
  if (!is.null(output_csv)) {
    dir.create(dirname(output_csv), recursive = TRUE, showWarnings = FALSE)
    utils::write.csv(combined, output_csv, row.names = FALSE)
  }
  list(metrics = combined, results = results)
}

plot_embedding_benchmark <- function(benchmark, output_dir) {
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
  metrics <- benchmark$metrics
  ok <- metrics[metrics$status == "ok", , drop = FALSE]
  paths <- character()
  if (nrow(ok) == 0L) return(paths)

  paths["speed_quality"] <- file.path(output_dir, "speed_quality.png")
  datasets <- unique(ok$dataset)
  implementations <- unique(ok$implementation)
  colors <- stats::setNames(grDevices::hcl.colors(length(implementations), "Dark 3"), implementations)
  nc <- ceiling(sqrt(length(datasets)))
  nr <- ceiling(length(datasets) / nc)

  grDevices::png(paths["speed_quality"], width = 1120, height = 700, res = 140)
  old_par <- graphics::par(no.readonly = TRUE)
  tryCatch({
    graphics::par(mfrow = c(nr, nc), mar = c(4.2, 4.2, 2.4, 1))
    for (dataset in datasets) {
      sub <- ok[ok$dataset == dataset, , drop = FALSE]
      graphics::plot(
        sub$elapsed,
        sub$knn_preservation,
        pch = 19,
        col = colors[sub$implementation],
        xlab = "Elapsed seconds",
        ylab = "KNN preservation",
        main = dataset
      )
      graphics::legend(
        "bottomright",
        legend = unique(sub$implementation),
        col = colors[unique(sub$implementation)],
        pch = 19,
        bty = "n",
        cex = 0.75
      )
    }
  }, finally = {
    graphics::par(old_par)
    grDevices::dev.off()
  })

  long <- rbind(
    data.frame(ok[, c("dataset", "implementation", "repeat")], metric = "silhouette", value = ok$silhouette),
    data.frame(ok[, c("dataset", "implementation", "repeat")], metric = "knn_preservation", value = ok$knn_preservation),
    data.frame(ok[, c("dataset", "implementation", "repeat")], metric = "stability", value = ok$stability)
  )
  long <- long[is.finite(long$value), , drop = FALSE]
  if (nrow(long) > 0L) {
    avg <- stats::aggregate(value ~ dataset + implementation + metric, data = long, FUN = mean)
    paths["metrics"] <- file.path(output_dir, "metrics.png")
    grDevices::png(paths["metrics"], width = 1260, height = 980, res = 140)
    old_par <- graphics::par(no.readonly = TRUE)
    tryCatch({
      metric_names <- unique(avg$metric)
      graphics::par(
        mfrow = c(length(metric_names), length(datasets)),
        mar = c(6.8, 4.2, 2.4, 1)
      )
      for (metric in metric_names) {
        for (dataset in datasets) {
          sub <- avg[avg$dataset == dataset & avg$metric == metric, , drop = FALSE]
          if (nrow(sub) == 0L) {
            graphics::plot.new()
            graphics::title(main = paste(dataset, metric, sep = " / "))
            next
          }
          graphics::barplot(
            sub$value,
            names.arg = sub$implementation,
            col = colors[sub$implementation],
            las = 2,
            ylab = "Score",
            main = paste(dataset, metric, sep = " / ")
          )
        }
      }
    }, finally = {
      graphics::par(old_par)
      grDevices::dev.off()
    })
  }
  paths
}

method_aliases <- function(methods) {
  aliases <- c(
    fast = "fastknnumap_sgd",
    native_umap = "fastknnumap_sgd",
    umap = "fastknnumap_sgd",
    spectral = "fastknnumap_spectral",
    hybrid = "fastknnumap_hybrid",
    landmark = "fastknnumap_landmark",
    tsne = "knn_tsne",
    rtsne = "knn_tsne",
    rtsne_neighbors = "knn_tsne",
    pacmap = "knn_pacmap",
    trimap = "knn_trimap",
    localmap = "knn_localmap"
  )
  if (any(methods == "all")) {
    return(unname(aliases[c("hybrid", "landmark", "fast", "spectral", "tsne", "pacmap", "trimap", "localmap")]))
  }
  unknown <- setdiff(methods, c(names(aliases), aliases))
  if (length(unknown) > 0L) {
    stop("Unknown methods: ", paste(unknown, collapse = ", "), call. = FALSE)
  }
  unname(ifelse(methods %in% names(aliases), aliases[methods], methods))
}

#' Benchmark embeddings with a minimal user interface
#'
#' @param datasets Public datasets to run. Defaults to a small practical suite.
#' @param n Row subset for datasets larger than Iris. Use `NULL` for each
#'   dataset's default.
#' @param methods Methods to compare. Use simple names such as `"fast"`,
#'   `"tsne"`, `"pacmap"`, `"trimap"`, `"localmap"`, or `"all"`.
#' @param output_csv Optional path to save the combined metrics table.
#' @return A list with combined `metrics` and per-dataset `results`.
#' @export
benchmark_embed <- function(datasets = c("iris", "pendigits", "fashion_mnist"),
                            n = 2000L,
                            methods = c("fast", "tsne"),
                            output_csv = NULL,
                            preset = c("balanced", "quick", "accuracy")) {
  preset <- match.arg(preset)
  datasets <- match.arg(
    datasets,
    c("iris", "mnist", "fashion_mnist", "pendigits", "shuttle"),
    several.ok = TRUE
  )
  subsets <- stats::setNames(rep(NA_real_, length(datasets)), datasets)
  if (!is.null(n)) {
    subsets[] <- as.numeric(n)
    subsets[datasets == "iris"] <- NA_real_
  }
  settings <- switch(
    preset,
    quick = list(n_epochs = 150L, repeats = 1L, preserve_sample = 2000L),
    balanced = list(n_epochs = 250L, repeats = 1L, preserve_sample = 5000L),
    accuracy = list(n_epochs = 500L, repeats = 3L, preserve_sample = 5000L)
  )
  result <- benchmark_embedding_datasets(
    datasets = datasets,
    subsets = subsets,
    implementations = method_aliases(methods),
    pca_dims = 50L,
    repeats = settings$repeats,
    output_csv = output_csv,
    k = 30L,
    n_epochs = settings$n_epochs,
    hybrid_epochs = 100L,
    n_threads = max(1L, parallel::detectCores(logical = FALSE) - 1L),
    silhouette_sample = 5000L,
    preserve_sample = settings$preserve_sample,
    verbose = FALSE
  )
  if (!is.null(output_csv)) {
    plot_dir <- paste0(tools::file_path_sans_ext(output_csv), "_plots")
    result$plot_paths <- plot_embedding_benchmark(result, plot_dir)
  }
  result
}
