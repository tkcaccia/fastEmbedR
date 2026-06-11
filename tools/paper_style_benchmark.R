#!/usr/bin/env Rscript

suppressPackageStartupMessages(library(fastEmbedR))

parse_scalar <- function(name, default) {
  args <- commandArgs(trailingOnly = TRUE)
  prefix <- paste0("--", name, "=")
  value <- args[startsWith(args, prefix)]
  if (length(value) == 0L) {
    Sys.getenv(paste0("FASTEMBEDR_PAPER_", toupper(gsub("-", "_", name))), default)
  } else {
    sub(prefix, "", value[[1L]], fixed = TRUE)
  }
}

parse_flag <- function(name, default = FALSE) {
  args <- commandArgs(trailingOnly = TRUE)
  env <- Sys.getenv(paste0("FASTEMBEDR_PAPER_", toupper(gsub("-", "_", name))), "")
  any(args == paste0("--", name)) ||
    identical(tolower(env), "1") ||
    identical(tolower(env), "true") ||
    identical(tolower(parse_scalar(name, if (default) "true" else "false")), "true") ||
    identical(parse_scalar(name, if (default) "1" else "0"), "1")
}

standardize_matrix <- function(x) {
  x <- as.matrix(x)
  storage.mode(x) <- "double"
  keep <- apply(x, 2L, function(col) all(is.finite(col)) && stats::sd(col) > 0)
  x <- x[, keep, drop = FALSE]
  x <- scale(x)
  storage.mode(x) <- "double"
  x
}

subsample_dataset <- function(dataset, max_n, seed = 4L) {
  if (is.na(max_n) || max_n < 1L || nrow(dataset$x) <= max_n) return(dataset)
  set.seed(seed)
  keep <- sort(sample.int(nrow(dataset$x), max_n))
  dataset$x <- dataset$x[keep, , drop = FALSE]
  if (!is.null(dataset$labels)) dataset$labels <- dataset$labels[keep]
  dataset
}

dataset_record <- function(name, x, labels = NULL, source = "generated") {
  list(
    name = name,
    x = standardize_matrix(x),
    labels = if (is.null(labels)) NULL else factor(labels),
    source = source
  )
}

make_gaussian_dataset <- function(seed = 4L) {
  set.seed(seed)
  n_per_class <- 250L
  p <- 24L
  labels <- factor(rep(seq_len(4L), each = n_per_class))
  centers <- matrix(0, 4L, p)
  centers[2L, 1L:6L] <- 2.0
  centers[3L, 7L:12L] <- 2.0
  centers[4L, 13L:18L] <- 2.0
  x <- matrix(stats::rnorm(length(labels) * p, sd = 0.85), length(labels), p)
  x <- x + centers[as.integer(labels), , drop = FALSE]
  dataset_record("gaussian_4class_1000", x, labels)
}

make_imbalanced_dataset <- function(seed = 4L) {
  set.seed(seed)
  sizes <- c(900L, 300L, 90L, 60L, 30L)
  p <- 20L
  labels <- factor(rep(seq_along(sizes), sizes))
  centers <- matrix(stats::rnorm(length(sizes) * p, sd = 2.2), length(sizes), p)
  x <- matrix(stats::rnorm(length(labels) * p, sd = 0.9), length(labels), p)
  x <- x + centers[as.integer(labels), , drop = FALSE]
  dataset_record("imbalanced_5class_1380", x, labels)
}

make_many_clusters_dataset <- function(seed = 4L) {
  set.seed(seed)
  n_clusters <- 12L
  n_per <- 125L
  p <- 18L
  labels <- factor(rep(seq_len(n_clusters), each = n_per))
  centers <- matrix(stats::rnorm(n_clusters * p, sd = 3.0), n_clusters, p)
  x <- matrix(stats::rnorm(length(labels) * p, sd = 0.75), length(labels), p)
  x <- x + centers[as.integer(labels), , drop = FALSE]
  dataset_record("many_clusters_1500", x, labels)
}

make_anisotropic_dataset <- function(seed = 4L) {
  set.seed(seed)
  n_per <- 400L
  labels <- factor(rep(1:3, each = n_per))
  x <- matrix(stats::rnorm(length(labels) * 12L), length(labels), 12L)
  x[, 1L] <- x[, 1L] * 4.0
  x[, 2L] <- x[, 2L] * 0.25
  shift <- c(-3.5, 0, 3.5)[as.integer(labels)]
  x[, 1L] <- x[, 1L] + shift
  x[, 3L] <- x[, 3L] + shift * 0.5
  dataset_record("anisotropic_3class_1200", x, labels)
}

make_sparse_signal_dataset <- function(seed = 4L) {
  set.seed(seed)
  n_per <- 350L
  p <- 80L
  labels <- factor(rep(1:4, each = n_per))
  x <- matrix(stats::rnorm(length(labels) * p, sd = 1.0), length(labels), p)
  for (class_id in seq_len(4L)) {
    cols <- ((class_id - 1L) * 5L + 1L):(class_id * 5L)
    x[labels == class_id, cols] <- x[labels == class_id, cols] + 2.2
  }
  dataset_record("sparse_signal_4class_1400", x, labels)
}

sklearn_datasets <- function() {
  if (!requireNamespace("reticulate", quietly = TRUE)) return(NULL)
  tryCatch(reticulate::import("sklearn.datasets"), error = function(e) NULL)
}

load_digits_dataset <- function() {
  sk <- sklearn_datasets()
  if (is.null(sk)) return(NULL)
  digits <- sk$load_digits()
  dataset_record("sklearn_digits_1797", digits$data, as.integer(digits$target), "sklearn load_digits")
}

load_sklearn_tuple_dataset <- function(kind, n = 1500L, seed = 4L) {
  sk <- sklearn_datasets()
  if (is.null(sk)) return(NULL)
  if (identical(kind, "moons")) {
    out <- sk$make_moons(n_samples = as.integer(n), noise = 0.08, random_state = as.integer(seed))
    return(dataset_record("sklearn_moons_1500", out[[1L]], out[[2L]], "sklearn make_moons"))
  }
  if (identical(kind, "circles")) {
    out <- sk$make_circles(n_samples = as.integer(n), noise = 0.05, factor = 0.45, random_state = as.integer(seed))
    return(dataset_record("sklearn_circles_1500", out[[1L]], out[[2L]], "sklearn make_circles"))
  }
  if (identical(kind, "blobs")) {
    out <- sk$make_blobs(n_samples = as.integer(n), centers = 8L, n_features = 12L,
                         cluster_std = 1.2, random_state = as.integer(seed))
    return(dataset_record("sklearn_blobs_1500", out[[1L]], out[[2L]], "sklearn make_blobs"))
  }
  if (identical(kind, "swiss_roll")) {
    out <- sk$make_swiss_roll(n_samples = as.integer(n), noise = 0.08, random_state = as.integer(seed))
    color <- as.numeric(out[[2L]])
    labels <- cut(color, breaks = stats::quantile(color, probs = seq(0, 1, length.out = 7), na.rm = TRUE),
                  include.lowest = TRUE, labels = FALSE)
    return(dataset_record("sklearn_swiss_roll_1500", out[[1L]], labels, "sklearn make_swiss_roll"))
  }
  if (identical(kind, "s_curve")) {
    out <- sk$make_s_curve(n_samples = as.integer(n), noise = 0.08, random_state = as.integer(seed))
    color <- as.numeric(out[[2L]])
    labels <- cut(color, breaks = stats::quantile(color, probs = seq(0, 1, length.out = 7), na.rm = TRUE),
                  include.lowest = TRUE, labels = FALSE)
    return(dataset_record("sklearn_s_curve_1500", out[[1L]], labels, "sklearn make_s_curve"))
  }
  NULL
}

download_text_file <- function(url, path) {
  if (file.exists(path)) return(path)
  dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
  ok <- tryCatch({
    utils::download.file(url, path, quiet = TRUE, mode = "wb")
    TRUE
  }, error = function(e) FALSE)
  if (isTRUE(ok) && file.exists(path)) path else NA_character_
}

load_pendigits_dataset <- function(cache_dir) {
  path <- download_text_file(
    "https://archive.ics.uci.edu/ml/machine-learning-databases/pendigits/pendigits.tra",
    file.path(cache_dir, "pendigits.tra")
  )
  if (is.na(path)) return(NULL)
  dat <- utils::read.table(path, sep = ",", header = FALSE)
  dataset_record("uci_pendigits_train", dat[, -ncol(dat), drop = FALSE], dat[[ncol(dat)]], "UCI PenDigits")
}

load_letter_dataset <- function(cache_dir) {
  path <- download_text_file(
    "https://archive.ics.uci.edu/ml/machine-learning-databases/letter-recognition/letter-recognition.data",
    file.path(cache_dir, "letter-recognition.data")
  )
  if (is.na(path)) return(NULL)
  dat <- utils::read.table(path, sep = ",", header = FALSE)
  dataset_record("uci_letter_recognition", dat[, -1L, drop = FALSE], dat[[1L]], "UCI Letter Recognition")
}

script_dir <- function() {
  file_arg <- commandArgs(FALSE)
  file_arg <- file_arg[startsWith(file_arg, "--file=")]
  if (length(file_arg) > 0L) {
    return(dirname(normalizePath(sub("^--file=", "", file_arg[[1L]]), mustWork = FALSE)))
  }
  file.path(getwd(), "tools")
}

ensure_kaggle_npz <- function(cache_dir, dataset_key, processed_file) {
  kaggle_dir <- file.path(cache_dir, "kaggle")
  out <- file.path(kaggle_dir, "processed", processed_file)
  if (file.exists(out)) return(out)
  if (!isTRUE(getOption("fastembedr.download_kaggle", FALSE))) return(NA_character_)
  prepare_script <- file.path(script_dir(), "prepare_kaggle_paper_datasets.py")
  if (!file.exists(prepare_script)) return(NA_character_)
  args <- c(prepare_script, "--dataset", dataset_key, "--cache-dir", kaggle_dir)
  status <- tryCatch({
    out_lines <- system2("python3", args, stdout = TRUE, stderr = TRUE)
    if (length(out_lines) > 0L) message(paste(out_lines, collapse = "\n"))
    status <- attr(out_lines, "status")
    if (is.null(status)) 0L else as.integer(status)
  }, error = function(e) 1L)
  if (identical(as.integer(status), 0L) && file.exists(out)) out else NA_character_
}

load_npz_dataset <- function(path, name, source) {
  if (is.na(path) || !file.exists(path) || !requireNamespace("reticulate", quietly = TRUE)) return(NULL)
  tryCatch({
    np <- reticulate::import("numpy", convert = FALSE)
    raw <- np$load(path, allow_pickle = TRUE)
    x <- reticulate::py_to_r(raw[["x"]])
    labels <- reticulate::py_to_r(raw[["labels"]])
    dataset_record(name, x, as.character(labels), source)
  }, error = function(e) NULL)
}

load_kaggle_usps_dataset <- function(cache_dir) {
  path <- ensure_kaggle_npz(cache_dir, "usps", "kaggle_usps.npz")
  load_npz_dataset(path, "kaggle_usps_9298", "Kaggle bistaumanga/usps-dataset")
}

load_kaggle_lyrics_dataset <- function(cache_dir) {
  processed_file <- Sys.getenv("FASTEMBEDR_KAGGLE_LYRICS_FILE", "kaggle_metrolyrics_svd100.npz")
  path <- ensure_kaggle_npz(cache_dir, "lyrics", processed_file)
  load_npz_dataset(path, "kaggle_metrolyrics_svd100", "Kaggle gyani95/380000-lyrics-from-metrolyrics")
}

dataset_max_n_from_name <- function(name) {
  hit <- regmatches(name, regexpr("_n[0-9]+$", name))
  if (length(hit) == 0L || !nzchar(hit)) return(NA_integer_)
  as.integer(sub("_n", "", hit))
}

load_dataset_for_result <- function(name, cache_dir, seed = 4L) {
  max_n <- dataset_max_n_from_name(name)
  base <- sub("_n[0-9]+$", "", name)
  out <- switch(
    base,
    iris = dataset_record("iris", iris[, 1L:4L], iris$Species, "R datasets"),
    sklearn_digits_1797 = load_digits_dataset(),
    gaussian_4class_1000 = make_gaussian_dataset(seed),
    imbalanced_5class_1380 = make_imbalanced_dataset(seed),
    many_clusters_1500 = make_many_clusters_dataset(seed),
    anisotropic_3class_1200 = make_anisotropic_dataset(seed),
    sparse_signal_4class_1400 = make_sparse_signal_dataset(seed),
    sklearn_moons_1500 = load_sklearn_tuple_dataset("moons", 1500L, seed),
    sklearn_circles_1500 = load_sklearn_tuple_dataset("circles", 1500L, seed),
    sklearn_blobs_1500 = load_sklearn_tuple_dataset("blobs", 1500L, seed),
    sklearn_swiss_roll_1500 = load_sklearn_tuple_dataset("swiss_roll", 1500L, seed),
    sklearn_s_curve_1500 = load_sklearn_tuple_dataset("s_curve", 1500L, seed),
    uci_pendigits_train = load_pendigits_dataset(cache_dir),
    uci_letter_recognition = load_letter_dataset(cache_dir),
    kaggle_usps_9298 = load_kaggle_usps_dataset(cache_dir),
    kaggle_metrolyrics_svd100 = load_kaggle_lyrics_dataset(cache_dir),
    NULL
  )
  if (is.null(out)) stop("Cannot reconstruct dataset: ", name, call. = FALSE)
  out <- subsample_dataset(out, max_n, seed = seed)
  out$name <- name
  out
}

dataset_cache <- new.env(parent = emptyenv())

get_dataset <- function(name, cache_dir, seed = 4L) {
  key <- paste(name, seed, sep = "\r")
  if (!exists(key, envir = dataset_cache, inherits = FALSE)) {
    assign(key, load_dataset_for_result(name, cache_dir, seed), envir = dataset_cache)
  }
  get(key, envir = dataset_cache, inherits = FALSE)
}

linear_reconstruction_error <- function(x, y) {
  x <- scale(as.matrix(x), center = TRUE, scale = FALSE)
  y <- scale(as.matrix(y), center = TRUE, scale = FALSE)
  gram <- crossprod(y)
  if (rcond(gram) < 1e-10) gram <- gram + diag(1e-8, nrow(gram))
  coef <- solve(gram, crossprod(y, x))
  recon <- y %*% coef
  sum((x - recon) * (x - recon))
}

global_score <- function(x, embedding) {
  x <- as.matrix(x)
  embedding <- as.matrix(embedding)
  if (ncol(x) < 2L) return(NA_real_)
  pca <- tryCatch(stats::prcomp(x, center = TRUE, scale. = FALSE)$x[, 1:2, drop = FALSE], error = function(e) NULL)
  if (is.null(pca)) return(NA_real_)
  epca <- linear_reconstruction_error(x, pca)
  if (!is.finite(epca) || epca <= .Machine$double.eps) return(NA_real_)
  e <- linear_reconstruction_error(x, embedding)
  score <- exp(-max(0, e - epca) / epca)
  max(0, min(1, score))
}

center_distance_pcc <- function(x, embedding, labels = NULL, seed = 4L) {
  x <- as.matrix(x)
  embedding <- as.matrix(embedding)
  if (!is.null(labels) && length(unique(labels)) >= 3L) {
    groups <- factor(labels)
  } else {
    set.seed(seed)
    k <- min(30L, max(3L, floor(sqrt(nrow(x)))))
    groups <- factor(stats::kmeans(x, centers = k, nstart = 3L, iter.max = 50L)$cluster)
  }
  if (length(levels(groups)) < 3L) return(NA_real_)
  x_centers <- do.call(rbind, lapply(levels(groups), function(g) colMeans(x[groups == g, , drop = FALSE])))
  y_centers <- do.call(rbind, lapply(levels(groups), function(g) colMeans(embedding[groups == g, , drop = FALSE])))
  xd <- as.numeric(stats::dist(x_centers))
  yd <- as.numeric(stats::dist(y_centers))
  if (stats::sd(xd) == 0 || stats::sd(yd) == 0) return(NA_real_)
  stats::cor(xd, yd, method = "pearson")
}

embedding_knn_accuracy <- function(embedding, labels, k = 10L) {
  if (is.null(labels) || length(unique(labels)) < 2L) return(NA_real_)
  labels <- factor(labels)
  d <- as.matrix(stats::dist(embedding))
  diag(d) <- Inf
  idx <- t(apply(d, 1L, order))[, seq_len(min(k, nrow(d) - 1L)), drop = FALSE]
  pred <- vapply(seq_len(nrow(idx)), function(i) {
    tab <- table(labels[idx[i, ]])
    names(tab)[which.max(tab)]
  }, character(1))
  mean(factor(pred, levels = levels(labels)) == labels)
}

augment_results <- function(results, cache_dir, out_dir, seed = 4L) {
  dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
  ok <- results$status == "success" & file.exists(results$layout_path)
  results$trimap_global_score <- NA_real_
  results$center_distance_pcc <- NA_real_
  results$local_nn_accuracy <- results$label_knn_accuracy
  for (row in which(ok)) {
    dataset <- get_dataset(results$dataset[[row]], cache_dir, seed = seed)
    layout <- readRDS(results$layout_path[[row]])
    results$trimap_global_score[[row]] <- global_score(dataset$x, layout)
    results$center_distance_pcc[[row]] <- center_distance_pcc(dataset$x, layout, dataset$labels, seed = seed)
    if (!is.finite(results$local_nn_accuracy[[row]])) {
      results$local_nn_accuracy[[row]] <- embedding_knn_accuracy(layout, dataset$labels)
    }
  }
  utils::write.csv(results, file.path(out_dir, "paper_style_results_augmented.csv"), row.names = FALSE)
  results
}

best_strict_rows <- function(results) {
  ok <- results[results$status == "success" & results$benchmark_scope == "strict_knn", , drop = FALSE]
  key <- paste(ok$dataset, ok$implementation, sep = "\r")
  out <- do.call(rbind, lapply(split(ok, key), function(x) {
    x <- x[order(-x$combined_score, -x$trustworthiness, x$total_time_sec), , drop = FALSE]
    x[1L, , drop = FALSE]
  }))
  rownames(out) <- NULL
  out
}

short_impl <- function(x) {
  out <- sub("^fastEmbedR::", "fE ", x)
  out <- sub("^Rtsne::", "Rtsne ", out)
  out <- sub("^uwot::", "uwot ", out)
  out <- sub("^umap::", "umap ", out)
  out <- sub("_knn_exact$", " exact", out)
  out <- sub("_knn$", "", out)
  out
}

write_wide_table <- function(data, value, path, digits = 3L) {
  tab <- data[data$status == "success" & data$benchmark_scope == "strict_knn", , drop = FALSE]
  tab <- best_strict_rows(tab)
  tab$value <- round(tab[[value]], digits)
  wide <- reshape(
    tab[, c("implementation", "dataset", "value")],
    idvar = "implementation",
    timevar = "dataset",
    direction = "wide"
  )
  names(wide) <- sub("^value\\.", "", names(wide))
  wide <- wide[order(wide$implementation), , drop = FALSE]
  utils::write.csv(wide, path, row.names = FALSE)
  wide
}

plot_heatmap <- function(data, value, title, path, fill_label = value) {
  if (!requireNamespace("ggplot2", quietly = TRUE)) return(invisible(NULL))
  gg <- asNamespace("ggplot2")
  tab <- best_strict_rows(data)
  tab$implementation_short <- short_impl(tab$implementation)
  p <- gg$ggplot(tab, gg$aes(x = implementation_short, y = dataset, fill = .data[[value]])) +
    gg$geom_tile(color = "white", linewidth = 0.25) +
    gg$scale_fill_viridis_c(option = "C", na.value = "grey90") +
    gg$labs(x = "Method", y = "Dataset", fill = fill_label, title = title) +
    gg$theme_bw(base_size = 10) +
    gg$theme(axis.text.x = gg$element_text(angle = 45, hjust = 1))
  ggplot2::ggsave(path, p, width = 11, height = 7, dpi = 190)
}

plot_speed_quality <- function(data, out_dir) {
  if (!requireNamespace("ggplot2", quietly = TRUE)) return(invisible(NULL))
  gg <- asNamespace("ggplot2")
  tab <- data[data$status == "success" & data$benchmark_scope == "strict_knn", , drop = FALSE]
  tab$implementation_short <- short_impl(tab$implementation)
  p <- gg$ggplot(tab, gg$aes(total_time_sec, trimap_global_score, color = implementation_short, shape = method_family)) +
    gg$geom_point(alpha = 0.76, size = 2.1) +
    gg$scale_x_log10() +
    gg$labs(x = "Runtime including shared KNN (seconds, log scale)", y = "TriMap-style global score",
            color = "Method", shape = "Family", title = "Runtime vs global structure preservation") +
    gg$theme_bw(base_size = 11)
  ggplot2::ggsave(file.path(out_dir, "paper_speed_vs_global_score.png"), p, width = 10, height = 6, dpi = 190)
}

plot_runtime_scaling <- function(data, out_dir) {
  if (!requireNamespace("ggplot2", quietly = TRUE)) return(invisible(NULL))
  gg <- asNamespace("ggplot2")
  tab <- data[data$status == "success" & data$benchmark_scope == "strict_knn", , drop = FALSE]
  tab <- best_strict_rows(tab)
  tab$implementation_short <- short_impl(tab$implementation)
  p <- gg$ggplot(tab, gg$aes(n, total_time_sec, color = implementation_short)) +
    gg$geom_point(alpha = 0.75, size = 2) +
    gg$geom_smooth(method = "lm", formula = y ~ x, se = FALSE, linewidth = 0.7) +
    gg$scale_x_log10() +
    gg$scale_y_log10() +
    gg$labs(x = "Number of samples (log scale)", y = "Runtime (seconds, log scale)",
            color = "Method", title = "Runtime scaling across benchmark datasets") +
    gg$theme_bw(base_size = 11)
  ggplot2::ggsave(file.path(out_dir, "paper_runtime_scaling.png"), p, width = 10, height = 6, dpi = 190)
}

plot_embedding_grid <- function(data, cache_dir, out_dir, seed = 4L) {
  if (!requireNamespace("ggplot2", quietly = TRUE)) return(invisible(NULL))
  gg <- asNamespace("ggplot2")
  methods <- c(
    "fastEmbedR::umap_knn",
    "fastEmbedR::trimap_knn",
    "fastEmbedR::localmap_knn",
    "uwot::umap_knn",
    "Rtsne::Rtsne_neighbors"
  )
  datasets <- c("iris", "sklearn_digits_1797", "sklearn_s_curve_1500", "uci_pendigits_train_n1800")
  tab <- best_strict_rows(data)
  tab <- tab[tab$implementation %in% methods & tab$dataset %in% datasets & file.exists(tab$layout_path), , drop = FALSE]
  if (nrow(tab) == 0L) return(invisible(NULL))
  pieces <- vector("list", nrow(tab))
  for (i in seq_len(nrow(tab))) {
    ds <- get_dataset(tab$dataset[[i]], cache_dir, seed = seed)
    layout <- readRDS(tab$layout_path[[i]])
    pieces[[i]] <- data.frame(
      x = layout[, 1L],
      y = layout[, 2L],
      label = if (is.null(ds$labels)) factor(seq_len(nrow(layout))) else factor(ds$labels),
      dataset = tab$dataset[[i]],
      method = sprintf(
        "%s\nNN %.2f, GS %.2f",
        short_impl(tab$implementation[[i]]),
        tab$local_nn_accuracy[[i]],
        tab$trimap_global_score[[i]]
      ),
      stringsAsFactors = FALSE
    )
  }
  plot_data <- do.call(rbind, pieces)
  p <- gg$ggplot(plot_data, gg$aes(x, y, color = label)) +
    gg$geom_point(size = 0.45, alpha = 0.75, show.legend = FALSE) +
    gg$facet_grid(dataset ~ method, scales = "free") +
    gg$labs(x = NULL, y = NULL, title = "Embedding grid annotated with local NN accuracy and global score") +
    gg$theme_void(base_size = 9) +
    gg$theme(
      strip.text = gg$element_text(size = 7),
      plot.title = gg$element_text(hjust = 0.5)
    )
  ggplot2::ggsave(file.path(out_dir, "paper_embedding_grid_nn_gs.png"), p, width = 12, height = 9, dpi = 200)
}

summarize_stability <- function(stability_dir, cache_dir, out_dir, seed = 4L) {
  path <- file.path(stability_dir, "latest_rjournal_benchmark_results.csv")
  if (!file.exists(path)) return(NULL)
  stability <- read.csv(path)
  stability <- augment_results(stability, cache_dir, file.path(out_dir, "stability_augmented"), seed = seed)
  ok <- stability[stability$status == "success" & stability$benchmark_scope == "strict_knn", , drop = FALSE]
  summary <- aggregate(
    cbind(total_time_sec, trustworthiness, local_nn_accuracy, silhouette,
          trimap_global_score, procrustes_rmsd, combined_score) ~ implementation,
    ok,
    mean,
    na.rm = TRUE
  )
  summary <- summary[order(summary$procrustes_rmsd), , drop = FALSE]
  utils::write.csv(summary, file.path(out_dir, "paper_stability_summary.csv"), row.names = FALSE)
  summary
}

write_report <- function(out_dir, augmented, stability_summary) {
  path <- file.path(out_dir, "paper_style_benchmark_report.md")
  ok <- augmented[augmented$status == "success" & augmented$benchmark_scope == "strict_knn", , drop = FALSE]
  mean_summary <- aggregate(
    cbind(total_time_sec, peak_ram_mb, trustworthiness, local_nn_accuracy,
          silhouette, trimap_global_score, center_distance_pcc, combined_score) ~ implementation,
    ok,
    mean,
    na.rm = TRUE
  )
  mean_summary <- mean_summary[order(-mean_summary$combined_score), , drop = FALSE]
  utils::write.csv(mean_summary, file.path(out_dir, "paper_mean_summary.csv"), row.names = FALSE)
  con <- file(path, "w")
  on.exit(close(con), add = TRUE)
  cat("# Paper-style benchmark for fastEmbedR\n\n", file = con)
  cat("This benchmark follows the evidence style of the TriMap and LocalMAP papers: local neighbor accuracy, global score, silhouette, posthoc classification proxy, runtime, robustness, and embedding grids.\n\n", file = con)
  cat("## Paper-inspired metrics\n\n", file = con)
  cat("- TriMap-style global score: PCA-normalized linear reconstruction score, where PCA is 1 by definition.\n", file = con)
  cat("- Local structure: trustworthiness and label nearest-neighbor accuracy in the embedding.\n", file = con)
  cat("- Cluster separation: silhouette using labels when available.\n", file = con)
  cat("- Global placement: centroid-distance Pearson correlation.\n", file = con)
  cat("- Robustness: Procrustes-aligned RMSD across seeds, lower is better.\n\n", file = con)
  cat("## Mean strict-KNN performance\n\n", file = con)
  if (requireNamespace("knitr", quietly = TRUE)) {
    cat(paste(knitr::kable(mean_summary, format = "markdown", digits = 4, row.names = FALSE), collapse = "\n"), "\n\n", file = con)
  } else {
    utils::write.table(mean_summary, con, sep = "|", row.names = FALSE, quote = FALSE)
    cat("\n\n", file = con)
  }
  if (!is.null(stability_summary)) {
    cat("## Stability supplement\n\n", file = con)
    if (requireNamespace("knitr", quietly = TRUE)) {
      cat(paste(knitr::kable(stability_summary, format = "markdown", digits = 4, row.names = FALSE), collapse = "\n"), "\n\n", file = con)
    } else {
      utils::write.table(stability_summary, con, sep = "|", row.names = FALSE, quote = FALSE)
      cat("\n\n", file = con)
    }
  }
  cat("## Figures\n\n", file = con)
  for (fig in c(
    "paper_speed_vs_global_score.png",
    "paper_global_score_heatmap.png",
    "paper_silhouette_heatmap.png",
    "paper_runtime_heatmap.png",
    "paper_runtime_scaling.png",
    "paper_embedding_grid_nn_gs.png"
  )) {
    if (file.exists(file.path(out_dir, fig))) {
      cat("![", fig, "](", fig, ")\n\n", sep = "", file = con)
    }
  }
  path
}

results_dir <- parse_scalar("results-dir", file.path("results", "rjournal_benchmark"))
stability_dir <- parse_scalar("stability-dir", file.path("results", "rjournal_benchmark_stability"))
out_dir <- parse_scalar("out-dir", file.path("results", "paper_style_benchmark"))
cache_dir <- parse_scalar("cache-dir", file.path(results_dir, "cache"))
seed <- as.integer(parse_scalar("seed", "4"))
options(fastembedr.download_kaggle = parse_flag("download-kaggle", default = FALSE))
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

results_file <- file.path(results_dir, "latest_rjournal_benchmark_results.csv")
if (!file.exists(results_file)) stop("Missing benchmark result file: ", results_file, call. = FALSE)
results <- read.csv(results_file)
augmented <- augment_results(results, cache_dir, out_dir, seed = seed)
best <- best_strict_rows(augmented)
utils::write.csv(best, file.path(out_dir, "paper_best_strict_rows.csv"), row.names = FALSE)

invisible(write_wide_table(augmented, "trimap_global_score", file.path(out_dir, "paper_table_global_score.csv")))
invisible(write_wide_table(augmented, "silhouette", file.path(out_dir, "paper_table_silhouette.csv")))
invisible(write_wide_table(augmented, "total_time_sec", file.path(out_dir, "paper_table_runtime_sec.csv")))
invisible(write_wide_table(augmented, "local_nn_accuracy", file.path(out_dir, "paper_table_local_nn_accuracy.csv")))

plot_heatmap(augmented, "trimap_global_score", "TriMap-style global score", file.path(out_dir, "paper_global_score_heatmap.png"), "GS")
plot_heatmap(augmented, "silhouette", "LocalMAP-style silhouette score", file.path(out_dir, "paper_silhouette_heatmap.png"), "Silhouette")
plot_heatmap(augmented, "total_time_sec", "Runtime by dataset", file.path(out_dir, "paper_runtime_heatmap.png"), "seconds")
plot_speed_quality(augmented, out_dir)
plot_runtime_scaling(augmented, out_dir)
plot_embedding_grid(augmented, cache_dir, out_dir, seed = seed)
stability_summary <- summarize_stability(stability_dir, cache_dir, out_dir, seed = seed)
report <- write_report(out_dir, augmented, stability_summary)

cat("Saved paper-style benchmark artifacts:\n")
cat("  ", normalizePath(file.path(out_dir, "paper_style_results_augmented.csv")), "\n", sep = "")
cat("  ", normalizePath(file.path(out_dir, "paper_best_strict_rows.csv")), "\n", sep = "")
cat("  ", normalizePath(file.path(out_dir, "paper_mean_summary.csv")), "\n", sep = "")
cat("  ", normalizePath(report), "\n", sep = "")
