#!/usr/bin/env Rscript

suppressPackageStartupMessages(library(fastEmbedR))

parse_csv_arg <- function(name, default) {
  args <- commandArgs(trailingOnly = TRUE)
  prefix <- paste0("--", name, "=")
  value <- args[startsWith(args, prefix)]
  if (length(value) == 0L) {
    value <- Sys.getenv(paste0("FASTEMBEDR_RJ_", toupper(gsub("-", "_", name))), default)
  } else {
    value <- sub(prefix, "", value[[1L]], fixed = TRUE)
  }
  value <- trimws(strsplit(value, ",", fixed = TRUE)[[1L]])
  value[nzchar(value)]
}

parse_scalar <- function(name, default) {
  args <- commandArgs(trailingOnly = TRUE)
  prefix <- paste0("--", name, "=")
  value <- args[startsWith(args, prefix)]
  if (length(value) == 0L) {
    Sys.getenv(paste0("FASTEMBEDR_RJ_", toupper(gsub("-", "_", name))), default)
  } else {
    sub(prefix, "", value[[1L]], fixed = TRUE)
  }
}

parse_flag <- function(name, default = FALSE) {
  args <- commandArgs(trailingOnly = TRUE)
  env <- Sys.getenv(paste0("FASTEMBEDR_RJ_", toupper(gsub("-", "_", name))), "")
  any(args == paste0("--", name)) ||
    identical(tolower(env), "1") ||
    identical(tolower(env), "true") ||
    isTRUE(default)
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

subsample_dataset <- function(dataset, max_n, seed) {
  max_n <- as.integer(max_n)
  if (!is.finite(max_n) || max_n < 1L || nrow(dataset$x) <= max_n) return(dataset)
  set.seed(seed)
  keep <- sort(sample.int(nrow(dataset$x), max_n))
  dataset$x <- dataset$x[keep, , drop = FALSE]
  dataset$labels <- if (is.null(dataset$labels)) NULL else dataset$labels[keep]
  dataset$name <- paste0(dataset$name, "_n", max_n)
  dataset
}

dataset_record <- function(name, x, labels = NULL, source = "generated", task = "classification") {
  if (!is.null(labels)) labels <- factor(labels)
  list(
    name = name,
    x = standardize_matrix(x),
    labels = labels,
    source = source,
    task = task
  )
}

make_gaussian_dataset <- function(seed = 11L) {
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
  dataset_record("gaussian_4class_1000", x, labels, "generated")
}

make_imbalanced_dataset <- function(seed = 12L) {
  set.seed(seed)
  sizes <- c(900L, 300L, 90L, 60L, 30L)
  p <- 20L
  labels <- factor(rep(seq_along(sizes), sizes))
  centers <- matrix(stats::rnorm(length(sizes) * p, sd = 2.2), length(sizes), p)
  x <- matrix(stats::rnorm(length(labels) * p, sd = 0.9), length(labels), p)
  x <- x + centers[as.integer(labels), , drop = FALSE]
  dataset_record("imbalanced_5class_1380", x, labels, "generated")
}

make_many_clusters_dataset <- function(seed = 13L) {
  set.seed(seed)
  n_clusters <- 12L
  n_per <- 125L
  p <- 18L
  labels <- factor(rep(seq_len(n_clusters), each = n_per))
  centers <- matrix(stats::rnorm(n_clusters * p, sd = 3.0), n_clusters, p)
  x <- matrix(stats::rnorm(length(labels) * p, sd = 0.75), length(labels), p)
  x <- x + centers[as.integer(labels), , drop = FALSE]
  dataset_record("many_clusters_1500", x, labels, "generated")
}

make_anisotropic_dataset <- function(seed = 14L) {
  set.seed(seed)
  n_per <- 400L
  labels <- factor(rep(1:3, each = n_per))
  x <- matrix(stats::rnorm(length(labels) * 12L), length(labels), 12L)
  x[, 1L] <- x[, 1L] * 4.0
  x[, 2L] <- x[, 2L] * 0.25
  shift <- c(-3.5, 0, 3.5)[as.integer(labels)]
  x[, 1L] <- x[, 1L] + shift
  x[, 3L] <- x[, 3L] + shift * 0.5
  dataset_record("anisotropic_3class_1200", x, labels, "generated")
}

make_sparse_signal_dataset <- function(seed = 15L) {
  set.seed(seed)
  n_per <- 350L
  p <- 80L
  labels <- factor(rep(1:4, each = n_per))
  x <- matrix(stats::rnorm(length(labels) * p, sd = 1.0), length(labels), p)
  for (class_id in seq_len(4L)) {
    cols <- ((class_id - 1L) * 5L + 1L):(class_id * 5L)
    x[labels == class_id, cols] <- x[labels == class_id, cols] + 2.2
  }
  dataset_record("sparse_signal_4class_1400", x, labels, "generated")
}

sklearn_datasets <- function() {
  if (!requireNamespace("reticulate", quietly = TRUE)) return(NULL)
  tryCatch(reticulate::import("sklearn.datasets"), error = function(e) NULL)
}

load_digits_dataset <- function() {
  sk <- sklearn_datasets()
  if (is.null(sk)) return(NULL)
  tryCatch({
    digits <- sk$load_digits()
    dataset_record("sklearn_digits_1797", digits$data, as.integer(digits$target), "sklearn load_digits")
  }, error = function(e) NULL)
}

load_sklearn_tuple_dataset <- function(name, n, seed) {
  sk <- sklearn_datasets()
  if (is.null(sk)) return(NULL)
  n <- as.integer(n)
  tryCatch({
    if (identical(name, "moons")) {
      out <- sk$make_moons(n_samples = as.integer(n), noise = 0.08, random_state = as.integer(seed))
      return(dataset_record(paste0("sklearn_moons_", n), out[[1L]], out[[2L]], "sklearn make_moons"))
    }
    if (identical(name, "circles")) {
      out <- sk$make_circles(n_samples = as.integer(n), noise = 0.05, factor = 0.45, random_state = as.integer(seed))
      return(dataset_record(paste0("sklearn_circles_", n), out[[1L]], out[[2L]], "sklearn make_circles"))
    }
    if (identical(name, "blobs")) {
      out <- sk$make_blobs(
        n_samples = as.integer(n),
        centers = 8L,
        n_features = 12L,
        cluster_std = 1.2,
        random_state = as.integer(seed)
      )
      return(dataset_record(paste0("sklearn_blobs_", n), out[[1L]], out[[2L]], "sklearn make_blobs"))
    }
    if (identical(name, "swiss_roll")) {
      out <- sk$make_swiss_roll(n_samples = as.integer(n), noise = 0.08, random_state = as.integer(seed))
      color <- as.numeric(out[[2L]])
      labels <- cut(color, breaks = stats::quantile(color, probs = seq(0, 1, length.out = 7), na.rm = TRUE),
                    include.lowest = TRUE, labels = FALSE)
      return(dataset_record(paste0("sklearn_swiss_roll_", n), out[[1L]], labels, "sklearn make_swiss_roll", "ordered manifold"))
    }
    if (identical(name, "s_curve")) {
      out <- sk$make_s_curve(n_samples = as.integer(n), noise = 0.08, random_state = as.integer(seed))
      color <- as.numeric(out[[2L]])
      labels <- cut(color, breaks = stats::quantile(color, probs = seq(0, 1, length.out = 7), na.rm = TRUE),
                    include.lowest = TRUE, labels = FALSE)
      return(dataset_record(paste0("sklearn_s_curve_", n), out[[1L]], labels, "sklearn make_s_curve", "ordered manifold"))
    }
    NULL
  }, error = function(e) NULL)
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

load_pendigits_dataset <- function(cache_dir, max_n, seed) {
  url <- "https://archive.ics.uci.edu/ml/machine-learning-databases/pendigits/pendigits.tra"
  path <- download_text_file(url, file.path(cache_dir, "pendigits.tra"))
  if (is.na(path)) return(NULL)
  dat <- tryCatch(utils::read.table(path, sep = ",", header = FALSE), error = function(e) NULL)
  if (is.null(dat) || ncol(dat) < 2L) return(NULL)
  x <- dat[, -ncol(dat), drop = FALSE]
  labels <- dat[[ncol(dat)]]
  subsample_dataset(dataset_record("uci_pendigits_train", x, labels, "UCI PenDigits"), max_n, seed)
}

load_letter_dataset <- function(cache_dir, max_n, seed) {
  url <- "https://archive.ics.uci.edu/ml/machine-learning-databases/letter-recognition/letter-recognition.data"
  path <- download_text_file(url, file.path(cache_dir, "letter-recognition.data"))
  if (is.na(path)) return(NULL)
  dat <- tryCatch(utils::read.table(path, sep = ",", header = FALSE), error = function(e) NULL)
  if (is.null(dat) || ncol(dat) < 2L) return(NULL)
  x <- dat[, -1L, drop = FALSE]
  labels <- dat[[1L]]
  subsample_dataset(dataset_record("uci_letter_recognition", x, labels, "UCI Letter Recognition"), max_n, seed)
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
  if (!isTRUE(getOption("fastembedr.download_kaggle", FALSE))) {
    warning(
      "Skipping Kaggle dataset `", dataset_key, "` because prepared file is missing: ", out,
      ". Run tools/prepare_kaggle_paper_datasets.py first, or pass --download-kaggle.",
      call. = FALSE
    )
    return(NA_character_)
  }
  prepare_script <- file.path(script_dir(), "prepare_kaggle_paper_datasets.py")
  if (!file.exists(prepare_script)) {
    warning("Cannot find Kaggle preparation script: ", prepare_script, call. = FALSE)
    return(NA_character_)
  }
  args <- c(
    prepare_script,
    "--dataset", dataset_key,
    "--cache-dir", kaggle_dir
  )
  message("Preparing optional Kaggle dataset: ", dataset_key)
  status <- tryCatch({
    out_lines <- system2("python3", args, stdout = TRUE, stderr = TRUE)
    if (length(out_lines) > 0L) message(paste(out_lines, collapse = "\n"))
    status <- attr(out_lines, "status")
    if (is.null(status)) 0L else as.integer(status)
  }, warning = function(w) {
    warning(conditionMessage(w), call. = FALSE)
    1L
  }, error = function(e) {
    warning(conditionMessage(e), call. = FALSE)
    1L
  })
  if (identical(status, 0L) && file.exists(out)) out else {
    warning("Kaggle dataset preparation did not produce: ", out, call. = FALSE)
    NA_character_
  }
}

load_npz_dataset <- function(path, name, source) {
  if (is.na(path) || !file.exists(path)) return(NULL)
  if (!requireNamespace("reticulate", quietly = TRUE)) {
    warning("Skipping ", name, " because reticulate is not installed.", call. = FALSE)
    return(NULL)
  }
  tryCatch({
    np <- reticulate::import("numpy", convert = FALSE)
    raw <- np$load(path, allow_pickle = TRUE)
    x <- reticulate::py_to_r(raw[["x"]])
    labels <- reticulate::py_to_r(raw[["labels"]])
    dataset_record(name, x, as.character(labels), source)
  }, error = function(e) {
    warning("Skipping ", name, ": ", conditionMessage(e), call. = FALSE)
    NULL
  })
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

load_named_dataset <- function(name, cache_dir, max_n, seed) {
  key <- tolower(name)
  large_n <- max(10001L, as.integer(max_n))
  out <- switch(
    key,
    iris = dataset_record("iris", iris[, 1L:4L], iris$Species, "R datasets"),
    digits = load_digits_dataset(),
    gaussian = make_gaussian_dataset(seed),
    gaussian_4class = make_gaussian_dataset(seed),
    imbalanced = make_imbalanced_dataset(seed),
    many_clusters = make_many_clusters_dataset(seed),
    anisotropic = make_anisotropic_dataset(seed),
    sparse_signal = make_sparse_signal_dataset(seed),
    moons = load_sklearn_tuple_dataset("moons", 1500L, seed),
    circles = load_sklearn_tuple_dataset("circles", 1500L, seed),
    blobs = load_sklearn_tuple_dataset("blobs", 1500L, seed),
    swiss_roll = load_sklearn_tuple_dataset("swiss_roll", 1500L, seed),
    s_curve = load_sklearn_tuple_dataset("s_curve", 1500L, seed),
    moons_large = load_sklearn_tuple_dataset("moons", large_n, seed),
    circles_large = load_sklearn_tuple_dataset("circles", large_n, seed),
    blobs_large = load_sklearn_tuple_dataset("blobs", large_n, seed),
    swiss_roll_large = load_sklearn_tuple_dataset("swiss_roll", large_n, seed),
    s_curve_large = load_sklearn_tuple_dataset("s_curve", large_n, seed),
    pendigits = load_pendigits_dataset(cache_dir, max_n, seed),
    letter = load_letter_dataset(cache_dir, max_n, seed),
    kaggle_usps = load_kaggle_usps_dataset(cache_dir),
    kaggle_lyrics = load_kaggle_lyrics_dataset(cache_dir),
    kaggle_metrolyrics = load_kaggle_lyrics_dataset(cache_dir),
    stop("Unknown dataset: ", name, call. = FALSE)
  )
  if (is.null(out)) {
    warning("Skipping unavailable dataset: ", name, call. = FALSE)
    return(NULL)
  }
  subsample_dataset(out, max_n, seed)
}

metric_k <- function(n) {
  values <- c(15L, 30L, 50L)
  values[values < n]
}

perplexity_from_k <- function(k, n) {
  max_valid <- floor((n - 1L) / 3L)
  max(2L, min(30L, max_valid, floor(k / 3L)))
}

json_or_text <- function(x) {
  if (requireNamespace("jsonlite", quietly = TRUE)) {
    return(as.character(jsonlite::toJSON(x, auto_unbox = TRUE, null = "null")))
  }
  paste(paste(names(x), unlist(x), sep = "="), collapse = ";")
}

package_version_or_na <- function(package) {
  if (!requireNamespace(package, quietly = TRUE)) return(NA_character_)
  as.character(utils::packageVersion(package))
}

has_export <- function(package, name) {
  requireNamespace(package, quietly = TRUE) &&
    exists(name, envir = asNamespace(package), inherits = FALSE)
}

fast_tsne_path <- function() {
  candidates <- c(
    Sys.getenv("FAST_TSNE_PATH", ""),
    file.path(Sys.getenv("HOME"), ".local", "bin", "fast_tsne"),
    Sys.which("fast_tsne")
  )
  candidates <- candidates[nzchar(candidates)]
  candidates <- candidates[file.exists(candidates) & file.access(candidates, 1L) == 0L]
  if (length(candidates) == 0L) "" else normalizePath(candidates[[1L]], mustWork = FALSE)
}

fft_rtsne_info <- function() {
  for (package in c("fftRtsne", "Spectre")) {
    if (has_export(package, "fftRtsne")) {
      fun <- get("fftRtsne", envir = asNamespace(package), inherits = FALSE)
      path <- fast_tsne_path()
      if ("fast_tsne_path" %in% names(formals(fun)) && !nzchar(path)) return(NULL)
      return(list(package = package, fun = fun, fast_tsne_path = path))
    }
  }
  NULL
}

current_rss_mb <- function() {
  pid <- Sys.getpid()
  out <- suppressWarnings(system2("ps", c("-o", "rss=", "-p", as.character(pid)), stdout = TRUE, stderr = FALSE))
  if (length(out) == 0L) return(NA_real_)
  rss_kb <- suppressWarnings(as.numeric(trimws(out[[1L]])))
  if (!is.finite(rss_kb)) NA_real_ else rss_kb / 1024
}

extract_layout <- function(x) {
  if (length(dim(x)) == 3L) return(as.matrix(x[, , 1L, drop = TRUE]))
  if (is.matrix(x) || is.data.frame(x)) return(as.matrix(x))
  if (is.list(x)) {
    for (name in c("layout", "Y", "embedding", "embeddings", "data")) {
      if (!is.null(x[[name]])) return(extract_layout(x[[name]]))
    }
  }
  as.matrix(x)
}

coerce_layout <- function(x, n) {
  x <- extract_layout(x)
  storage.mode(x) <- "double"
  if (nrow(x) != n && ncol(x) == n) x <- t(x)
  if (nrow(x) != n || ncol(x) < 2L) {
    stop("The returned object is not an n x 2 embedding matrix.", call. = FALSE)
  }
  x[, seq_len(2L), drop = FALSE]
}

measure_layout <- function(expr, n) {
  invisible(gc())
  rss_before <- current_rss_mb()
  elapsed <- system.time({
    layout <- force(expr)
  })[["elapsed"]]
  rss_after <- current_rss_mb()
  list(
    layout = coerce_layout(layout, n),
    elapsed = as.numeric(elapsed),
    rss_before_mb = rss_before,
    rss_after_mb = rss_after,
    rss_delta_mb = if (is.finite(rss_before) && is.finite(rss_after)) rss_after - rss_before else NA_real_,
    peak_ram_mb = if (is.finite(rss_before) && is.finite(rss_after)) max(rss_before, rss_after) else NA_real_
  )
}

failure_row <- function(ctx, adapter, status, message) {
  data.frame(
    dataset = ctx$dataset$name,
    dataset_source = ctx$dataset$source,
    method_family = adapter$family,
    implementation = adapter$id,
    package = adapter$package,
    function_name = adapter$function_name,
    package_version = package_version_or_na(adapter$package),
    input_type = adapter$input_type,
    benchmark_scope = if (identical(adapter$input_type, "precomputed_knn")) "strict_knn" else "raw_x_separate",
    supports_precomputed_knn = isTRUE(adapter$supports_precomputed_knn),
    n = nrow(ctx$dataset$x),
    p = ncol(ctx$dataset$x),
    seed = ctx$seed,
    knn_k_without_self = ctx$k_without_self,
    perplexity = ctx$perplexity,
    n_epochs = ctx$n_epochs,
    max_iter = ctx$max_iter,
    knn_time_sec = if (identical(adapter$input_type, "precomputed_knn")) ctx$knn_time else NA_real_,
    embedding_time_sec = NA_real_,
    total_time_sec = NA_real_,
    rss_before_mb = NA_real_,
    rss_after_mb = NA_real_,
    rss_delta_mb = NA_real_,
    peak_ram_mb = NA_real_,
    status = status,
    error_message = message,
    parameters_json = json_or_text(adapter$params(ctx)),
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
    layout_path = NA_character_,
    stringsAsFactors = FALSE
  )
}

save_layout <- function(layout, ctx, adapter, layout_dir) {
  dir.create(layout_dir, recursive = TRUE, showWarnings = FALSE)
  safe_name <- gsub("[^A-Za-z0-9_.-]+", "_", paste(ctx$dataset$name, adapter$id, "k", ctx$k_without_self, "seed", ctx$seed, sep = "_"))
  path <- file.path(layout_dir, paste0(safe_name, ".rds"))
  saveRDS(layout, path)
  path
}

success_row <- function(ctx, adapter, measured, layout_dir) {
  k_eval <- metric_k(nrow(ctx$dataset$x))
  metrics <- fastEmbedR::evaluate_embedding(
    ctx$dataset$x,
    measured$layout,
    labels = ctx$dataset$labels,
    k = k_eval,
    reference_nn = ctx$eval_reference,
    sample_size_for_global_metrics = min(ctx$global_sample_size, nrow(ctx$dataset$x)),
    sample_size_for_local_metrics = min(ctx$local_sample_size, nrow(ctx$dataset$x)),
    use_cache = FALSE,
    seed = ctx$seed,
    method = adapter$id,
    backend = "cpu",
    dataset = ctx$dataset$name
  )
  data.frame(
    dataset = ctx$dataset$name,
    dataset_source = ctx$dataset$source,
    method_family = adapter$family,
    implementation = adapter$id,
    package = adapter$package,
    function_name = adapter$function_name,
    package_version = package_version_or_na(adapter$package),
    input_type = adapter$input_type,
    benchmark_scope = if (identical(adapter$input_type, "precomputed_knn")) "strict_knn" else "raw_x_separate",
    supports_precomputed_knn = isTRUE(adapter$supports_precomputed_knn),
    n = nrow(ctx$dataset$x),
    p = ncol(ctx$dataset$x),
    seed = ctx$seed,
    knn_k_without_self = ctx$k_without_self,
    perplexity = ctx$perplexity,
    n_epochs = ctx$n_epochs,
    max_iter = ctx$max_iter,
    knn_time_sec = if (identical(adapter$input_type, "precomputed_knn")) ctx$knn_time else NA_real_,
    embedding_time_sec = measured$elapsed,
    total_time_sec = measured$elapsed + if (identical(adapter$input_type, "precomputed_knn")) ctx$knn_time else 0,
    rss_before_mb = measured$rss_before_mb,
    rss_after_mb = measured$rss_after_mb,
    rss_delta_mb = measured$rss_delta_mb,
    peak_ram_mb = measured$peak_ram_mb,
    status = "success",
    error_message = NA_character_,
    parameters_json = json_or_text(adapter$params(ctx)),
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
    layout_path = save_layout(measured$layout, ctx, adapter, layout_dir),
    stringsAsFactors = FALSE
  )
}

run_adapter <- function(ctx, adapter, include_raw_x, layout_dir) {
  if (!requireNamespace(adapter$package, quietly = TRUE)) {
    return(failure_row(ctx, adapter, "not_installed", paste0("Package ", adapter$package, " is not installed.")))
  }
  if (!isTRUE(adapter$supports_precomputed_knn) && !isTRUE(include_raw_x)) {
    return(failure_row(ctx, adapter, "not_supported", "This adapter does not accept the shared precomputed KNN matrix. Enable --include-raw-x to run it as a separate raw-X baseline."))
  }
  if (!is.null(adapter$available) && !isTRUE(adapter$available())) {
    return(failure_row(ctx, adapter, "not_supported", adapter$unavailable_message))
  }
  tryCatch({
    set.seed(ctx$seed)
    measured <- measure_layout(adapter$run(ctx), nrow(ctx$dataset$x))
    success_row(ctx, adapter, measured, layout_dir)
  }, error = function(e) {
    failure_row(ctx, adapter, "failed", conditionMessage(e))
  })
}

uwot_umap <- function(ctx) {
  uwot::umap(
    X = ctx$dataset$x,
    n_neighbors = ctx$k_without_self,
    n_components = 2L,
    metric = "euclidean",
    nn_method = list(idx = ctx$index, dist = ctx$distance),
    n_epochs = ctx$n_epochs,
    min_dist = 0.1,
    init = "spectral",
    n_threads = 1L,
    n_sgd_threads = 0L,
    fast_sgd = FALSE,
    verbose = FALSE,
    ret_model = FALSE,
    seed = ctx$seed
  )
}

umap_package_umap <- function(ctx) {
  uknn <- umap::umap.knn(ctx$index, ctx$distance)
  config <- umap::umap.defaults
  config$knn <- uknn
  config$n_neighbors <- ctx$k_without_self
  config$n_epochs <- ctx$n_epochs
  config$min_dist <- 0.1
  config$random_state <- ctx$seed
  set.seed(ctx$seed)
  umap::umap(ctx$dataset$x, knn = uknn, config = config)$layout
}

rtsne_neighbors <- function(ctx) {
  Rtsne::Rtsne_neighbors(
    ctx$index,
    ctx$distance,
    dims = 2L,
    perplexity = ctx$perplexity,
    max_iter = ctx$max_iter,
    theta = 0.5,
    eta = 200,
    stop_lying_iter = 250L,
    mom_switch_iter = 250L,
    momentum = 0.5,
    final_momentum = 0.8,
    exaggeration_factor = 12,
    num_threads = 1L,
    verbose = FALSE
  )$Y
}

raw_tsne <- function(ctx) {
  tsne::tsne(ctx$dataset$x, k = 2L, perplexity = ctx$perplexity, max_iter = ctx$max_iter)
}

raw_fft_rtsne <- function(ctx) {
  info <- fft_rtsne_info()
  if (is.null(info)) stop("No usable fftRtsne() wrapper and fast_tsne executable found.", call. = FALSE)
  if (identical(info$package, "Spectre") && requireNamespace("rsvd", quietly = TRUE)) {
    suppressPackageStartupMessages(library(rsvd))
  }
  fun <- info$fun
  args <- list(
    X = ctx$dataset$x,
    dims = 2L,
    perplexity = ctx$perplexity,
    max_iter = ctx$max_iter,
    rand_seed = ctx$seed,
    theta = 0.5,
    nthreads = 1L,
    fast_tsne_path = info$fast_tsne_path,
    verbose = FALSE
  )
  args <- args[names(args) %in% names(formals(fun))]
  do.call(fun, args)
}

raw_mmtsne <- function(ctx) {
  fun_name <- c("mmtsne", "MMtsne", "mmt_sne")
  fun_name <- fun_name[vapply(fun_name, function(name) has_export("mmtsne", name), logical(1))]
  if (length(fun_name) == 0L) stop("No exported mmtsne function found.", call. = FALSE)
  fun <- get(fun_name[[1L]], envir = asNamespace("mmtsne"), inherits = FALSE)
  args <- list(X = ctx$dataset$x, x = ctx$dataset$x, dims = 2L, perplexity = ctx$perplexity,
               max_iter = ctx$max_iter, n_iter = ctx$max_iter, verbose = FALSE)
  args <- args[names(args) %in% names(formals(fun))]
  do.call(fun, args)
}

rdimtools_call <- function(ctx, function_name) {
  fun <- get(function_name, envir = asNamespace("Rdimtools"), inherits = FALSE)
  args <- list(
    X = ctx$dataset$x,
    ndim = 2L,
    n_neighbors = ctx$k_without_self,
    perplexity = ctx$perplexity,
    maxiter = ctx$max_iter,
    max_iter = ctx$max_iter,
    niter = ctx$max_iter,
    verbose = FALSE
  )
  args <- args[names(args) %in% names(formals(fun))]
  if (!"X" %in% names(args)) args <- c(list(ctx$dataset$x), args)
  coerce_layout(do.call(fun, args), nrow(ctx$dataset$x))
}

make_adapters <- function() {
  fft_info <- fft_rtsne_info()
  list(
    list(
      id = "fastEmbedR::umap_knn",
      package = "fastEmbedR",
      function_name = "embed_knn(method = 'umap')",
      family = "umap",
      input_type = "precomputed_knn",
      supports_precomputed_knn = TRUE,
      params = function(ctx) list(k = ctx$k_without_self, n_epochs = ctx$n_epochs, min_dist = 0.1),
      run = function(ctx) fastEmbedR:::fast_knn_umap_core(ctx$knn, seed = ctx$seed, backend = "cpu", n_epochs = ctx$n_epochs)
    ),
    list(
      id = "fastEmbedR::tsne_knn",
      package = "fastEmbedR",
      function_name = "embed_knn(method = 'tsne')",
      family = "tsne",
      input_type = "precomputed_knn",
      supports_precomputed_knn = TRUE,
      params = function(ctx) list(k = ctx$k_without_self, perplexity = ctx$perplexity, quality = "auto"),
      run = function(ctx) fastEmbedR::embed_knn(ctx$knn, method = "tsne", seed = ctx$seed, backend = "cpu")
    ),
    list(
      id = "fastEmbedR::pacmap_knn",
      package = "fastEmbedR",
      function_name = "embed_knn(method = 'pacmap')",
      family = "pacmap",
      input_type = "precomputed_knn",
      supports_precomputed_knn = TRUE,
      params = function(ctx) list(k = ctx$k_without_self, n_epochs = ctx$n_epochs),
      run = function(ctx) fastEmbedR:::knn_embed_core(ctx$knn, objective = "pacmap", seed = ctx$seed, backend = "cpu", n_epochs = ctx$n_epochs)
    ),
    list(
      id = "fastEmbedR::trimap_knn",
      package = "fastEmbedR",
      function_name = "embed_knn(method = 'trimap')",
      family = "trimap",
      input_type = "precomputed_knn",
      supports_precomputed_knn = TRUE,
      params = function(ctx) list(k = ctx$k_without_self, n_epochs = ctx$n_epochs),
      run = function(ctx) fastEmbedR:::knn_embed_core(ctx$knn, objective = "trimap", seed = ctx$seed, backend = "cpu", n_epochs = ctx$n_epochs)
    ),
    list(
      id = "fastEmbedR::localmap_knn",
      package = "fastEmbedR",
      function_name = "embed_knn(method = 'localmap')",
      family = "localmap",
      input_type = "precomputed_knn",
      supports_precomputed_knn = TRUE,
      params = function(ctx) list(k = ctx$k_without_self, n_epochs = ctx$n_epochs),
      run = function(ctx) fastEmbedR:::knn_embed_core(ctx$knn, objective = "localmap", seed = ctx$seed, backend = "cpu", n_epochs = ctx$n_epochs)
    ),
    list(
      id = "uwot::umap_knn",
      package = "uwot",
      function_name = "umap",
      family = "umap",
      input_type = "precomputed_knn",
      supports_precomputed_knn = TRUE,
      params = function(ctx) list(k = ctx$k_without_self, n_epochs = ctx$n_epochs, min_dist = 0.1, fast_sgd = FALSE),
      run = uwot_umap
    ),
    list(
      id = "umap::umap_knn",
      package = "umap",
      function_name = "umap + umap.knn",
      family = "umap",
      input_type = "precomputed_knn",
      supports_precomputed_knn = TRUE,
      params = function(ctx) list(k = ctx$k_without_self, n_epochs = ctx$n_epochs, min_dist = 0.1),
      run = umap_package_umap
    ),
    list(
      id = "Rtsne::Rtsne_neighbors",
      package = "Rtsne",
      function_name = "Rtsne_neighbors",
      family = "tsne",
      input_type = "precomputed_knn",
      supports_precomputed_knn = TRUE,
      params = function(ctx) list(k = ctx$k_without_self, perplexity = ctx$perplexity, max_iter = ctx$max_iter, theta = 0.5, eta = 200),
      run = rtsne_neighbors
    ),
    list(
      id = "tsne::tsne",
      package = "tsne",
      function_name = "tsne",
      family = "tsne",
      input_type = "raw_x",
      supports_precomputed_knn = FALSE,
      params = function(ctx) list(perplexity = ctx$perplexity, max_iter = ctx$max_iter),
      run = raw_tsne
    ),
    list(
      id = if (is.null(fft_info)) "fftRtsne::fftRtsne" else paste0(fft_info$package, "::fftRtsne"),
      package = if (is.null(fft_info)) "fftRtsne" else fft_info$package,
      function_name = "fftRtsne",
      family = "tsne",
      input_type = "raw_x",
      supports_precomputed_knn = FALSE,
      available = function() !is.null(fft_rtsne_info()),
      unavailable_message = "No usable fftRtsne() wrapper and fast_tsne executable found.",
      params = function(ctx) list(perplexity = ctx$perplexity, max_iter = ctx$max_iter),
      run = raw_fft_rtsne
    ),
    list(
      id = "mmtsne::mmtsne",
      package = "mmtsne",
      function_name = "mmtsne",
      family = "tsne",
      input_type = "raw_x",
      supports_precomputed_knn = FALSE,
      params = function(ctx) list(perplexity = ctx$perplexity, max_iter = ctx$max_iter),
      run = raw_mmtsne
    ),
    list(
      id = "Rdimtools::do.tsne",
      package = "Rdimtools",
      function_name = "do.tsne",
      family = "tsne",
      input_type = "raw_x",
      supports_precomputed_knn = FALSE,
      available = function() has_export("Rdimtools", "do.tsne"),
      unavailable_message = "Rdimtools::do.tsne is not exported.",
      params = function(ctx) list(perplexity = ctx$perplexity, max_iter = ctx$max_iter),
      run = function(ctx) rdimtools_call(ctx, "do.tsne")
    )
  )
}

adapter_subset <- function(adapters, methods, implementations = character()) {
  methods <- tolower(methods)
  if (!(length(methods) == 0L || any(methods %in% c("all", "all_strict")))) {
    keep <- vapply(adapters, function(a) {
      any(tolower(a$family) %in% methods) ||
        any(tolower(a$package) %in% methods) ||
        any(tolower(a$id) %in% methods) ||
        (any(methods == "fastembedr") && identical(a$package, "fastEmbedR")) ||
        (any(methods == "references") && !identical(a$package, "fastEmbedR"))
    }, logical(1))
    adapters <- adapters[keep]
  }
  implementations <- tolower(implementations)
  implementations <- implementations[nzchar(implementations) & !implementations %in% c("all", "all_strict")]
  if (length(implementations) > 0L) {
    keep <- vapply(adapters, function(a) {
      id <- tolower(a$id)
      any(id %in% implementations) || any(vapply(implementations, grepl, logical(1L), x = id, fixed = TRUE))
    }, logical(1L))
    adapters <- adapters[keep]
  }
  adapters
}

run_one_context <- function(dataset, k_without_self, seed, n_epochs, max_iter,
                            include_raw_x, adapters, layout_dir,
                            local_sample_size, global_sample_size) {
  k_without_self <- min(as.integer(k_without_self), nrow(dataset$x) - 1L)
  if (k_without_self < 2L) stop("KNN k must be at least 2 after dataset-size clipping.", call. = FALSE)
  message("Dataset ", dataset$name, ", seed ", seed, ": building shared KNN k = ", k_without_self)
  knn_time <- system.time({
    raw_knn <- fastEmbedR::nn(dataset$x, dataset$x, k_without_self + 1L, backend = "cpu")
  })[["elapsed"]]
  knn_no_self <- fastEmbedR:::coerce_knn_input(raw_knn)
  eval_k <- max(metric_k(nrow(dataset$x)))
  eval_reference <- fastEmbedR::nn(dataset$x, dataset$x, eval_k + 1L, backend = "cpu")
  ctx <- list(
    dataset = dataset,
    seed = as.integer(seed),
    k_without_self = as.integer(k_without_self),
    perplexity = perplexity_from_k(k_without_self, nrow(dataset$x)),
    n_epochs = as.integer(n_epochs),
    max_iter = as.integer(max_iter),
    knn = knn_no_self,
    index = knn_no_self$indices,
    distance = knn_no_self$distances,
    knn_time = as.numeric(knn_time),
    eval_reference = eval_reference,
    local_sample_size = as.integer(local_sample_size),
    global_sample_size = as.integer(global_sample_size)
  )
  rows <- lapply(adapters, function(adapter) {
    message("  ", adapter$id)
    run_adapter(ctx, adapter, include_raw_x = include_raw_x, layout_dir = layout_dir)
  })
  do.call(rbind, rows)
}

scale01 <- function(x, inverse = FALSE) {
  if (inverse) x <- -x
  ok <- is.finite(x)
  out <- rep(NA_real_, length(x))
  if (sum(ok) == 0L) return(out)
  rng <- range(x[ok], na.rm = TRUE)
  if (!is.finite(diff(rng)) || diff(rng) == 0) {
    out[ok] <- 1
  } else {
    out[ok] <- (x[ok] - rng[1L]) / diff(rng)
  }
  out
}

procrustes_rmsd <- function(a, b) {
  a <- as.matrix(a)
  b <- as.matrix(b)
  a <- sweep(a, 2L, colMeans(a), "-")
  b <- sweep(b, 2L, colMeans(b), "-")
  a_scale <- sqrt(sum(a * a))
  b_scale <- sqrt(sum(b * b))
  if (a_scale == 0 || b_scale == 0) return(NA_real_)
  a <- a / a_scale
  b <- b / b_scale
  sv <- svd(t(b) %*% a)
  rot <- sv$u %*% t(sv$v)
  sqrt(mean((a - b %*% rot)^2))
}

compute_stability <- function(results) {
  results$procrustes_rmsd <- NA_real_
  ok <- results$status == "success" & file.exists(results$layout_path)
  if (sum(ok) == 0L) return(results)
  groups <- split(which(ok), paste(results$dataset[ok], results$implementation[ok], results$knn_k_without_self[ok], sep = "\r"))
  for (idx in groups) {
    seeds <- unique(results$seed[idx])
    if (length(seeds) < 2L) next
    layouts <- lapply(results$layout_path[idx], readRDS)
    rmsd <- numeric(0)
    for (i in seq_len(length(layouts) - 1L)) {
      for (j in (i + 1L):length(layouts)) {
        rmsd <- c(rmsd, procrustes_rmsd(layouts[[i]], layouts[[j]]))
      }
    }
    results$procrustes_rmsd[idx] <- mean(rmsd, na.rm = TRUE)
  }
  results
}

add_combined_score <- function(results) {
  results$combined_score <- NA_real_
  ok <- results$status == "success"
  groups <- split(which(ok), paste(results$dataset[ok], results$knn_k_without_self[ok], sep = "\r"))
  for (idx in groups) {
    results$combined_score[idx] <-
      0.30 * scale01(results$trustworthiness[idx]) +
      0.25 * scale01(results$knn_preservation_15[idx]) +
      0.20 * scale01(results$label_knn_accuracy[idx]) +
      0.10 * scale01(results$silhouette[idx]) +
      0.10 * scale01(results$total_time_sec[idx], inverse = TRUE) +
      0.05 * scale01(results$peak_ram_mb[idx], inverse = TRUE)
  }
  results
}

best_rows <- function(results) {
  ok <- results[results$status == "success" & results$benchmark_scope == "strict_knn", , drop = FALSE]
  if (nrow(ok) == 0L) return(ok)
  split_key <- paste(ok$dataset, ok$method_family, ok$implementation, sep = "\r")
  out <- do.call(rbind, lapply(split(ok, split_key), function(x) {
    x <- x[order(-x$combined_score, -x$trustworthiness, x$total_time_sec), , drop = FALSE]
    x[1L, , drop = FALSE]
  }))
  rownames(out) <- NULL
  out
}

short_impl <- function(x) {
  out <- x
  out <- sub("^fastEmbedR::", "fastEmbedR ", out)
  out <- sub("^uwot::", "uwot ", out)
  out <- sub("^umap::", "umap ", out)
  out <- sub("^Rtsne::", "Rtsne ", out)
  out <- sub("_knn_exact$", " exact", out)
  out <- sub("_knn$", "", out)
  out
}

plot_outputs <- function(results, best, datasets, out_dir, max_embedding_plots = 12L) {
  fig_dir <- file.path(out_dir, "figures")
  dir.create(fig_dir, recursive = TRUE, showWarnings = FALSE)
  paths <- character(0)
  if (!requireNamespace("ggplot2", quietly = TRUE)) return(paths)
  gg <- asNamespace("ggplot2")
  ok <- results[results$status == "success" & results$benchmark_scope == "strict_knn", , drop = FALSE]
  if (nrow(ok) == 0L) return(paths)
  ok$short_implementation <- short_impl(ok$implementation)
  best$short_implementation <- short_impl(best$implementation)

  p <- gg$ggplot(ok, gg$aes(x = total_time_sec, y = trustworthiness, color = short_implementation, shape = method_family)) +
    gg$geom_point(alpha = 0.78, size = 2.2) +
    gg$scale_x_log10() +
    gg$labs(x = "Total runtime including shared KNN (seconds, log scale)", y = "Trustworthiness", color = "Implementation", shape = "Method", title = "Speed-quality trade-off") +
    gg$theme_bw(base_size = 11) +
    gg$theme(legend.position = "right")
  path <- file.path(fig_dir, "speed_quality_scatter.png")
  ggplot2::ggsave(path, p, width = 10, height = 6, dpi = 180)
  paths <- c(paths, path)

  heat <- best
  heat$dataset <- factor(heat$dataset, levels = unique(heat$dataset))
  heat$short_implementation <- factor(heat$short_implementation, levels = unique(heat$short_implementation))
  p <- gg$ggplot(heat, gg$aes(x = short_implementation, y = dataset, fill = trustworthiness)) +
    gg$geom_tile(color = "white", linewidth = 0.25) +
    gg$scale_fill_viridis_c(option = "C", na.value = "grey90") +
    gg$labs(x = "Implementation", y = "Dataset", fill = "Trust", title = "Best trustworthiness by dataset") +
    gg$theme_bw(base_size = 10) +
    gg$theme(axis.text.x = gg$element_text(angle = 45, hjust = 1))
  path <- file.path(fig_dir, "trustworthiness_heatmap.png")
  ggplot2::ggsave(path, p, width = 11, height = 7, dpi = 180)
  paths <- c(paths, path)

  p <- gg$ggplot(heat, gg$aes(x = short_implementation, y = dataset, fill = log10(total_time_sec))) +
    gg$geom_tile(color = "white", linewidth = 0.25) +
    gg$scale_fill_viridis_c(option = "B", na.value = "grey90") +
    gg$labs(x = "Implementation", y = "Dataset", fill = "log10 sec", title = "Runtime by dataset") +
    gg$theme_bw(base_size = 10) +
    gg$theme(axis.text.x = gg$element_text(angle = 45, hjust = 1))
  path <- file.path(fig_dir, "runtime_heatmap.png")
  ggplot2::ggsave(path, p, width = 11, height = 7, dpi = 180)
  paths <- c(paths, path)

  p <- gg$ggplot(heat, gg$aes(x = short_implementation, y = dataset, fill = label_knn_accuracy)) +
    gg$geom_tile(color = "white", linewidth = 0.25) +
    gg$scale_fill_viridis_c(option = "D", na.value = "grey90") +
    gg$labs(x = "Implementation", y = "Dataset", fill = "Label kNN", title = "Label accuracy by dataset") +
    gg$theme_bw(base_size = 10) +
    gg$theme(axis.text.x = gg$element_text(angle = 45, hjust = 1))
  path <- file.path(fig_dir, "label_accuracy_heatmap.png")
  ggplot2::ggsave(path, p, width = 11, height = 7, dpi = 180)
  paths <- c(paths, path)

  label_lookup <- setNames(lapply(datasets, function(d) d$labels), vapply(datasets, function(d) d$name, character(1)))
  emb <- best[best$package == "fastEmbedR" & best$status == "success" & file.exists(best$layout_path), , drop = FALSE]
  emb <- emb[order(emb$dataset, emb$method_family), , drop = FALSE]
  if (nrow(emb) > 0L) {
    emb <- emb[seq_len(min(max_embedding_plots, nrow(emb))), , drop = FALSE]
    for (i in seq_len(nrow(emb))) {
      layout <- readRDS(emb$layout_path[[i]])
      labels <- label_lookup[[emb$dataset[[i]]]]
      if (is.null(labels)) labels <- seq_len(nrow(layout))
      dat <- data.frame(x = layout[, 1L], y = layout[, 2L], label = factor(labels))
      p <- gg$ggplot(dat, gg$aes(x, y, color = label)) +
        gg$geom_point(size = 0.8, alpha = 0.75, show.legend = FALSE) +
        gg$labs(x = NULL, y = NULL, title = paste(emb$dataset[[i]], short_impl(emb$implementation[[i]]))) +
        gg$theme_void(base_size = 11) +
        gg$theme(plot.title = gg$element_text(hjust = 0.5))
      path <- file.path(fig_dir, paste0("embedding_", gsub("[^A-Za-z0-9_.-]+", "_", paste(emb$dataset[[i]], emb$implementation[[i]], sep = "_")), ".png"))
      ggplot2::ggsave(path, p, width = 5, height = 4, dpi = 180)
      paths <- c(paths, path)
    }
  }
  paths
}

write_md_table <- function(x, con) {
  rownames(x) <- NULL
  if (requireNamespace("knitr", quietly = TRUE)) {
    cat(paste(knitr::kable(x, format = "markdown", digits = 4, row.names = FALSE), collapse = "\n"), "\n", file = con)
  } else {
    utils::write.table(x, con, sep = "|", row.names = FALSE, quote = FALSE)
  }
}

write_report <- function(results, best, datasets, figures, out_dir, command_line) {
  path <- file.path(out_dir, "rjournal_benchmark_report.md")
  strict <- results[results$benchmark_scope == "strict_knn", , drop = FALSE]
  success <- strict[strict$status == "success", , drop = FALSE]
  dataset_table <- do.call(rbind, lapply(datasets, function(d) {
    data.frame(dataset = d$name, source = d$source, n = nrow(d$x), p = ncol(d$x),
               n_labels = if (is.null(d$labels)) NA_integer_ else length(levels(factor(d$labels))),
               stringsAsFactors = FALSE)
  }))
  package_table <- unique(results[, c("package", "package_version")])
  package_table <- package_table[order(package_table$package), , drop = FALSE]

  top <- best[, c("dataset", "method_family", "implementation", "knn_k_without_self",
                  "total_time_sec", "peak_ram_mb", "trustworthiness",
                  "knn_preservation_15", "label_knn_accuracy", "combined_score")]
  top <- top[order(top$dataset, top$method_family, -top$combined_score), , drop = FALSE]

  con <- file(path, open = "w")
  on.exit(close(con), add = TRUE)
  cat("# fastEmbedR R Journal benchmark draft\n\n", file = con)
  cat("Generated: ", format(Sys.time(), "%Y-%m-%d %H:%M:%S %Z"), "\n\n", sep = "", file = con)
  cat("Command:\n\n```sh\n", paste(command_line, collapse = " "), "\n```\n\n", sep = "", file = con)
  cat("## Design\n\n", file = con)
  cat("The primary comparison uses a shared precomputed Euclidean KNN matrix for each dataset, k, and seed. Methods that cannot consume precomputed KNN are either skipped or run only when `--include-raw-x` is supplied, and are labelled as `raw_x_separate` rather than mixed into the strict KNN ranking.\n\n", file = con)
  cat("Quality metrics include trustworthiness, continuity, KNN preservation at k = 15/30/50, global distance correlation, stress, silhouette, label-neighbour accuracy, ARI, NMI, and rare-class recall where labels are available. RSS memory is measured before and after each embedding call; this is useful for relative local runs but is not a true sampled peak profiler.\n\n", file = con)
  cat("## Dataset panel\n\n", file = con)
  write_md_table(dataset_table, con)
  cat("\n\n## Package versions\n\n", file = con)
  write_md_table(package_table, con)
  cat("\n\n## Strict-KNN run status\n\n", file = con)
  status_tab <- as.data.frame.matrix(table(strict$status, strict$implementation))
  status_tab$status <- rownames(status_tab)
  status_tab <- status_tab[, c("status", setdiff(names(status_tab), "status")), drop = FALSE]
  write_md_table(status_tab, con)
  cat("\n\n## Best strict-KNN rows\n\n", file = con)
  write_md_table(top, con)
  cat("\n\n## Figures\n\n", file = con)
  rel_figures <- file.path("figures", basename(figures))
  for (fig in rel_figures) {
    cat("![", basename(fig), "](", fig, ")\n\n", sep = "", file = con)
  }
  if (nrow(success) > 0L) {
    cat("## Main takeaways for this run\n\n", file = con)
    by_impl <- aggregate(
      cbind(total_time_sec, trustworthiness, knn_preservation_15, label_knn_accuracy, combined_score) ~ implementation,
      data = success,
      FUN = mean,
      na.rm = TRUE
    )
    by_impl <- by_impl[order(-by_impl$combined_score), , drop = FALSE]
    write_md_table(by_impl, con)
    cat("\n", file = con)
  }
  path
}

write_session_info <- function(out_dir) {
  utils::capture.output(sessionInfo(), file = file.path(out_dir, "session_info.txt"))
  info <- list(
    date = as.character(Sys.time()),
    platform = R.version$platform,
    r_version = as.character(getRversion()),
    fastEmbedR_backend = tryCatch(fastEmbedR::backend_info(), error = function(e) conditionMessage(e))
  )
  if (requireNamespace("jsonlite", quietly = TRUE)) {
    jsonlite::write_json(info, file.path(out_dir, "system_info.json"), pretty = TRUE, auto_unbox = TRUE)
  }
}

default_datasets <- paste(
  c("iris", "digits", "gaussian", "imbalanced", "many_clusters", "anisotropic",
    "sparse_signal", "moons", "circles", "blobs", "swiss_roll", "s_curve",
    "pendigits", "letter"),
  collapse = ","
)

out_dir <- parse_scalar("out-dir", file.path("results", "rjournal_benchmark"))
cache_dir <- parse_scalar("cache-dir", file.path(out_dir, "cache"))
layout_dir <- file.path(out_dir, "layouts")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(cache_dir, recursive = TRUE, showWarnings = FALSE)

dataset_names <- parse_csv_arg("datasets", default_datasets)
k_values <- as.integer(parse_csv_arg("k", "15,30"))
k_values <- k_values[is.finite(k_values) & k_values > 0L]
if (length(k_values) == 0L) k_values <- c(15L, 30L)
seeds <- as.integer(parse_csv_arg("seeds", "4"))
seeds <- seeds[is.finite(seeds)]
if (length(seeds) == 0L) seeds <- 4L
n_epochs <- as.integer(parse_scalar("n-epochs", "500"))
max_iter <- as.integer(parse_scalar("max-iter", "500"))
max_n <- as.integer(parse_scalar("max-n", "2500"))
local_sample_size <- as.integer(parse_scalar("local-sample-size", "1000"))
global_sample_size <- as.integer(parse_scalar("global-sample-size", "1000"))
include_raw_x <- parse_flag("include-raw-x", default = FALSE)
options(fastembedr.download_kaggle = parse_flag("download-kaggle", default = FALSE))
methods <- parse_csv_arg("methods", "all_strict")
implementations <- parse_csv_arg("implementations", "")
max_embedding_plots <- as.integer(parse_scalar("max-embedding-plots", "12"))

datasets <- Filter(Negate(is.null), lapply(dataset_names, load_named_dataset, cache_dir = cache_dir, max_n = max_n, seed = min(seeds)))
if (length(datasets) == 0L) stop("No datasets could be loaded.", call. = FALSE)

adapters <- adapter_subset(make_adapters(), methods, implementations)
if (!include_raw_x) {
  # Keep rows for unsupported raw-X adapters in the output, but they will be labelled not_supported.
  adapters <- adapters
}

result_list <- list()
for (dataset in datasets) {
  for (k in k_values) {
    for (seed in seeds) {
      result_list[[length(result_list) + 1L]] <- run_one_context(
        dataset = dataset,
        k_without_self = k,
        seed = seed,
        n_epochs = n_epochs,
        max_iter = max_iter,
        include_raw_x = include_raw_x,
        adapters = adapters,
        layout_dir = layout_dir,
        local_sample_size = local_sample_size,
        global_sample_size = global_sample_size
      )
    }
  }
}

results <- do.call(rbind, result_list)
results <- compute_stability(results)
results <- add_combined_score(results)
best <- best_rows(results)

stamp <- format(Sys.time(), "%Y%m%d_%H%M%S")
results_file <- file.path(out_dir, paste0("rjournal_benchmark_results_", stamp, ".csv"))
best_file <- file.path(out_dir, paste0("rjournal_benchmark_best_", stamp, ".csv"))
latest_results <- file.path(out_dir, "latest_rjournal_benchmark_results.csv")
latest_best <- file.path(out_dir, "latest_rjournal_benchmark_best.csv")
utils::write.csv(results, results_file, row.names = FALSE)
utils::write.csv(best, best_file, row.names = FALSE)
utils::write.csv(results, latest_results, row.names = FALSE)
utils::write.csv(best, latest_best, row.names = FALSE)

dataset_meta <- do.call(rbind, lapply(datasets, function(d) {
  data.frame(dataset = d$name, source = d$source, n = nrow(d$x), p = ncol(d$x),
             n_labels = if (is.null(d$labels)) NA_integer_ else length(levels(factor(d$labels))),
             stringsAsFactors = FALSE)
}))
utils::write.csv(dataset_meta, file.path(out_dir, "dataset_metadata.csv"), row.names = FALSE)

figures <- plot_outputs(results, best, datasets, out_dir, max_embedding_plots = max_embedding_plots)
report <- write_report(results, best, datasets, figures, out_dir, commandArgs())
write_session_info(out_dir)

print(results[, c(
  "dataset",
  "method_family",
  "implementation",
  "input_type",
  "knn_k_without_self",
  "seed",
  "total_time_sec",
  "peak_ram_mb",
  "trustworthiness",
  "knn_preservation_15",
  "silhouette",
  "label_knn_accuracy",
  "combined_score",
  "status",
  "error_message"
)], row.names = FALSE)

cat("\nSaved benchmark artifacts:\n")
cat("  ", normalizePath(results_file), "\n", sep = "")
cat("  ", normalizePath(best_file), "\n", sep = "")
cat("  ", normalizePath(latest_results), "\n", sep = "")
cat("  ", normalizePath(latest_best), "\n", sep = "")
cat("  ", normalizePath(report), "\n", sep = "")
for (fig in figures) cat("  ", normalizePath(fig), "\n", sep = "")
