#!/usr/bin/env Rscript

suppressPackageStartupMessages(library(fastEmbedR))

parse_csv_arg <- function(name, default) {
  args <- commandArgs(trailingOnly = TRUE)
  prefix <- paste0("--", name, "=")
  value <- args[startsWith(args, prefix)]
  if (length(value) == 0L) {
    value <- Sys.getenv(paste0("FASTEMBEDR_BENCH_", toupper(gsub("-", "_", name))), default)
  } else {
    value <- sub(prefix, "", value[[1L]], fixed = TRUE)
  }
  trimws(strsplit(value, ",", fixed = TRUE)[[1L]])
}

parse_flag <- function(name, default = FALSE) {
  args <- commandArgs(trailingOnly = TRUE)
  env <- Sys.getenv(paste0("FASTEMBEDR_BENCH_", toupper(gsub("-", "_", name))), "")
  any(args == paste0("--", name)) || identical(tolower(env), "1") ||
    identical(tolower(env), "true") || default
}

parse_scalar <- function(name, default) {
  args <- commandArgs(trailingOnly = TRUE)
  prefix <- paste0("--", name, "=")
  value <- args[startsWith(args, prefix)]
  if (length(value) == 0L) {
    Sys.getenv(paste0("FASTEMBEDR_BENCH_", toupper(gsub("-", "_", name))), default)
  } else {
    sub(prefix, "", value[[1L]], fixed = TRUE)
  }
}

standardize_matrix <- function(x) {
  x <- as.matrix(x)
  storage.mode(x) <- "double"
  keep <- apply(x, 2L, function(col) {
    all(is.finite(col)) && stats::sd(col) > 0
  })
  x <- x[, keep, drop = FALSE]
  x <- scale(x)
  storage.mode(x) <- "double"
  x
}

make_synthetic_dataset <- function(seed = 11L) {
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
  list(name = "synthetic_1000", x = standardize_matrix(x), labels = labels)
}

load_digits_dataset <- function() {
  if (!requireNamespace("reticulate", quietly = TRUE)) return(NULL)
  tryCatch({
    sklearn <- reticulate::import("sklearn.datasets")
    digits <- sklearn$load_digits()
    list(
      name = "sklearn_digits",
      x = standardize_matrix(digits$data),
      labels = factor(as.integer(digits$target))
    )
  }, error = function(e) NULL)
}

load_named_dataset <- function(name) {
  name <- tolower(name)
  if (identical(name, "iris")) {
    return(list(
      name = "iris",
      x = standardize_matrix(iris[, 1L:4L]),
      labels = iris$Species
    ))
  }
  if (identical(name, "synthetic") || identical(name, "synthetic_1000")) {
    return(make_synthetic_dataset())
  }
  if (identical(name, "digits") || identical(name, "sklearn_digits")) {
    return(load_digits_dataset())
  }
  stop("Unknown dataset: ", name, call. = FALSE)
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
  requireNamespace(package, quietly = TRUE) && exists(name, envir = asNamespace(package), inherits = FALSE)
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
  gc()
  peak_ram_mb <- NA_real_
  elapsed <- system.time({
    layout <- force(expr)
  })[["elapsed"]]
  list(layout = coerce_layout(layout, n), elapsed = as.numeric(elapsed), peak_ram_mb = peak_ram_mb)
}

failure_row <- function(ctx, adapter, status, message) {
  data.frame(
    dataset = ctx$dataset$name,
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
    stringsAsFactors = FALSE
  )
}

success_row <- function(ctx, adapter, measured) {
  k_eval <- metric_k(nrow(ctx$dataset$x))
  metrics <- fastEmbedR::evaluate_embedding(
    ctx$dataset$x,
    measured$layout,
    labels = ctx$dataset$labels,
    k = k_eval,
    reference_nn = ctx$eval_reference,
    sample_size_for_global_metrics = min(500L, nrow(ctx$dataset$x)),
    sample_size_for_local_metrics = min(1000L, nrow(ctx$dataset$x)),
    use_cache = FALSE,
    seed = ctx$seed,
    method = adapter$id,
    backend = "cpu",
    dataset = ctx$dataset$name
  )
  data.frame(
    dataset = ctx$dataset$name,
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
    stringsAsFactors = FALSE
  )
}

run_adapter <- function(ctx, adapter, include_raw_x = FALSE) {
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
    success_row(ctx, adapter, measured)
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
  tsne::tsne(
    ctx$dataset$x,
    k = 2L,
    perplexity = ctx$perplexity,
    max_iter = ctx$max_iter
  )
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
  args <- list(
    X = ctx$dataset$x,
    x = ctx$dataset$x,
    Y = ctx$dataset$x,
    dims = 2L,
    no_dims = 2L,
    ndim = 2L,
    perplexity = ctx$perplexity,
    max_iter = ctx$max_iter,
    n_iter = ctx$max_iter,
    verbose = FALSE
  )
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

m3addon_trimap <- function(ctx) {
  if (!requireNamespace("SingleCellExperiment", quietly = TRUE)) {
    stop("SingleCellExperiment is required to build the cell_data_set-like input expected by m3addon::trimap.", call. = FALSE)
  }
  if (!requireNamespace("reticulate", quietly = TRUE)) {
    stop("reticulate is required by m3addon::trimap.", call. = FALSE)
  }
  suppressPackageStartupMessages(library(SingleCellExperiment))
  trimap_python <- file.path(Sys.getenv("HOME"), ".virtualenvs", "r-reticulate-py310-trimap", "bin", "python")
  if (!reticulate::py_available(initialize = FALSE) && file.exists(trimap_python)) {
    reticulate::use_python(trimap_python, required = FALSE)
  }
  sce <- SingleCellExperiment::SingleCellExperiment(
    assays = list(counts = t(ctx$dataset$x))
  )
  SingleCellExperiment::reducedDim(sce, "PCA") <- ctx$dataset$x
  knn_tuple <- reticulate::tuple(
    ctx$index - 1L,
    ctx$distance
  )
  out <- m3addon::trimap(
    sce,
    preprocess_method = "PCA",
    num_dims = ncol(ctx$dataset$x),
    n_dims = 2L,
    n_inliers = ctx$k_without_self,
    n_outliers = min(5L, ctx$k_without_self),
    n_random = min(5L, ctx$k_without_self),
    n_iters = ctx$max_iter,
    knn_tuple = knn_tuple,
    python_home = trimap_python,
    apply_pca_trimap = FALSE,
    verbose = FALSE
  )
  SingleCellExperiment::reducedDim(out, "trimap")
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
    ),
    list(
      id = "m3addon::trimap_knn_tuple",
      package = "m3addon",
      function_name = "trimap",
      family = "trimap",
      input_type = "precomputed_knn",
      supports_precomputed_knn = TRUE,
      params = function(ctx) list(k = ctx$k_without_self, n_iters = ctx$max_iter, knn_tuple = TRUE),
      run = m3addon_trimap
    )
  )
}

rank_strict_knn <- function(results) {
  ok <- results[
    results$status == "success" &
      results$benchmark_scope == "strict_knn",
    ,
    drop = FALSE
  ]
  if (nrow(ok) == 0L) return(ok)
  split_key <- paste(ok$dataset, ok$method_family, ok$implementation, sep = "\r")
  do.call(rbind, lapply(split(ok, split_key), function(x) {
    x <- x[order(
      -x$trustworthiness,
      -x$label_knn_accuracy,
      -x$knn_preservation_15,
      x$total_time_sec
    ), , drop = FALSE]
    x[1L, , drop = FALSE]
  }))
}

run_one_dataset_k <- function(dataset, k_without_self, seed, n_epochs, max_iter, include_raw_x) {
  k_without_self <- min(as.integer(k_without_self), nrow(dataset$x) - 1L)
  if (k_without_self < 2L) stop("KNN k must be at least 2 after dataset-size clipping.", call. = FALSE)
  message("Dataset ", dataset$name, ": building shared KNN k = ", k_without_self)
  knn_time <- system.time({
    knn <- fastEmbedR::nn(dataset$x, dataset$x, k_without_self + 1L, backend = "cpu")
  })[["elapsed"]]
  knn_no_self <- fastEmbedR:::coerce_knn_input(knn)
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
    eval_reference = eval_reference
  )
  rows <- lapply(make_adapters(), function(adapter) {
    message("  ", adapter$id)
    run_adapter(ctx, adapter, include_raw_x = include_raw_x)
  })
  do.call(rbind, rows)
}

datasets_requested <- parse_csv_arg("datasets", "iris,synthetic")
datasets <- Filter(Negate(is.null), lapply(datasets_requested, load_named_dataset))
if (length(datasets) == 0L) stop("No datasets could be loaded.", call. = FALSE)

k_values <- as.integer(parse_csv_arg("k", "15,30"))
k_values <- k_values[is.finite(k_values) & k_values > 0L]
if (length(k_values) == 0L) k_values <- c(15L, 30L)

seed <- as.integer(parse_scalar("seed", "4"))
n_epochs <- as.integer(parse_scalar("n-epochs", "500"))
max_iter <- as.integer(parse_scalar("max-iter", "500"))
include_raw_x <- parse_flag("include-raw-x", default = FALSE)
out_dir <- parse_scalar("out-dir", file.path("results", "performance"))
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

result_list <- list()
for (dataset in datasets) {
  for (k in k_values) {
    result_list[[length(result_list) + 1L]] <- run_one_dataset_k(
      dataset,
      k,
      seed,
      n_epochs,
      max_iter,
      include_raw_x
    )
  }
}
results <- do.call(rbind, result_list)

best <- rank_strict_knn(results)
stamp <- format(Sys.time(), "%Y%m%d_%H%M%S")
scope_suffix <- if (isTRUE(include_raw_x)) "raw_x" else "strict_knn"
results_file <- file.path(out_dir, paste0("r_package_embedding_comparison_", stamp, ".csv"))
best_file <- file.path(out_dir, paste0("r_package_embedding_comparison_best_", stamp, ".csv"))
latest_file <- file.path(out_dir, "latest_r_package_embedding_comparison.csv")
latest_best_file <- file.path(out_dir, "latest_r_package_embedding_comparison_best.csv")
latest_scope_file <- file.path(out_dir, paste0("latest_r_package_embedding_comparison_", scope_suffix, ".csv"))
latest_scope_best_file <- file.path(out_dir, paste0("latest_r_package_embedding_comparison_best_", scope_suffix, ".csv"))
utils::write.csv(results, results_file, row.names = FALSE)
utils::write.csv(best, best_file, row.names = FALSE)
utils::write.csv(results, latest_file, row.names = FALSE)
utils::write.csv(best, latest_best_file, row.names = FALSE)
utils::write.csv(results, latest_scope_file, row.names = FALSE)
utils::write.csv(best, latest_scope_best_file, row.names = FALSE)

print(results[, c(
  "dataset",
  "method_family",
  "implementation",
  "input_type",
  "knn_k_without_self",
  "embedding_time_sec",
  "total_time_sec",
  "trustworthiness",
  "knn_preservation_15",
  "silhouette",
  "label_knn_accuracy",
  "status",
  "error_message"
)], row.names = FALSE)

cat("\nStrict KNN best rows:\n")
if (nrow(best) > 0L) {
  print(best[, c(
    "dataset",
    "method_family",
    "implementation",
    "knn_k_without_self",
    "total_time_sec",
    "trustworthiness",
    "knn_preservation_15",
    "silhouette",
    "label_knn_accuracy"
  )], row.names = FALSE)
} else {
  cat("No successful strict-KNN rows.\n")
}

cat("\nSaved:\n")
cat("  ", normalizePath(results_file), "\n", sep = "")
cat("  ", normalizePath(best_file), "\n", sep = "")
cat("  ", normalizePath(latest_file), "\n", sep = "")
cat("  ", normalizePath(latest_best_file), "\n", sep = "")
cat("  ", normalizePath(latest_scope_file), "\n", sep = "")
cat("  ", normalizePath(latest_scope_best_file), "\n", sep = "")
