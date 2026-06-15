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
out_dir <- args$out_dir %||% file.path("/mnt/sata_ssd", paste0("fastEmbedR_BENCHMARK3_", format(Sys.time(), "%Y%m%d_%H%M%S")))
benchmark1_dir <- args$benchmark1_dir %||% ""
k <- as.integer(args$k %||% "50")
n_threads <- as.integer(args$threads %||% "4")
timeout_sec <- as.integer(args$timeout %||% "600")
seed <- as.integer(args$seed %||% "42")
metric_n <- as.integer(args$metric_n %||% "5000")
worker <- isTRUE(as.logical(args$worker %||% FALSE))

dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

log_msg <- function(...) {
  cat(format(Sys.time(), "%Y-%m-%d %H:%M:%S"), " ", sprintf(...), "\n", sep = "")
  flush.console()
}

available_pkg <- function(pkg) requireNamespace(pkg, quietly = TRUE)

json_or_text <- function(x) {
  if (requireNamespace("jsonlite", quietly = TRUE)) {
    as.character(jsonlite::toJSON(x, auto_unbox = TRUE, null = "null", digits = NA))
  } else {
    paste(capture.output(str(x)), collapse = " ")
  }
}

read_peak_rss_gb <- function() {
  status <- "/proc/self/status"
  if (!file.exists(status)) return(NA_real_)
  x <- readLines(status, warn = FALSE)
  v <- x[grepl("^VmHWM:", x)]
  if (!length(v)) return(NA_real_)
  kb <- suppressWarnings(as.numeric(gsub("[^0-9]", "", v[[1L]])))
  if (!is.finite(kb)) NA_real_ else kb / 1024^2
}

coerce_matrix <- function(x) {
  if (inherits(x, "Matrix")) x <- as.matrix(x)
  if (is.data.frame(x)) x <- as.matrix(x)
  if (!is.matrix(x)) x <- as.matrix(x)
  storage.mode(x) <- "double"
  x
}

load_rdata_object <- function(path, name) {
  env <- new.env(parent = emptyenv())
  load(path, envir = env)
  if (!exists(name, envir = env, inherits = FALSE)) {
    stop("No object named `", name, "` in ", path, call. = FALSE)
  }
  get(name, envir = env, inherits = FALSE)
}

load_dataset <- function(dataset, data_path) {
  obj <- load_rdata_object(data_path, "dataset")
  list(
    data = coerce_matrix(obj$data),
    labels = if (is.null(obj$labels)) NULL else as.factor(obj$labels),
    metadata = obj$metadata %||% list(),
    source = data_path
  )
}

manifest_path <- file.path(data_root, "dataset_manifest.csv")
if (!file.exists(manifest_path)) stop("Missing dataset manifest: ", manifest_path, call. = FALSE)
manifest <- utils::read.csv(manifest_path, stringsAsFactors = FALSE)
if (!all(c("dataset", "path") %in% names(manifest))) {
  stop("dataset_manifest.csv must contain columns `dataset` and `path`.", call. = FALSE)
}

pca_manifest_path <- file.path(data_root, "pca_init_manifest.csv")
pca_manifest <- if (file.exists(pca_manifest_path)) {
  utils::read.csv(pca_manifest_path, stringsAsFactors = FALSE)
} else {
  data.frame(dataset = character(), pca_init_path = character(), stringsAsFactors = FALSE)
}

dataset_filter <- strsplit(args$datasets %||% paste(manifest$dataset, collapse = ","), ",", fixed = TRUE)[[1L]]
dataset_filter <- trimws(dataset_filter)
manifest <- manifest[manifest$dataset %in% dataset_filter, , drop = FALSE]
if (nrow(manifest) == 0L) stop("No requested datasets found in manifest.", call. = FALSE)

standardize_knn <- function(obj) {
  if (!is.null(obj$indices) && !is.null(obj$distances)) return(list(indices = obj$indices, distances = obj$distances))
  if (!is.null(obj$idx) && !is.null(obj$dist)) return(list(indices = obj$idx, distances = obj$dist))
  if (!is.null(obj$nn.idx) && !is.null(obj$nn.dists)) return(list(indices = obj$nn.idx, distances = obj$nn.dists))
  if (!is.null(obj$index) && !is.null(obj$distance)) return(list(indices = obj$index, distances = obj$distance))
  stop("Cannot identify KNN indices/distances in object.", call. = FALSE)
}

drop_self_if_first <- function(indices, distances, target_k) {
  if (ncol(indices) > target_k) {
    self_first <- all(indices[, 1L] == seq_len(nrow(indices)))
    zero_first <- all(abs(distances[, 1L]) < 1e-12)
    if (isTRUE(self_first) || isTRUE(zero_first)) {
      indices <- indices[, -1L, drop = FALSE]
      distances <- distances[, -1L, drop = FALSE]
    }
  }
  if (ncol(indices) > target_k) {
    indices <- indices[, seq_len(target_k), drop = FALSE]
    distances <- distances[, seq_len(target_k), drop = FALSE]
  }
  list(indices = indices, distances = distances)
}

find_existing_cuvs_knn <- function(dataset, k) {
  candidates <- character()
  if (nzchar(benchmark1_dir)) {
    candidates <- c(candidates, file.path(benchmark1_dir, "knn_cuvs_nndescent", paste0(dataset, "_cuvs_nndescent_k", k, ".RData")))
  }
  b1_dirs <- list.dirs("/mnt/sata_ssd", recursive = FALSE, full.names = TRUE)
  b1_dirs <- b1_dirs[grepl("fastEmbedR_BENCHMARK1_", basename(b1_dirs))]
  b1_dirs <- b1_dirs[order(file.info(b1_dirs)$mtime, decreasing = TRUE)]
  candidates <- c(candidates, file.path(b1_dirs, "knn_cuvs_nndescent", paste0(dataset, "_cuvs_nndescent_k", k, ".RData")))
  candidates <- candidates[file.exists(candidates)]
  if (length(candidates)) candidates[[1L]] else ""
}

preferred_cpu_knn_backend <- function() {
  if (available_pkg("fastEmbedR") && isTRUE(tryCatch(fastEmbedR::faiss_available(), error = function(e) FALSE))) {
    return("faiss_hnsw")
  }
  "hnsw"
}

load_or_compute_knn <- function(dataset_name, x) {
  cache_dir <- file.path(out_dir, "knn_cache")
  dir.create(cache_dir, recursive = TRUE, showWarnings = FALSE)
  local_cache <- file.path(cache_dir, paste0(dataset_name, "_cuvs_nndescent_k", k, ".RData"))
  source <- "benchmark3_cache"
  if (!file.exists(local_cache)) {
    existing <- find_existing_cuvs_knn(dataset_name, k)
    if (nzchar(existing)) {
      file.copy(existing, local_cache, overwrite = TRUE)
      source <- paste0("precomputed:", existing)
    }
  }
  if (file.exists(local_cache)) {
    obj <- load_rdata_object(local_cache, "nn_cuvs_nndescent")
    sx <- standardize_knn(obj)
    if (ncol(sx$indices) >= k) {
      return(list(knn = drop_self_if_first(sx$indices, sx$distances, k), source = source, knn_sec = 0))
    }
  }
  if (!available_pkg("fastEmbedR")) stop("fastEmbedR is required to compute KNN.", call. = FALSE)
  knn_backends <- if (isTRUE(tryCatch(fastEmbedR::cuvs_available() && fastEmbedR::cuda_available(), error = function(e) FALSE))) {
    c("cuda_cuvs_nndescent", "cuda_cuvs_bruteforce", "cuda_exact", "faiss_hnsw", "rcpphnsw")
  } else {
    c(preferred_cpu_knn_backend(), "rcpphnsw")
  }
  knn_backends <- unique(knn_backends)
  nn_cuvs_nndescent <- NULL
  last_error <- NULL
  used_backend <- NA_character_
  t <- system.time({
    for (candidate_backend in knn_backends) {
      attempt <- tryCatch(
        fastEmbedR::nn(
          x,
          k = k,
          backend = candidate_backend,
          n_threads = n_threads,
          metric = "euclidean"
        ),
        error = function(e) {
          last_error <<- paste(candidate_backend, conditionMessage(e), sep = ": ")
          NULL
        }
      )
      if (!is.null(attempt)) {
        nn_cuvs_nndescent <- attempt
        used_backend <- candidate_backend
        break
      }
    }
  })[["elapsed"]]
  if (is.null(nn_cuvs_nndescent)) {
    stop("Could not compute fallback KNN for BENCHMARK #3. Last error: ", last_error, call. = FALSE)
  }
  save(nn_cuvs_nndescent, file = local_cache, compress = "gzip")
  sx <- standardize_knn(nn_cuvs_nndescent)
  list(knn = drop_self_if_first(sx$indices, sx$distances, k), source = paste0("computed_benchmark3_", used_backend), knn_sec = as.numeric(t))
}

load_pca_init <- function(dataset_name, n) {
  row <- pca_manifest[pca_manifest$dataset == dataset_name, , drop = FALSE]
  paths <- character()
  if (nrow(row) > 0L) {
    pcol <- intersect(c("pca_init_path", "path"), names(row))
    if (length(pcol)) paths <- as.character(row[[pcol[[1L]]]])
  }
  paths <- c(paths, Sys.glob(file.path(data_root, dataset_name, "*_fastPLS_pca2_init.RData")))
  paths <- paths[nzchar(paths) & file.exists(paths)]
  if (!length(paths)) return(NULL)
  env <- new.env(parent = emptyenv())
  load(paths[[1L]], envir = env)
  candidates <- mget(ls(env), env, ifnotfound = list(NULL))
  layout <- NULL
  for (obj in candidates) {
    if (is.matrix(obj) || is.data.frame(obj)) {
      mat <- as.matrix(obj)
      if (nrow(mat) == n && ncol(mat) >= 2L) {
        layout <- mat[, 1:2, drop = FALSE]
        break
      }
    }
    if (is.list(obj)) {
      for (nm in c("layout", "pca", "scores", "x", "rotation")) {
        if (!is.null(obj[[nm]])) {
          mat <- as.matrix(obj[[nm]])
          if (nrow(mat) == n && ncol(mat) >= 2L) {
            layout <- mat[, 1:2, drop = FALSE]
            break
          }
        }
      }
    }
    if (!is.null(layout)) break
  }
  if (is.null(layout)) return(NULL)
  storage.mode(layout) <- "double"
  scale(layout, center = TRUE, scale = FALSE)
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
  x[, 1:2, drop = FALSE]
}

metric_columns <- c(
  "trustworthiness", "continuity", "knn_preservation_15", "knn_preservation_30",
  "knn_preservation_50", "distance_spearman", "distance_pearson", "stress",
  "silhouette", "label_knn_accuracy", "ari", "nmi", "rare_class_recall"
)

empty_metrics <- function() {
  out <- as.data.frame(as.list(rep(NA_real_, length(metric_columns))), stringsAsFactors = FALSE)
  names(out) <- metric_columns
  out
}

evaluate_layout <- function(dataset_name, x, labels, layout, method, backend) {
  if (!available_pkg("fastEmbedR")) return(empty_metrics())
  n <- nrow(x)
  keep <- seq_len(n)
  if (n > metric_n) {
    set.seed(seed)
    keep <- sort(sample.int(n, metric_n))
  }
  metrics <- tryCatch(
    fastEmbedR::evaluate_embedding(
      x[keep, , drop = FALSE],
      layout[keep, , drop = FALSE],
      labels = if (is.null(labels)) NULL else labels[keep],
      k = c(15L, 30L, 50L),
      sample_size_for_global_metrics = min(3000L, length(keep)),
      sample_size_for_local_metrics = min(3000L, length(keep)),
      seed = seed,
      method = method,
      backend = "cpu",
      n_threads = n_threads,
      dataset = dataset_name
    ),
    error = function(e) {
      warning("Evaluation failed for ", dataset_name, " / ", method, ": ", conditionMessage(e))
      empty_metrics()
    }
  )
  out <- empty_metrics()
  for (nm in intersect(names(out), names(metrics))) out[[nm]] <- metrics[[nm]][[1L]]
  out
}

plot_layout <- function(layout, labels, path, title) {
  dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
  png(path, width = 1800, height = 1500, res = 180)
  on.exit(dev.off(), add = TRUE)
  par(mar = c(3, 3, 3, 1), bg = "white")
  if (is.null(labels)) {
    cols <- "#1f77b4"
  } else {
    f <- as.factor(labels)
    pal <- grDevices::hcl.colors(max(3L, nlevels(f)), "Dark 3")
    cols <- pal[as.integer(f)]
  }
  plot(layout[, 1], layout[, 2], pch = 16, cex = 0.35, col = cols,
       xlab = "UMAP 1", ylab = "UMAP 2", main = title)
}

sanitize <- function(x) gsub("[^A-Za-z0-9_+-]+", "_", x)

run_with_timing <- function(expr, n) {
  gc()
  before <- read_peak_rss_gb()
  t <- system.time({
    value <- force(expr)
  })[["elapsed"]]
  after <- read_peak_rss_gb()
  peak <- suppressWarnings(max(before, after, na.rm = TRUE))
  if (!is.finite(peak)) peak <- NA_real_
  list(
    layout = coerce_layout(value, n),
    embedding_sec = as.numeric(t),
    peak_ram_gb = peak
  )
}

method_specs <- function() {
  list(
    list(
      method = "fastEmbedR_umap_cpu_binary",
      package = "fastEmbedR",
      backend = "cpu",
      uses_precomputed_nn = TRUE,
      graph_mode = "binary",
      init_policy = "internal_spectral_from_knn_binary_graph",
      runner = function(ctx) {
        fastEmbedR::umap_knn(
          ctx$knn$indices,
          ctx$knn$distances,
          backend = "cpu",
          graph_mode = "binary",
          n_threads = n_threads,
          seed = seed,
          verbose = FALSE
        )
      }
    ),
    list(
      method = "fastEmbedR_umap_cpu_fuzzy",
      package = "fastEmbedR",
      backend = "cpu",
      uses_precomputed_nn = TRUE,
      graph_mode = "fuzzy",
      init_policy = "internal_spectral_from_knn_fuzzy_graph",
      runner = function(ctx) {
        fastEmbedR::umap_knn(
          ctx$knn$indices,
          ctx$knn$distances,
          backend = "cpu",
          graph_mode = "fuzzy",
          n_threads = n_threads,
          seed = seed,
          verbose = FALSE
        )
      }
    ),
    list(
      method = "fastEmbedR_umap_metal_binary",
      package = "fastEmbedR",
      backend = "metal",
      uses_precomputed_nn = TRUE,
      graph_mode = "binary",
      init_policy = "internal_spectral_from_knn_binary_graph",
      runner = function(ctx) {
        fastEmbedR::umap_knn(
          ctx$knn$indices,
          ctx$knn$distances,
          backend = "metal",
          graph_mode = "binary",
          n_threads = n_threads,
          seed = seed,
          verbose = FALSE
        )
      }
    ),
    list(
      method = "fastEmbedR_umap_metal_fuzzy",
      package = "fastEmbedR",
      backend = "metal",
      uses_precomputed_nn = TRUE,
      graph_mode = "fuzzy",
      init_policy = "internal_spectral_from_knn_fuzzy_graph",
      runner = function(ctx) {
        fastEmbedR::umap_knn(
          ctx$knn$indices,
          ctx$knn$distances,
          backend = "metal",
          graph_mode = "fuzzy",
          n_threads = n_threads,
          seed = seed,
          verbose = FALSE
        )
      }
    ),
    list(
      method = "fastEmbedR_umap_cuda_binary",
      package = "fastEmbedR",
      backend = "cuda",
      uses_precomputed_nn = TRUE,
      graph_mode = "binary",
      init_policy = "native_cuda_fused_spectral_from_knn_binary_graph",
      runner = function(ctx) {
        fastEmbedR::umap_knn(
          ctx$knn$indices,
          ctx$knn$distances,
          backend = "cuda",
          graph_mode = "binary",
          n_threads = n_threads,
          seed = seed,
          verbose = FALSE
        )
      }
    ),
    list(
      method = "fastEmbedR_umap_cuda_fuzzy",
      package = "fastEmbedR",
      backend = "cuda",
      uses_precomputed_nn = TRUE,
      graph_mode = "fuzzy",
      init_policy = "internal_spectral_from_knn_fuzzy_graph_cuda_optimizer",
      runner = function(ctx) {
        fastEmbedR::umap_knn(
          ctx$knn$indices,
          ctx$knn$distances,
          backend = "cuda",
          graph_mode = "fuzzy",
          n_threads = n_threads,
          seed = seed,
          verbose = FALSE
        )
      }
    ),
    list(
      method = "uwot_umap_fast_sgd",
      package = "uwot",
      backend = "cpu",
      uses_precomputed_nn = TRUE,
      graph_mode = "fuzzy",
      init_policy = "uwot_internal_spectral",
      runner = function(ctx) {
        uwot::umap(
          ctx$x,
          n_neighbors = k,
          nn_method = ctx$knn,
          n_threads = n_threads,
          n_sgd_threads = n_threads,
          fast_sgd = TRUE,
          init = "spectral",
          min_dist = 0.1,
          ret_model = FALSE,
          verbose = FALSE,
          seed = seed
        )
      }
    ),
    list(
      method = "uwot_umap_default",
      package = "uwot",
      backend = "cpu",
      uses_precomputed_nn = TRUE,
      graph_mode = "fuzzy",
      init_policy = "uwot_internal_spectral",
      runner = function(ctx) {
        uwot::umap(
          ctx$x,
          n_neighbors = k,
          nn_method = ctx$knn,
          n_threads = n_threads,
          n_sgd_threads = 0,
          fast_sgd = FALSE,
          init = "spectral",
          min_dist = 0.1,
          ret_model = FALSE,
          verbose = FALSE,
          seed = seed
        )
      }
    ),
    list(
      method = "umap_package",
      package = "umap",
      backend = "cpu",
      uses_precomputed_nn = TRUE,
      graph_mode = "fuzzy",
      init_policy = "umap_package_internal_spectral",
      runner = function(ctx) {
        uknn <- umap::umap.knn(ctx$knn$indices, ctx$knn$distances)
        cfg <- umap::umap.defaults
        cfg$knn <- uknn
        cfg$n_neighbors <- k
        cfg$init <- "spectral"
        cfg$min_dist <- 0.1
        umap::umap(ctx$x, knn = uknn, config = cfg)$layout
      }
    )
  )
}
result_template <- function(dataset_name, method, package, backend, status, error_message = NA_character_) {
  data.frame(
    dataset = dataset_name,
    method = method,
    package = package,
    backend_requested = backend,
    backend_used = if (status == "success") backend else NA_character_,
    status = status,
    error_message = error_message,
    n = NA_integer_,
    p = NA_integer_,
    seed = seed,
    k = k,
    graph_mode = NA_character_,
    uses_precomputed_nn = NA,
    pca_init_available = NA,
    init_policy = NA_character_,
    knn_source = NA_character_,
    knn_sec = NA_real_,
    embedding_sec = NA_real_,
    total_sec = NA_real_,
    peak_ram_gb = NA_real_,
    layout_rdata = NA_character_,
    plot_png = NA_character_,
    parameters_json = NA_character_,
    empty_metrics(),
    stringsAsFactors = FALSE
  )
}

run_one <- function(dataset_name, method_name, row_out) {
  specs <- method_specs()
  spec <- specs[vapply(specs, function(z) identical(z$method, method_name), logical(1))][[1L]]
  ds_row <- manifest[manifest$dataset == dataset_name, , drop = FALSE]
  if (nrow(ds_row) != 1L) stop("Dataset not found in manifest: ", dataset_name, call. = FALSE)
  ds <- load_dataset(dataset_name, ds_row$path[[1L]])
  x <- ds$data
  labels <- ds$labels
  n <- nrow(x)
  p <- ncol(x)
  if (!available_pkg(spec$package)) {
    row <- result_template(dataset_name, spec$method, spec$package, spec$backend, "not_installed",
                           paste0("Package ", spec$package, " is not installed."))
    row$n <- n
    row$p <- p
    utils::write.csv(row, row_out, row.names = FALSE)
    return(invisible(row))
  }
  Y_init <- load_pca_init(dataset_name, n)
  knn_info <- load_or_compute_knn(dataset_name, x)
  ctx <- list(dataset = dataset_name, x = x, labels = labels, knn = knn_info$knn, Y_init = Y_init)
  status <- "success"
  error <- NA_character_
  measured <- NULL
  tryCatch({
    set.seed(seed)
    measured <- run_with_timing(spec$runner(ctx), n)
  }, error = function(e) {
    status <<- "failed"
    error <<- conditionMessage(e)
  })
  row <- result_template(dataset_name, spec$method, spec$package, spec$backend, status, error)
  row$n <- n
  row$p <- p
  row$uses_precomputed_nn <- isTRUE(spec$uses_precomputed_nn)
  row$graph_mode <- spec$graph_mode %||% NA_character_
  row$pca_init_available <- !is.null(Y_init)
  row$init_policy <- spec$init_policy
  row$knn_source <- knn_info$source
  row$knn_sec <- knn_info$knn_sec
  row$parameters_json <- json_or_text(list(
    k = k,
    seed = seed,
    n_threads = n_threads,
    min_dist = 0.1,
    graph_mode = spec$graph_mode %||% NA_character_,
    precomputed_knn = TRUE,
    pca_init_available = !is.null(Y_init),
    init_policy = spec$init_policy
  ))
  if (!is.null(measured)) {
    layout <- measured$layout
    layout_file <- file.path(out_dir, "layouts", paste0(sanitize(dataset_name), "__", sanitize(spec$method), ".RData"))
    plot_file <- file.path(out_dir, "plots", paste0(sanitize(dataset_name), "__", sanitize(spec$method), ".png"))
    dir.create(dirname(layout_file), recursive = TRUE, showWarnings = FALSE)
    metrics <- evaluate_layout(dataset_name, x, labels, layout, spec$method, spec$backend)
    layout_result <- list(
      dataset = dataset_name,
      method = spec$method,
      backend = spec$backend,
      layout = layout,
      labels = labels,
      metrics = metrics,
      parameters = list(
        k = k,
        seed = seed,
        n_threads = n_threads,
        graph_mode = spec$graph_mode %||% NA_character_,
        init_policy = spec$init_policy
      )
    )
    save(layout_result, file = layout_file, compress = "gzip")
    plot_layout(layout, labels, plot_file, paste(dataset_name, spec$method))
    row$embedding_sec <- measured$embedding_sec
    row$total_sec <- measured$embedding_sec + if (is.finite(knn_info$knn_sec)) knn_info$knn_sec else 0
    row$peak_ram_gb <- measured$peak_ram_gb
    row$layout_rdata <- layout_file
    row$plot_png <- plot_file
    for (nm in names(metrics)) if (nm %in% names(row)) row[[nm]] <- metrics[[nm]][[1L]]
  }
  utils::write.csv(row, row_out, row.names = FALSE)
  invisible(row)
}

combine_worker_rows <- function(worker_dir) {
  files <- list.files(worker_dir, pattern = "\\.csv$", full.names = TRUE)
  if (!length(files)) return(data.frame())
  rows <- lapply(files, function(f) utils::read.csv(f, stringsAsFactors = FALSE))
  common <- Reduce(union, lapply(rows, names))
  rows <- lapply(rows, function(x) {
    miss <- setdiff(common, names(x))
    for (m in miss) x[[m]] <- NA
    x[, common, drop = FALSE]
  })
  do.call(rbind, rows)
}

make_barplots <- function(results, out_dir) {
  ok <- results[results$status == "success", , drop = FALSE]
  if (!nrow(ok)) return(invisible(NULL))
  png(file.path(out_dir, "benchmark3_runtime_barplot.png"), width = 2200, height = 1500, res = 180)
  par(mar = c(10, 5, 4, 1), bg = "white")
  labels <- paste(ok$dataset, ok$method, sep = "\n")
  cols <- ifelse(
    grepl("cuda", ok$backend_requested, ignore.case = TRUE), "#D55E00",
    ifelse(grepl("metal", ok$backend_requested, ignore.case = TRUE), "#009E73", "#0072B2")
  )
  barplot(ok$embedding_sec, names.arg = labels, las = 2, cex.names = 0.55, col = cols,
          ylab = "Embedding seconds", main = "BENCHMARK #3 UMAP embedding runtime")
  legend("topright", legend = c("CPU", "Metal", "CUDA"), fill = c("#0072B2", "#009E73", "#D55E00"), bty = "n")
  dev.off()

  png(file.path(out_dir, "benchmark3_trust_barplot.png"), width = 2200, height = 1500, res = 180)
  par(mar = c(10, 5, 4, 1), bg = "white")
  barplot(ok$trustworthiness, names.arg = labels, las = 2, cex.names = 0.55, col = cols,
          ylab = "Trustworthiness", main = "BENCHMARK #3 UMAP embedding quality")
  legend("topright", legend = c("CPU", "Metal", "CUDA"), fill = c("#0072B2", "#009E73", "#D55E00"), bty = "n")
  dev.off()
}

write_methods_file <- function(out_dir) {
  txt <- c(
    "# BENCHMARK #3 Material and Methods",
    "",
    "This benchmark compares UMAP implementations across the curated fastEmbedR datasets. On macOS it includes CPU and native Metal rows; on the chiamaka GPU workstation it includes CPU and native CUDA rows.",
    "",
    "Datasets are loaded from `dataset_manifest.csv`. Labels are used only for post-hoc quality metrics and plots, not for fitting.",
    "",
    "Nearest-neighbour input: all methods use the same saved cuVS NN-descent KNN matrix from BENCHMARK #1 when present. If the required KNN cache is absent, the script computes it once with `fastEmbedR::nn()`, preferring `backend = \"cuda_cuvs_nndescent\"` when CUDA/cuVS is available.",
    "",
    "Initialization: spectral initialization is used wherever the implementation exposes it. `fastEmbedR::umap_knn()` uses its internal KNN spectral initialization, including native CUDA fused spectral initialization on the CUDA path. `uwot::umap()` is called with `init = \"spectral\"`. The `umap` package is called with `config$init = \"spectral\"`. Precomputed fastPLS PCA initialization files are loaded only to record availability in the result table and are not used for this spectral-initialization benchmark.",
    "",
    "Compared implementations:",
    "",
    "- `fastEmbedR_umap_cpu_binary` and `fastEmbedR_umap_cpu_fuzzy`: native fastEmbedR UMAP from precomputed KNN, CPU backend, four CPU threads.",
    "- `fastEmbedR_umap_metal_binary` and `fastEmbedR_umap_metal_fuzzy`: native fastEmbedR UMAP from precomputed KNN, Metal backend when available.",
    "- `fastEmbedR_umap_cuda_binary` and `fastEmbedR_umap_cuda_fuzzy`: native fastEmbedR UMAP from precomputed KNN, CUDA backend when available.",
    "- `uwot_umap_fast_sgd`: `uwot::umap()` with shared precomputed KNN and `fast_sgd = TRUE`.",
    "- `uwot_umap_default`: `uwot::umap()` with shared precomputed KNN and `fast_sgd = FALSE`.",
    "- `umap_package`: `umap::umap()` with `umap::umap.knn()` built from the shared precomputed KNN.",
    "",
    sprintf("Default settings: k = %d non-self neighbours, min_dist = 0.1, seed = %d, CPU threads = %d. fastEmbedR UMAP is evaluated with graph_mode = \"binary\" and graph_mode = \"fuzzy\".", k, seed, n_threads),
    "",
    sprintf("Each method-dataset worker is executed with a %d second timeout. Failed, unavailable, or timed-out rows are retained in the result table.", timeout_sec),
    "",
    "Quality metrics are computed on a reproducible sample of up to 5000 cells/samples using `fastEmbedR::evaluate_embedding()`: trustworthiness, continuity, kNN preservation at 15/30/50, global distance correlations, stress, silhouette, label kNN accuracy, ARI, NMI, and rare-class recall when labels are available.",
    "",
    "Outputs: `benchmark3_umap_results.csv`, per-method `.RData` layout files, per-method PNG plots, runtime/quality barplots, and a narrative result summary."
  )
  writeLines(txt, file.path(out_dir, "BENCHMARK3_MATERIALS_AND_METHODS.md"))
}

write_summary_file <- function(results, out_dir) {
  ok <- results[results$status == "success", , drop = FALSE]
  lines <- c("# BENCHMARK #3 Results Summary", "")
  if (!nrow(results)) {
    lines <- c(lines, "No worker rows were collected.")
  } else {
    lines <- c(lines, sprintf("Rows collected: %d. Successful rows: %d.", nrow(results), nrow(ok)), "")
    if (nrow(ok)) {
      best_time <- ok[order(ok$dataset, ok$embedding_sec), c("dataset", "method", "backend_requested", "embedding_sec", "trustworthiness", "label_knn_accuracy")]
      best_time <- do.call(rbind, lapply(split(best_time, best_time$dataset), head, 1L))
      lines <- c(lines, "Fastest successful method per dataset:", "", capture.output(print(best_time, row.names = FALSE)), "")
      best_quality <- ok[order(ok$dataset, -ok$trustworthiness), c("dataset", "method", "backend_requested", "embedding_sec", "trustworthiness", "label_knn_accuracy")]
      best_quality <- do.call(rbind, lapply(split(best_quality, best_quality$dataset), head, 1L))
      lines <- c(lines, "Highest trustworthiness per dataset:", "", capture.output(print(best_quality, row.names = FALSE)), "")
    }
    failed <- results[results$status != "success", c("dataset", "method", "status", "error_message"), drop = FALSE]
    if (nrow(failed)) lines <- c(lines, "Failed/unavailable rows:", "", capture.output(print(failed, row.names = FALSE)), "")
    lines <- c(lines, "Comment:", "",
               "Rows share the same KNN input whenever possible. The `knn_sec` column is zero when a saved KNN cache was reused and positive when BENCHMARK #3 had to compute the missing KNN. UMAP initialization differs by implementation where APIs differ; this is recorded in `init_policy` and `parameters_json`.")
  }
  writeLines(lines, file.path(out_dir, "BENCHMARK3_RESULTS_SUMMARY.md"))
}

if (worker) {
  dataset_name <- args$dataset %||% stop("--dataset is required in worker mode", call. = FALSE)
  method_name <- args$method %||% stop("--method is required in worker mode", call. = FALSE)
  row_out <- args$row_out %||% stop("--row_out is required in worker mode", call. = FALSE)
  row <- tryCatch(
    run_one(dataset_name, method_name, row_out),
    error = function(e) {
      specs <- method_specs()
      spec <- specs[vapply(specs, function(z) identical(z$method, method_name), logical(1))][[1L]]
      row <- result_template(dataset_name, method_name, spec$package, spec$backend, "failed", conditionMessage(e))
      utils::write.csv(row, row_out, row.names = FALSE)
      row
    }
  )
  quit(save = "no", status = 0L)
}

write_methods_file(out_dir)
worker_dir <- file.path(out_dir, "worker_rows")
dir.create(worker_dir, recursive = TRUE, showWarnings = FALSE)

methods <- vapply(method_specs(), `[[`, character(1L), "method")
if (!is.null(args$methods)) {
  method_filter <- trimws(strsplit(args$methods, ",", fixed = TRUE)[[1L]])
  methods <- methods[methods %in% method_filter]
  if (!length(methods)) stop("No requested methods found in BENCHMARK #3 method list.", call. = FALSE)
}

log_msg("BENCHMARK #3 output: %s", out_dir)
log_msg("Datasets: %s", paste(manifest$dataset, collapse = ", "))
log_msg("Methods: %s", paste(methods, collapse = ", "))

cmd_args_all <- commandArgs(FALSE)
file_arg <- cmd_args_all[grepl("^--file=", cmd_args_all)]
script_path <- if (length(file_arg)) sub("^--file=", "", file_arg[[1L]]) else "tools/benchmark3_umap_speed_accuracy.R"
script_path <- normalizePath(script_path, mustWork = FALSE)

for (dataset_name in manifest$dataset) {
  for (method_name in methods) {
    row_file <- file.path(worker_dir, paste0(sanitize(dataset_name), "__", sanitize(method_name), ".csv"))
    if (file.exists(row_file)) {
      log_msg("Skipping existing %s / %s", dataset_name, method_name)
      next
    }
    log_msg("Running %s / %s", dataset_name, method_name)
    cmd <- sprintf(
      "timeout %d Rscript %s --worker=TRUE --data_root=%s --out_dir=%s --benchmark1_dir=%s --dataset=%s --method=%s --k=%d --threads=%d --timeout=%d --seed=%d --metric_n=%d --row_out=%s",
      timeout_sec,
      shQuote(script_path),
      shQuote(data_root),
      shQuote(out_dir),
      shQuote(benchmark1_dir),
      shQuote(dataset_name),
      shQuote(method_name),
      k,
      n_threads,
      timeout_sec,
      seed,
      metric_n,
      shQuote(row_file)
    )
    code <- system(cmd)
    if (!file.exists(row_file)) {
      specs <- method_specs()
      spec <- specs[vapply(specs, function(z) identical(z$method, method_name), logical(1))][[1L]]
      row <- result_template(dataset_name, method_name, spec$package, spec$backend, "timeout",
                             paste0("Worker exceeded timeout or terminated with code ", code, "."))
      utils::write.csv(row, row_file, row.names = FALSE)
    }
  }
  results_partial <- combine_worker_rows(worker_dir)
  utils::write.csv(results_partial, file.path(out_dir, "benchmark3_umap_results_partial.csv"), row.names = FALSE)
}

results <- combine_worker_rows(worker_dir)
utils::write.csv(results, file.path(out_dir, "benchmark3_umap_results.csv"), row.names = FALSE)
if (nrow(results)) {
  ok <- results[results$status == "success", , drop = FALSE]
  if (nrow(ok)) {
    best_by_dataset <- do.call(rbind, lapply(split(ok, ok$dataset), function(z) z[order(z$embedding_sec), , drop = FALSE][1L, , drop = FALSE]))
    utils::write.csv(best_by_dataset, file.path(out_dir, "benchmark3_best_by_dataset_runtime.csv"), row.names = FALSE)
  }
  make_barplots(results, out_dir)
  write_summary_file(results, out_dir)
}

log_msg("BENCHMARK #3 finished: %s", out_dir)
