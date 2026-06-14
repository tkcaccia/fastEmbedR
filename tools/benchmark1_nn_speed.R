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

`%||%` <- function(x, y) if (is.null(x) || length(x) == 0L || is.na(x)) y else x

args <- parse_args(commandArgs(trailingOnly = TRUE))

data_root <- args$data_root %||% "/mnt/sata_ssd/fastEmbedR_Data"
out_dir <- args$out_dir %||% file.path("/mnt/sata_ssd", paste0("fastEmbedR_BENCHMARK1_", format(Sys.time(), "%Y%m%d_%H%M%S")))
k <- as.integer(args$k %||% "50")
n_threads <- as.integer(args$threads %||% "4")
timeout_sec <- as.integer(args$timeout %||% "600")
worker <- isTRUE(as.logical(args$worker %||% FALSE))

dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

log_msg <- function(...) {
  cat(format(Sys.time(), "%Y-%m-%d %H:%M:%S"), " ", sprintf(...), "\n", sep = "")
  flush.console()
}

write_csv_one <- function(path, row) {
  utils::write.csv(row, path, row.names = FALSE)
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

available_pkg <- function(pkg) requireNamespace(pkg, quietly = TRUE)

coerce_matrix <- function(x) {
  if (inherits(x, "Matrix")) x <- as.matrix(x)
  if (is.data.frame(x)) x <- as.matrix(x)
  if (!is.matrix(x)) x <- as.matrix(x)
  storage.mode(x) <- "double"
  x
}

load_dataset <- function(dataset, data_path) {
  if (identical(dataset, "SimulatedUniform2D")) {
    set.seed(1)
    data <- matrix(runif(1000000), ncol = 2)
    colnames(data) <- c("x", "y")
    return(list(data = data, labels = NULL, source = "simulated runif(1000000), ncol=2"))
  }
  env <- new.env(parent = emptyenv())
  load(data_path, envir = env)
  if (!exists("dataset", envir = env, inherits = FALSE)) {
    stop("No object named `dataset` in ", data_path)
  }
  ds <- get("dataset", envir = env, inherits = FALSE)
  list(data = coerce_matrix(ds$data), labels = ds$labels, source = data_path)
}

standardize_knn <- function(obj) {
  if (is.null(obj)) return(list(indices = NULL, distances = NULL))
  if (!is.null(obj$indices) && !is.null(obj$distances)) {
    return(list(indices = obj$indices, distances = obj$distances))
  }
  if (!is.null(obj$idx) && !is.null(obj$dist)) {
    return(list(indices = obj$idx, distances = obj$dist))
  }
  if (!is.null(obj$nn.idx) && !is.null(obj$nn.dists)) {
    return(list(indices = obj$nn.idx, distances = obj$nn.dists))
  }
  if (!is.null(obj$index) && !is.null(obj$distance)) {
    return(list(indices = obj$index, distances = obj$distance))
  }
  list(indices = NULL, distances = NULL)
}

drop_self_if_first <- function(indices, distances, target_k) {
  if (is.null(indices) || is.null(distances)) return(list(indices = indices, distances = distances))
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

save_cuvs_knn <- function(obj, dataset, out_dir) {
  knn_dir <- file.path(out_dir, "knn_cuvs_nndescent")
  dir.create(knn_dir, recursive = TRUE, showWarnings = FALSE)
  nn_cuvs_nndescent <- obj
  save(
    nn_cuvs_nndescent,
    file = file.path(knn_dir, paste0(dataset, "_cuvs_nndescent_k", k, ".RData")),
    compress = "gzip"
  )
}

annoy_knn <- function(x, k, n_trees = 50L) {
  if (!available_pkg("RcppAnnoy")) stop("RcppAnnoy unavailable")
  p <- ncol(x)
  index <- new(RcppAnnoy::AnnoyEuclidean, p)
  for (i in seq_len(nrow(x))) index$addItem(i - 1L, x[i, ])
  index$build(as.integer(n_trees))
  idx <- matrix(NA_integer_, nrow(x), k)
  dst <- matrix(NA_real_, nrow(x), k)
  for (i in seq_len(nrow(x))) {
    ans <- index$getNNsByVectorList(x[i, ], k + 1L, search_k = -1L, include_distances = TRUE)
    ii <- as.integer(ans$item + 1L)
    dd <- as.numeric(ans$distance)
    keep <- ii != i
    ii <- ii[keep]
    dd <- dd[keep]
    if (length(ii) < k) {
      ii <- c(ii, rep(NA_integer_, k - length(ii)))
      dd <- c(dd, rep(NA_real_, k - length(dd)))
    }
    idx[i, ] <- ii[seq_len(k)]
    dst[i, ] <- dd[seq_len(k)]
  }
  list(indices = idx, distances = dst)
}

run_method <- function(method, x, k, n_threads, dataset, out_dir) {
  if (startsWith(method, "fastEmbedR_")) {
    if (!available_pkg("fastEmbedR")) stop("fastEmbedR unavailable")
    backend <- sub("^fastEmbedR_", "", method)
    backend <- switch(
      backend,
      cpu_exact = "cpu",
      rcpphnsw = "hnsw",
      faiss_flat_l2 = "faiss_flat_l2",
      faiss_flat_ip = "faiss_flat_ip",
      faiss_ivf = "faiss_ivf",
      faiss_ivfpq = "faiss_ivfpq",
      faiss_hnsw = "faiss_hnsw",
      faiss_nsg = "faiss_nsg",
      faiss_nndescent = "faiss_nndescent",
      cuda_exact = "cuda",
      cuda_ivf = "cuda_ivf",
      cuda_cuvs_bruteforce = "cuda_cuvs_bruteforce",
      cuda_cuvs_cagra = "cuda_cuvs_cagra",
      cuda_cuvs_nndescent = "cuda_cuvs_nndescent",
      backend
    )
    obj <- fastEmbedR::nn(x, k = k, backend = backend, n_threads = n_threads)
    if (identical(method, "fastEmbedR_cuda_cuvs_nndescent")) {
      save_cuvs_knn(obj, dataset, out_dir)
    }
    return(obj)
  }

  switch(
    method,
    Rnanoflann_standard = {
      if (!available_pkg("Rnanoflann")) stop("Rnanoflann unavailable")
      out <- Rnanoflann::nn(x, x, k + 1L, parallel = TRUE, cores = n_threads, sorted = TRUE)
      keep <- drop_self_if_first(out$indices, out$distances, k)
      list(indices = keep$indices, distances = keep$distances)
    },
    RANN_kd = {
      if (!available_pkg("RANN")) stop("RANN unavailable")
      out <- RANN::nn2(x, x, k = k + 1L, treetype = "kd")
      keep <- drop_self_if_first(out$nn.idx, out$nn.dists, k)
      list(indices = keep$indices, distances = keep$distances)
    },
    RANN_bd = {
      if (!available_pkg("RANN")) stop("RANN unavailable")
      out <- RANN::nn2(x, x, k = k + 1L, treetype = "bd")
      keep <- drop_self_if_first(out$nn.idx, out$nn.dists, k)
      list(indices = keep$indices, distances = keep$distances)
    },
    rnndescent_rpf = {
      if (!available_pkg("rnndescent")) stop("rnndescent unavailable")
      rnndescent::rpf_knn(x, k = k, n_threads = n_threads, include_self = FALSE, progress = "none")
    },
    rnndescent_rnnd = {
      if (!available_pkg("rnndescent")) stop("rnndescent unavailable")
      rnndescent::rnnd_knn(x, k = k, n_threads = n_threads, progress = "none")
    },
    rnndescent_nnd = {
      if (!available_pkg("rnndescent")) stop("rnndescent unavailable")
      rnndescent::nnd_knn(x, k = k, n_threads = n_threads, progress = "none")
    },
    rnndescent_bruteforce = {
      if (!available_pkg("rnndescent")) stop("rnndescent unavailable")
      rnndescent::brute_force_knn(x, k = k, n_threads = n_threads)
    },
    RcppHNSW_hnsw = {
      if (!available_pkg("RcppHNSW")) stop("RcppHNSW unavailable")
      RcppHNSW::hnsw_knn(x, k = k, distance = "euclidean", M = 16, ef_construction = 200, ef = max(50, 3 * k), n_threads = n_threads, progress = "none")
    },
    RcppAnnoy_euclidean = annoy_knn(x, k),
    BiocNeighbors_vptree = {
      if (!available_pkg("BiocNeighbors")) stop("BiocNeighbors unavailable")
      BiocNeighbors::findKNN(x, k = k, BNPARAM = BiocNeighbors::VptreeParam(distance = "Euclidean"), num.threads = n_threads)
    },
    BiocNeighbors_hnsw = {
      if (!available_pkg("BiocNeighbors")) stop("BiocNeighbors unavailable")
      BiocNeighbors::findKNN(x, k = k, BNPARAM = BiocNeighbors::HnswParam(distance = "Euclidean", nlinks = 16, ef.construction = 200, ef.search = max(50, 3 * k)), num.threads = n_threads)
    },
    BiocNeighbors_annoy = {
      if (!available_pkg("BiocNeighbors")) stop("BiocNeighbors unavailable")
      BiocNeighbors::findKNN(x, k = k, BNPARAM = BiocNeighbors::AnnoyParam(distance = "Euclidean", ntrees = 50), num.threads = n_threads)
    },
    uwot_similarity_graph_fnn = {
      if (!available_pkg("uwot")) stop("uwot unavailable")
      uwot::similarity_graph(x, n_neighbors = k, nn_method = "fnn", n_threads = n_threads, verbose = FALSE)
    },
    uwot_similarity_graph_annoy = {
      if (!available_pkg("uwot")) stop("uwot unavailable")
      uwot::similarity_graph(x, n_neighbors = k, nn_method = "annoy", n_threads = n_threads, verbose = FALSE)
    },
    uwot_similarity_graph_hnsw = {
      if (!available_pkg("uwot")) stop("uwot unavailable")
      uwot::similarity_graph(x, n_neighbors = k, nn_method = "hnsw", n_threads = n_threads, verbose = FALSE)
    },
    uwot_similarity_graph_nndescent = {
      if (!available_pkg("uwot")) stop("uwot unavailable")
      uwot::similarity_graph(x, n_neighbors = k, nn_method = "nndescent", n_threads = n_threads, verbose = FALSE)
    },
    umap_umap_knn_from_cuvs = {
      if (!available_pkg("umap")) stop("umap unavailable")
      if (!available_pkg("fastEmbedR")) stop("fastEmbedR unavailable")
      knn <- fastEmbedR::nn(x, k = k, backend = "cuda_cuvs_nndescent", n_threads = n_threads)
      sx <- standardize_knn(knn)
      umap::umap.knn(sx$indices, sx$distances)
    },
    Rtsne_neighbors = {
      stop("Rtsne::Rtsne_neighbors consumes precomputed neighbours and optimizes t-SNE; it is not a standalone KNN search method.")
    },
    stop("Unknown method: ", method)
  )
}

method_table <- function() {
  data.frame(
    method = c(
      "fastEmbedR_cpu_exact",
      "fastEmbedR_rcpphnsw",
      "fastEmbedR_faiss_flat_l2",
      "fastEmbedR_faiss_flat_ip",
      "fastEmbedR_faiss_ivf",
      "fastEmbedR_faiss_ivfpq",
      "fastEmbedR_faiss_hnsw",
      "fastEmbedR_faiss_nsg",
      "fastEmbedR_faiss_nndescent",
      "fastEmbedR_cuda_exact",
      "fastEmbedR_cuda_ivf",
      "fastEmbedR_cuda_cuvs_bruteforce",
      "fastEmbedR_cuda_cuvs_cagra",
      "fastEmbedR_cuda_cuvs_nndescent",
      "Rnanoflann_standard",
      "RANN_kd",
      "RANN_bd",
      "rnndescent_rpf",
      "rnndescent_rnnd",
      "rnndescent_nnd",
      "rnndescent_bruteforce",
      "RcppHNSW_hnsw",
      "RcppAnnoy_euclidean",
      "BiocNeighbors_vptree",
      "BiocNeighbors_hnsw",
      "BiocNeighbors_annoy",
      "uwot_similarity_graph_fnn",
      "uwot_similarity_graph_annoy",
      "uwot_similarity_graph_hnsw",
      "uwot_similarity_graph_nndescent",
      "umap_umap_knn_from_cuvs",
      "Rtsne_neighbors"
    ),
    implementation = c(
      rep("fastEmbedR", 14),
      "Rnanoflann", "RANN", "RANN",
      "rnndescent", "rnndescent", "rnndescent", "rnndescent",
      "RcppHNSW", "RcppAnnoy",
      "BiocNeighbors", "BiocNeighbors", "BiocNeighbors",
      "uwot", "uwot", "uwot", "uwot",
      "umap", "Rtsne"
    ),
    backend = c(
      "CPU", "CPU", "CPU", "CPU", "CPU", "CPU", "CPU", "CPU", "CPU",
      "CUDA", "CUDA", "CUDA", "CUDA", "CUDA",
      rep("CPU", 18)
    ),
    kind = c(
      rep("knn_search", 30),
      "knn_consumer",
      "not_applicable"
    ),
    stringsAsFactors = FALSE
  )
}

if (worker) {
  dataset <- args$dataset
  data_path <- args$data_path
  method <- args$method
  result_path <- args$result_path
  dir.create(dirname(result_path), recursive = TRUE, showWarnings = FALSE)
  meta <- method_table()
  mm <- meta[match(method, meta$method), , drop = FALSE]
  started_total <- proc.time()[["elapsed"]]
  row <- data.frame(
    dataset = dataset,
    method = method,
    implementation = mm$implementation %||% NA_character_,
    backend = mm$backend %||% NA_character_,
    kind = mm$kind %||% NA_character_,
    n = NA_integer_,
    p = NA_integer_,
    k = k,
    n_threads = n_threads,
    status = "failed",
    time_sec = NA_real_,
    load_sec = NA_real_,
    peak_rss_gb = NA_real_,
    output_rows = NA_integer_,
    output_cols = NA_integer_,
    error = "",
    stringsAsFactors = FALSE
  )
  tryCatch({
    if (identical(mm$kind, "not_applicable")) {
      row$status <- "not_applicable"
      row$error <- "Rtsne::Rtsne_neighbors is not a standalone KNN search method."
      write_csv_one(result_path, row)
      quit(status = 0L)
    }
    load_start <- proc.time()[["elapsed"]]
    ds <- load_dataset(dataset, data_path)
    x <- ds$data
    row$n <- nrow(x)
    row$p <- ncol(x)
    row$load_sec <- proc.time()[["elapsed"]] - load_start
    gc()
    start <- proc.time()[["elapsed"]]
    obj <- run_method(method, x, k, n_threads, dataset, out_dir)
    row$time_sec <- proc.time()[["elapsed"]] - start
    sx <- standardize_knn(obj)
    if (!is.null(sx$indices)) {
      row$output_rows <- nrow(sx$indices)
      row$output_cols <- ncol(sx$indices)
    }
    row$status <- "success"
    row$peak_rss_gb <- read_peak_rss_gb()
  }, error = function(e) {
    row$status <- "failed"
    row$error <- conditionMessage(e)
    row$time_sec <- proc.time()[["elapsed"]] - started_total
    row$peak_rss_gb <- read_peak_rss_gb()
  })
  write_csv_one(result_path, row)
  quit(status = 0L)
}

manifest_path <- file.path(data_root, "dataset_manifest.csv")
if (!file.exists(manifest_path)) stop("Missing dataset manifest: ", manifest_path)
manifest <- read.csv(manifest_path, stringsAsFactors = FALSE)
manifest$path <- file.path(data_root, manifest$dataset, paste0(manifest$dataset, ".RData"))

datasets <- manifest[, c("dataset", "path", "n", "p")]
datasets <- rbind(
  datasets,
  data.frame(dataset = "SimulatedUniform2D", path = "SIMULATED", n = 500000L, p = 2L)
)
if (!is.null(args$datasets)) {
  wanted <- strsplit(args$datasets, ",", fixed = TRUE)[[1L]]
  datasets <- datasets[datasets$dataset %in% wanted, , drop = FALSE]
}

methods <- method_table()
if (!is.null(args$methods)) {
  wanted_methods <- strsplit(args$methods, ",", fixed = TRUE)[[1L]]
  methods <- methods[methods$method %in% wanted_methods, , drop = FALSE]
}

dir.create(file.path(out_dir, "worker_results"), recursive = TRUE, showWarnings = FALSE)

utils::write.csv(datasets, file.path(out_dir, "benchmark1_datasets.csv"), row.names = FALSE)
utils::write.csv(methods, file.path(out_dir, "benchmark1_methods.csv"), row.names = FALSE)

cmdline <- commandArgs(FALSE)
file_arg <- grep("^--file=", cmdline, value = TRUE)
script <- if (length(file_arg)) sub("^--file=", "", file_arg[[1L]]) else "tools/benchmark1_nn_speed.R"
if (!file.exists(script)) {
  script <- normalizePath("tools/benchmark1_nn_speed.R", mustWork = TRUE)
} else {
  script <- normalizePath(script, mustWork = TRUE)
}

results <- list()
job_id <- 0L
for (di in seq_len(nrow(datasets))) {
  for (mi in seq_len(nrow(methods))) {
    job_id <- job_id + 1L
    dataset <- datasets$dataset[[di]]
    method <- methods$method[[mi]]
    result_path <- file.path(out_dir, "worker_results", sprintf("%03d_%s__%s.csv", job_id, dataset, method))
    if (file.exists(result_path)) {
      log_msg("Skipping existing %s / %s", dataset, method)
      next
    }
    log_msg("[%03d/%03d] %s / %s", job_id, nrow(datasets) * nrow(methods), dataset, method)
    cmd_args <- c(
      as.character(timeout_sec),
      "Rscript",
      script,
      "--worker=TRUE",
      paste0("--dataset=", dataset),
      paste0("--data_path=", datasets$path[[di]]),
      paste0("--method=", method),
      paste0("--result_path=", result_path),
      paste0("--out_dir=", out_dir),
      paste0("--k=", k),
      paste0("--threads=", n_threads)
    )
    status <- system2("timeout", cmd_args, stdout = file.path(out_dir, "benchmark1_worker_stdout.log"), stderr = file.path(out_dir, "benchmark1_worker_stderr.log"))
    if (!file.exists(result_path)) {
      timeout_row <- data.frame(
        dataset = dataset,
        method = method,
        implementation = methods$implementation[[mi]],
        backend = methods$backend[[mi]],
        kind = methods$kind[[mi]],
        n = datasets$n[[di]],
        p = datasets$p[[di]],
        k = k,
        n_threads = n_threads,
        status = if (identical(status, 124L)) "timeout" else "failed",
        time_sec = if (identical(status, 124L)) timeout_sec else NA_real_,
        load_sec = NA_real_,
        peak_rss_gb = NA_real_,
        output_rows = NA_integer_,
        output_cols = NA_integer_,
        error = paste("worker did not produce result; exit status", status),
        stringsAsFactors = FALSE
      )
      write_csv_one(result_path, timeout_row)
    }
  }
}

files <- list.files(file.path(out_dir, "worker_results"), pattern = "[.]csv$", full.names = TRUE)
results <- do.call(rbind, lapply(files, read.csv, stringsAsFactors = FALSE))
results <- results[order(results$dataset, results$backend, results$implementation, results$method), ]
utils::write.csv(results, file.path(out_dir, "benchmark1_nn_speed_results.csv"), row.names = FALSE)

success <- results[results$status == "success" & results$kind == "knn_search", , drop = FALSE]
best <- success[order(success$dataset, success$time_sec), ]
best <- best[!duplicated(best$dataset), ]
utils::write.csv(best, file.path(out_dir, "benchmark1_best_by_dataset.csv"), row.names = FALSE)

png(file.path(out_dir, "benchmark1_nn_speed_barplot.png"), width = 2200, height = 1400, res = 160)
op <- par(mar = c(12, 5, 4, 2), mfrow = c(ceiling(length(unique(results$dataset)) / 2), 2))
for (dataset in unique(results$dataset)) {
  sub <- results[results$dataset == dataset & results$kind != "not_applicable", , drop = FALSE]
  sub$plot_time <- ifelse(sub$status == "success", sub$time_sec, timeout_sec)
  sub <- sub[order(sub$plot_time), ]
  cols <- ifelse(sub$status == "success", ifelse(sub$backend == "CUDA", "#2b8cbe", "#7bccc4"), "#d95f0e")
  barplot(
    sub$plot_time,
    names.arg = sub$method,
    las = 2,
    col = cols,
    main = dataset,
    ylab = "seconds (timeouts shown at cap)",
    cex.names = 0.45
  )
  legend("topright", fill = c("#7bccc4", "#2b8cbe", "#d95f0e"), legend = c("CPU success", "CUDA success", "failed/timeout"), cex = 0.7, bty = "n")
}
par(op)
dev.off()

materials <- c(
  "# BENCHMARK #1 Materials and Methods",
  "",
  "Benchmark #1 measures nearest-neighbour construction speed across fastEmbedR native backends, CUDA/cuVS backends, FAISS CPU backends, and external R package implementations.",
  "",
  paste0("Datasets were read from `", data_root, "`. The manifest datasets were MNIST, FashionMNIST, USPS, COIL20, MetRef, and TabulaMuris. A simulated reference dataset was generated as `matrix(runif(1000000), ncol = 2)` with columns `x` and `y`, giving 500,000 observations and 2 variables."),
  paste0("All methods used k = ", k, ". CPU methods were run with n_threads/cores = ", n_threads, " when the package exposed a thread argument. Each dataset-method pair was executed in a separate R process with GNU `timeout` set to ", timeout_sec, " seconds."),
  "The fastEmbedR CUDA/cuVS NN-descent output was saved for every dataset where the method completed successfully.",
  "",
  "fastEmbedR methods tested: exact CPU, RcppHNSW wrapper, FAISS Flat L2, FAISS Flat IP, FAISS IVF, FAISS IVFPQ, FAISS HNSW, FAISS NSG, FAISS NNDescent, native CUDA exact, native CUDA IVF, cuVS brute force, cuVS CAGRA, and cuVS NN-descent.",
  "External R package methods tested: Rnanoflann, RANN kd-tree and bd-tree, rnndescent RPF/RNND/NND/brute-force, RcppHNSW, RcppAnnoy, BiocNeighbors VP-tree/HNSW/Annoy, and uwot::similarity_graph with nn_method = fnn, annoy, hnsw, and nndescent.",
  "umap::umap.knn was included as a precomputed-neighbour consumer test, not as a standalone KNN search algorithm. Rtsne::Rtsne_neighbors was marked not applicable because it consumes precomputed neighbours and optimizes t-SNE rather than exporting a standalone KNN search.",
  "",
  "The benchmark records elapsed method time, load/conversion time, peak resident memory when available from `/proc/self/status`, output dimensions where an index matrix is returned, status, and error messages."
)
writeLines(materials, file.path(out_dir, "BENCHMARK1_MATERIALS_AND_METHODS.md"))

summary_lines <- c(
  "# BENCHMARK #1 Results Summary",
  "",
  paste0("Run directory: `", out_dir, "`"),
  "",
  "## Best Successful KNN Search Per Dataset",
  "",
  paste(capture.output(print(best[, c("dataset", "method", "implementation", "backend", "time_sec", "status")], row.names = FALSE)), collapse = "\n"),
  "",
  "## Comments",
  "",
  "This benchmark separates pure KNN search methods from graph/consumer functions. The fastest method can differ by dataset shape: low-dimensional simulated data favours tree/grid-like methods, while high-dimensional image matrices favour approximate graph or GPU methods. Exact brute-force methods are included as references but are expected to time out or be uncompetitive on the largest datasets. cuVS NN-descent outputs are saved to allow later embedding benchmarks to reuse the same neighbour graph rather than recomputing KNN."
)
writeLines(summary_lines, file.path(out_dir, "BENCHMARK1_RESULTS_SUMMARY.md"))

log_msg("DONE: %s", out_dir)
