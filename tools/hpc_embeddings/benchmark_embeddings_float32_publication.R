#!/usr/bin/env Rscript

parse_args <- function(args) {
  out <- list()
  for (arg in args) {
    if (!startsWith(arg, "--")) next
    kv <- strsplit(sub("^--", "", arg), "=", fixed = TRUE)[[1L]]
    out[[kv[[1L]]]] <- if (length(kv) > 1L) paste(kv[-1L], collapse = "=") else TRUE
  }
  out
}

`%||%` <- function(x, y) {
  if (is.null(x) || length(x) == 0L || (length(x) == 1L && is.na(x))) y else x
}

args <- parse_args(commandArgs(trailingOnly = TRUE))

script_path <- normalizePath(
  sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)[1L] %||% ""),
  mustWork = FALSE
)
if (!nzchar(script_path) || !file.exists(script_path)) {
  script_path <- normalizePath(args$script %||% "benchmark_embeddings_float32_publication.R", mustWork = FALSE)
}

bool_arg <- function(name, default = FALSE) {
  value <- args[[name]]
  if (is.null(value)) return(default)
  isTRUE(as.logical(value))
}

int_arg <- function(name, default) {
  value <- suppressWarnings(as.integer(args[[name]] %||% default))
  if (length(value) != 1L || is.na(value)) as.integer(default) else value
}

num_arg <- function(name, default) {
  value <- suppressWarnings(as.numeric(args[[name]] %||% default))
  if (length(value) != 1L || is.na(value)) as.numeric(default) else value
}

csv_arg <- function(name, default) {
  value <- args[[name]] %||% default
  x <- trimws(strsplit(value, ",", fixed = TRUE)[[1L]])
  x[nzchar(x)]
}

base_dir <- normalizePath(args$base_dir %||% "/scratch/firenze/NN", mustWork = FALSE)
data_root <- normalizePath(args$data_root %||% file.path(base_dir, "Data"), mustWork = FALSE)
out_dir <- normalizePath(args$out_dir %||% file.path(base_dir, "benchmark_embeddings_float32_publication"), mustWork = FALSE)
threads <- int_arg("threads", 12L)
timeout <- int_arg("timeout", 10800L)
seed <- int_arg("seed", 4L)
k <- int_arg("k", 30L)
perplexity <- num_arg("perplexity", 15)
backend_group <- args$backend_group %||% "cpu"
force <- bool_arg("force", FALSE)
worker <- bool_arg("worker", FALSE)

datasets <- csv_arg(
  "datasets",
  "COIL20,USPS,FashionMNIST,FlowRepository_FR-FCM-ZYRM_files,flow18,MNIST,imagenet,MetRef,mass41,TabulaMuris"
)

default_methods <- if (identical(backend_group, "cuda")) {
  "fastEmbedR_opentsne_cuda,fastEmbedR_umap_cuda_fuzzy,fastEmbedR_umap_cuda_binary"
} else {
  "fastEmbedR_opentsne_cpu,fastEmbedR_umap_cpu_fuzzy,fastEmbedR_umap_cpu_binary,Rtsne_full,KlugerLab_FItSNE,umap_package,uwot_default,uwot_fast_sgd"
}
methods <- csv_arg("methods", default_methods)

dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(file.path(out_dir, "logs"), recursive = TRUE, showWarnings = FALSE)
dir.create(file.path(out_dir, "layouts"), recursive = TRUE, showWarnings = FALSE)
dir.create(file.path(out_dir, "plots"), recursive = TRUE, showWarnings = FALSE)
dir.create(file.path(out_dir, "worker_results"), recursive = TRUE, showWarnings = FALSE)

Sys.setenv(
  OMP_NUM_THREADS = as.character(threads),
  OPENBLAS_NUM_THREADS = as.character(threads),
  MKL_NUM_THREADS = as.character(threads),
  VECLIB_MAXIMUM_THREADS = as.character(threads),
  RCPP_PARALLEL_NUM_THREADS = as.character(threads)
)

log_file <- file.path(out_dir, "benchmark.log")
log_msg <- function(...) {
  line <- paste0(format(Sys.time(), "%Y-%m-%d %H:%M:%S"), " ", sprintf(...))
  cat(line, "\n")
  cat(line, "\n", file = log_file, append = TRUE)
  flush.console()
}

find_source_rdata <- function(dataset) {
  folder <- file.path(data_root, dataset)
  hits <- list.files(folder, pattern = "\\.[Rr][Dd]ata$", full.names = TRUE)
  hits <- hits[!grepl(
    "float32|_nn|pca|manifest|summary|backup|faissR|reference|worker|benchmark",
    basename(hits),
    ignore.case = TRUE
  )]
  if (!length(hits)) return(NA_character_)
  exact <- hits[tolower(tools::file_path_sans_ext(basename(hits))) == tolower(dataset)]
  if (length(exact)) return(exact[[1L]])
  hits <- hits[order(nchar(basename(hits)), basename(hits))]
  hits[[1L]]
}

find_float_rdata <- function(dataset) {
  folder <- file.path(data_root, dataset)
  hits <- list.files(folder, pattern = "float32.*\\.[Rr][Dd]ata$", full.names = TRUE, ignore.case = TRUE)
  if (!length(hits)) stop("No float32 RData found for ", dataset, call. = FALSE)
  hits[[1L]]
}

pick_dataset_object <- function(path) {
  env <- new.env(parent = emptyenv())
  object_names <- load(path, envir = env)
  objects <- mget(object_names, env, inherits = FALSE)
  for (nm in names(objects)) {
    obj <- objects[[nm]]
    if (is.list(obj) && !is.null(obj$data)) {
      return(list(data = obj$data, labels = obj$labels %||% NULL, object_name = nm))
    }
  }
  for (nm in names(objects)) {
    obj <- objects[[nm]]
    if (is.matrix(obj) || is.data.frame(obj) || inherits(obj, "Matrix") || inherits(obj, "float32")) {
      labels <- NULL
      for (candidate in c("labels", "label", "Y", "y", "classes", "class")) {
        if (exists(candidate, envir = env, inherits = FALSE)) {
          lab <- get(candidate, envir = env, inherits = FALSE)
          if (length(lab) == nrow(obj)) labels <- lab
        }
      }
      return(list(data = obj, labels = labels, object_name = nm))
    }
  }
  stop("Could not find a data matrix/list in ", path, call. = FALSE)
}

as_double_matrix <- function(x) {
  if (inherits(x, "float32")) {
    if (!requireNamespace("float", quietly = TRUE)) stop("float package is required.", call. = FALSE)
    x <- float::dbl(x)
  }
  if (inherits(x, "Matrix")) x <- as.matrix(x)
  if (is.data.frame(x)) x <- as.matrix(x)
  if (!is.matrix(x)) x <- as.matrix(x)
  storage.mode(x) <- "double"
  x
}

as_fastembedr_float_input <- function(x) {
  if (inherits(x, "float32")) return(x)
  if (!requireNamespace("float", quietly = TRUE)) {
    stop("The float package is required for fastEmbedR float32 benchmark input.", call. = FALSE)
  }
  float::fl(as_double_matrix(x))
}

layout_matrix <- function(x) {
  if (is.list(x) && !is.null(x$layout)) x <- x$layout
  if (is.list(x) && !is.null(x$Y)) x <- x$Y
  if (inherits(x, "float32")) {
    if (!requireNamespace("float", quietly = TRUE)) stop("float package required to coerce layout.", call. = FALSE)
    x <- float::dbl(x)
  }
  y <- as.matrix(x)
  y[, 1:2, drop = FALSE]
}

sample_for_metrics <- function(n, size, seed) {
  if (n <= size) return(seq_len(n))
  set.seed(seed)
  sort(sample.int(n, size))
}

score_layout <- function(x_standard, layout, labels) {
  out <- list(trust = NA_real_, knn_preservation = NA_real_, label_acc = NA_real_)
  if (!requireNamespace("fastEmbedR", quietly = TRUE)) return(out)
  rows <- sample_for_metrics(nrow(layout), min(5000L, nrow(layout)), seed + 19L)
  score <- tryCatch(
    fastEmbedR::evaluate_embedding(
      x_standard[rows, , drop = FALSE],
      layout[rows, , drop = FALSE],
      labels = if (is.null(labels)) NULL else labels[rows],
      k = c(15L, 30L, 50L),
      sample_size_for_global_metrics = min(3000L, length(rows)),
      sample_size_for_local_metrics = min(3000L, length(rows)),
      seed = seed,
      n_threads = threads,
      dataset = args$dataset %||% NA_character_
    ),
    error = function(e) e
  )
  if (!inherits(score, "error")) {
    out$trust <- as.numeric(score$trustworthiness %||% NA_real_)
    out$knn_preservation <- as.numeric(score$knn_preservation %||% score$knn_preservation_15 %||% NA_real_)
    out$label_acc <- as.numeric(score$nn_accuracy %||% score$label_knn_accuracy %||% NA_real_)
  }
  out
}

plot_layout <- function(layout, labels, path, title) {
  png(path, width = 1800, height = 1400, res = 180)
  on.exit(dev.off(), add = TRUE)
  par(mar = c(2, 2, 3, 1))
  if (is.null(labels)) {
    plot(layout[, 1], layout[, 2], pch = 16, cex = 0.28, col = "#1f77b4",
         axes = FALSE, xlab = "", ylab = "", main = title)
  } else {
    labels <- as.factor(labels)
    pal <- grDevices::hcl.colors(nlevels(labels), "Dark 3")
    plot(layout[, 1], layout[, 2], pch = 16, cex = 0.28, col = pal[as.integer(labels)],
         axes = FALSE, xlab = "", ylab = "", main = title)
  }
  box(col = "grey70")
}

fast_tsne_path <- function() {
  candidates <- c(
    Sys.getenv("FASTEMBEDR_FAST_TSNE_PATH", ""),
    Sys.getenv("FAST_TSNE_PATH", ""),
    "/opt/fit-sne/bin/fast_tsne",
    "/mnt/sata_ssd/FIt-SNE/bin/fast_tsne",
    file.path(Sys.getenv("HOME"), ".local", "bin", "fast_tsne"),
    Sys.which("fast_tsne")
  )
  candidates <- candidates[nzchar(candidates)]
  candidates <- candidates[file.exists(candidates) & file.access(candidates, 1L) == 0L]
  if (length(candidates)) normalizePath(candidates[[1L]], mustWork = FALSE) else ""
}

fitsne_wrapper <- function() {
  for (package in c("fftRtsne", "Spectre")) {
    if (requireNamespace(package, quietly = TRUE) &&
        exists("fftRtsne", envir = asNamespace(package), inherits = FALSE)) {
      return(list(name = package, fun = get("fftRtsne", envir = asNamespace(package), inherits = FALSE)))
    }
  }
  wrappers <- c(
    Sys.getenv("FASTEMBEDR_FAST_TSNE_R", ""),
    Sys.getenv("FAST_TSNE_R", ""),
    "/opt/fit-sne/bin/fast_tsne.R",
    "/mnt/sata_ssd/FIt-SNE/fast_tsne.R",
    "/mnt/sata_ssd/FIt-SNE/bin/fast_tsne.R"
  )
  wrappers <- wrappers[nzchar(wrappers) & file.exists(wrappers)]
  if (!length(wrappers)) return(NULL)
  env <- new.env(parent = .GlobalEnv)
  old_wd <- getwd()
  on.exit(setwd(old_wd), add = TRUE)
  setwd(dirname(wrappers[[1L]]))
  source(wrappers[[1L]], local = env, chdir = FALSE)
  setwd(old_wd)
  if (!exists("fftRtsne", envir = env, inherits = FALSE)) return(NULL)
  fun <- get("fftRtsne", envir = env, inherits = FALSE)
  list(name = "KlugerLab_FItSNE_source", fun = fun, exe = fast_tsne_path())
}

run_fitsne <- function(x, y_init = NULL) {
  wrapper <- fitsne_wrapper()
  exe <- fast_tsne_path()
  if (!nzchar(exe)) stop("fast_tsne executable not found.", call. = FALSE)
  if (is.null(wrapper)) stop("FIt-SNE R wrapper not found.", call. = FALSE)
  formals_names <- names(formals(wrapper$fun))
  call_args <- list(
    X = x,
    dims = 2L,
    perplexity = perplexity,
    max_iter = 750L,
    rand_seed = seed,
    theta = 0.5,
    nthreads = threads,
    fast_tsne_path = exe,
    verbose = FALSE
  )
  if (!is.null(y_init)) {
    if ("Y_init" %in% formals_names) call_args$Y_init <- y_init
    if ("initial_config" %in% formals_names) call_args$initial_config <- y_init
    if ("init" %in% formals_names) call_args$init <- y_init
    if ("initialization" %in% formals_names) call_args$initialization <- y_init
  }
  call_args <- call_args[names(call_args) %in% formals_names]
  do.call(wrapper$fun, call_args)
}

method_backend <- function(method) {
  if (grepl("_cuda$", method) || grepl("_cuda_", method)) "cuda" else "cpu"
}

cuda_status_message <- function() {
  parts <- c("CUDA backend unavailable.")
  if (requireNamespace("fastEmbedR", quietly = TRUE)) {
    parts <- c(
      parts,
      paste0(
        " fastEmbedR version=",
        paste(capture.output(print(try(utils::packageVersion("fastEmbedR"), silent = TRUE))), collapse = " ")
      ),
      paste0(
        " fastEmbedR exports cuda_available=",
        "cuda_available" %in% getNamespaceExports("fastEmbedR")
      )
    )
  } else {
    parts <- c(parts, " fastEmbedR is not installed.")
  }
  if (requireNamespace("faissR", quietly = TRUE)) {
    parts <- c(
      parts,
      paste0(
        " faissR::backend_info()=",
        paste(capture.output(print(try(faissR::backend_info(), silent = TRUE))), collapse = " ")
      ),
      paste0(
        " faissR::cuda_available()=",
        paste(capture.output(print(try(faissR::cuda_available(), silent = TRUE))), collapse = " ")
      ),
      paste0(
        " faissR::cuvs_available()=",
        paste(capture.output(print(try(faissR::cuvs_available(), silent = TRUE))), collapse = " ")
      )
    )
  } else {
    parts <- c(parts, " faissR is not installed.")
  }
  paste(parts, collapse = "")
}

cuda_ready_for_benchmark <- function() {
  if (!requireNamespace("fastEmbedR", quietly = TRUE)) return(FALSE)
  if (!requireNamespace("faissR", quietly = TRUE)) return(FALSE)
  isTRUE(tryCatch(faissR::cuda_available(), error = function(e) FALSE))
}

run_embedding_method <- function(method, dataset, x_fast, x_ref, labels) {
  set.seed(seed)
  if (method == "fastEmbedR_opentsne_cpu") {
    return(fastEmbedR::opentsne(x_fast, perplexity = perplexity, backend = "cpu",
                                n_threads = threads, seed = seed))
  }
  if (method == "fastEmbedR_opentsne_cuda") {
    return(fastEmbedR::opentsne(x_fast, perplexity = perplexity, backend = "cuda",
                                n_threads = threads, seed = seed))
  }
  if (method == "fastEmbedR_umap_cpu_fuzzy") {
    return(fastEmbedR::umap(x_fast, n_neighbors = k, backend = "cpu",
                            graph_mode = "fuzzy", n_threads = threads, seed = seed))
  }
  if (method == "fastEmbedR_umap_cpu_binary") {
    return(fastEmbedR::umap(x_fast, n_neighbors = k, backend = "cpu",
                            graph_mode = "binary", n_threads = threads, seed = seed))
  }
  if (method == "fastEmbedR_umap_cuda_fuzzy") {
    return(fastEmbedR::umap(x_fast, n_neighbors = k, backend = "cuda",
                            graph_mode = "fuzzy", n_threads = threads, seed = seed))
  }
  if (method == "fastEmbedR_umap_cuda_binary") {
    return(fastEmbedR::umap(x_fast, n_neighbors = k, backend = "cuda",
                            graph_mode = "binary", n_threads = threads, seed = seed))
  }
  if (method == "Rtsne_full") {
    if (is.null(x_ref)) stop("Standard R dataset is required for Rtsne_full.", call. = FALSE)
    if (!requireNamespace("Rtsne", quietly = TRUE)) stop("Rtsne is not installed.", call. = FALSE)
    return(Rtsne::Rtsne(x_ref, perplexity = perplexity, check_duplicates = FALSE,
                        pca = TRUE, num_threads = threads)$Y)
  }
  if (method == "KlugerLab_FItSNE") {
    if (is.null(x_ref)) stop("Standard R dataset is required for KlugerLab_FItSNE.", call. = FALSE)
    return(run_fitsne(x_ref))
  }
  if (method == "umap_package") {
    if (is.null(x_ref)) stop("Standard R dataset is required for umap_package.", call. = FALSE)
    if (!requireNamespace("umap", quietly = TRUE)) stop("umap is not installed.", call. = FALSE)
    cfg <- umap::umap.defaults
    cfg$n_neighbors <- k
    return(umap::umap(x_ref, config = cfg)$layout)
  }
  if (method == "uwot_default") {
    if (is.null(x_ref)) stop("Standard R dataset is required for uwot_default.", call. = FALSE)
    if (!requireNamespace("uwot", quietly = TRUE)) stop("uwot is not installed.", call. = FALSE)
    return(uwot::umap(x_ref, n_neighbors = k, n_threads = threads,
                      n_sgd_threads = 1, fast_sgd = FALSE, verbose = FALSE))
  }
  if (method == "uwot_fast_sgd") {
    if (is.null(x_ref)) stop("Standard R dataset is required for uwot_fast_sgd.", call. = FALSE)
    if (!requireNamespace("uwot", quietly = TRUE)) stop("uwot is not installed.", call. = FALSE)
    return(uwot::umap(x_ref, n_neighbors = k, n_threads = threads,
                      n_sgd_threads = threads, fast_sgd = TRUE, verbose = FALSE))
  }
  stop("Unknown method: ", method, call. = FALSE)
}

worker_main <- function() {
  dataset <- args$dataset
  method <- args$method
  worker_out <- args$worker_out
  if (is.null(dataset) || is.null(method) || is.null(worker_out)) {
    stop("--dataset, --method, and --worker_out are required in worker mode.", call. = FALSE)
  }
  if (method_backend(method) == "cuda" && !cuda_ready_for_benchmark()) {
    stop(cuda_status_message(), call. = FALSE)
  }

  float_obj <- pick_dataset_object(find_float_rdata(dataset))
  standard_path <- find_source_rdata(dataset)
  standard <- if (is.na(standard_path)) NULL else pick_dataset_object(standard_path)
  labels <- if (is.null(standard)) float_obj$labels else (standard$labels %||% float_obj$labels)
  if (!is.null(labels)) labels <- as.factor(labels)
  # Reference packages must receive the standard R object loaded from the
  # dataset .RData file, not the float32 object and not a converted copy.
  x_ref <- if (is.null(standard)) NULL else standard$data
  x_fast <- as_fastembedr_float_input(float_obj$data)
  if (!is.null(x_ref) && nrow(x_ref) != nrow(x_fast)) {
    stop("Standard and float32 datasets have different number of rows for ", dataset, call. = FALSE)
  }
  x_score <- if (is.null(x_ref)) as_double_matrix(float_obj$data) else as_double_matrix(x_ref)

  gc()
  t <- system.time({
    fit <- run_embedding_method(method, dataset, x_fast, x_ref, labels)
  })
  layout <- layout_matrix(fit)
  elapsed <- unname(t[["elapsed"]])
  layout_file <- file.path(out_dir, "layouts", paste0(dataset, "_", method, "_seed", seed, ".rds"))
  plot_file <- file.path(out_dir, "plots", paste0(dataset, "_", method, "_seed", seed, ".png"))
  saveRDS(list(layout = layout, labels = labels, method = method, dataset = dataset), layout_file)
  plot_layout(layout, labels, plot_file, sprintf("%s %s %.2fs", dataset, method, elapsed))
  # Metrics need numeric matrix arithmetic; this conversion is only for scoring,
  # never for the reference package calls above.
  scores <- score_layout(x_score, layout, labels)

  result <- data.frame(
    dataset = dataset,
    method = method,
    backend = method_backend(method),
    status = "success",
    n = nrow(x_fast),
    p = ncol(x_fast),
    k = if (grepl("umap", method, ignore.case = TRUE)) k else NA_integer_,
    perplexity = if (grepl("tsne|Rtsne|FItSNE", method, ignore.case = TRUE)) perplexity else NA_real_,
    input_fastEmbedR = if (grepl("^fastEmbedR", method)) {
      "float32"
    } else {
      "standard_R_matrix"
    },
    elapsed_sec = elapsed,
    trust = scores$trust,
    knn_preservation = scores$knn_preservation,
    label_acc = scores$label_acc,
    max_rss_kb = NA_real_,
    max_rss_gb = NA_real_,
    layout_file = layout_file,
    plot_file = plot_file,
    error = NA_character_,
    stringsAsFactors = FALSE
  )
  write.csv(result, worker_out, row.names = FALSE)
}

parse_time_v <- function(path) {
  if (!file.exists(path)) return(list(max_rss_kb = NA_real_, exit_status = NA_integer_))
  txt <- readLines(path, warn = FALSE)
  rss_line <- grep("Maximum resident set size", txt, value = TRUE)
  rss <- NA_real_
  if (length(rss_line)) {
    rss <- suppressWarnings(as.numeric(sub(".*: *", "", rss_line[[length(rss_line)]])))
  }
  status_line <- grep("Exit status", txt, value = TRUE)
  exit_status <- NA_integer_
  if (length(status_line)) {
    exit_status <- suppressWarnings(as.integer(sub(".*: *", "", status_line[[length(status_line)]])))
  }
  list(max_rss_kb = rss, exit_status = exit_status)
}

write_combined_outputs <- function(results) {
  if (!length(results)) return(invisible(NULL))
  tab <- do.call(rbind, results)
  write.csv(tab, file.path(out_dir, "embedding_benchmark_results.csv"), row.names = FALSE)
  ok <- tab$status == "success" & is.finite(tab$elapsed_sec)
  if (any(ok)) {
    png(file.path(out_dir, "embedding_time_barplot.png"), width = 1800, height = 1100, res = 150)
    par(mar = c(11, 5, 3, 1))
    labs <- paste(tab$dataset[ok], tab$method[ok], sep = "\n")
    barplot(tab$elapsed_sec[ok], names.arg = labs, las = 2, cex.names = 0.55,
            ylab = "Seconds", main = "Embedding runtime")
    dev.off()
    mem_ok <- ok & is.finite(tab$max_rss_gb)
    if (any(mem_ok)) {
      png(file.path(out_dir, "embedding_memory_barplot.png"), width = 1800, height = 1100, res = 150)
      par(mar = c(11, 5, 3, 1))
      mem_labs <- paste(tab$dataset[mem_ok], tab$method[mem_ok], sep = "\n")
      barplot(tab$max_rss_gb[mem_ok], names.arg = mem_labs, las = 2, cex.names = 0.55,
              ylab = "Peak RSS (GB)", main = "Embedding peak memory")
      dev.off()
    }
  }
  invisible(tab)
}

if (worker) {
  tryCatch(worker_main(), error = function(e) {
    dataset <- args$dataset %||% NA_character_
    method <- args$method %||% NA_character_
    worker_out <- args$worker_out %||% file.path(out_dir, "worker_failed.csv")
    row <- data.frame(
      dataset = dataset,
      method = method,
      backend = method_backend(method),
      status = "failed",
      n = NA_integer_,
      p = NA_integer_,
      k = if (grepl("umap", method, ignore.case = TRUE)) k else NA_integer_,
      perplexity = if (grepl("tsne|Rtsne|FItSNE", method, ignore.case = TRUE)) perplexity else NA_real_,
      input_fastEmbedR = if (grepl("^fastEmbedR", method)) {
        "float32"
      } else {
        "standard_R_matrix"
      },
      elapsed_sec = NA_real_,
      trust = NA_real_,
      knn_preservation = NA_real_,
      label_acc = NA_real_,
      max_rss_kb = NA_real_,
      max_rss_gb = NA_real_,
      layout_file = NA_character_,
      plot_file = NA_character_,
      error = conditionMessage(e),
      stringsAsFactors = FALSE
    )
    write.csv(row, worker_out, row.names = FALSE)
    message(conditionMessage(e))
    quit(status = 1L)
  })
  quit(status = 0L)
}

main_results <- list()
timeout_bin <- Sys.getenv("TIMEOUT_BIN", Sys.which("timeout"))
time_bin <- Sys.getenv("TIME_V_BIN", "/usr/bin/time")
if (!file.exists(time_bin)) time_bin <- Sys.which("time")
if (nzchar(time_bin)) {
  has_time_v <- suppressWarnings(
    system2(time_bin, c("-v", "true"), stdout = FALSE, stderr = FALSE) == 0L
  )
  if (!isTRUE(has_time_v)) time_bin <- ""
}

log_msg("Starting benchmark: backend_group=%s threads=%d timeout=%d", backend_group, threads, timeout)
log_msg("Datasets: %s", paste(datasets, collapse = ","))
log_msg("Methods: %s", paste(methods, collapse = ","))

input_audit <- do.call(rbind, lapply(datasets, function(dataset) {
  float_path <- tryCatch(find_float_rdata(dataset), error = function(e) NA_character_)
  standard_path <- find_source_rdata(dataset)
  data.frame(
    dataset = dataset,
    float32_file = if (is.na(float_path)) NA_character_ else float_path,
    standard_rdata_file = if (is.na(standard_path)) NA_character_ else standard_path,
    has_float32 = !is.na(float_path),
    has_standard_rdata = !is.na(standard_path),
    stringsAsFactors = FALSE
  )
}))
write.csv(input_audit, file.path(out_dir, "dataset_input_audit.csv"), row.names = FALSE)
missing_standard <- input_audit$dataset[!input_audit$has_standard_rdata]
if (length(missing_standard)) {
  log_msg("Datasets missing standard .RData for reference methods: %s", paste(missing_standard, collapse = ","))
}

for (dataset in datasets) {
  for (method in methods) {
    worker_csv <- file.path(out_dir, "worker_results", paste0(dataset, "_", method, ".csv"))
    worker_log <- file.path(out_dir, "logs", paste0(dataset, "_", method, ".log"))
    time_log <- file.path(out_dir, "logs", paste0(dataset, "_", method, "_time.txt"))
    if (!force && file.exists(worker_csv)) {
      log_msg("%s/%s: existing worker result, reusing", dataset, method)
      row <- read.csv(worker_csv, stringsAsFactors = FALSE)
      main_results[[length(main_results) + 1L]] <- row
      write_combined_outputs(main_results)
      next
    }
    cmd <- c()
    if (nzchar(timeout_bin)) cmd <- c(cmd, timeout_bin, as.character(timeout))
    if (nzchar(time_bin)) cmd <- c(cmd, time_bin, "-v", "-o", time_log)
    cmd <- c(
      cmd,
      file.path(R.home("bin"), "Rscript"),
      script_path,
      "--worker=TRUE",
      paste0("--base_dir=", base_dir),
      paste0("--data_root=", data_root),
      paste0("--out_dir=", out_dir),
      paste0("--dataset=", dataset),
      paste0("--method=", method),
      paste0("--worker_out=", worker_csv),
      paste0("--threads=", threads),
      paste0("--timeout=", timeout),
      paste0("--seed=", seed),
      paste0("--k=", k),
      paste0("--perplexity=", perplexity)
    )
    log_msg("%s/%s: running", dataset, method)
    status <- system2(cmd[[1L]], args = cmd[-1L], stdout = worker_log, stderr = worker_log)
    time_info <- parse_time_v(time_log)
    if (file.exists(worker_csv)) {
      row <- read.csv(worker_csv, stringsAsFactors = FALSE)
    } else {
      row <- data.frame(
        dataset = dataset,
        method = method,
        backend = method_backend(method),
        status = if (identical(status, 124L)) "timeout" else "failed",
        n = NA_integer_, p = NA_integer_,
        k = if (grepl("umap", method, ignore.case = TRUE)) k else NA_integer_,
        perplexity = if (grepl("tsne|Rtsne|FItSNE", method, ignore.case = TRUE)) perplexity else NA_real_,
        input_fastEmbedR = if (grepl("^fastEmbedR", method)) "float32" else "standard_R_matrix",
        elapsed_sec = NA_real_, trust = NA_real_, knn_preservation = NA_real_,
        label_acc = NA_real_, max_rss_kb = NA_real_, max_rss_gb = NA_real_,
        layout_file = NA_character_, plot_file = NA_character_,
        error = paste("worker exited with status", status),
        stringsAsFactors = FALSE
      )
      write.csv(row, worker_csv, row.names = FALSE)
    }
    row$max_rss_kb <- time_info$max_rss_kb
    row$max_rss_gb <- time_info$max_rss_kb / 1024^2
    write.csv(row, worker_csv, row.names = FALSE)
    main_results[[length(main_results) + 1L]] <- row
    log_msg("%s/%s: %s sec=%s rss_gb=%s", dataset, method, row$status[1],
            format(row$elapsed_sec[1], digits = 4), format(row$max_rss_gb[1], digits = 4))
    write_combined_outputs(main_results)
  }
}

tab <- write_combined_outputs(main_results)
print(tab)
log_msg("DONE: %s", out_dir)
