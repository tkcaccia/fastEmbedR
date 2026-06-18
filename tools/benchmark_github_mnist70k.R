#!/usr/bin/env Rscript

arg_value <- function(name, default = NULL) {
  prefix <- paste0("--", name, "=")
  args <- commandArgs(trailingOnly = TRUE)
  hit <- args[startsWith(args, prefix)]
  if (!length(hit)) return(default)
  sub(prefix, "", hit[[length(hit)]], fixed = TRUE)
}

arg_flag <- function(name, default = FALSE) {
  value <- arg_value(name, NA_character_)
  if (is.na(value)) return(isTRUE(default))
  tolower(value) %in% c("1", "true", "yes", "y")
}

arg_int <- function(name, default) {
  value <- suppressWarnings(as.integer(arg_value(name, as.character(default))))
  if (length(value) != 1L || is.na(value)) default else value
}

timed <- function(expr) {
  gc()
  t <- system.time(value <- force(expr))
  list(value = value, sec = unname(t[["elapsed"]]))
}

first_nonempty <- function(x, default = NA_character_) {
  x <- x[nzchar(x)]
  if (length(x)) x[[1L]] else default
}

run_cmd <- function(cmd, args = character()) {
  out <- tryCatch(
    suppressWarnings(system2(cmd, args, stdout = TRUE, stderr = FALSE)),
    error = function(e) character()
  )
  first_nonempty(out)
}

machine_specs <- function(n_threads) {
  os <- Sys.info()
  mem_bytes <- suppressWarnings(as.numeric(run_cmd("sysctl", c("-n", "hw.memsize"))))
  if (!is.finite(mem_bytes) && file.exists("/proc/meminfo")) {
    meminfo <- readLines("/proc/meminfo", warn = FALSE)
    mem_total <- sub("^MemTotal:[[:space:]]*([0-9]+).*", "\\1", meminfo[grepl("^MemTotal:", meminfo)][1L])
    mem_bytes <- suppressWarnings(as.numeric(mem_total) * 1024)
  }
  total_ram_gb <- if (is.finite(mem_bytes)) round(mem_bytes / 1024^3, 2) else NA_real_
  cpu <- run_cmd("sysctl", c("-n", "machdep.cpu.brand_string"))
  if (is.na(cpu) && file.exists("/proc/cpuinfo")) {
    cpu_lines <- readLines("/proc/cpuinfo", warn = FALSE)
    model <- sub("^model name[[:space:]]*:[[:space:]]*", "", cpu_lines[grepl("^model name", cpu_lines)][1L])
    cpu <- first_nonempty(model)
  }
  gpu <- run_cmd("nvidia-smi", c("--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"))
  data.frame(
    run_timestamp_utc = format(Sys.time(), "%Y-%m-%d %H:%M:%S %Z", tz = "UTC"),
    machine = unname(os[["nodename"]]),
    system = paste(unname(os[["sysname"]]), unname(os[["release"]])),
    platform = R.version$platform,
    cpu = cpu,
    gpu = gpu,
    logical_cores = parallel::detectCores(logical = TRUE),
    total_ram_gb = total_ram_gb,
    r_version = as.character(getRversion()),
    fastEmbedR_version = as.character(utils::packageVersion("fastEmbedR")),
    faissR_version = if (requireNamespace("faissR", quietly = TRUE)) as.character(utils::packageVersion("faissR")) else NA_character_,
    uwot_version = if (requireNamespace("uwot", quietly = TRUE)) as.character(utils::packageVersion("uwot")) else NA_character_,
    Rtsne_version = if (requireNamespace("Rtsne", quietly = TRUE)) as.character(utils::packageVersion("Rtsne")) else NA_character_,
    requested_threads = n_threads,
    stringsAsFactors = FALSE
  )
}

write_machine_specs <- function(specs, out_dir) {
  csv_path <- file.path(out_dir, "machine-specs.csv")
  md_path <- file.path(out_dir, "machine-specs.md")
  utils::write.csv(specs, csv_path, row.names = FALSE)
  lines <- c(
    "# Machine Specification",
    "",
    sprintf("- Run timestamp: %s", specs$run_timestamp_utc),
    sprintf("- Machine: %s", specs$machine),
    sprintf("- System: %s", specs$system),
    sprintf("- Platform: %s", specs$platform),
    sprintf("- CPU: %s", specs$cpu),
    sprintf("- GPU: %s", specs$gpu),
    sprintf("- Logical cores: %s", specs$logical_cores),
    sprintf("- RAM: %s GB", specs$total_ram_gb),
    sprintf("- R: %s", specs$r_version),
    sprintf("- fastEmbedR: %s", specs$fastEmbedR_version),
    sprintf("- faissR: %s", specs$faissR_version),
    sprintf("- uwot: %s", specs$uwot_version),
    sprintf("- Rtsne: %s", specs$Rtsne_version),
    sprintf("- Requested benchmark threads: %s", specs$requested_threads)
  )
  writeLines(lines, md_path)
  invisible(list(csv = csv_path, md = md_path))
}

plot_time_barplot <- function(results, path) {
  ok <- results[results$status == "success" & is.finite(results$total_sec), , drop = FALSE]
  if (!nrow(ok)) return(invisible(FALSE))
  labels <- gsub("^fastEmbedR ", "", ok$method)
  labels <- gsub(" UMAP fast_sgd full$", "\nfast_sgd full", labels)
  labels <- gsub(" UMAP ", "\nUMAP ", labels)
  labels <- gsub(" openTSNE ", "\nopenTSNE ", labels)
  nn <- ifelse(is.finite(ok$nn_sec), ok$nn_sec, 0)
  embed <- ifelse(is.finite(ok$embed_sec), ok$embed_sec, ok$total_sec)
  other <- pmax(0, ok$total_sec - nn - embed)
  values <- rbind(
    `NN search` = nn,
    Embedding = embed,
    Other = other
  )
  png(path, width = 1700, height = 1050, res = 150)
  on.exit(dev.off(), add = TRUE)
  old <- par(no.readonly = TRUE)
  on.exit(par(old), add = TRUE)
  par(mar = c(8.8, 4.8, 4.2, 1.2))
  cols <- c("#4C78A8", "#F58518", "#54A24B")
  bp <- barplot(
    values,
    names.arg = labels,
    las = 2,
    col = cols,
    border = NA,
    ylab = "Seconds",
    main = "MNIST 70k computational time",
    ylim = c(0, max(ok$total_sec, na.rm = TRUE) * 1.18)
  )
  legend("topright", fill = cols, legend = rownames(values), bty = "n")
  text(bp, ok$total_sec, labels = sprintf("%.2fs", ok$total_sec), pos = 3, cex = 0.85)
  invisible(TRUE)
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

load_mnist <- function(cache_dir) {
  cache_file <- file.path(cache_dir, "mnist70k_flattened.rds")
  if (file.exists(cache_file)) return(readRDS(cache_file))
  base <- "https://storage.googleapis.com/cvdf-datasets/mnist"
  files <- c(
    train_images = "train-images-idx3-ubyte.gz",
    train_labels = "train-labels-idx1-ubyte.gz",
    test_images = "t10k-images-idx3-ubyte.gz",
    test_labels = "t10k-labels-idx1-ubyte.gz"
  )
  paths <- vapply(files, function(file) {
    download_file(file.path(base, file), file.path(cache_dir, "mnist", file))
  }, character(1L))
  out <- list(
    data = rbind(read_idx_images(paths[["train_images"]]), read_idx_images(paths[["test_images"]])),
    labels = factor(c(
      as.character(read_idx_labels(paths[["train_labels"]])),
      as.character(read_idx_labels(paths[["test_labels"]]))
    ))
  )
  dir.create(cache_dir, recursive = TRUE, showWarnings = FALSE)
  saveRDS(out, cache_file, version = 2)
  out
}

load_mnist_rdata <- function(path) {
  env <- new.env(parent = emptyenv())
  load(path, envir = env)
  objects <- mget(ls(env), env, inherits = FALSE)
  if ("mnist" %in% names(objects) && is.list(objects$mnist)) objects <- objects$mnist
  if ("dataset" %in% names(objects) && is.list(objects$dataset)) objects <- objects$dataset
  if (length(objects) == 1L && is.list(objects[[1L]])) objects <- objects[[1L]]
  data_candidates <- c("data", "x", "X", "images", "train")
  label_candidates <- c("labels", "y", "Y", "label", "Label")
  data_name <- data_candidates[data_candidates %in% names(objects)][1L]
  label_name <- label_candidates[label_candidates %in% names(objects)][1L]
  if (is.na(data_name) || is.na(label_name)) {
    stop("Could not find MNIST data and labels in ", path, call. = FALSE)
  }
  list(
    data = as.matrix(objects[[data_name]]),
    labels = factor(objects[[label_name]])
  )
}

sample_rows <- function(labels, n, seed) {
  if (n >= length(labels)) return(seq_along(labels))
  set.seed(seed)
  split_ids <- split(seq_along(labels), labels)
  rows <- unlist(lapply(split_ids, function(idx) {
    sample(idx, max(1L, floor(length(idx) / length(labels) * n)))
  }), use.names = FALSE)
  rows <- unique(rows)
  if (length(rows) < n) rows <- c(rows, sample(setdiff(seq_along(labels), rows), n - length(rows)))
  sort(rows[seq_len(min(length(rows), n))])
}

layout_matrix <- function(x) {
  if (is.list(x) && !is.null(x$layout)) x <- x$layout
  if (is.list(x) && !is.null(x$Y)) x <- x$Y
  as.matrix(x)[, 1:2, drop = FALSE]
}

score <- function(x, layout, labels, seed, n_threads) {
  rows <- sample_rows(labels, min(5000L, nrow(x)), seed + 19L)
  fastEmbedR::evaluate_embedding(
    x[rows, , drop = FALSE],
    layout[rows, , drop = FALSE],
    labels = labels[rows],
    k = c(15L, 30L, 50L),
    sample_size_for_global_metrics = min(3000L, length(rows)),
    sample_size_for_local_metrics = min(3000L, length(rows)),
    seed = seed,
    n_threads = n_threads,
    dataset = "MNIST70k"
  )
}

plot_layouts <- function(layouts, labels, rows, path, seed) {
  ok <- names(layouts)[vapply(layouts, function(x) !is.null(x), logical(1L))]
  if (!length(ok)) return(invisible(FALSE))
  keep <- sample_rows(labels, min(70000L, length(labels)), seed + 29L)
  png(path, width = 1000 * min(3L, length(ok)), height = 780 * ceiling(length(ok) / 3), res = 150)
  on.exit(dev.off(), add = TRUE)
  old <- par(no.readonly = TRUE)
  on.exit(par(old), add = TRUE)
  par(mfrow = c(ceiling(length(ok) / 3), min(3L, length(ok))), mar = c(1.4, 1.4, 3.0, 0.6))
  pal <- grDevices::hcl.colors(nlevels(labels), "Dark 3")
  cols <- pal[as.integer(labels)]
  for (nm in ok) {
    row <- rows[rows$method == nm, , drop = FALSE]
    main <- sprintf("%s\n%.2fs trust %.3f", nm, row$total_sec[[1L]], row$trustworthiness[[1L]])
    y <- layouts[[nm]]
    plot(y[keep, 1], y[keep, 2], pch = 16, cex = 0.22, col = cols[keep],
         axes = FALSE, xlab = "", ylab = "", main = main)
    box(col = "grey70")
  }
  invisible(TRUE)
}

suppressPackageStartupMessages(library(fastEmbedR))

seed <- arg_int("seed", 4L)
n <- arg_int("n", 70000L)
k <- arg_int("k", 15L)
perplexity <- arg_int("perplexity", 15L)
n_threads <- arg_int("threads", 4L)
cache_dir <- arg_value("cache-dir", file.path("results", "dataset_cache"))
mnist_rdata <- arg_value("mnist-rdata", "")
out_dir <- arg_value("out-dir", file.path("results", paste0("github_mnist70k_", format(Sys.time(), "%Y%m%d_%H%M%S"))))
run_metal <- arg_flag("run-metal", TRUE)
run_cuda <- arg_flag("run-cuda", FALSE)
run_refs <- arg_flag("run-references", TRUE)
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
specs <- machine_specs(n_threads)
write_machine_specs(specs, out_dir)

mnist <- if (nzchar(mnist_rdata)) load_mnist_rdata(mnist_rdata) else load_mnist(cache_dir)
rows <- sample_rows(mnist$labels, min(n, nrow(mnist$data)), seed)
x <- mnist$data[rows, , drop = FALSE]
labels <- droplevels(mnist$labels[rows])

layouts <- list()
results <- list()

add_row <- function(method, family, backend, nn_sec, embed_sec, total_sec, layout, status = "success", error = NA_character_) {
  metrics <- if (!is.null(layout)) score(x, layout, labels, seed, n_threads) else NULL
  results[[length(results) + 1L]] <<- data.frame(
    method = method,
    family = family,
    backend = backend,
    n = nrow(x),
    p = ncol(x),
    k = k,
    perplexity = perplexity,
    machine = specs$machine,
    cpu = specs$cpu,
    requested_threads = n_threads,
    nn_sec = nn_sec,
    embed_sec = embed_sec,
    total_sec = total_sec,
    trustworthiness = if (is.null(metrics)) NA_real_ else metrics$trustworthiness[[1L]],
    label_knn_accuracy = if (is.null(metrics)) NA_real_ else metrics$label_knn_accuracy[[1L]],
    status = status,
    error_message = error,
    stringsAsFactors = FALSE
  )
  if (!is.null(layout)) layouts[[method]] <<- layout
}

extract_stage_time <- function(fit, stage) {
  if (!is.list(fit) || is.null(fit$timings)) return(NA_real_)
  timings <- fit$timings
  if (!is.matrix(timings) && !is.data.frame(timings)) return(NA_real_)
  if (!stage %in% rownames(timings) || !"elapsed" %in% colnames(timings)) return(NA_real_)
  as.numeric(timings[stage, "elapsed"])
}

run_one <- function(method, family, backend, expr) {
  tryCatch({
    t <- timed(expr())
    y <- layout_matrix(t$value)
    nn_sec <- extract_stage_time(t$value, "knn")
    embed_sec <- extract_stage_time(t$value, "embedding")
    if (!is.finite(embed_sec)) embed_sec <- t$sec
    add_row(method, family, backend, nn_sec, embed_sec, t$sec, y)
  }, error = function(e) {
    add_row(method, family, backend, NA_real_, NA_real_, NA_real_, NULL, "failed", conditionMessage(e))
  })
}

run_one("fastEmbedR openTSNE CPU", "openTSNE", "cpu", function() {
  fastEmbedR::opentsne(
    x,
    n_neighbors = k,
    perplexity = perplexity,
    backend = "cpu",
    n_threads = n_threads,
    seed = seed
  )
})

if (run_metal) {
  run_one("fastEmbedR openTSNE Metal", "openTSNE", "metal", function() {
    fastEmbedR::opentsne(
      x,
      n_neighbors = k,
      perplexity = perplexity,
      backend = "metal",
      n_threads = n_threads,
      seed = seed
    )
  })
}

if (run_cuda) {
  run_one("fastEmbedR openTSNE CUDA", "openTSNE", "cuda", function() {
    fastEmbedR::opentsne(
      x,
      n_neighbors = k,
      perplexity = perplexity,
      backend = "cuda",
      n_threads = n_threads,
      seed = seed
    )
  })
}

if (run_refs && requireNamespace("Rtsne", quietly = TRUE)) {
  run_one("Rtsne full", "Rtsne", "cpu", function() {
    Rtsne::Rtsne(x, perplexity = perplexity, check_duplicates = FALSE, pca = TRUE)
  })
}

run_one("fastEmbedR UMAP CPU fuzzy", "UMAP", "cpu", function() {
  fastEmbedR::umap(
    x,
    n_neighbors = k,
    backend = "cpu",
    graph_mode = "fuzzy",
    n_threads = n_threads,
    seed = seed
  )
})

if (run_metal) {
  run_one("fastEmbedR UMAP Metal fuzzy", "UMAP", "metal", function() {
    fastEmbedR::umap(
      x,
      n_neighbors = k,
      backend = "metal",
      graph_mode = "fuzzy",
      n_threads = n_threads,
      seed = seed
    )
  })
}

if (run_cuda) {
  run_one("fastEmbedR UMAP CUDA fuzzy", "UMAP", "cuda", function() {
    fastEmbedR::umap(
      x,
      n_neighbors = k,
      backend = "cuda",
      graph_mode = "fuzzy",
      n_threads = n_threads,
      seed = seed
    )
  })
}

if (run_refs && requireNamespace("uwot", quietly = TRUE)) {
  run_one("uwot UMAP fast_sgd full", "UMAP", "cpu", function() {
    uwot::umap(
      x,
      n_neighbors = k,
      fast_sgd = TRUE,
      n_threads = n_threads,
      n_sgd_threads = n_threads,
      ret_model = FALSE,
      verbose = FALSE
    )
  })
}

tab <- do.call(rbind, results)
utils::write.csv(tab, file.path(out_dir, "mnist70k_github_benchmark.csv"), row.names = FALSE)
plot_layouts(layouts, labels, tab, file.path(out_dir, "mnist70k_github_benchmark.png"), seed)
plot_time_barplot(tab, file.path(out_dir, "mnist70k_github_benchmark_time_barplot.png"))
print(tab)
message("Machine specs: ", normalizePath(file.path(out_dir, "machine-specs.md"), mustWork = FALSE))
message("Timing barplot: ", normalizePath(file.path(out_dir, "mnist70k_github_benchmark_time_barplot.png"), mustWork = FALSE))
