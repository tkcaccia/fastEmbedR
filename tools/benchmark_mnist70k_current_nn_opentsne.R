arg_value <- function(name, default = NULL) {
  prefix <- paste0("--", name, "=")
  hit <- grep(paste0("^", prefix), commandArgs(trailingOnly = TRUE), value = TRUE)
  if (length(hit) == 0L) return(default)
  sub(prefix, "", hit[[1L]])
}

arg_int <- function(name, default) {
  value <- suppressWarnings(as.integer(arg_value(name, as.character(default))))
  if (length(value) != 1L || is.na(value)) default else value
}

arg_flag <- function(name, default = FALSE) {
  value <- tolower(arg_value(name, if (isTRUE(default)) "true" else "false"))
  value %in% c("1", "true", "yes", "on")
}

`%||%` <- function(lhs, rhs) {
  if (is.null(lhs) || length(lhs) == 0L) rhs else lhs
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

load_mnist_flat_cache <- function(cache) {
  if (file.exists(cache)) return(readRDS(cache))
  cache_dir <- dirname(cache)
  base <- "https://storage.googleapis.com/cvdf-datasets/mnist"
  files <- c(
    train_images = "train-images-idx3-ubyte.gz",
    train_labels = "train-labels-idx1-ubyte.gz",
    test_images = "t10k-images-idx3-ubyte.gz",
    test_labels = "t10k-labels-idx1-ubyte.gz"
  )
  paths <- vapply(
    files,
    function(file) download_file(file.path(base, file), file.path(cache_dir, "mnist", file)),
    character(1L)
  )
  mnist <- list(
    x = rbind(read_idx_images(paths[["train_images"]]), read_idx_images(paths[["test_images"]])),
    labels = factor(c(
      as.character(read_idx_labels(paths[["train_labels"]])),
      as.character(read_idx_labels(paths[["test_labels"]]))
    )),
    source = "MNIST IDX public files, raw flattened 28x28 pixels"
  )
  dir.create(dirname(cache), recursive = TRUE, showWarnings = FALSE)
  saveRDS(mnist, cache, version = 2)
  mnist
}

stratified_sample <- function(labels, n, seed) {
  labels <- factor(labels)
  set.seed(seed)
  rows <- unlist(
    lapply(split(seq_along(labels), labels), function(ii) {
      sample(ii, min(length(ii), ceiling(n / nlevels(labels))))
    }),
    use.names = FALSE
  )
  sort(rows[seq_len(min(length(rows), n))])
}

empty_result <- function(spec, status, message, n, p, k) {
  data.frame(
    machine = Sys.info()[["nodename"]],
    backend_requested = spec$backend,
    label = spec$label,
    status = status,
    error_message = message,
    n = n,
    p = p,
    k = k,
    nn_sec = NA_real_,
    backend_used = NA_character_,
    exact = NA,
    metric = NA_character_,
    stringsAsFactors = FALSE
  )
}

suppressPackageStartupMessages(library(fastEmbedR))

seed <- arg_int("seed", 6L)
n_target <- arg_int("n", 70000L)
k <- arg_int("k", 50L)
threads <- arg_int("threads", 4L)
exact_threshold <- arg_int("exact-threshold", 5000L)
run_embeddings <- arg_flag("embed", TRUE)
cache <- arg_value("cache", "results/dataset_cache/mnist_idx_70000_flattened.rds")
out_dir <- arg_value(
  "out-dir",
  file.path("results", paste0("mnist70k_current_nn_opentsne_", format(Sys.time(), "%Y%m%d_%H%M%S")))
)
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

mnist <- load_mnist_flat_cache(cache)
rows <- seq_len(min(n_target, nrow(mnist$x)))
x <- as.matrix(mnist$x[rows, , drop = FALSE])
storage.mode(x) <- "double"
labels <- droplevels(factor(mnist$labels[rows]))
n <- nrow(x)
p <- ncol(x)
perplexity <- min(30L, floor(k / 3L), floor((n - 1L) / 3L))
plot_rows <- stratified_sample(labels, min(20000L, n), seed)
metric_rows <- stratified_sample(labels, min(5000L, n), seed + 1009L)
init_file <- file.path(out_dir, "mnist70k_opentsne_pca_init.rds")

message("MNIST 70k current nn() + openTSNE benchmark")
message("  n=", n, " p=", p, " k=", k, " threads=", threads, " perplexity=", perplexity)
message("  faiss_available=", faiss_available(), " cuvs_available=", cuvs_available(),
        " cuda_available=", cuda_available(), " metal_available=", metal_available())
message("  out_dir=", normalizePath(out_dir, mustWork = FALSE))

init_sec <- system.time({
  init <- opentsne_pca_init(
    x,
    n_components = 2L,
    seed = seed,
    backend = "cpu",
    cache_file = init_file
  )
})[["elapsed"]]
message("  PCA init sec=", sprintf("%.3f", init_sec))

specs <- list(
  list(label = "cpu_exact", backend = "cpu", requires = "cpu_exact"),
  list(label = "auto", backend = "auto", requires = "cpu"),
  list(label = "cpu_approx", backend = "cpu_approx", requires = "cpu"),
  list(label = "hnsw", backend = "hnsw", requires = "RcppHNSW"),
  list(label = "rcpphnsw", backend = "rcpphnsw", requires = "RcppHNSW"),
  list(label = "cpu_hnsw", backend = "cpu_hnsw", requires = "RcppHNSW"),
  list(label = "faiss_flat_l2", backend = "faiss_flat_l2", requires = "faiss"),
  list(label = "faiss_flat_ip", backend = "faiss_flat_ip", requires = "faiss"),
  list(label = "faiss_ivf", backend = "faiss_ivf", requires = "faiss"),
  list(label = "faiss_ivfpq", backend = "faiss_ivfpq", requires = "faiss"),
  list(label = "faiss_hnsw", backend = "faiss_hnsw", requires = "faiss"),
  list(label = "faiss_nsg", backend = "faiss_nsg", requires = "faiss"),
  list(label = "faiss_nndescent", backend = "faiss_nndescent", requires = "faiss"),
  list(label = "cuda_cuvs_nndescent", backend = "cuda_cuvs_nndescent", requires = "cuvs"),
  list(label = "cuda_cuvs_cagra", backend = "cuda_cuvs_cagra", requires = "cuvs"),
  list(label = "cuda_cuvs_bruteforce", backend = "cuda_cuvs_bruteforce", requires = "cuvs")
)

available_spec <- function(spec) {
  if (identical(spec$requires, "cpu")) return(list(ok = TRUE, reason = NA_character_))
  if (identical(spec$requires, "cpu_exact")) {
    if (n > exact_threshold) {
      return(list(
        ok = FALSE,
        reason = paste0("skipped: exact all-pairs CPU KNN on n=", n,
                        " exceeds exact-threshold=", exact_threshold)
      ))
    }
    return(list(ok = TRUE, reason = NA_character_))
  }
  if (identical(spec$requires, "RcppHNSW")) {
    ok <- requireNamespace("RcppHNSW", quietly = TRUE)
    return(list(ok = ok, reason = if (ok) NA_character_ else "RcppHNSW is not installed"))
  }
  if (identical(spec$requires, "faiss")) {
    ok <- isTRUE(faiss_available())
    return(list(ok = ok, reason = if (ok) NA_character_ else "package not built with FAISS"))
  }
  if (identical(spec$requires, "cuvs")) {
    ok <- isTRUE(cuvs_available())
    return(list(ok = ok, reason = if (ok) NA_character_ else "package not built with cuVS/CUDA"))
  }
  list(ok = FALSE, reason = paste0("unknown requirement: ", spec$requires))
}

knn_rows <- list()
embed_rows <- list()
layouts <- list()
seen_embedding_keys <- character()

for (spec in specs) {
  message("  nn(): ", spec$label, " [", spec$backend, "]")
  availability <- available_spec(spec)
  if (!isTRUE(availability$ok)) {
    knn_rows[[length(knn_rows) + 1L]] <- empty_result(
      spec, "skipped", availability$reason, n, p, k
    )
    message("    skipped: ", availability$reason)
    next
  }

  knn <- NULL
  row <- tryCatch({
    elapsed <- system.time({
      knn <- nn(x, k = k + 1L, backend = spec$backend, n_threads = threads)
    })[["elapsed"]]
    saveRDS(knn, file.path(out_dir, paste0("knn_", spec$label, ".rds")), version = 2)
    out <- data.frame(
      machine = Sys.info()[["nodename"]],
      backend_requested = spec$backend,
      label = spec$label,
      status = "success",
      error_message = NA_character_,
      n = n,
      p = p,
      k = k,
      nn_sec = elapsed,
      backend_used = attr(knn, "backend") %||% NA_character_,
      exact = isTRUE(attr(knn, "exact")),
      metric = attr(knn, "metric") %||% NA_character_,
      stringsAsFactors = FALSE
    )
    message("    success: ", sprintf("%.3fs", elapsed), " backend_used=", out$backend_used)
    out
  }, error = function(e) {
    message("    failed: ", conditionMessage(e))
    empty_result(spec, "failed", conditionMessage(e), n, p, k)
  })
  knn_rows[[length(knn_rows) + 1L]] <- row

  if (!run_embeddings || !identical(row$status, "success")) next
  if (identical(spec$label, "faiss_flat_ip")) {
    message("    openTSNE skipped: inner-product KNN is not a Euclidean distance graph")
    next
  }
  embedding_key <- paste(row$backend_used, row$metric, row$exact, sep = "_")
  if (embedding_key %in% seen_embedding_keys) {
    message("    openTSNE skipped: duplicate resolved KNN algorithm already plotted")
    next
  }
  seen_embedding_keys <- c(seen_embedding_keys, embedding_key)

  message("    openTSNE: ", spec$label)
  embed_result <- tryCatch({
    embed_sec <- system.time({
      y <- opentsne_knn(
        knn,
        perplexity = perplexity,
        Y_init = init_file,
        early_exaggeration_iter = 100L,
        n_iter = 150L,
        backend = "cpu",
        n_threads = threads,
        seed = seed,
        verbose = FALSE
      )
    })[["elapsed"]]
    saveRDS(y, file.path(out_dir, paste0("layout_", spec$label, ".rds")), version = 2)
    metrics <- evaluate_embedding(
      x[metric_rows, , drop = FALSE],
      y[metric_rows, , drop = FALSE],
      labels = labels[metric_rows],
      k = c(15L, 30L, 50L),
      dataset = "mnist70k_raw_flattened",
      method = paste0("opentsne_", spec$label),
      backend = "cpu",
      seed = seed,
      sample_size_for_global_metrics = length(metric_rows),
      sample_size_for_local_metrics = length(metric_rows),
      use_cache = FALSE,
      n_threads = threads
    )
    list(
      layout = y,
      row = data.frame(
        machine = Sys.info()[["nodename"]],
        method = "opentsne",
        nn_label = spec$label,
        backend = "cpu",
        n = n,
        p = p,
        k = k,
        nn_sec = row$nn_sec,
        init_sec = init_sec,
        embed_sec = embed_sec,
        trust = metrics$trustworthiness[1],
        knn_preservation_15 = metrics$knn_preservation_15[1],
        knn_preservation_30 = metrics$knn_preservation_30[1],
        knn_preservation_50 = metrics$knn_preservation_50[1],
        label_acc = metrics$label_knn_accuracy[1],
        stringsAsFactors = FALSE
      )
    )
  }, error = function(e) {
    message("    openTSNE failed: ", conditionMessage(e))
    list(
      layout = NULL,
      row = data.frame(
        machine = Sys.info()[["nodename"]],
        method = "opentsne",
        nn_label = spec$label,
        backend = "cpu",
        n = n,
        p = p,
        k = k,
        nn_sec = row$nn_sec,
        init_sec = init_sec,
        embed_sec = NA_real_,
        trust = NA_real_,
        knn_preservation_15 = NA_real_,
        knn_preservation_30 = NA_real_,
        knn_preservation_50 = NA_real_,
        label_acc = NA_real_,
        stringsAsFactors = FALSE
      )
    )
  })
  if (!is.null(embed_result$layout)) {
    layouts[[spec$label]] <- embed_result$layout
  }
  embed_rows[[length(embed_rows) + 1L]] <- embed_result$row
}

knn_table <- do.call(rbind, knn_rows)
embed_table <- if (length(embed_rows)) do.call(rbind, embed_rows) else data.frame()
write.csv(knn_table, file.path(out_dir, "mnist70k_nn_timing.csv"), row.names = FALSE)
write.csv(embed_table, file.path(out_dir, "mnist70k_opentsne_quality.csv"), row.names = FALSE)

if (length(layouts)) {
  n_panels <- length(layouts)
  n_cols <- min(3L, n_panels)
  n_rows <- ceiling(n_panels / n_cols)
  png(file.path(out_dir, "mnist70k_opentsne_from_nn_gallery.png"),
      width = 900L * n_cols, height = 820L * n_rows, res = 140)
  par(mfrow = c(n_rows, n_cols), mar = c(1, 1, 4, 1))
  for (nm in names(layouts)) {
    y <- layouts[[nm]]
    row <- embed_table[embed_table$nn_label == nm, , drop = FALSE]
    plot(
      y[plot_rows, , drop = FALSE],
      pch = 21,
      bg = as.integer(labels[plot_rows]),
      col = "#00000022",
      cex = 0.35,
      axes = FALSE,
      xlab = "",
      ylab = "",
      main = sprintf(
        "%s\nNN %.2fs | embed %.2fs | trust %.3f | label %.3f",
        nm, row$nn_sec[1], row$embed_sec[1], row$trust[1], row$label_acc[1]
      )
    )
    box(col = "grey70")
  }
  dev.off()
}

saveRDS(
  list(
    knn = knn_table,
    opentsne = embed_table,
    labels = labels,
    plot_rows = plot_rows,
    metric_rows = metric_rows,
    out_dir = out_dir
  ),
  file.path(out_dir, "mnist70k_current_nn_opentsne_results.rds"),
  version = 2
)

message("Wrote: ", file.path(out_dir, "mnist70k_nn_timing.csv"))
message("Wrote: ", file.path(out_dir, "mnist70k_opentsne_quality.csv"))
if (length(layouts)) {
  message("Wrote: ", file.path(out_dir, "mnist70k_opentsne_from_nn_gallery.png"))
}
print(knn_table)
print(embed_table)
cat("OUT_DIR=", normalizePath(out_dir), "\n", sep = "")
