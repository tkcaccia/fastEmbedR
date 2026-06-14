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

arg_csv <- function(name, default) {
  value <- arg_value(name, default)
  trimws(strsplit(value, ",", fixed = TRUE)[[1L]])
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

suppressPackageStartupMessages(library(fastEmbedR))

seed <- arg_int("seed", 6L)
n_target <- arg_int("n", 70000L)
k <- arg_int("k", 50L)
threads <- arg_int("threads", 4L)
cache <- arg_value("cache", "results/dataset_cache/mnist_idx_70000_flattened.rds")
out_dir <- arg_value(
  "out-dir",
  file.path("results", paste0("opentsne_knn_fourway_70k_", format(Sys.time(), "%Y%m%d_%H%M%S")))
)
methods <- arg_csv(
  "methods",
  paste(
    c(
      "FAISS_Flat_L2",
      "FAISS_Flat_IP",
      "FAISS_IVF",
      "FAISS_IVFPQ",
      "FAISS_HNSW",
      "FAISS_NSG",
      "FAISS_NNDescent",
      "RcppHNSW_hnsw",
      "fastEmbedR_auto"
    ),
    collapse = ","
  )
)
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

mnist <- load_mnist_flat_cache(cache)
labels_all <- factor(mnist$labels)
rows <- seq_len(min(n_target, nrow(mnist$x)))
x <- as.matrix(mnist$x[rows, , drop = FALSE])
storage.mode(x) <- "double"
labels <- droplevels(labels_all[rows])
perplexity <- min(30L, floor(k / 3L), floor((nrow(x) - 1L) / 3L))
init_file <- file.path(out_dir, "mnist70k_opentsne_pca_init.rds")
table_file <- file.path(out_dir, "opentsne_knn_fourway_70k.csv")

message("MNIST 70k four-way KNN + openTSNE")
message("  faiss_available=", fastEmbedR::faiss_available())
message("  n=", nrow(x), " p=", ncol(x), " k=", k, " perplexity=", perplexity, " threads=", threads)
message("  methods=", paste(methods, collapse = ","))
message("  out_dir=", out_dir)

init_sec <- system.time({
  init <- fastEmbedR::opentsne_pca_init(
    x,
    n_components = 2L,
    seed = seed,
    backend = "cpu",
    cache_file = init_file
  )
})[["elapsed"]]
message("  PCA init sec=", sprintf("%.3f", init_sec), " cache=", init_file)

plot_rows <- stratified_sample(labels, min(20000L, length(labels)), seed)
metric_rows <- stratified_sample(labels, min(5000L, length(labels)), seed + 1009L)

build_knn <- function(method) {
  if (method %in% c("FAISS_Flat", "FAISS_Flat_L2")) {
    return(fastEmbedR::nn(x, k = k + 1L, backend = "faiss_flat_l2", n_threads = threads))
  }
  if (identical(method, "FAISS_Flat_IP")) {
    return(fastEmbedR::nn(x, k = k + 1L, backend = "faiss_flat_ip", n_threads = threads))
  }
  if (identical(method, "FAISS_IVF")) {
    return(fastEmbedR::nn(x, k = k + 1L, backend = "faiss_ivf", n_threads = threads))
  }
  if (identical(method, "FAISS_IVFPQ")) {
    return(fastEmbedR::nn(x, k = k + 1L, backend = "faiss_ivfpq", n_threads = threads))
  }
  if (identical(method, "FAISS_HNSW")) {
    return(fastEmbedR::nn(x, k = k + 1L, backend = "faiss_hnsw", n_threads = threads))
  }
  if (identical(method, "FAISS_NSG")) {
    return(fastEmbedR::nn(x, k = k + 1L, backend = "faiss_nsg", n_threads = threads))
  }
  if (identical(method, "FAISS_NNDescent")) {
    return(fastEmbedR::nn(x, k = k + 1L, backend = "faiss_nndescent", n_threads = threads))
  }
  if (identical(method, "RcppHNSW_hnsw")) {
    if (!requireNamespace("RcppHNSW", quietly = TRUE)) {
      stop("R package `RcppHNSW` is not installed.", call. = FALSE)
    }
    h <- RcppHNSW::hnsw_knn(
      x,
      k = k + 1L,
      distance = "euclidean",
      M = 16,
      ef_construction = 200,
      ef = 100,
      n_threads = threads,
      verbose = FALSE,
      progress = "none"
    )
    return(list(indices = h$idx[, -1L, drop = FALSE], distances = h$dist[, -1L, drop = FALSE]))
  }
  if (identical(method, "fastEmbedR_auto")) {
    return(fastEmbedR::nn(x, k = k + 1L, backend = "auto", n_threads = threads))
  }
  if (identical(method, "fastEmbedR_cuda_cuvs_nndescent")) {
    return(fastEmbedR::nn(x, k = k + 1L, backend = "cuda_cuvs_nndescent", n_threads = threads))
  }
  if (identical(method, "fastEmbedR_cuda_cuvs_cagra")) {
    return(fastEmbedR::nn(x, k = k + 1L, backend = "cuda_cuvs_cagra", n_threads = threads))
  }
  if (identical(method, "fastEmbedR_cuda_cuvs_bruteforce")) {
    return(fastEmbedR::nn(x, k = k + 1L, backend = "cuda_cuvs_bruteforce", n_threads = threads))
  }
  stop("Unknown method: ", method, call. = FALSE)
}

can_embed_with_opentsne <- function(method, knn) {
  if (identical(method, "FAISS_Flat_IP")) {
    return(FALSE)
  }
  if (isTRUE(attr(knn, "metric") %in% "inner_product_similarity_shifted_to_distance")) {
    return(FALSE)
  }
  TRUE
}

method_backend <- function(method) {
  if (grepl("cuda", method, ignore.case = TRUE)) return("cuda")
  if (grepl("metal", method, ignore.case = TRUE)) return("metal")
  "cpu"
}

opentsne_backend_for_method <- function(method) {
  if (grepl("cuda", method, ignore.case = TRUE)) return("cuda")
  if (grepl("metal", method, ignore.case = TRUE)) return("metal")
  "cpu"
}

rows_out <- list()
layouts <- list()

for (method in methods) {
  message("  KNN method: ", method)
  row <- tryCatch({
    nn_sec <- system.time({
      knn <- build_knn(method)
    })[["elapsed"]]
    saveRDS(knn, file.path(out_dir, paste0("knn_", method, ".rds")), version = 2)
    message("    NN sec=", sprintf("%.3f", nn_sec))

    if (!can_embed_with_opentsne(method, knn)) {
      message("    embed skipped: inner-product similarities are not Euclidean distances")
      data.frame(
        machine = Sys.info()[["nodename"]],
        method = "opentsne",
        knn = method,
        backend = method_backend(method),
        n = nrow(x),
        p = ncol(x),
        k = k,
        nn_sec = nn_sec,
        init_sec = init_sec,
        embed_sec = NA_real_,
        trust = NA_real_,
        knn_preservation_15 = NA_real_,
        knn_preservation_30 = NA_real_,
        knn_preservation_50 = NA_real_,
        label_acc = NA_real_,
        status = "knn_only",
        error_message = "Inner-product FAISS output is not a Euclidean distance KNN for this openTSNE affinity path.",
        stringsAsFactors = FALSE
      )
    } else {
      embed_sec <- system.time({
        y <- fastEmbedR::opentsne_knn(
          knn,
          perplexity = perplexity,
          Y_init = init_file,
          early_exaggeration_iter = 100L,
          n_iter = 150L,
          backend = opentsne_backend_for_method(method),
          n_threads = threads,
          seed = seed,
          verbose = FALSE
        )
      })[["elapsed"]]
      saveRDS(y, file.path(out_dir, paste0("layout_", method, ".rds")), version = 2)
      layouts[[method]] <- y
      message("    embed sec=", sprintf("%.3f", embed_sec))

      metrics <- fastEmbedR::evaluate_embedding(
        x[metric_rows, , drop = FALSE],
        y[metric_rows, , drop = FALSE],
        labels = labels[metric_rows],
        k = c(15, 30, 50),
        dataset = "mnist70k_raw_flattened",
        method = method,
        backend = method_backend(method),
        seed = seed,
        sample_size_for_global_metrics = min(5000L, length(metric_rows))
      )
      data.frame(
        machine = Sys.info()[["nodename"]],
        method = "opentsne",
        knn = method,
        backend = method_backend(method),
        n = nrow(x),
        p = ncol(x),
        k = k,
        nn_sec = nn_sec,
        init_sec = init_sec,
        embed_sec = embed_sec,
        trust = metrics$trustworthiness[1],
        knn_preservation_15 = metrics$knn_preservation_15[1],
        knn_preservation_30 = metrics$knn_preservation_30[1],
        knn_preservation_50 = metrics$knn_preservation_50[1],
        label_acc = metrics$label_knn_accuracy[1],
        status = "success",
        error_message = NA_character_,
        stringsAsFactors = FALSE
      )
    }
  }, error = function(e) {
    data.frame(
      machine = Sys.info()[["nodename"]],
      method = "opentsne",
      knn = method,
      backend = method_backend(method),
      n = nrow(x),
      p = ncol(x),
      k = k,
      nn_sec = NA_real_,
      init_sec = init_sec,
      embed_sec = NA_real_,
      trust = NA_real_,
      knn_preservation_15 = NA_real_,
      knn_preservation_30 = NA_real_,
      knn_preservation_50 = NA_real_,
      label_acc = NA_real_,
      status = "failed",
      error_message = conditionMessage(e),
      stringsAsFactors = FALSE
    )
  })
  rows_out[[length(rows_out) + 1L]] <- row
  write.csv(do.call(rbind, rows_out), table_file, row.names = FALSE)
}

tab <- do.call(rbind, rows_out)
saveRDS(
  list(table = tab, labels = labels, plot_rows = plot_rows,
       metric_rows = metric_rows, init_file = init_file,
       layouts = layouts),
  file.path(out_dir, "opentsne_knn_fourway_70k.rds"),
  version = 2
)

success <- tab$status == "success"
if (any(success)) {
  png(
    file.path(out_dir, "opentsne_knn_fourway_70k.png"),
    width = 950 * sum(success),
    height = 950,
    res = 145
  )
  par(mfrow = c(1, sum(success)), mar = c(1, 1, 4.5, 1))
  for (method in tab$knn[success]) {
    y <- layouts[[method]]
    row <- tab[tab$knn == method, , drop = FALSE]
    plot(
      y[plot_rows, , drop = FALSE],
      pch = 21,
      bg = as.integer(labels[plot_rows]),
      col = "#00000022",
      cex = 0.30,
      axes = FALSE,
      xlab = "",
      ylab = "",
      main = sprintf(
        "%s\nNN %.2fs | embed %.2fs | trust %.3f | label %.3f",
        method, row$nn_sec, row$embed_sec, row$trust, row$label_acc
      )
    )
    box(col = "grey70")
  }
  mtext("MNIST 70k raw flattened openTSNE, same saved PCA init", outer = TRUE, line = -1, cex = 1.05)
  dev.off()
}

print(tab)
cat("OUT_DIR=", normalizePath(out_dir), "\n", sep = "")
