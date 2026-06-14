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
  file.path("results", paste0("opentsne_faiss_ivf_70k_", format(Sys.time(), "%Y%m%d_%H%M%S")))
)
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

if (!file.exists(cache)) stop("MNIST cache not found: ", cache, call. = FALSE)
mnist <- readRDS(cache)
labels_all <- factor(mnist$labels)
rows <- seq_len(min(n_target, nrow(mnist$x)))
x <- as.matrix(mnist$x[rows, , drop = FALSE])
storage.mode(x) <- "double"
labels <- droplevels(labels_all[rows])
perplexity <- min(30L, floor(k / 3L), floor((nrow(x) - 1L) / 3L))
init_file <- file.path(out_dir, "mnist70k_opentsne_pca_init.rds")

message("MNIST 70k FAISS IVF openTSNE")
message("  faiss_available=", fastEmbedR::faiss_available())
message("  n=", nrow(x), " p=", ncol(x), " k=", k, " perplexity=", perplexity, " threads=", threads)
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

run_one <- function(name, backend, prefer_faiss = NULL) {
  old_prefer <- getOption("fastEmbedR.cpu_nndescent_prefer_faiss", NULL)
  on.exit({
    if (is.null(old_prefer)) {
      options(fastEmbedR.cpu_nndescent_prefer_faiss = NULL)
    } else {
      options(fastEmbedR.cpu_nndescent_prefer_faiss = old_prefer)
    }
  }, add = TRUE)
  if (!is.null(prefer_faiss)) {
    options(fastEmbedR.cpu_nndescent_prefer_faiss = isTRUE(prefer_faiss))
  }

  message("  KNN backend: ", name)
  nn_sec <- system.time({
    knn <- fastEmbedR::nn(x, k = k + 1L, backend = backend, n_threads = threads)
  })[["elapsed"]]
  saveRDS(knn, file.path(out_dir, paste0("knn_", name, ".rds")), version = 2)
  message("    NN sec=", sprintf("%.3f", nn_sec))

  message("  openTSNE embedding: ", name)
  embed_sec <- system.time({
    y <- fastEmbedR::opentsne_knn(
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
  saveRDS(y, file.path(out_dir, paste0("layout_", name, ".rds")), version = 2)
  message("    embed sec=", sprintf("%.3f", embed_sec))

  metrics <- fastEmbedR::evaluate_embedding(
    x[metric_rows, , drop = FALSE],
    y[metric_rows, , drop = FALSE],
    labels = labels[metric_rows],
    k = c(15, 30, 50),
    dataset = "mnist70k_raw_flattened",
    method = name,
    backend = "cpu",
    seed = seed,
    sample_size_for_global_metrics = min(5000L, length(metric_rows))
  )
  list(
    name = name,
    layout = y,
    row = data.frame(
      machine = Sys.info()[["nodename"]],
      method = "opentsne",
      knn = name,
      backend = "cpu",
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
      stringsAsFactors = FALSE
    )
  )
}

res <- list(
  run_one("FAISS_IVF", "faiss_ivf"),
  run_one("fastEmbedR_cpu_nndescent_native", "cpu_nndescent", prefer_faiss = FALSE)
)

tab <- do.call(rbind, lapply(res, `[[`, "row"))
write.csv(tab, file.path(out_dir, "opentsne_faiss_ivf_70k.csv"), row.names = FALSE)
saveRDS(
  list(results = res, table = tab, labels = labels, plot_rows = plot_rows,
       metric_rows = metric_rows, init_file = init_file),
  file.path(out_dir, "opentsne_faiss_ivf_70k.rds"),
  version = 2
)

png(file.path(out_dir, "opentsne_faiss_ivf_70k.png"), width = 1900, height = 950, res = 145)
par(mfrow = c(1, 2), mar = c(1, 1, 4.5, 1))
for (r in res) {
  y <- r$layout
  row <- r$row
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
      r$name, row$nn_sec, row$embed_sec, row$trust, row$label_acc
    )
  )
  box(col = "grey70")
}
mtext("MNIST 70k raw flattened openTSNE, saved PCA init reused", outer = TRUE, line = -1, cex = 1.05)
dev.off()

print(tab)
cat("OUT_DIR=", normalizePath(out_dir), "\n", sep = "")
