#!/usr/bin/env Rscript

arg_value <- function(name, default = NULL) {
  prefix <- paste0("--", name, "=")
  args <- commandArgs(trailingOnly = TRUE)
  hit <- args[startsWith(args, prefix)]
  if (length(hit) == 0L) return(default)
  sub(prefix, "", hit[[length(hit)]], fixed = TRUE)
}

latest_dir <- function(pattern) {
  paths <- Sys.glob(pattern)
  if (!length(paths)) stop("No paths match: ", pattern, call. = FALSE)
  paths[order(file.info(paths)$mtime, decreasing = TRUE)][[1L]]
}

as_layout <- function(x) {
  if (is.matrix(x)) return(x)
  if (is.data.frame(x)) return(as.matrix(x))
  if (is.list(x) && !is.null(x$layout)) return(as.matrix(x$layout))
  as.matrix(x)
}

read_results <- function(path) {
  out <- read.csv(path, stringsAsFactors = FALSE, check.names = FALSE)
  out[out$status == "success" & out$method == "opentsne" & out$variant == "full", , drop = FALSE]
}

local_dir <- arg_value(
  "local-dir",
  latest_dir("results/mnist70k_pca_opentsne_github_local_*")
)
cuda_dir <- arg_value(
  "cuda-dir",
  "results/chiamaka_mnist70k_cpu_cuda_20260612_150230/results"
)
dataset_cache <- arg_value(
  "dataset-cache",
  "results/dataset_cache/mnist_idx_70000_flattened.rds"
)
out_dir <- arg_value("out-dir", "docs/assets")
plot_n <- as.integer(arg_value("plot-n", "30000"))
seed <- as.integer(arg_value("seed", "6"))
point_cex <- as.numeric(arg_value("point-cex", "0.33"))

dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

mnist <- readRDS(dataset_cache)
labels <- droplevels(factor(mnist$labels))
pal <- grDevices::hcl.colors(nlevels(labels), "Dark 3")

set.seed(seed + 29L)
plot_rows <- unlist(tapply(
  seq_along(labels),
  labels,
  function(ii) sample(ii, min(length(ii), ceiling(plot_n / nlevels(labels))))
))
plot_rows <- sort(plot_rows[seq_len(min(length(plot_rows), plot_n))])

local_results <- read_results(file.path(local_dir, "mnist70k_current_backends_results.csv"))
cuda_results <- read_results(file.path(cuda_dir, "mnist70k_current_backends_results.csv"))
rows <- rbind(
  local_results[local_results$backend_used %in% c("cpu", "metal"), , drop = FALSE],
  cuda_results[cuda_results$backend_used == "cuda", , drop = FALSE]
)
rows <- rows[match(c("cpu", "metal", "cuda"), rows$backend_used), , drop = FALSE]

layout_paths <- c(
  cpu = file.path(local_dir, "MNIST70K_OPENTSNE_FULL_CPU_FFT.rds"),
  metal = file.path(local_dir, "MNIST70K_OPENTSNE_FULL_METAL_FFT.rds"),
  cuda = file.path(cuda_dir, "MNIST70K_OPENTSNE_FULL_CUDA_FFT.rds")
)
missing_layouts <- layout_paths[!file.exists(layout_paths)]
if (length(missing_layouts)) {
  stop("Missing layout files: ", paste(missing_layouts, collapse = ", "), call. = FALSE)
}
layouts <- lapply(layout_paths, function(path) as_layout(readRDS(path)))

timing <- data.frame(
  method = "openTSNE PCA init",
  backend = rows$backend_used,
  machine = rows$machine,
  nn_sec = rows$nn_sec,
  embedding_sec = rows$embedding_sec,
  total_sec = rows$nn_sec + rows$embedding_sec,
  trust = rows$trustworthiness,
  label_knn_accuracy = rows$label_knn_accuracy,
  stringsAsFactors = FALSE
)

timing_csv <- file.path(out_dir, "mnist70k-opentsne-pca-timing.csv")
write.csv(timing, timing_csv, row.names = FALSE)

plot_one <- function(y, main) {
  plot(
    y[plot_rows, 1L],
    y[plot_rows, 2L],
    pch = 16,
    cex = point_cex,
    col = pal[as.integer(labels[plot_rows])],
    axes = FALSE,
    xlab = "",
    ylab = "",
    main = main
  )
  box(col = "grey70")
}

embedding_png <- file.path(out_dir, "mnist70k-opentsne-pca-embeddings-cpu-metal-cuda.png")
png(embedding_png, width = 2400, height = 1050, res = 140)
par(mfrow = c(1, 3), mar = c(1.0, 1.0, 3.4, 1.0), oma = c(0, 0, 2.2, 0))
for (backend in c("cpu", "metal", "cuda")) {
  rr <- timing[timing$backend == backend, , drop = FALSE]
  plot_one(
    layouts[[backend]],
    sprintf(
      "%s\nNN %.2fs | embed %.2fs | trust %.3f",
      toupper(backend),
      rr$nn_sec,
      rr$embedding_sec,
      rr$trust
    )
  )
}
mtext("MNIST 70k raw pixels: fastEmbedR openTSNE, KNN input, PCA initialization", outer = TRUE, cex = 1.15)
dev.off()

timing_png <- file.path(out_dir, "mnist70k-opentsne-pca-timing-stacked.png")
png(timing_png, width = 1500, height = 950, res = 140)
par(mar = c(5.6, 4.6, 4.4, 1.4))
bar_values <- rbind(
  `NN search` = timing$nn_sec,
  Embedding = timing$embedding_sec
)
cols <- c("#4C78A8", "#F58518")
bp <- barplot(
  bar_values,
  names.arg = toupper(timing$backend),
  col = cols,
  border = NA,
  ylab = "Seconds",
  main = "MNIST 70k openTSNE PCA init: NN + embedding time",
  ylim = c(0, max(colSums(bar_values), na.rm = TRUE) * 1.18)
)
legend("topright", fill = cols, legend = rownames(bar_values), bty = "n")
total <- colSums(bar_values)
text(bp, total, labels = sprintf("%.2fs", total), pos = 3, cex = 0.9)
text(
  bp,
  rep(0, nrow(timing)),
  labels = sprintf("trust %.3f", timing$trust),
  pos = 1,
  cex = 0.8,
  xpd = NA
)
dev.off()

message("Local results: ", normalizePath(local_dir))
message("CUDA results: ", normalizePath(cuda_dir))
message("Embedding plot: ", normalizePath(embedding_png))
message("Timing plot: ", normalizePath(timing_png))
message("Timing CSV: ", normalizePath(timing_csv))
print(timing)
