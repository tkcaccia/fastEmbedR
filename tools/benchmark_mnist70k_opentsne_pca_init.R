#!/usr/bin/env Rscript

suppressPackageStartupMessages(library(fastEmbedR))

args <- commandArgs(trailingOnly = TRUE)
arg_value <- function(name, default = NULL) {
  hit <- grep(paste0("^--", name, "="), args, value = TRUE)
  if (length(hit)) sub(paste0("^--", name, "="), "", hit[[length(hit)]]) else default
}

out_dir <- arg_value(
  "out-dir",
  file.path("results", paste0("mnist70k_opentsne_pca_init_", format(Sys.time(), "%Y%m%d_%H%M%S")))
)
dataset_cache <- arg_value("dataset-cache", "results/dataset_cache/mnist_idx_70000_flattened.rds")
knn_cache <- arg_value(
  "knn-cache",
  "results/knn_cache/nn_mnist70k_raw_flattened_backend-cpu_knn-cpu_nndescent_n70000_p784_k51_seed6.rds"
)
seed <- as.integer(arg_value("seed", "6"))
n_neighbors <- as.integer(arg_value("n-neighbors", "50"))
perplexity <- as.numeric(arg_value("perplexity", "16"))
early_iter <- as.integer(arg_value("early-iter", "100"))
normal_iter <- as.integer(arg_value("normal-iter", "150"))
metric_n <- as.integer(arg_value("metric-n", "5000"))
plot_n <- as.integer(arg_value("plot-n", "20000"))
point_cex <- as.numeric(arg_value("point-cex", "0.32"))

dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

message("Loading dataset: ", dataset_cache)
mnist <- readRDS(dataset_cache)
stopifnot(all(c("x", "labels") %in% names(mnist)))
x <- as.matrix(mnist$x)
storage.mode(x) <- "double"
labels <- droplevels(factor(mnist$labels))

message("Loading cached KNN: ", knn_cache)
cache <- readRDS(knn_cache)
knn <- fastEmbedR:::normalize_opentsne_knn_input(cache$knn, NULL, n_neighbors)

message("Computing PCA init once")
pca_time <- system.time({
  x_centered <- sweep(x, 2L, colMeans(x), check.margin = FALSE)
  pca <- fastEmbedR:::fastpls_rsvd_pca_scores(
    x_centered,
    rank = 2L,
    seed = seed,
    backend = "cpu"
  )
  y_init <- as.matrix(pca$scores[, 1:2, drop = FALSE])
  y_init <- sweep(y_init, 2L, colMeans(y_init), check.margin = FALSE)
  init_scale <- max(stats::sd(y_init[, 1L]), stats::sd(y_init[, 2L]))
  if (is.finite(init_scale) && init_scale > 0) {
    y_init <- y_init * (1e-4 / init_scale)
  }
})

run_one <- function(backend) {
  message("Running openTSNE backend=", backend, " with PCA init")
  embedding_time <- system.time({
    layout <- fastEmbedR:::fast_knn_opentsne_materialized(
      knn$indices,
      knn$distances,
      n_components = 2L,
      perplexity = perplexity,
      seed = seed,
      backend = backend,
      Y_init = y_init,
      early_exaggeration_iter = early_iter,
      n_iter = normal_iter,
      early_exaggeration = 12,
      exaggeration = 1,
      learning_rate = "auto",
      initial_momentum = 0.8,
      final_momentum = 0.8,
      min_gain = 0.01,
      max_step_norm = 0.5,
      negative_gradient_method = "fft",
      record_costs = FALSE,
      auto_config = FALSE,
      input_had_self = knn$has_self,
      input_backend = knn$input_backend
    )
  })
  attr(layout, "embedding_sec") <- unname(embedding_time[["elapsed"]])
  layout
}

y_cpu <- run_one("cpu")
y_metal <- run_one("metal")

set.seed(seed + 17L)
metric_rows <- unlist(tapply(
  seq_along(labels),
  labels,
  function(ii) sample(ii, min(length(ii), ceiling(metric_n / nlevels(labels))))
))
metric_rows <- sort(metric_rows[seq_len(min(length(metric_rows), metric_n))])
score_one <- function(layout) {
  fastEmbedR::evaluate_embedding(
    x[metric_rows, , drop = FALSE],
    layout[metric_rows, , drop = FALSE],
    labels = labels[metric_rows],
    k = 15
  )
}
score_cpu <- score_one(y_cpu)
score_metal <- score_one(y_metal)

results <- data.frame(
  dataset = "mnist70k_raw_flattened",
  init = "pca_rsvd2_scaled_1e-4",
  backend = c("cpu", "metal"),
  knn_sec_cached = cache$nn_sec,
  pca_init_sec = unname(pca_time[["elapsed"]]),
  embed_sec = c(attr(y_cpu, "embedding_sec"), attr(y_metal, "embedding_sec")),
  trust = c(score_cpu$trustworthiness[1], score_metal$trustworthiness[1]),
  label_acc = c(score_cpu$label_knn_accuracy[1], score_metal$label_knn_accuracy[1]),
  stringsAsFactors = FALSE
)
write.csv(results, file.path(out_dir, "mnist70k_opentsne_pca_init_results.csv"), row.names = FALSE)
saveRDS(y_cpu, file.path(out_dir, "opentsne_cpu_pca_init.rds"))
saveRDS(y_metal, file.path(out_dir, "opentsne_metal_pca_init.rds"))
saveRDS(y_init, file.path(out_dir, "pca_init_scaled.rds"))

set.seed(seed + 29L)
plot_rows <- unlist(tapply(
  seq_along(labels),
  labels,
  function(ii) sample(ii, min(length(ii), ceiling(plot_n / nlevels(labels))))
))
plot_rows <- sort(plot_rows[seq_len(min(length(plot_rows), plot_n))])
pal <- grDevices::hcl.colors(nlevels(labels), "Dark 3")
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
plot_path <- file.path(out_dir, "mnist70k_opentsne_pca_init_cpu_vs_metal.png")
png(plot_path, width = 1800, height = 1300, res = 140)
par(mfrow = c(2, 1), mar = c(1.2, 1.2, 2.2, 1.2), oma = c(0, 0, 2, 0))
plot_one(
  y_cpu,
  sprintf("openTSNE CPU FFT PCA init (embed %.3fs, trust %.3f)",
          results$embed_sec[results$backend == "cpu"],
          results$trust[results$backend == "cpu"])
)
plot_one(
  y_metal,
  sprintf("openTSNE Metal FFT PCA init (embed %.3fs, trust %.3f)",
          results$embed_sec[results$backend == "metal"],
          results$trust[results$backend == "metal"])
)
mtext("MNIST 70k raw: cached KNN + PCA initialization", outer = TRUE, cex = 1.15)
dev.off()

message("Results CSV: ", normalizePath(file.path(out_dir, "mnist70k_opentsne_pca_init_results.csv"), mustWork = FALSE))
message("Plot PNG: ", normalizePath(plot_path, mustWork = FALSE))
print(results)
