#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(fastEmbedR)
})

args <- commandArgs(trailingOnly = TRUE)
get_arg <- function(name, default = NULL) {
  prefix <- paste0("--", name, "=")
  hit <- grep(paste0("^", prefix), args, value = TRUE)
  if (!length(hit)) return(default)
  sub(prefix, "", hit[[1L]], fixed = TRUE)
}

data_path <- get_arg("data", "/mnt/sata_ssd/fastEmbedR_Data/USPS/USPS.RData")
knn_path <- get_arg("knn", "/mnt/sata_ssd/fastEmbedR_BENCHMARK3_20260615_083543/knn_cache/USPS_cuvs_nndescent_k50.RData")
out_dir <- get_arg("out", file.path(getwd(), "results", paste0("cuda_umap_graph_replay_", format(Sys.time(), "%Y%m%d_%H%M%S"))))
seed <- as.integer(get_arg("seed", "42"))
n_threads <- as.integer(get_arg("n_threads", "4"))

dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

load_first_object <- function(path) {
  e <- new.env(parent = emptyenv())
  nm <- load(path, envir = e)
  if (!length(nm)) stop("No objects found in ", path, call. = FALSE)
  e[[nm[[1L]]]]
}

load_knn_object <- function(path) {
  e <- new.env(parent = emptyenv())
  nm <- load(path, envir = e)
  for (name in nm) {
    obj <- e[[name]]
    if (is.list(obj) && !is.null(obj$indices) && !is.null(obj$distances)) {
      return(obj)
    }
  }
  stop("No KNN object with $indices and $distances found in ", path, call. = FALSE)
}

dataset <- load_first_object(data_path)
if (is.list(dataset) && !is.null(dataset$data)) {
  x <- as.matrix(dataset$data)
  labels <- dataset$labels
} else {
  stop("Expected dataset list with $data and $labels in ", data_path, call. = FALSE)
}
knn_raw <- load_knn_object(knn_path)
knn <- fastEmbedR:::coerce_knn_input(knn_raw)
indices <- knn$indices
distances <- knn$distances
labels_factor <- if (is.null(labels)) NULL else as.factor(labels)

time_it <- function(expr) {
  gc()
  elapsed <- system.time(value <- force(expr))[["elapsed"]]
  list(value = value, seconds = as.numeric(elapsed))
}

score_layout <- function(layout) {
  out <- tryCatch(
    fastEmbedR::evaluate_embedding(
      x,
      layout,
      labels = labels_factor,
      k = c(15L, 30L, 50L),
      sample_size_for_global_metrics = min(3000L, nrow(x)),
      n_threads = n_threads,
      seed = seed
    ),
    error = function(e) data.frame(error_message = conditionMessage(e))
  )
  out[1L, , drop = FALSE]
}

plot_one <- function(layout, labels, title) {
  cols <- if (is.null(labels)) "#1f77b4" else as.integer(as.factor(labels))
  plot(layout[, 1], layout[, 2], pch = 20, cex = 0.35, col = cols,
       xlab = "UMAP 1", ylab = "UMAP 2", main = title)
}

message("Running CPU reference...")
cpu_ref <- time_it(fastEmbedR::umap_knn(
  indices, distances, backend = "cpu", n_threads = n_threads, seed = seed, verbose = FALSE
))

message("Running CUDA reference...")
cuda_ref <- time_it(fastEmbedR::umap_knn(
  indices, distances, backend = "cuda", n_threads = n_threads, seed = seed, verbose = FALSE
))
cfg <- attr(cuda_ref$value, "fastEmbedR_config")
if (is.null(cfg)) stop("CUDA layout did not return fastEmbedR_config", call. = FALSE)

message("Dumping CUDA graph...")
cuda_graph <- fastEmbedR:::umap_cuda_graph_dump_cpp(indices, distances)
valid <- cuda_graph$heads >= 0L & cuda_graph$tails >= 0L & cuda_graph$weights > 0

message("Computing CUDA initializer...")
cuda_init <- fastEmbedR:::spectral_knn_init_cuda_cpp(
  indices,
  distances,
  2L,
  as.integer(cfg$spectral_n_iter),
  seed
)

message("Replaying exact CUDA COO graph on CPU...")
cpu_replay <- time_it(fastEmbedR:::fast_knn_umap_coo_replay_cpp(
  cuda_graph$heads,
  cuda_graph$tails,
  cuda_graph$weights,
  cuda_graph$epochs_per_sample,
  cuda_init,
  as.integer(cfg$n_epochs),
  as.numeric(cfg$min_dist),
  as.integer(cfg$negative_sample_rate),
  as.numeric(cfg$learning_rate),
  as.numeric(cfg$repulsion_strength),
  n_threads,
  seed,
  TRUE,
  FALSE
))

layouts <- list(
  cpu_ref = cpu_ref$value,
  cuda_ref = cuda_ref$value,
  cpu_replay_cuda_graph = cpu_replay$value
)
timings <- c(
  cpu_ref = cpu_ref$seconds,
  cuda_ref = cuda_ref$seconds,
  cpu_replay_cuda_graph = cpu_replay$seconds
)

scores <- do.call(rbind, lapply(names(layouts), function(name) {
  s <- score_layout(layouts[[name]])
  data.frame(
    method = name,
    seconds = unname(timings[[name]]),
    trustworthiness = if ("trustworthiness" %in% names(s)) s$trustworthiness[[1L]] else NA_real_,
    silhouette = if ("silhouette" %in% names(s)) s$silhouette[[1L]] else NA_real_,
    label_knn_accuracy = if ("label_knn_accuracy" %in% names(s)) s$label_knn_accuracy[[1L]] else NA_real_,
    stringsAsFactors = FALSE
  )
}))

meta <- data.frame(
  n = nrow(indices),
  k = ncol(indices),
  cuda_graph_width = as.integer(cuda_graph$width),
  cuda_graph_capacity = as.integer(cuda_graph$capacity),
  cuda_graph_valid_edges = sum(valid),
  n_epochs = as.integer(cfg$n_epochs),
  negative_sample_rate = as.integer(cfg$negative_sample_rate),
  learning_rate = as.numeric(cfg$learning_rate),
  repulsion_strength = as.numeric(cfg$repulsion_strength),
  spectral_n_iter = as.integer(cfg$spectral_n_iter)
)

utils::write.csv(scores, file.path(out_dir, "cuda_umap_graph_replay_scores.csv"), row.names = FALSE)
utils::write.csv(meta, file.path(out_dir, "cuda_umap_graph_replay_meta.csv"), row.names = FALSE)
save(list = c("layouts", "scores", "meta", "cuda_graph"), file = file.path(out_dir, "cuda_umap_graph_replay.RData"), compress = "gzip")

png(file.path(out_dir, "cuda_umap_graph_replay.png"), width = 2400, height = 800, res = 150)
par(mfrow = c(1, 3), mar = c(4, 4, 3, 1), bg = "white")
plot_one(cpu_ref$value, labels_factor, "CPU reference")
plot_one(cuda_ref$value, labels_factor, "CUDA reference")
plot_one(cpu_replay$value, labels_factor, "CPU replay of CUDA graph")
dev.off()

print(scores)
print(meta)
cat("Output directory:", out_dir, "\n")
