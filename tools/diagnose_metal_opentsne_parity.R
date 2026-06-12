#!/usr/bin/env Rscript

suppressPackageStartupMessages(library(fastEmbedR))

args <- commandArgs(trailingOnly = TRUE)
arg_value <- function(name, default = NULL) {
  hit <- grep(paste0("^--", name, "="), args, value = TRUE)
  if (length(hit)) sub(paste0("^--", name, "="), "", hit[[length(hit)]]) else default
}

cache_path <- arg_value(
  "knn-cache",
  "results/knn_cache/nn_mnist70k_raw_flattened_backend-cpu_knn-cpu_nndescent_n70000_p784_k51_seed6.rds"
)
out_dir <- arg_value(
  "out-dir",
  file.path("results", paste0("metal_opentsne_parity_", format(Sys.time(), "%Y%m%d_%H%M%S")))
)
iters <- as.integer(arg_value("iters", "5"))
seed <- as.integer(arg_value("seed", "6"))
n_neighbors <- as.integer(arg_value("n-neighbors", "50"))
perplexity <- as.numeric(arg_value("perplexity", "16"))
max_step_norm <- as.numeric(arg_value("max-step-norm", "0.5"))
n_threads <- as.integer(arg_value("n-threads", "4"))

dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

cache <- readRDS(cache_path)
knn <- fastEmbedR:::normalize_opentsne_knn_input(cache$knn, NULL, n_neighbors)
init <- fastEmbedR:::make_opentsne_random_init(nrow(knn$indices), 2L, seed)

cpu <- fastEmbedR:::opentsne_cpu_trace_cpp(
  knn$indices,
  knn$distances,
  init,
  perplexity = perplexity,
  n_iter = iters,
  early_exaggeration = 12,
  learning_rate = NA_real_,
  learning_rate_auto = TRUE,
  momentum = 0.8,
  min_gain = 0.01,
  max_step_norm = max_step_norm,
  n_threads = n_threads
)

metal_layout <- fastEmbedR:::fast_knn_opentsne_materialized(
  knn$indices,
  knn$distances,
  n_components = 2L,
  perplexity = perplexity,
  seed = seed,
  backend = "metal",
  Y_init = init,
  early_exaggeration_iter = iters,
  n_iter = 0L,
  early_exaggeration = 12,
  exaggeration = 1,
  learning_rate = "auto",
  initial_momentum = 0.8,
  final_momentum = 0.8,
  min_gain = 0.01,
  max_step_norm = max_step_norm,
  negative_gradient_method = "fft",
  record_costs = TRUE,
  auto_config = FALSE,
  input_had_self = knn$has_self,
  input_backend = knn$input_backend
)
metal_trace <- attr(metal_layout, "metal_trace")
cpu_trace <- cpu$trace

cmp <- merge(
  transform(cpu_trace, backend = "cpu"),
  transform(metal_trace, backend = "metal"),
  by = "iter",
  suffixes = c("_cpu", "_metal")
)
metric_names <- c(
  "sum_q", "repulsive_norm", "attractive_norm", "gradient_norm",
  "update_norm", "embedding_norm"
)
for (metric in metric_names) {
  cpu_col <- paste0(metric, "_cpu")
  metal_col <- paste0(metric, "_metal")
  cmp[[paste0(metric, "_abs_diff")]] <- abs(cmp[[cpu_col]] - cmp[[metal_col]])
  cmp[[paste0(metric, "_rel_diff")]] <- cmp[[paste0(metric, "_abs_diff")]] /
    pmax(abs(cmp[[cpu_col]]), .Machine$double.eps)
}

write.csv(cpu_trace, file.path(out_dir, "cpu_trace.csv"), row.names = FALSE)
write.csv(metal_trace, file.path(out_dir, "metal_trace.csv"), row.names = FALSE)
write.csv(cmp, file.path(out_dir, "trace_comparison.csv"), row.names = FALSE)
saveRDS(
  list(
    cache_path = cache_path,
    n_neighbors = n_neighbors,
    perplexity = perplexity,
    seed = seed,
    iters = iters,
    max_step_norm = max_step_norm,
    cpu_trace = cpu_trace,
    metal_trace = metal_trace,
    comparison = cmp
  ),
  file.path(out_dir, "trace_diagnostic.rds")
)

print(cmp[, c(
  "iter",
  "sum_q_cpu", "sum_q_metal", "sum_q_rel_diff",
  "repulsive_norm_cpu", "repulsive_norm_metal", "repulsive_norm_rel_diff",
  "attractive_norm_cpu", "attractive_norm_metal", "attractive_norm_rel_diff",
  "update_norm_cpu", "update_norm_metal", "update_norm_rel_diff",
  "embedding_norm_cpu", "embedding_norm_metal", "embedding_norm_rel_diff"
)], digits = 4)
cat("Trace output:", normalizePath(out_dir), "\n")
