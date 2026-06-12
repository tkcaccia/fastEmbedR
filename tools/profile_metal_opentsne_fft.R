#!/usr/bin/env Rscript

arg_value <- function(name, default = NULL) {
  prefix <- paste0("--", name, "=")
  args <- commandArgs(trailingOnly = TRUE)
  hit <- args[startsWith(args, prefix)]
  if (length(hit) == 0L) return(default)
  sub(prefix, "", hit[[length(hit)]], fixed = TRUE)
}

arg_int <- function(name, default) {
  value <- suppressWarnings(as.integer(arg_value(name, as.character(default))))
  if (length(value) != 1L || is.na(value)) default else value
}

stratified_rows <- function(labels, n, seed) {
  n <- min(as.integer(n), length(labels))
  if (n >= length(labels)) return(seq_along(labels))
  set.seed(seed)
  labels <- as.factor(labels)
  by_label <- split(seq_along(labels), labels)
  take <- lapply(by_label, function(idx) ceiling(length(idx) / length(labels) * n))
  rows <- unlist(Map(function(idx, m) sample(idx, min(length(idx), m)), by_label, take), use.names = FALSE)
  if (length(rows) > n) rows <- sample(rows, n)
  if (length(rows) < n) rows <- c(rows, sample(setdiff(seq_along(labels), rows), n - length(rows)))
  sort(rows)
}

timed <- function(expr) {
  gc()
  t <- system.time(value <- force(expr))
  list(value = value, sec = unname(t[["elapsed"]]))
}

cache <- arg_value("cache", "results/dataset_cache/mnist_idx_70000_flattened.rds")
out_dir <- arg_value("out-dir", file.path("results", "metal_fft_profiles"))
n <- arg_int("n", 10000L)
k <- arg_int("k", 50L)
seed <- arg_int("seed", 6L)
early_iter <- arg_int("early-iter", 20L)
normal_iter <- arg_int("normal-iter", 30L)
n_threads <- arg_int("threads", 4L)
perplexity <- arg_int("perplexity", min(30L, floor(k / 3L)))

dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
Sys.setenv(FASTEMBEDR_METAL_STAGE_TIMING = "1")

suppressPackageStartupMessages(library(fastEmbedR))
if (!isTRUE(fastEmbedR:::embedding_metal_available_cpp()) ||
    !isTRUE(fastEmbedR:::metal_opentsne_native_available())) {
  stop("Native Metal openTSNE is unavailable on this machine.", call. = FALSE)
}
if (!file.exists(cache)) stop("MNIST cache not found: ", cache, call. = FALSE)

mnist <- readRDS(cache)
if (!all(c("x", "labels") %in% names(mnist))) {
  stop("Cache must contain `x` and `labels`.", call. = FALSE)
}
rows <- stratified_rows(mnist$labels, n, seed)
x <- mnist$x[rows, , drop = FALSE]
storage.mode(x) <- "double"

message("Profiling Metal openTSNE FFT")
message("  n=", nrow(x), " p=", ncol(x), " k=", k, " perplexity=", perplexity)
message("  early_iter=", early_iter, " normal_iter=", normal_iter)

knn_run <- timed(fastEmbedR::nn(
  x,
  k = k + 1L,
  backend = "metal_nndescent",
  n_threads = n_threads
))

embed_run <- timed(fastEmbedR::opentsne_knn(
  knn_run$value,
  backend = "metal",
  perplexity = perplexity,
  early_exaggeration_iter = early_iter,
  n_iter = normal_iter,
  seed = seed,
  negative_gradient_method = "fft"
))

timing <- attr(embed_run$value, "metal_stage_timing")
if (is.null(timing) || !NROW(timing)) {
  stop("No Metal stage timing was returned.", call. = FALSE)
}

timing$n <- nrow(x)
timing$p <- ncol(x)
timing$k <- k
timing$seed <- seed
timing$nn_sec <- knn_run$sec
timing$embed_sec <- embed_run$sec
timing$fft_gpu_sec <- sum(timing$gpu_sec[timing$stage %in% c("fft_forward", "fft_convolution")], na.rm = TRUE)
timing$total_gpu_sec <- sum(timing$gpu_sec, na.rm = TRUE)
timing$fft_gpu_fraction <- timing$fft_gpu_sec / pmax(timing$total_gpu_sec, .Machine$double.eps)

stamp <- format(Sys.time(), "%Y%m%d_%H%M%S")
csv <- file.path(out_dir, paste0("metal_opentsne_fft_profile_", stamp, ".csv"))
latest <- file.path(out_dir, "latest_metal_opentsne_fft_profile.csv")
write.csv(timing, csv, row.names = FALSE)
write.csv(timing, latest, row.names = FALSE)

print(timing)
message("Profile CSV: ", normalizePath(csv, mustWork = FALSE))
message("Latest CSV: ", normalizePath(latest, mustWork = FALSE))

