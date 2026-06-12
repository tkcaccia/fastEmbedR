#!/usr/bin/env Rscript

suppressPackageStartupMessages(library(fastEmbedR))

args <- commandArgs(trailingOnly = TRUE)
`%||%` <- function(x, y) if (is.null(x)) y else x

arg_value <- function(name, default) {
  prefix <- paste0("--", name, "=")
  hit <- grep(paste0("^", prefix), args, value = TRUE)
  if (length(hit)) sub(prefix, "", hit[[1]], fixed = TRUE) else default
}

sizes <- as.integer(strsplit(arg_value("sizes", "256,512"), ",", fixed = TRUE)[[1]])
seed <- as.integer(arg_value("seed", "1"))
n_repeats <- as.integer(arg_value("n-repeats", "5"))
out_dir <- arg_value("out-dir", file.path("results", "mpsgraph_fft_diagnostics"))
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

rows <- lapply(sizes, function(size) {
  message("Testing MPSGraph FFT convolution size ", size)
  out <- fastEmbedR:::metal_mpsgraph_convolution_diagnostic_cpp(
    fft_size = size,
    seed = seed,
    n_repeats = n_repeats
  )
  data.frame(
    fft_size = size,
    available = isTRUE(out$available),
    current_metal_time_sec = as.numeric(out$current_metal_time_sec %||% NA_real_),
    mpsgraph_median_time_sec = as.numeric(out$mpsgraph_median_time_sec %||% NA_real_),
    mpsgraph_first_time_sec = as.numeric(out$mpsgraph_first_time_sec %||% NA_real_),
    max_abs_error = as.numeric(out$max_abs_error %||% NA_real_),
    rms_abs_error = as.numeric(out$rms_abs_error %||% NA_real_),
    rms_relative_error = as.numeric(out$rms_relative_error %||% NA_real_),
    status = as.character(out$status %||% "success"),
    error_message = as.character(out$error_message %||% ""),
    stringsAsFactors = FALSE
  )
})

res <- do.call(rbind, rows)
stamp <- format(Sys.time(), "%Y%m%d_%H%M%S")
csv <- file.path(out_dir, paste0("mpsgraph_convolution_diagnostic_", stamp, ".csv"))
write.csv(res, csv, row.names = FALSE)
write.csv(res, file.path(out_dir, "latest_mpsgraph_convolution_diagnostic.csv"), row.names = FALSE)
print(res)
message("Wrote ", csv)
