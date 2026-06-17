#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)
data_root <- if (length(args) >= 1L) args[[1L]] else "/mnt/sata_ssd/fastEmbedR/Data"
out_dir <- if (length(args) >= 2L) args[[2L]] else file.path("/mnt/sata_ssd", paste0("fastEmbedR_singularity_smoke_", format(Sys.time(), "%Y%m%d_%H%M%S")))
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

log_msg <- function(...) {
  cat(format(Sys.time(), "%Y-%m-%d %H:%M:%S"), sprintf(...), "\n")
  flush.console()
}

time_it <- function(expr) {
  t <- system.time(value <- force(expr))
  list(value = value, sec = unname(t[["elapsed"]]))
}

load_mnist <- function(data_root) {
  path <- file.path(data_root, "MNIST", "MNIST.RData")
  if (!file.exists(path)) stop("Missing MNIST RData: ", path, call. = FALSE)
  env <- new.env(parent = emptyenv())
  load(path, envir = env)
  obj <- NULL
  for (nm in ls(env)) {
    z <- get(nm, env)
    if (is.list(z) && !is.null(z$data)) obj <- z
  }
  if (is.null(obj)) stop("No list with $data in ", path, call. = FALSE)
  x <- as.matrix(obj$data)
  storage.mode(x) <- "double"
  x <- scale(x, center = TRUE, scale = FALSE)
  x[!is.finite(x)] <- 0
  labels <- if (is.null(obj$labels)) NULL else as.factor(obj$labels)
  list(data = x, labels = labels)
}

plot_layout <- function(layout, labels, file, main) {
  png(file, width = 1800, height = 1500, res = 180)
  on.exit(dev.off(), add = TRUE)
  cols <- if (is.null(labels)) "#222222" else {
    pal <- grDevices::hcl.colors(length(unique(labels)), "Dark 3")
    pal[as.integer(as.factor(labels))]
  }
  plot(layout[, 1], layout[, 2], pch = 16, cex = 0.28, col = cols,
       xlab = "dim 1", ylab = "dim 2", main = main)
}

log_msg("Loading packages")
library(faissR)
library(fastEmbedR)
log_msg("faissR backends")
print(faissR::backend_info())
log_msg("fastEmbedR backends")
print(fastEmbedR::backend_info())

ds <- load_mnist(data_root)
x <- ds$data
labels <- ds$labels
log_msg("Loaded MNIST n=%d p=%d", nrow(x), ncol(x))

results <- list()

log_msg("Running faissR::nn_without_self backend=faiss_gpu_ivf_flat k=100")
knn_run <- time_it(faissR::nn_without_self(x, k = 100, backend = "faiss_gpu_ivf_flat", metric = "euclidean", n_threads = 4))
knn <- knn_run$value
save(knn, file = file.path(out_dir, "mnist70k_faiss_gpu_ivf_flat_k100.RData"), compress = FALSE)
results[[length(results) + 1L]] <- data.frame(method = "nn", backend = "faiss_gpu_ivf_flat", nn_sec = knn_run$sec, embed_sec = NA_real_, status = "success")

log_msg("Computing PCA init")
pca_run <- time_it(fastEmbedR::opentsne_pca_init(x, n_components = 2L, backend = "cuda", seed = 4))
Y_init <- pca_run$value
save(Y_init, file = file.path(out_dir, "mnist70k_opentsne_pca_init.RData"), compress = FALSE)

log_msg("Running fastEmbedR::opentsne_knn backend=cuda")
tsne_run <- time_it(fastEmbedR::opentsne_knn(knn, perplexity = 15, Y_init = Y_init, backend = "cuda", seed = 4, n_threads = 4, verbose = FALSE))
layout_tsne <- as.matrix(tsne_run$value)
save(layout_tsne, file = file.path(out_dir, "mnist70k_fastEmbedR_opentsne_cuda.RData"), compress = FALSE)
plot_layout(layout_tsne, labels, file.path(out_dir, "mnist70k_fastEmbedR_opentsne_cuda.png"), "MNIST70k fastEmbedR openTSNE CUDA")
results[[length(results) + 1L]] <- data.frame(method = "opentsne", backend = "cuda", nn_sec = 0, embed_sec = tsne_run$sec, status = "success")

log_msg("Running fastEmbedR::umap_knn backend=cuda graph_mode=binary")
umap_run <- time_it(fastEmbedR::umap_knn(knn, backend = "cuda", graph_mode = "binary", seed = 4, n_threads = 4, verbose = FALSE))
layout_umap <- as.matrix(umap_run$value)
save(layout_umap, file = file.path(out_dir, "mnist70k_fastEmbedR_umap_cuda_binary.RData"), compress = FALSE)
plot_layout(layout_umap, labels, file.path(out_dir, "mnist70k_fastEmbedR_umap_cuda_binary.png"), "MNIST70k fastEmbedR UMAP CUDA binary")
results[[length(results) + 1L]] <- data.frame(method = "umap", backend = "cuda_binary", nn_sec = 0, embed_sec = umap_run$sec, status = "success")

fitsne_status <- "not_run"
fitsne_sec <- NA_real_
fitsne_bin <- Sys.which("fast_tsne")
if (nzchar(fitsne_bin)) {
  log_msg("FIt-SNE executable found: %s", fitsne_bin)
  fitsne_status <- "available"
} else {
  log_msg("FIt-SNE executable not found in PATH")
  fitsne_status <- "not_available"
}
results[[length(results) + 1L]] <- data.frame(method = "KlugerLab_FItSNE", backend = "cpu_fft", nn_sec = NA_real_, embed_sec = fitsne_sec, status = fitsne_status)

res <- do.call(rbind, results)
write.csv(res, file.path(out_dir, "mnist70k_singularity_smoke_results.csv"), row.names = FALSE)
log_msg("Wrote %s", out_dir)
print(res)

