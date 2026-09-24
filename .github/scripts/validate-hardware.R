args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2L) {
  stop(
    "usage: validate-hardware.R <cpu|metal|cuda> <output-directory>",
    call. = FALSE
  )
}

backend <- match.arg(args[[1L]], c("cpu", "metal", "cuda"))
out_dir <- normalizePath(args[[2L]], mustWork = TRUE)

suppressPackageStartupMessages(library(fastEmbedR))

writeLines(capture.output(sessionInfo()), file.path(out_dir, "sessionInfo.txt"))
utils::write.csv(
  data.frame(
    package_version = as.character(packageVersion("fastEmbedR")),
    backend_requested = backend,
    stringsAsFactors = FALSE
  ),
  file.path(out_dir, "validation-request.csv"),
  row.names = FALSE
)

seed <- 20260822L
set.seed(seed)
n <- 768L
p <- 32L
group <- rep(seq_len(6L), length.out = n)
x <- matrix(rnorm(n * p, sd = 0.35), nrow = n, ncol = p)
x[, seq_len(6L)] <- x[, seq_len(6L), drop = FALSE] +
  stats::model.matrix(~ factor(group) - 1)

elapsed <- function(label, expression) {
  gc()
  timing <- system.time(value <- force(expression))
  list(label = label, value = value, elapsed = unname(timing[["elapsed"]]))
}

pca_run <- elapsed("pca", pca(
  x, ncomp = 2L, backend = backend, n.cores = 4L, seed = seed
))
knn_run <- elapsed("knn", precompute_knn(
  x, k = 15L, metric = "euclidean", backend = backend, n.cores = 4L
))

if (backend == "cuda") {
  cuda_knn_metadata <- data.frame(
    class = paste(class(knn_run$value), collapse = ";"),
    result_residency = as.character(knn_run$value$result_residency),
    result_copies = as.integer(
      knn_run$value$device_to_host_result_copies
    ),
    cpu_fallback = isTRUE(knn_run$value$cpu_fallback),
    backend_used = as.character(knn_run$value$backend_used),
    provider = as.character(knn_run$value$gpu_provider)
  )
  utils::write.csv(
    cuda_knn_metadata,
    file.path(out_dir, "cuda-knn-metadata.csv"),
    row.names = FALSE
  )
  if (!inherits(knn_run$value, "fastEmbedR_gpu_knn") ||
      !identical(knn_run$value$result_residency, "cuda") ||
      !isTRUE(knn_run$value$device_to_host_result_copies == 0L) ||
      isTRUE(knn_run$value$cpu_fallback)) {
    stop("CUDA KNN did not remain on the device.", call. = FALSE)
  }
  if (!startsWith(knn_run$value$gpu_provider, "fastEmbedR_native_")) {
    stop("CUDA KNN did not report a native GPU provider.", call. = FALSE)
  }
  observed <- fastEmbedR:::fastembedr_gpu_knn_to_host(knn_run$value)
  distance <- as.matrix(stats::dist(x))
  diag(distance) <- Inf
  reference <- t(apply(distance, 1L, function(values) {
    order(values)[seq_len(15L)]
  }))
  recall <- mean(vapply(seq_len(n), function(index) {
    length(intersect(reference[index, ], observed$indices[index, ])) / 15
  }, numeric(1L)))
  if (!isTRUE(all.equal(recall, 1, tolerance = 1e-7))) {
    stop("CUDA exact KNN disagrees with the independent CPU oracle.",
         call. = FALSE)
  }
  utils::write.csv(
    data.frame(
      operation = "exact_knn",
      reference = "base_R_dist",
      recall_at_15 = recall,
      backend_used = knn_run$value$backend_used,
      provider = knn_run$value$gpu_provider,
      result_residency = knn_run$value$result_residency,
      cpu_fallback = isTRUE(knn_run$value$cpu_fallback)
    ),
    file.path(out_dir, "cuda-knn-reference.csv"),
    row.names = FALSE
  )
}
umap_run <- elapsed("umap", umap(
  x, n_neighbors = 15L, metric = "euclidean", backend = backend,
  n.cores = 4L, seed = seed, graph_mode = "fuzzy"
))
tsne_run <- elapsed("tsne", tsne(
  x, perplexity = 15L, metric = "euclidean", backend = backend,
  n.cores = 4L, seed = seed, early_exaggeration_iter = 10L,
  n_iter = 20L, auto_config = FALSE, negative_gradient_method = "fft"
))
umap_knn_run <- elapsed("umap_knn", umap_knn(
  knn_run$value, backend = backend, seed = seed, graph_mode = "fuzzy"
))
tsne_knn_run <- elapsed("tsne_knn", tsne_knn(
  knn_run$value, perplexity = 15L, backend = backend, seed = seed,
  early_exaggeration_iter = 10L, n_iter = 20L, auto_config = FALSE,
  negative_gradient_method = "fft"
))

selection <- select_landmarks(x, landmarks = 0.5, seed = seed)
landmark_umap_run <- elapsed("landmark_umap", landmark_umap(
  x, landmarks = selection$indices, n_neighbors = 15L,
  backend = backend, n.cores = 4L, seed = seed,
  transform_k = 15L, graph_mode = "fuzzy"
))
landmark_tsne_run <- elapsed("landmark_tsne", landmark_tsne(
  x, landmarks = selection$indices, perplexity = 15,
  backend = backend, n.cores = 4L, seed = seed,
  transform_k = 15L, transform_perplexity = 5,
  transform_iter = 10L, early_exaggeration_iter = 10L,
  n_iter = 20L, auto_config = FALSE,
  negative_gradient_method = "fft"
))

graph <- knn_graph(
  knn_run$value, weight = "snn", backend = backend, n.cores = 4L
)
cluster_run <- elapsed("leiden", graph_cluster(
  graph, method = "leiden", backend = backend, n_iterations = 3L,
  n_runs = 1L, seed = seed
))

if (!identical(attr(knn_run$value, "backend"), backend)) {
  stop("KNN backend mismatch", call. = FALSE)
}
if (!startsWith(pca_run$value$backend, backend)) {
  stop("PCA backend mismatch: ", pca_run$value$backend, call. = FALSE)
}
umap_backend <- attr(umap_run$value$layout, "fastEmbedR_config")$backend
tsne_backend <- attr(tsne_run$value$layout, "fastEmbedR_config")$backend
umap_knn_backend <- attr(
  umap_knn_run$value, "fastEmbedR_config"
)$backend
tsne_knn_backend <- attr(
  tsne_knn_run$value, "fastEmbedR_config"
)$backend
if (!identical(umap_backend, backend)) {
  stop("UMAP backend mismatch", call. = FALSE)
}
if (!identical(tsne_backend, backend)) {
  stop("t-SNE backend mismatch", call. = FALSE)
}
if (!identical(umap_knn_backend, backend)) {
  stop("precomputed-KNN UMAP backend mismatch", call. = FALSE)
}
if (!identical(tsne_knn_backend, backend)) {
  stop("precomputed-KNN t-SNE backend mismatch", call. = FALSE)
}
if (!identical(landmark_umap_run$value$parameters$backend, backend) ||
    !identical(landmark_umap_run$value$parameters$projection_backend,
               backend)) {
  stop("landmark UMAP backend mismatch", call. = FALSE)
}
if (!identical(landmark_tsne_run$value$parameters$backend, backend) ||
    !identical(landmark_tsne_run$value$parameters$projection_backend,
               backend)) {
  stop("landmark t-SNE backend mismatch", call. = FALSE)
}
if (!identical(cluster_run$value$backend_requested, backend) ||
    !identical(cluster_run$value$backend, backend)) {
  stop("Leiden backend mismatch", call. = FALSE)
}

layouts <- list(
  pca = pca_run$value$scores,
  umap = umap_run$value$layout,
  tsne = tsne_run$value$layout,
  umap_knn = umap_knn_run$value,
  tsne_knn = tsne_knn_run$value,
  landmark_umap = landmark_umap_run$value$layout,
  landmark_tsne = landmark_tsne_run$value$layout,
  leiden_membership = cluster_run$value$membership
)
saveRDS(layouts, file.path(out_dir, "validation-results.rds"), version = 3L)

summary <- data.frame(
  operation = c(
    "pca", "knn", "umap", "tsne", "umap_knn", "tsne_knn",
    "landmark_umap", "landmark_tsne", "leiden"
  ),
  backend_requested = backend,
  backend_used = c(
    pca_run$value$backend,
    attr(knn_run$value, "backend"),
    umap_backend,
    tsne_backend,
    umap_knn_backend,
    tsne_knn_backend,
    landmark_umap_run$value$parameters$projection_backend,
    landmark_tsne_run$value$parameters$projection_backend,
    cluster_run$value$backend
  ),
  elapsed_sec = c(
    pca_run$elapsed, knn_run$elapsed, umap_run$elapsed,
    tsne_run$elapsed, umap_knn_run$elapsed, tsne_knn_run$elapsed,
    landmark_umap_run$elapsed, landmark_tsne_run$elapsed,
    cluster_run$elapsed
  ),
  n = n,
  p = p,
  seed = seed,
  stringsAsFactors = FALSE
)
utils::write.csv(
  summary, file.path(out_dir, "hardware-benchmark.csv"),
  row.names = FALSE
)

cat("All public operations used the requested", backend, "backend.\n")
print(summary)
