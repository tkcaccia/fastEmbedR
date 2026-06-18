make_cluster_data <- function(n = 18L, p = 5L) {
  labels <- rep(seq_len(3L), length.out = n)
  x <- matrix(rnorm(n * p, sd = 0.25), nrow = n, ncol = p)
  x <- x + labels
  list(x = x)
}

expect_embedding <- function(layout, n) {
  expect_equal(dim(layout), c(n, 2L))
  expect_true(all(is.finite(layout)))
}

test_that("public API is KNN and openTSNE focused", {
  exports <- getNamespaceExports("fastEmbedR")
  expect_true(all(c(
    "umap", "umap_knn", "opentsne", "opentsne_knn", "embed_knn", "backend_info",
    "metal_available", "cuda_available",
    "evaluate_embedding", "transform_tsne", "landmark_tsne",
    "nn", "nn_without_self", "candidate_knn", "fast_kmeans",
    "knn_fit", "predict_proba", "knn_recall", "faiss_available", "cuvs_available"
  ) %in% exports))
  expect_false(any(c(
    "supervised_umap", "tsne", "infotsne", "pacmap", "trimap",
    "localmap", "transform_embedding", "knn_graph"
  ) %in% exports))

  expect_true("n_threads" %in% names(formals(opentsne)))
  expect_true("n_threads" %in% names(formals(opentsne_knn)))
  expect_true("n_threads" %in% names(formals(umap)))
  expect_true("n_threads" %in% names(formals(umap_knn)))
  expect_true("n_threads" %in% names(formals(embed_knn)))
  expect_true("n_threads" %in% names(formals(landmark_tsne)))
  expect_true("n_threads" %in% names(formals(transform_tsne)))
  expect_true("n_threads" %in% names(formals(evaluate_embedding)))
})

test_that("core exported functions have tiny openTSNE smoke tests", {
  set.seed(101)
  fixture <- make_cluster_data()
  x <- fixture$x
  labels <- fixture$labels
  n <- nrow(x)

  expect_type(metal_available(), "logical")
  expect_length(metal_available(), 1L)
  expect_type(cuda_available(), "logical")
  expect_length(cuda_available(), 1L)
  expect_type(faissR::cuvs_available(), "logical")
  expect_length(faissR::cuvs_available(), 1L)

  info <- backend_info()
  expect_s3_class(info, "data.frame")
  expect_true(all(c("backend", "available", "knn_available", "embedding_available") %in% names(info)))
  expect_true(all(c("cpu", "cuvs", "cuda", "metal") %in% info$backend))
  expect_true(isTRUE(info$available[info$backend == "cpu"]))

  knn <- nn(x, backend = "cpu")
  expect_s3_class(knn, "fastEmbedR_nn")
  expect_equal(dim(knn$indices), c(n, fastEmbedR:::auto_k(x, include_self = TRUE)))
  expect_equal(dim(knn$distances), c(n, fastEmbedR:::auto_k(x, include_self = TRUE)))
  expect_equal(attr(knn, "backend"), "cpu")
  expect_identical(faiss_available(), faissR::faiss_available())
  expect_identical(cuvs_available(), faissR::cuvs_available())

  layout <- embed_knn(knn, method = "opentsne", perplexity = 1, early_exaggeration_iter = 2L, n_iter = 3L)
  expect_embedding(layout, n)
  expect_equal(attr(layout, "fastEmbedR_config")$method, "opentsne")

  layout_umap <- embed_knn(knn, method = "umap")
  expect_embedding(layout_umap, n)
  expect_equal(attr(layout_umap, "fastEmbedR_config")$method, "umap")

  layout_umap_knn <- umap_knn(knn)
  expect_embedding(layout_umap_knn, n)
  expect_equal(attr(layout_umap_knn, "fastEmbedR_config")$method, "umap")
  expect_equal(
    attr(layout_umap_knn, "fastEmbedR_config")$auto_parameter_backend,
    "cpp_knn_distance_profile"
  )

  layout_knn <- opentsne_knn(knn, perplexity = 1, early_exaggeration_iter = 2L, n_iter = 3L)
  expect_embedding(layout_knn, n)
  expect_equal(attr(layout_knn, "fastEmbedR_config")$method, "opentsne")

  fit <- opentsne(x, n_neighbors = 4L, perplexity = 1,
    early_exaggeration_iter = 2L, n_iter = 3L,
    n_threads = 2L)
  expect_s3_class(fit, "fastEmbedR_embedding")
  expect_embedding(fit$layout, n)
  expect_equal(fit$parameters$method, "opentsne")
  expect_equal(fit$parameters$n_threads, 2L)
  expect_null(fit$knn)

  fit_knn <- opentsne(knn, perplexity = 1,
    early_exaggeration_iter = 2L, n_iter = 3L)
  expect_s3_class(fit_knn, "fastEmbedR_embedding")
  expect_embedding(fit_knn$layout, n)
  expect_equal(fit_knn$parameters$input, "knn")
  expect_equal(fit_knn$metrics$knn_elapsed, 0)

  scores <- evaluate_embedding(
    x,
    fit$layout,
    k = c(4L, 5L),
    sample_size_for_global_metrics = n,
    sample_size_for_local_metrics = n,
    use_cache = FALSE,
    method = "opentsne",
    backend = "cpu",
    dataset = "toy"
  )
  expect_true(all(c(
    "trustworthiness", "knn_preservation", "silhouette",
    "density_spearman", "density_log_radius_rmse", "rare_class_recall"
  ) %in% names(scores)))

  gpu_scores <- evaluate_embedding(
    x,
    fit$layout,
    k = c(4L, 5L),
    sample_size_for_global_metrics = n,
    sample_size_for_local_metrics = n,
    use_cache = FALSE,
    method = "opentsne",
    backend = "gpu",
    n_threads = 2L,
    dataset = "toy"
  )
  if (isTRUE(fastEmbedR:::cuda_metric_available())) {
    expect_equal(gpu_scores$metric_backend, "cuda")
  } else {
    expect_equal(gpu_scores$metric_backend, "cpu")
    expect_match(gpu_scores$metric_backend_reason, "metric_backend_unavailable")
  }
})
