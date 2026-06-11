make_cluster_data <- function(n = 18L, p = 5L) {
  labels <- rep(seq_len(3L), length.out = n)
  x <- matrix(rnorm(n * p, sd = 0.25), nrow = n, ncol = p)
  x <- x + labels
  list(x = x, labels = labels)
}

expect_embedding <- function(layout, n) {
  expect_equal(dim(layout), c(n, 2L))
  expect_true(all(is.finite(layout)))
}

test_that("public API is UMAP, t-SNE, and KNN focused", {
  exports <- getNamespaceExports("fastEmbedR")
  expect_true(all(c(
    "umap", "tsne", "embed_knn", "nn", "backend_info",
    "metal_available", "cuda_available", "cuvs_available", "evaluate_embedding",
    "transform_embedding", "transform_tsne", "landmark_tsne"
  ) %in% exports))
  expect_false(any(c("pacmap", "trimap", "localmap") %in% exports))
  expect_false(any(c("knn_tsne", "knn_pacmap", "knn_trimap", "knn_localmap") %in% exports))
})

test_that("core exported functions have tiny smoke tests", {
  set.seed(101)
  fixture <- make_cluster_data()
  x <- fixture$x
  labels <- fixture$labels
  n <- nrow(x)

  expect_type(metal_available(), "logical")
  expect_length(metal_available(), 1L)
  expect_type(cuda_available(), "logical")
  expect_length(cuda_available(), 1L)
  expect_type(cuvs_available(), "logical")
  expect_length(cuvs_available(), 1L)

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

  layout_umap <- embed_knn(knn, method = "umap")
  expect_embedding(layout_umap, n)
  layout_tsne <- embed_knn(knn, method = "tsne", perplexity = 1, max_iter = 5L)
  expect_embedding(layout_tsne, n)

  fit <- umap(x, labels = labels, n_neighbors = 4L,
    silhouette_sample = NULL, preserve_sample = NULL)
  expect_s3_class(fit, "fastEmbedR_embedding")
  expect_embedding(fit$layout, n)
  expect_null(fit$knn)

  fit_tsne <- tsne(x, labels = labels, n_neighbors = 4L, perplexity = 1,
    max_iter = 5L, silhouette_sample = NULL, preserve_sample = NULL)
  expect_s3_class(fit_tsne, "fastEmbedR_embedding")
  expect_embedding(fit_tsne$layout, n)
  expect_equal(fit_tsne$parameters$method, "tsne")

  sup <- supervised_umap(x, labels = labels, n_neighbors = 4L,
    silhouette_sample = NULL, preserve_sample = NULL)
  expect_embedding(sup$layout, n)
  expect_true(isTRUE(sup$parameters$supervised))

  projected <- transform_embedding(fit$layout, nn(x, x[1:3, , drop = FALSE],
    k = 4L, backend = "cpu"))
  expect_embedding(projected, 3L)
  expect_equal(attr(projected, "backend"), "cpu")

  scores <- evaluate_embedding(
    x,
    fit$layout,
    labels = labels,
    k = c(4L, 5L),
    sample_size_for_global_metrics = n,
    sample_size_for_local_metrics = n,
    use_cache = FALSE,
    method = "umap",
    backend = "cpu",
    dataset = "toy"
  )
  expect_true(all(c(
    "trustworthiness", "knn_preservation", "silhouette",
    "density_spearman", "density_log_radius_rmse", "rare_class_recall"
  ) %in% names(scores)))
})
