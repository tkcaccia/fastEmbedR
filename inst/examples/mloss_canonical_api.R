x <- scale(as.matrix(iris[, 1:4]))
knn <- fastEmbedR::precompute_knn(x, k = 15L, backend = "cpu")
fit_umap <- fastEmbedR::umap(x, n_neighbors = 15L, backend = "cpu")
layout_tsne <- fastEmbedR::tsne_knn(knn, perplexity = 5, init_data = x)
landmark_fit <- fastEmbedR::umap(
  x, landmarks = 0.5, n_neighbors = 15L, backend = "cpu"
)
stopifnot(fit_umap$parameters$backend == "cpu",
  attr(knn, "backend") == "cpu", nrow(landmark_fit$layout) == 150L)
