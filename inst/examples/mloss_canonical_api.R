x <- scale(as.matrix(iris[, 1:4]))
capabilities <- fastEmbedR::fastEmbedR_capabilities()
knn <- fastEmbedR::precompute_knn(x, k = 15L, backend = "cpu")
fit_umap <- fastEmbedR::umap(x, n_neighbors = 15L, backend = "cpu")
layout_tsne <- fastEmbedR::tsne_knn(knn, perplexity = 5, init_data = x)
selection <- fastEmbedR::select_landmarks(x, landmarks = 0.5, seed = 4L)
model <- fastEmbedR::fit_landmark_model(x, selection, method = "umap",
  n_neighbors = 15L, backend = "cpu")
projected <- fastEmbedR::project_landmark_model(model, x[1:5, ],
  transform_k = 15L, refinement_epochs = 2L)
stopifnot(fit_umap$parameters$backend == "cpu",
  attr(knn, "backend") == "cpu", nrow(projected) == 5L)
