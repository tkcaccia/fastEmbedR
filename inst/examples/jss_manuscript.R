library("fastEmbedR")

x <- scale(as.matrix(iris[, 1:4]))

old_options <- options(backend = "cpu", n.cores = 1L)

fit_tsne <- fastEmbedR::tsne(
    x, perplexity = 5, backend = "cpu", n.cores = 1,
    early_exaggeration_iter = 50, n_iter = 200, seed = 4
)
fit_umap <- fastEmbedR::umap(
    x, n_neighbors = 15, graph_mode = "fuzzy",
    backend = "cpu", n.cores = 1, seed = 4
)

plot_file <- tempfile(fileext = ".pdf")
grDevices::pdf(plot_file)
plot(fit_tsne, labels = iris$Species)
plot(fit_umap, labels = iris$Species)
grDevices::dev.off()
unlink(plot_file)

knn <- fastEmbedR::precompute_knn(
    x, k = 15, metric = "euclidean",
    backend = "cpu", n.cores = 1
)
y_umap <- fastEmbedR::umap_knn(
    knn, graph_mode = "fuzzy", backend = "cpu", seed = 4
)
y_tsne <- fastEmbedR::tsne_knn(
    knn, perplexity = 5, backend = "cpu", seed = 4
)

pc <- fastEmbedR::pca(
    x, ncomp = 2, backend = "cpu", n.cores = 1,
    seed = 4, tsne_init = TRUE
)
fit <- fastEmbedR::tsne(
    x, perplexity = 5, Y_init = pc$tsne_init,
    backend = "cpu", n.cores = 1, seed = 4
)

landmark_fit <- fastEmbedR::umap(
    x, landmarks = 0.5, n_neighbors = 15,
    backend = "cpu", n.cores = 1, seed = 4
)

options(old_options)

stopifnot(
    identical(dim(fit_tsne$layout), c(150L, 2L)),
    identical(dim(fit_umap$layout), c(150L, 2L)),
    identical(dim(y_umap), c(150L, 2L)),
    identical(dim(y_tsne), c(150L, 2L)),
    identical(dim(fit$layout), c(150L, 2L)),
    identical(dim(landmark_fit$layout), c(150L, 2L))
)
