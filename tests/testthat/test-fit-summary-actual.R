test_that("four resident entry points return a common summary schema", {
    x <- as.matrix(iris[, 1:4])
    fitted <- list(
        pca = pca(x, ncomp = 2L, backend = "cpu"),
        tsne = tsne(x, perplexity = 5, affinity_k = 15L,
            n_iter = 20L, early_exaggeration_iter = 5L,
            backend = "cpu"),
        umap = umap(x, n_neighbors = 10L, backend = "cpu"),
        cluster = graph_cluster(x, method = "louvain",
            k = 10L, backend = "cpu"))
    required <- c("backend", "devices", "device_identity",
        "scope", "stage_seconds", "memory_use", "output_rows")
    for (fit in fitted) {
        info <- summary(fit)
        expect_identical(names(info), required)
        expect_match(info$backend, "cpu")
        expect_identical(info$scope, "full")
        expect_true(is.list(info$stage_seconds))
        expect_true(is.list(info$memory_use))
    }
})
