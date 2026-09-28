test_that("Metal UMAP optimizes all three output coordinates", {
    skip_if_not(embedding_metal_available_cpp())
    x <- scale(as.matrix(iris[seq_len(60L), 1:4]))
    for (mode in c("fuzzy", "binary")) {
        fit <- umap(
            x, n_neighbors = 10L, n_components = 3L,
            backend = "metal", graph_mode = mode, seed = 4L
        )
        layout <- as.matrix(fit$layout)
        expect_identical(dim(layout), c(60L, 3L))
        expect_true(all(is.finite(layout)))
        expect_gt(stats::sd(layout[, 3L]), 0)
        expect_identical(fit$parameters$optimizer_backend, "metal")
        expect_identical(fit$parameters$backend, "metal")
        expect_match(fit$parameters$init_backend, "cpu")
    }
})

test_that("Metal 3D UMAP preserves local neighborhoods", {
    skip_if_not(embedding_metal_available_cpp())
    x <- scale(as.matrix(iris[, 1:4]))
    fit <- umap(
        x, n_neighbors = 15L, n_components = 3L,
        backend = "metal", seed = 4L
    )
    quality <- evaluate_embedding(x, fit$layout, k = 15L)
    expect_gt(quality$trustworthiness, 0.9)
    expect_identical(fit$parameters$backend, "metal")
    expect_identical(fit$parameters$optimizer_backend, "metal")
})

test_that("Metal UMAP accepts precomputed neighbors in 3D", {
    skip_if_not(embedding_metal_available_cpp())
    x <- scale(as.matrix(iris[seq_len(45L), 1:4]))
    knn <- precompute_knn(x, k = 10L, backend = "cpu")
    for (mode in c("fuzzy", "binary")) {
        layout <- umap_knn(
            knn, n_components = 3L, backend = "metal",
            graph_mode = mode, seed = 11L
        )
        expect_identical(dim(layout), c(45L, 3L))
        expect_true(all(is.finite(layout)))
        expect_gt(stats::sd(layout[, 3L]), 0)
        expect_identical(
            attr(layout, "fastEmbedR_config")$backend, "metal"
        )
    }

    prepared <- prepare_umap_knn(knn, backend = "metal")
    reused <- umap_knn(
        prepared, n_components = 3L, backend = "metal", seed = 11L
    )
    expect_identical(dim(reused), c(45L, 3L))
    expect_true(all(is.finite(as.matrix(reused))))
    expect_identical(
        attr(reused, "fastEmbedR_config")$optimizer_backend,
        "metal"
    )
})

test_that("Metal 3D landmarks never use CPU refinement silently", {
    skip_if_not(embedding_metal_available_cpp())
    x <- scale(as.matrix(iris[seq_len(45L), 1:4]))
    expect_error(
        umap(
            x, n_neighbors = 10L, n_components = 3L,
            landmarks = 0.5, backend = "metal", seed = 11L
        ),
        "Metal 3D landmark refinement is not available"
    )
})
