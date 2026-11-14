test_that("CUDA UMAP keeps three-dimensional KNN on device", {
    skip_if_not(embedding_cuda_available_cpp())
    skip_if_not(fastembedr_cuda_knn_available())
    set.seed(81)
    x <- matrix(rnorm(60L * 4L), nrow = 60L)
    gpu_knn <- precompute_knn(x, k = 10L, backend = "cuda")
    expect_s3_class(gpu_knn, "fastEmbedR_gpu_knn")

    for (mode in c("fuzzy", "binary")) {
        fit <- umap(
            x, nn = gpu_knn, n_neighbors = 10L,
            n_components = 3L, backend = "cuda",
            graph_mode = mode, seed = 81L
        )
        layout <- as.matrix(fit$layout)
        expect_identical(dim(layout), c(60L, 3L))
        expect_true(all(is.finite(layout)))
        expect_gt(sd(layout[, 3L]), 0)
        expect_identical(fit$parameters$optimizer_backend, "cuda")
        expect_identical(fit$parameters$knn_residency, "cuda_device")
    }

    full <- umap(
        x, n_neighbors = 10L, n_components = 3L,
        backend = "cuda", seed = 81L
    )
    expect_identical(dim(full$layout), c(60L, 3L))
    expect_identical(full$parameters$optimizer_backend, "cuda")
    expect_identical(full$parameters$knn_residency, "cuda_device")

    float_fit <- umap(
        float::fl(x), n_neighbors = 10L, n_components = 3L,
        backend = "cuda", seed = 81L
    )
    expect_s4_class(float_fit$layout, "float32")
    expect_identical(dim(float_fit$layout), c(60L, 3L))
})

test_that("CUDA UMAP accepts host KNN in three dimensions", {
    skip_if_not(embedding_cuda_available_cpp())
    set.seed(82)
    x <- matrix(rnorm(48L * 4L), nrow = 48L)
    host_knn <- precompute_knn(x, k = 8L, backend = "cpu")
    layout <- umap_knn(
        host_knn, n_components = 3L, backend = "cuda",
        graph_mode = "fuzzy", seed = 82L
    )
    expect_identical(dim(layout), c(48L, 3L))
    expect_true(all(is.finite(as.matrix(layout))))
    expect_gt(sd(as.matrix(layout)[, 3L]), 0)
    expect_identical(
        attr(layout, "fastEmbedR_config")$optimizer_backend,
        "cuda"
    )
})

test_that("CUDA 3D UMAP reports nonfinite optimizer coordinates", {
    skip_if_not(embedding_cuda_available_cpp())
    set.seed(83)
    x <- matrix(rnorm(48L * 4L), nrow = 48L)
    knn <- precompute_knn(x, k = 8L, backend = "cpu")
    expect_error(
        fastEmbedR:::fast_knn_umap_core(
            knn, n_components = 3L, backend = "cuda",
            n_epochs = 2L, seed = 83L,
            config_override = list(learning_rate = 1e100)
        ),
        "nonfinite coordinates"
    )
})

test_that("CUDA landmark UMAP keeps fixed references in three dimensions", {
    skip_if_not(embedding_cuda_available_cpp())
    skip_if_not(fastembedr_cuda_knn_available())
    x <- scale(as.matrix(iris[seq_len(45L), 1:4]))
    for (mode in c("fuzzy", "binary")) {
        fit <- umap(
            x, n_neighbors = 10L, n_components = 3L,
            landmarks = 0.5, backend = "cuda",
            graph_mode = mode, seed = 11L
        )
        layout <- as.matrix(fit$layout)
        expect_identical(dim(layout), c(45L, 3L))
        expect_true(all(is.finite(layout)))
        expect_gt(sd(layout[, 3L]), 0)
        expect_equal(layout[fit$landmarks$indices, , drop = FALSE],
            as.matrix(fit$landmarks$layout), tolerance = 1e-5)
        expect_identical(fit$parameters$landmark_refinement_backend,
            "cuda")
        held_out <- project_landmark_model(
            fit$model, x[1:3, , drop = FALSE], transform_k = 10L,
            refinement_epochs = 5L
        )
        expect_identical(dim(held_out$layout), c(3L, 3L))
        expect_true(all(is.finite(as.matrix(held_out$layout))))
    }
})
