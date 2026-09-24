test_that(paste(
    "CUDA preprocessing, projection, interpolation, and",
    "scoring match CPU"
), {
    if (!isTRUE(fastEmbedR:::embedding_cuda_available_cpp())) {
        skip("CUDA embedding backend is not available in this build.")
    }

    set.seed(71)
    x <- matrix(rnorm(160), nrow = 40L, ncol = 4L)
    cpu_pre <- fastEmbedR:::prepare_embedding_data(
        x,
        standardize = TRUE,
        pca_dims = NULL,
        seed = 71L,
        backend = "cpu"
    )
    cuda_pre <- fastEmbedR:::prepare_embedding_data(
        x,
        standardize = TRUE,
        pca_dims = NULL,
        seed = 71L,
        backend = "cuda"
    )
    expect_equal(cuda_pre$preprocess$standardize_backend, "cuda")
    expect_equal(cuda_pre$data, cpu_pre$data, tolerance = 1e-10)

    x_pca <- matrix(rnorm(60L * 30L), nrow = 60L, ncol = 30L)
    cuda_pca <- fastEmbedR:::prepare_embedding_data(
        x_pca,
        standardize = TRUE,
        pca_dims = 4L,
        seed = 71L,
        backend = "cuda"
    )
    raft_compiled <- isTRUE(
        fastEmbedR:::fastembedr_build_config_cpp()$raft_compiled
    )
    expected_pca_backend <- if (raft_compiled) {
        "cuda_raft_tsvd"
    } else {
        "cuda_native_rsvd"
    }
    expected_pca_method <- if (raft_compiled) {
        "raft_tsvd"
    } else {
        "rsvd"
    }
    expect_equal(dim(cuda_pca$data), c(60L, 4L))
    expect_equal(cuda_pca$preprocess$pca_backend, expected_pca_backend)
    expect_equal(cuda_pca$preprocess$pca_method, expected_pca_method)

    reference_layout <- cbind(rnorm(8L), rnorm(8L))
    projection_indices <- matrix(
        c(
            1L, 2L, 3L,
            3L, 4L, 5L,
            5L, 6L, 7L,
            2L, 4L, 8L,
            1L, 7L, 8L
        ),
        nrow = 5L,
        byrow = TRUE
    )
    projection_distances <- matrix(
        c(
            0.0, 0.4, 0.7,
            0.2, 0.3, 0.9,
            0.1, 0.6, 0.8,
            0.5, 0.6, 0.7,
            0.3, 0.4, 0.5
        ),
        nrow = 5L,
        byrow = TRUE
    )
    cpu_project <- fastEmbedR:::project_embedding_knn_cpp(
        reference_layout,
        projection_indices,
        projection_distances
    )
    cuda_project <- fastEmbedR:::project_embedding_knn_cuda_cpp(
        reference_layout,
        projection_indices,
        projection_distances
    )
    expect_equal(cuda_project, cpu_project, tolerance = 1e-7)

    landmark_indices <- c(2L, 5L, 8L, 11L, 14L, 17L, 20L, 23L)
    landmark_projection_indices <- matrix(
        rep(c(1L, 2L, 3L, 4L), length.out = 24L * 4L),
        nrow = 24L,
        ncol = 4L
    )
    landmark_projection_distances <- matrix(
        abs(rnorm(24L * 4L)) + 0.01,
        nrow = 24L,
        ncol = 4L
    )
    for (i in seq_along(landmark_indices)) {
        landmark_projection_indices[landmark_indices[i], 1L] <- i
        landmark_projection_distances[landmark_indices[i], 1L] <- 0
    }
    cpu_interp <- fastEmbedR:::interpolate_landmark_layout_cpp(
        reference_layout,
        as.integer(landmark_indices),
        landmark_projection_indices,
        landmark_projection_distances,
        24L
    )
    cuda_interp <- fastEmbedR:::interpolate_landmark_layout_cuda_cpp(
        reference_layout,
        as.integer(landmark_indices),
        landmark_projection_indices,
        landmark_projection_distances,
        24L
    )
    expect_equal(cuda_interp, cpu_interp, tolerance = 1e-7)

    layout <- cbind(rnorm(30L), rnorm(30L))
    labels <- rep(1:3, each = 10L)
    knn <- test_exact_knn(layout, layout, k = 7L, backend = "cpu")
    high_indices <- knn$indices[, -1L, drop = FALSE]
    keep <- seq_len(nrow(layout))
    cpu_score <- fastEmbedR:::knn_structure_score_cpp(
        layout,
        high_indices,
        as.integer(keep),
        6L,
        as.integer(labels),
        3L
    )
    cuda_score <- fastEmbedR:::knn_structure_score_cuda_cpp(
        layout,
        high_indices,
        as.integer(keep),
        6L,
        as.integer(labels),
        3L
    )
    expect_equal(cuda_score, cpu_score, tolerance = 1e-8)

    cpu_sil <- fastEmbedR:::silhouette_score_cpp(layout, as.integer(labels))
    cuda_sil <- fastEmbedR:::silhouette_score_cuda_cpp(
        layout, as.integer(labels), 3L
    )
    expect_equal(cuda_sil, cpu_sil, tolerance = 1e-10)
})

test_that("CUDA PCA preserves float32 input and never falls back", {
    skip_if_not_installed("float")
    if (!isTRUE(fastEmbedR:::embedding_cuda_available_cpp())) {
        skip("CUDA embedding backend is not available in this build.")
    }

    set.seed(72)
    x <- float::fl(matrix(rnorm(80L * 12L), nrow = 80L))
    fit <- fastEmbedR::pca(
        x,
        ncomp = 4L,
        center = TRUE,
        scale = TRUE,
        backend = "cuda",
        seed = 72L
    )

    raft_compiled <- isTRUE(
        fastEmbedR:::fastembedr_build_config_cpp()$raft_compiled
    )
    expected_backend <- if (raft_compiled) {
        "cuda_raft_tsvd"
    } else {
        "cuda_native_rsvd"
    }
    expected_method <- if (raft_compiled) "raft_tsvd" else "rsvd"
    expect_equal(fit$backend, expected_backend)
    expect_equal(fit$method, expected_method)
    expect_equal(fit$precision, "float32")
    expect_s4_class(fit$scores, "float32")
    expect_s4_class(fit$loadings, "float32")
    expect_equal(dim(fit$scores), c(80L, 4L))
    expect_equal(dim(fit$loadings), c(12L, 4L))

    prepared <- fastEmbedR:::prepare_embedding_data(
        x,
        standardize = FALSE,
        pca_dims = 4L,
        seed = 72L,
        backend = "cuda"
    )
    expect_equal(prepared$preprocess$pca_backend, expected_backend)
    expect_s4_class(prepared$data, "float32")
})

test_that("CUDA PCA selects rSVD for wide low-rank input", {
    if (!isTRUE(fastEmbedR:::embedding_cuda_available_cpp())) {
        skip("CUDA embedding backend is not available in this build.")
    }

    set.seed(73)
    left <- matrix(rnorm(320L * 5L), nrow = 320L)
    right <- matrix(rnorm(1536L * 5L), nrow = 1536L)
    x <- left %*% t(right)
    x <- x + matrix(rnorm(length(x), sd = 0.01), nrow = nrow(x))

    fit <- fastEmbedR:::fastembedr_cuda_pca(
        x, ncomp = 2L, seed = 73L, method = "auto"
    )
    repeated <- fastEmbedR:::fastembedr_cuda_pca(
        x, ncomp = 2L, seed = 73L, method = "rsvd"
    )

    expect_identical(fit$backend, "cuda_native_rsvd")
    expect_identical(fit$method, "rsvd")
    expect_identical(fit$selection, "auto")
    expect_equal(fit$scores, repeated$scores, tolerance = 1e-6)
    raft_compiled <- isTRUE(
        fastEmbedR:::fastembedr_build_config_cpp()$raft_compiled
    )
    if (raft_compiled) {
        reference <- fastEmbedR:::fastembedr_cuda_pca(
            x, ncomp = 2L, seed = 73L, method = "tsvd"
        )
        agreement <- svd(
            crossprod(
                qr.Q(qr(fit$loadings)),
                qr.Q(qr(reference$loadings))
            ),
            nu = 0L,
            nv = 0L
        )$d
        expect_gte(min(agreement), 0.995)
    } else {
        expect_error(
            fastEmbedR:::fastembedr_cuda_pca(
                x, ncomp = 2L, seed = 73L, method = "tsvd"
            ),
            "RAFT TSVD was requested"
        )
    }

    embedding <- fastEmbedR::tsne(
        x,
        perplexity = 5,
        backend = "cuda",
        seed = 73L,
        early_exaggeration_iter = 1L,
        n_iter = 2L,
        auto_config = FALSE
    )
    expect_identical(
        embedding$parameters$initialization,
        "pca_cuda_native_rsvd_device"
    )
    expect_identical(
        embedding$parameters$initialization_requested,
        "pca_cuda_auto_device"
    )
})
