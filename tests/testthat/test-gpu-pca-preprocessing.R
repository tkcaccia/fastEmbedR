test_that("CUDA PCA preprocesses numeric and float32 input on CUDA", {
    skip_if_not_installed("float")
    if (!isTRUE(fastEmbedR:::embedding_cuda_available_cpp())) {
        skip("CUDA embedding backend is unavailable.")
    }

    set.seed(921)
    x <- matrix(rnorm(120L * 18L, mean = 7), 120L, 18L)
    expected_center <- colMeans(x)
    expected_scale <- apply(x, 2L, stats::sd)
    numeric_fit <- fastEmbedR::pca(
        x, ncomp = 3L, center = TRUE, scale = TRUE,
        backend = "cuda", seed = 921L
    )
    float_fit <- fastEmbedR::pca(
        float::fl(x), ncomp = 3L, center = TRUE, scale = TRUE,
        backend = "cuda", seed = 921L
    )

    expect_match(numeric_fit$backend, "^cuda_")
    expect_match(float_fit$backend, "^cuda_")
    expect_s4_class(float_fit$scores, "float32")
    expect_equal(numeric_fit$center, expected_center, tolerance = 2e-5)
    expect_equal(numeric_fit$scale, expected_scale, tolerance = 2e-5)
    expect_equal(float_fit$center, expected_center, tolerance = 2e-5)
    expect_equal(float_fit$scale, expected_scale, tolerance = 2e-5)

    x[1L, 1L] <- Inf
    expect_error(
        fastEmbedR::pca(x, ncomp = 2L, backend = "cuda"),
        "finite"
    )
})

test_that("CUDA randomized PCA scores project onto the fitted loadings", {
    skip_if_not_installed("float")
    skip_if_not(isTRUE(fastEmbedR:::embedding_cuda_available_cpp()))

    set.seed(944)
    data <- matrix(rnorm(320L * 72L, mean = 7), 320L, 72L)
    as_double <- function(x) if (inherits(x, "float32")) float::dbl(x) else x
    for (input in list(data, float::fl(data))) {
        fit <- fastEmbedR:::fastembedr_cuda_pca(
            input, ncomp = 3L, center = TRUE, scale = TRUE,
            seed = 944L, method = "rsvd")
        expect_identical(fit$backend, "cuda_native_rsvd")
        training <- scale(as_double(input), fit$center, fit$scale)
        expected <- training %*% as_double(fit$loadings)
        scores <- as_double(fit$scores)
        expect_lt(max(abs(scores - expected)) / max(abs(expected)), 1e-4)
    }
})

test_that("Metal PCA preprocesses numeric and float32 input on Metal", {
    skip_if_not_installed("float")
    if (!isTRUE(fastEmbedR:::embedding_metal_available_cpp())) {
        skip("Metal embedding backend is unavailable.")
    }

    set.seed(922)
    x <- matrix(rnorm(120L * 18L, mean = 7), 120L, 18L)
    expected_center <- colMeans(x)
    expected_scale <- apply(x, 2L, stats::sd)
    numeric_fit <- fastEmbedR::pca(
        x, ncomp = 3L, center = TRUE, scale = TRUE,
        backend = "metal", seed = 922L
    )
    float_fit <- fastEmbedR::pca(
        float::fl(x), ncomp = 3L, center = TRUE, scale = TRUE,
        backend = "metal", seed = 922L
    )

    expect_identical(numeric_fit$backend, "metal_mps_rsvd")
    expect_identical(float_fit$backend, "metal_mps_rsvd")
    expect_s4_class(float_fit$scores, "float32")
    expect_equal(numeric_fit$center, expected_center, tolerance = 2e-5)
    expect_equal(numeric_fit$scale, expected_scale, tolerance = 2e-5)
    expect_equal(float_fit$center, expected_center, tolerance = 2e-5)
    expect_equal(float_fit$scale, expected_scale, tolerance = 2e-5)

    x[1L, 1L] <- Inf
    expect_error(
        fastEmbedR::pca(x, ncomp = 2L, backend = "metal"),
        "finite"
    )
})

test_that("CUDA t-SNE reuses the resident KNN feature matrix", {
    skip_if_not_installed("float")
    if (!isTRUE(fastEmbedR:::embedding_cuda_available_cpp())) {
        skip("CUDA embedding backend is unavailable.")
    }

    set.seed(923)
    x <- float::fl(matrix(rnorm(160L * 20L), 160L, 20L))
    plain_knn <- fastEmbedR::precompute_knn(
        x, k = 8L, backend = "cuda"
    )
    fit <- fastEmbedR::tsne(
        x, perplexity = 8, backend = "cuda", keep_knn = TRUE,
        early_exaggeration_iter = 1L, n_iter = 2L,
        auto_config = FALSE, seed = 923L
    )

    expect_identical(plain_knn$data_residency, "unavailable")
    expect_equal(plain_knn$resident_result_bytes, 160 * 8 * 8)
    expect_identical(
        fit$parameters$pca_input_residency,
        "shared_cuda_knn_data"
    )
    expect_identical(fit$knn$data_residency, "cuda_device")
    expect_equal(
        fit$knn$resident_result_bytes,
        160 * 8 * 8 + 160 * 20 * 4
    )
    expect_identical(
        fit$knn$data_layout,
        "row_major_observation_by_feature"
    )
    expect_false(isTRUE(fit$parameters$cpu_fallback))

    cosine_fit <- fastEmbedR::tsne(
        x, perplexity = 8, metric = "cosine", backend = "cuda",
        keep_knn = TRUE, early_exaggeration_iter = 1L, n_iter = 2L,
        auto_config = FALSE, seed = 923L
    )
    expect_identical(
        cosine_fit$parameters$pca_input_residency,
        "host_to_cuda"
    )
    expect_identical(cosine_fit$knn$data_residency, "unavailable")
})
