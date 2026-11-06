test_that("resident PCA landmarks fit selected rows and project all rows", {
    set.seed(617)
    x <- matrix(rnorm(360L), 60L, 6L)
    xtest <- matrix(rnorm(48L), 8L, 6L)
    ids <- seq.int(5L, 60L, by = 5L)
    fit <- pca(x, ncomp = 3L, landmarks = ids,
        xtest = xtest, tsne_init = TRUE, backend = "cpu",
        n.cores = 2L, seed = 617L)
    reference <- pca(x[ids, , drop = FALSE], ncomp = 3L,
        backend = "cpu", n.cores = 2L, seed = 617L)
    expected <- fastEmbedR:::project_pca_test_scores(x, reference)
    expect_identical(fit$fit_scope,
        "landmark_fit_plus_projection")
    expect_identical(fit$projection_backend, "cpu")
    expect_identical(fit$landmarks$indices, ids)
    expect_equal(fit$scores_reference, reference$scores)
    expect_equal(fit$scores, expected, tolerance = 1e-8)
    expect_equal(dim(fit$scores), c(60L, 3L))
    expect_equal(dim(fit$scores_test), c(8L, 3L))
    expect_equal(dim(fit$tsne_init), c(60L, 3L))
    expect_error(pca(x, ncomp = 3L, landmarks = 1:3),
        "more selected landmarks")
    expect_error(pca(x, landmarks = ids,
        massive = "out_of_core"), "massive_project_pca")
    expect_error(pca(x, landmarks = ids, checkpoint = TRUE),
        "checkpoint controls")
    full <- pca(x, landmarks = seq_len(nrow(x)), backend = "cpu")
    expect_identical(full$fit_scope, "all_rows_optimized")
    expect_equal(dim(full$scores), c(60L, 2L))
})

test_that("resident landmark PCA preserves float32 all-row scores", {
    skip_if_not_installed("float")
    set.seed(17)
    x <- float::fl(matrix(rnorm(2400L), 120L, 20L))
    fit <- pca(x, ncomp = 3L, landmarks = 0.5,
        backend = "cpu", n.cores = 2L, seed = 17L)
    expect_s4_class(fit$scores, "float32")
    expect_equal(dim(fit$scores), c(120L, 3L))
    expect_equal(fit$landmarks$n_landmarks, 60L)
    expect_true(all(is.finite(as.matrix(fit$scores))))
})

test_that("resident Metal PCA projection uses the native multiply route", {
    set.seed(45L)
    x <- matrix(rnorm(1600L), 100L, 16L)
    reference <- pca(x[seq.int(1L, 100L, by = 2L), ],
        ncomp = 3L, backend = "cpu", center = TRUE,
        scale = TRUE)
    expected <- fastEmbedR:::project_pca_test_scores(x, reference)
    reference$backend <- "metal_mps_rsvd"
    calls <- 0L
    with_mocked_bindings({
        projected <- fastEmbedR:::pca_resident_project_scores(
            x, reference)
        expect_equal(projected, expected, tolerance = 1e-8)
        expect_gt(calls, 0L)
        if (requireNamespace("float", quietly = TRUE)) {
            reference$loadings <- float::fl(reference$loadings)
            projected_float <-
                fastEmbedR:::pca_resident_project_scores(
                    float::fl(x), reference)
            expect_s4_class(projected_float, "float32")
            expect_equal(float::dbl(projected_float), expected,
                tolerance = 1e-5)
        }
    }, rsvd_multiply_metal_cpp = function(left, right,
            transpose_left) {
        calls <<- calls + 1L
        expect_false(transpose_left)
        left %*% right
    }, .package = "fastEmbedR")
})

test_that("landmark PCA labels Metal projection in the fit", {
    set.seed(91L)
    x <- matrix(rnorm(240L), 40L, 6L)
    ids <- seq.int(2L, 40L, by = 2L)
    reference <- pca(x[ids, , drop = FALSE], ncomp = 2L,
        backend = "cpu")
    expected <- fastEmbedR:::project_pca_test_scores(x, reference)
    reference$backend <- "metal_mps_rsvd"
    with_mocked_bindings({
        fit <- fastEmbedR:::pca_resident_landmarks(x, 2L,
            NULL, TRUE, FALSE, "metal", 1L, 91L, FALSE,
            NULL, ids, "off", FALSE, FALSE)
        expect_identical(fit$projection_backend, "metal")
        expect_equal(fit$scores, expected, tolerance = 1e-8)
        expect_equal(fit$scores_reference, reference$scores)
    }, pca = function(...) reference,
    rsvd_multiply_metal_cpp = function(left, right,
            transpose_left) left %*% right,
    .package = "fastEmbedR")
})

test_that("multi-GPU landmark PCA projects external rows on CUDA", {
    set.seed(92L)
    x <- matrix(rnorm(240L), 40L, 6L)
    ids <- seq.int(2L, 40L, by = 2L)
    extra <- x[1:3, , drop = FALSE]
    reference <- pca(x[ids, , drop = FALSE], ncomp = 2L,
        backend = "cpu")
    expected <- fastEmbedR:::project_pca_test_scores(extra,
        reference)
    reference$backend <- "cuda"
    reference$scores_test <- fastEmbedR:::project_pca_test_scores(
        x, reference)
    reference$test_projection_devices <- c(0L, 1L)
    calls <- 0L
    with_mocked_bindings({
        fit <- fastEmbedR:::pca_resident_landmarks(x, 2L,
            extra, TRUE, FALSE, "cuda", 1L, 92L, FALSE,
            c(0L, 1L), ids, "off", FALSE, FALSE)
        expect_identical(fit$projection_backend, "cuda")
        expect_identical(fit$projection_devices, c(0L, 1L))
        expect_identical(fit$test_projection_backend, "cuda")
        expect_identical(fit$test_projection_devices,
            c(0L, 1L))
        expect_equal(fit$scores_test, expected,
            tolerance = 1e-8)
        expect_equal(calls, 1L)
    }, pca = function(...) reference,
    pca_cuda_resident_project_cpp = function(data, devices,
            center, scale, loadings) {
        calls <<- calls + 1L
        expect_identical(devices, c(0L, 1L))
        sweep(sweep(data, 2L, center, "-"), 2L,
            scale, "/") %*% loadings
    }, .package = "fastEmbedR")
})
