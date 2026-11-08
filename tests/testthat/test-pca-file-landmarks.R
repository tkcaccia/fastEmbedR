test_that("public file-backed landmark PCA projects all rows", {
    skip_if(.Platform$OS.type != "unix")
    x <- as.matrix(iris[, 1:4])
    work <- tempfile("file-pca-landmarks-")
    dir.create(work)
    on.exit(unlink(work, recursive = TRUE), add = TRUE)
    source <- file.path(work, "input.f32")
    writeBin(as.numeric(t(x)), source, size = 4L,
        endian = "little")
    input <- massive_matrix(source, nrow(x), ncol(x))
    output <- file.path(work, "fit.f32")
    fit <- pca(input, ncomp = 2L, landmarks = 40L,
        backend = "cpu", n.cores = 2L, seed = 4L,
        storage = "file_backed", output = output,
        memory_limit = "512MB", xtest = input,
        tsne_init = TRUE)
    info <- summary(fit)
    expect_identical(info$backend, "cpu")
    expect_identical(info$scope, "landmark")
    expect_identical(length(fit$landmarks$indices), 40L)
    expect_equal(fit$scores$nrow, nrow(x))
    expect_equal(fit$scores_reference$nrow, 40)
    expect_true(all(c("selection", "reference_fit", "projection",
        "test_projection", "tsne_init") %in%
        names(info$stage_seconds)))
    expect_gt(info$memory_use$ram_bytes, 0)
    scores <- massive_read_rows(fit, 1L, nrow(x))
    reference <- sweep(sweep(x, 2L, fit$center, "-"),
        2L, fit$scale, "/") %*% fit$loadings
    expect_equal(scores, reference, tolerance = 1e-3)
    expect_identical(dim(massive_read_rows(fit, 5L, 8L)),
        c(8L, 2L))
    expect_identical(dim(massive_read_rows(fit$scores_test,
        5L, 8L)), c(8L, 2L))
    expect_identical(dim(massive_read_rows(fit$tsne_init,
        5L, 8L)), c(8L, 2L))
    reopened <- massive_open_pca(output)
    expect_identical(summary(reopened)$scope, "landmark")
    expect_equal(massive_read_rows(reopened, 1L, nrow(x)),
        scores, tolerance = 1e-6)
})

test_that("file-backed PCA landmark requests fail before writing", {
    skip_if(.Platform$OS.type != "unix")
    x <- as.matrix(iris[, 1:4])
    work <- tempfile("file-pca-invalid-")
    dir.create(work)
    on.exit(unlink(work, recursive = TRUE), add = TRUE)
    source <- file.path(work, "input.f32")
    writeBin(as.numeric(t(x)), source, size = 4L,
        endian = "little")
    input <- massive_matrix(source, nrow(x), ncol(x))
    output <- file.path(work, "fit.f32")
    expect_error(pca(input, ncomp = 2L, landmarks = 1L,
        output = output, storage = "file_backed"),
        "more landmarks")
    expect_error(pca(input, ncomp = 2L, landmarks = c(1L, 4L),
        output = output, storage = "file_backed"),
        "explicit row IDs")
    expect_error(pca(input, ncomp = 2L, landmarks = 40L,
        output = output, storage = "file_backed",
        checkpoint = TRUE), "not yet supported")
    expect_false(file.exists(output))
})

test_that("file-backed PCA accepts fractional and all-row landmarks", {
    skip_if(.Platform$OS.type != "unix")
    x <- as.matrix(iris[, 1:4])
    work <- tempfile("file-pca-fractions-")
    dir.create(work)
    on.exit(unlink(work, recursive = TRUE), add = TRUE)
    source <- file.path(work, "input.f32")
    writeBin(as.numeric(t(x)), source, size = 4L,
        endian = "little")
    input <- massive_matrix(source, nrow(x), ncol(x))
    partial <- pca(input, ncomp = 2L, landmarks = 0.25,
        backend = "cpu", output = file.path(work, "partial.f32"),
        storage = "file_backed", memory_limit = "512MB")
    expect_identical(summary(partial)$scope, "landmark")
    expect_length(partial$landmarks$indices, 38L)
    complete <- pca(input, ncomp = 2L, landmarks = TRUE,
        backend = "cpu", output = file.path(work, "complete.f32"),
        storage = "file_backed", memory_limit = "512MB")
    expect_identical(summary(complete)$scope, "full")
    expect_equal(complete$scores$nrow, nrow(x))
})
