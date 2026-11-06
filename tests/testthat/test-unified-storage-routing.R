test_that("PCA infers storage mode from a file-backed input", {
    x <- matrix(seq_len(60L) / 13, nrow = 15L, ncol = 4L)
    path <- tempfile(fileext = ".f32")
    on.exit(unlink(path), add = TRUE)
    writeBin(as.vector(t(x)), path, size = 4L,
        endian = "little")
    source <- massive_matrix(path, nrow(x), ncol(x))
    fit <- pca(source, ncomp = 2L, backend = "cpu",
        memory_limit = "1GB")
    expect_identical(fit$massive_auto, "in_memory")
    expect_equal(dim(fit$scores), c(nrow(x), 2L))
    explicit <- pca(source, ncomp = 2L, backend = "cpu",
        massive = "off")
    expect_null(explicit$massive_auto)
})
