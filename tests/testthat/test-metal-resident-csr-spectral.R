test_that("resident Metal UMAP initializes directly from prepared CSR", {
    skip_if_not(embedding_metal_available_cpp())
    previous <- options(fastEmbedR.metal_umap_csr_spectral = TRUE)
    on.exit(options(previous))
    local_mocked_bindings(
        spectral_knn_init = function(...) stop("Separate KNN initialization"),
        .package = "fastEmbedR")
    n <- 48L
    ids <- outer(seq_len(n), c(-2L, -1L, 1L, 2L),
        function(i, j) as.integer((i + j - 1L) %% n + 1L))
    fit <- umap_knn(ids, float::fl(matrix(1, n, ncol(ids))),
        backend = "metal", n.cores = 1L)
    cfg <- attr(fit, "fastEmbedR_config")
    expect_identical(cfg$init_backend, "metal_csr_diffusion")
    expect_identical(cfg$optimizer_backend, "metal")
    expect_gt(cfg$metal_initialization_allocated_bytes, 0)
    expect_identical(dim(fit), c(n, 2L))
    expect_true(all(is.finite(as.numeric(fit))))
})
