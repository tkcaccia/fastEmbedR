test_that("UMAP initialization is deterministic and reusable", {
    set.seed(701)
    x <- matrix(rnorm(240L), 40L, 6L)
    knn <- test_exact_knn(x, k = 8L, backend = "cpu")

    init_a <- umap_init(
        knn,
        backend = "cpu",
        graph_mode = "fuzzy",
        seed = 701L,
        n.cores = 2L
    )
    init_b <- umap_init(
        knn,
        backend = "cpu",
        graph_mode = "fuzzy",
        seed = 701L,
        n.cores = 2L
    )

    expect_s3_class(init_a, "fastEmbedR_umap_initialization")
    expect_equal(init_a$layout, init_b$layout, tolerance = 1e-7)
    expect_identical(init_a$parameters$init_backend, "cpu_fuzzy_csr")
    expect_identical(init_a$parameters$n.cores, 2L)
    expect_null(init_a$parameters$n_threads)
    expect_s3_class(init_a$prepared, "fastEmbedR_umap_prepared")
    expect_equal(
        init_a$timings$stage,
        c("knn", "graph", "initialization", "total")
    )

    layout <- umap_knn(
        init_a,
        backend = "cpu",
        seed = 701L,
        n.cores = 2L
    )
    expect_equal(dim(layout), c(nrow(x), 2L))
    expect_true(all(is.finite(layout)))
    cfg <- attr(layout, "fastEmbedR_config")
    expect_true(isTRUE(cfg$prepared_reuse_hit))
    expect_true(isTRUE(cfg$initialization_reuse_hit))
})

test_that("UMAP initialization accepts a raw data matrix", {
    set.seed(704)
    x <- matrix(rnorm(180L), 30L, 6L)

    initialization <- umap_init(
        x,
        n_neighbors = 8L,
        backend = "cpu",
        seed = 704L,
        n.cores = 2L
    )

    expect_s3_class(initialization, "fastEmbedR_umap_initialization")
    expect_equal(dim(initialization$layout), c(nrow(x), 2L))
    expect_true(all(is.finite(initialization$layout)))
})

test_that("standard fuzzy graph is the UMAP default", {
    set.seed(700)
    x <- matrix(rnorm(180L), 30L, 6L)
    knn <- test_exact_knn(x, k = 8L, backend = "cpu")

    prepared <- prepare_umap_knn(knn, backend = "cpu", n.cores = 2L)
    expect_identical(prepared$config$graph_mode, "fuzzy")

    default_layout <- umap_knn(
        knn,
        backend = "cpu",
        seed = 700L,
        n.cores = 2L
    )
    explicit_layout <- umap_knn(
        knn,
        backend = "cpu",
        graph_mode = "fuzzy",
        seed = 700L,
        n.cores = 2L
    )

    expect_identical(
        attr(default_layout, "fastEmbedR_config")$graph_mode,
        "fuzzy"
    )
    expect_equal(default_layout, explicit_layout, tolerance = 1e-7)
})

test_that("UMAP initialization validates prepared graph mode", {
    set.seed(702)
    x <- matrix(rnorm(120L), 30L, 4L)
    knn <- test_exact_knn(x, k = 6L, backend = "cpu")
    prepared <- prepare_umap_knn(knn, graph_mode = "binary")

    expect_error(
        umap_init(prepared, graph_mode = "fuzzy"),
        "does not match"
    )
    init <- umap_init(prepared, seed = 702L)
    expect_identical(init$parameters$graph_mode, "binary")
})

test_that("Metal can optimize a reusable CPU UMAP initialization", {
    skip_if_not(fastEmbedR:::embedding_metal_available_cpp())
    set.seed(703)
    x <- matrix(rnorm(240L), 40L, 6L)
    knn <- test_exact_knn(x, k = 8L, backend = "cpu")
    init <- umap_init(
        knn,
        backend = "cpu",
        graph_mode = "fuzzy",
        seed = 703L,
        n.cores = 2L
    )

    layout <- umap_knn(
        init,
        backend = "metal",
        seed = 703L,
        n.cores = 2L
    )
    expect_equal(dim(layout), c(nrow(x), 2L))
    expect_true(all(is.finite(layout)))
    cfg <- attr(layout, "fastEmbedR_config")
    expect_identical(cfg$optimizer_backend, "metal")
    expect_true(isTRUE(cfg$initialization_reuse_hit))
})

test_that("CUDA computes a native 3D UMAP initialization", {
    skip_if_not(fastEmbedR:::embedding_cuda_available_cpp())
    set.seed(705)
    x <- matrix(rnorm(48L * 6L), 48L, 6L)
    knn <- test_exact_knn(x, k = 10L, backend = "cpu")
    first <- umap_init(
        knn, backend = "cuda", n_components = 3L, seed = 705L
    )
    second <- umap_init(
        knn, backend = "cuda", n_components = 3L, seed = 705L
    )
    expect_identical(dim(first$layout), c(48L, 3L))
    expect_true(all(is.finite(first$layout)))
    expect_gt(stats::sd(first$layout[, 3L]), 0)
    expect_equal(
        as.numeric(first$layout), as.numeric(second$layout),
        tolerance = 1e-6
    )
    expect_identical(first$parameters$init_backend, "cuda")

    layout <- umap_knn(
        first, backend = "cuda", n_components = 3L, seed = 705L
    )
    expect_identical(dim(layout), c(48L, 3L))
    expect_true(all(is.finite(layout)))
    expect_gt(stats::sd(layout[, 3L]), 0)
    cfg <- attr(layout, "fastEmbedR_config")
    expect_true(isTRUE(cfg$initialization_reuse_hit))
    expect_identical(cfg$optimizer_backend, "cuda")
    expect_identical(cfg$sgd_loop, "coo_float32_snapshot_delta")
})

test_that("Metal computes a native 3D UMAP initialization", {
    skip_if_not(fastEmbedR:::embedding_metal_available_cpp())
    set.seed(706L)
    x <- matrix(rnorm(48L * 6L), 48L, 6L)
    knn <- test_exact_knn(x, k = 10L, backend = "cpu")
    first <- umap_init(
        knn, backend = "metal", n_components = 3L, seed = 706L
    )
    second <- umap_init(
        knn, backend = "metal", n_components = 3L, seed = 706L
    )
    expect_identical(dim(first$layout), c(48L, 3L))
    expect_true(all(is.finite(first$layout)))
    expect_gt(stats::sd(first$layout[, 3L]), 0)
    expect_equal(first$layout, second$layout, tolerance = 1e-6)
    expect_identical(first$parameters$init_backend, "metal")
})
