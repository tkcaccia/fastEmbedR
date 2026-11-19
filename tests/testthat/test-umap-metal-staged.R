test_that("Metal 2D UMAP reports ordered CPU and GPU stages", {
    skip_if_not(embedding_metal_available_cpp())
    x <- scale(as.matrix(iris[seq_len(60L), 1:4]))
    prepared <- umap_init(
        x, n_neighbors = 10L, backend = "cpu", seed = 7L
    )
    layout <- umap_knn(
        prepared, n_components = 2L, backend = "metal", seed = 4L
    )
    cfg <- attr(layout, "fastEmbedR_config")
    expect_identical(dim(layout), c(60L, 2L))
    expect_true(all(is.finite(as.matrix(layout))))
    expect_identical(cfg$backend, "metal")
    expect_identical(cfg$metal_optimizer_pipeline,
        "cpu_ordered_warm_metal_parallel")
    expect_gt(cfg$metal_cpu_warm_epochs, 0L)
    expect_gt(cfg$metal_gpu_epochs, 0L)
    expect_identical(cfg$metal_cpu_warm_epochs +
        cfg$metal_gpu_epochs, cfg$n_epochs)
})

test_that("one-call Metal UMAP retains stage metadata", {
    skip_if_not(embedding_metal_available_cpp())
    x <- scale(as.matrix(iris[seq_len(60L), 1:4]))
    fit <- umap(x, n_neighbors = 10L,
        backend = "metal", seed = 4L)
    expect_identical(fit$parameters$backend, "metal")
    expect_identical(fit$parameters$metal_optimizer_pipeline,
        "cpu_ordered_warm_metal_parallel")
    expect_gt(fit$parameters$metal_cpu_warm_epochs, 0L)
    expect_gt(fit$parameters$metal_gpu_epochs, 0L)
    expect_true(all(is.finite(as.matrix(fit$layout))))
})

test_that("3D Metal UMAP reports an all-Metal optimizer", {
    skip_if_not(embedding_metal_available_cpp())
    x <- scale(as.matrix(iris[seq_len(60L), 1:4]))
    fit <- umap(x, n_neighbors = 10L, n_components = 3L,
        backend = "metal", seed = 4L)
    expect_identical(fit$parameters$metal_optimizer_pipeline,
        "metal_parallel")
    expect_identical(fit$parameters$metal_cpu_warm_epochs, 0L)
    expect_identical(fit$parameters$metal_gpu_epochs,
        fit$parameters$n_epochs)
    expect_identical(fit$parameters$gpu_optimizer_mode,
        "clean_linear_atomic_inplace")
})

test_that("ordered CPU UMAP resumes the same epoch schedule", {
    x <- scale(as.matrix(iris[seq_len(45L), 1:4]))
    prep <- umap_init(x, n_neighbors = 10L,
        backend = "cpu", seed = 7L)
    graph <- prep$prepared$graph
    call_fit <- function(init, epochs, start) {
        fast_knn_umap_csr_clean_atomic_cpp(
            graph$offsets, graph$neighbors, graph$weights,
            init, epochs, 0.1, 5L, 0.25, 1,
            1L, 4L, 0L, FALSE,
            start_epoch = start, total_epochs = 20L
        )
    }
    full <- call_fit(prep$layout, 20L, 0L)
    warm <- call_fit(prep$layout, 5L, 0L)
    resumed <- call_fit(warm, 15L, 5L)
    expect_equal(resumed, full, tolerance = 1e-6)
    expect_error(call_fit(warm, 16L, 5L),
        "Invalid resumed UMAP epoch range")
})

test_that("large 2D Metal UMAP runs the full compact GPU schedule", {
    graph <- list(offset_dtype = "int32", max_weight = 1)
    init <- matrix(0, 100000L, 2L)
    cfg <- list(n_components = 2L, n_epochs = 20L,
        negative_sample_rate = 5L, learning_rate = 1,
        min_dist = 0.01, repulsion_strength = 1)
    with_mocked_bindings({
        out <- optimize_prepared_umap_metal(graph, init, cfg, 42L)
        expect_identical(attr(out, "metal_optimizer_pipeline"),
            "metal_compact_edges")
        expect_identical(attr(out, "metal_cpu_warm_epochs"), 0L)
        expect_identical(attr(out, "metal_gpu_epochs"), 20L)
    }, embedding_metal_available_cpp = function() TRUE,
    knn_embed_metal_csr_cpp = function(..., start_epoch, total_epochs) {
        expect_identical(list(...)[[12L]], 3L)
        expect_identical(start_epoch, 0L)
        expect_identical(total_epochs, 20L)
        init
    }, fast_knn_umap_csr_clean_atomic_cpp = function(...) {
        stop("Unexpected CPU warm-up")
    }, .package = "fastEmbedR")
})

test_that("compact Metal UMAP schedules edges without a degree cap", {
    skip_if_not(embedding_metal_available_cpp())
    init <- rbind(c(0, 0), c(1, 1))
    run <- function(epochs) knn_embed_metal_csr_cpp(
        c(0L, 257L, 257L), rep.int(1L, 257L), rep(1, 257L),
        init, epochs, 0L, 0.0005, 0.01, 1, 1, 42L, 3L)
    expect_equal(as.numeric(run(1L)), as.numeric(init))
    out <- run(2L)
    expect_identical(attr(out, "metal_truncated_edges"), 0L)
    expect_identical(attr(out, "metal_graph_input"),
        "resident_compact_edges")
    expect_true(all(is.finite(out)))
    expect_lt(sum((out[1, ] - out[2, ])^2), 2)
})
