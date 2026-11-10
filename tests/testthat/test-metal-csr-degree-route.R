test_that("Metal CSR rows over 256 edges use complete tiled traversal", {
    skip_if_not_installed("float")
    graph <- list(offsets = as.integer(c(0, 257, 257)),
        neighbors = rep.int(2L, 257L),
        weights = float::fl(rep(1, 257L)),
        offset_dtype = "int32")
    init <- matrix(seq_len(6L) / 100, 2L, 3L)
    cfg <- list(n_components = 3L, n_epochs = 2L,
        negative_sample_rate = 0L, learning_rate = 0.1,
        min_dist = 0.1, repulsion_strength = 1)
    seen <- FALSE
    with_mocked_bindings({
        out <- fastEmbedR:::optimize_prepared_umap_metal(
            graph, init, cfg, 42L)
        expect_true(seen)
        expect_identical(dim(out), c(2L, 3L))
        expect_identical(attr(out, "metal_truncated_edges"), 0L)
        expect_identical(attr(out, "metal_graph_input"),
            "resident_csr_int32_edge_tiles")
        reported <- fastEmbedR:::record_metal_tiled_policy(cfg, out)
        expect_identical(reported$gpu_optimizer_mode,
            "csr_edge_tiles")
        expect_identical(reported$gpu_optimizer_schedule,
            "stateless_edge_tiles")
    },
    embedding_metal_available_cpp = function() TRUE,
    umap_metal_optimize_csr_wide_cpp = function(...) {
        args <- list(...)
        seen <<- is.double(args[[1L]]) &&
            identical(args[[1L]], c(0, 257, 257)) &&
            length(args[[2L]]) == 257L
        out <- args[[4L]]
        attr(out, "metal_optimizer_pipeline") <-
            "metal_wide_edge_tiles"
        out
    }, .package = "fastEmbedR")
})

test_that("Metal CSR rows within 256 edges retain dense path", {
    skip_if_not_installed("float")
    graph <- list(offsets = as.integer(c(0, 256, 256)),
        neighbors = rep.int(2L, 256L),
        weights = float::fl(rep(1, 256L)),
        offset_dtype = "int32", max_weight = 1)
    init <- matrix(seq_len(6L) / 100, 2L, 3L)
    cfg <- list(n_components = 3L, n_epochs = 2L,
        negative_sample_rate = 0L, learning_rate = 0.1,
        min_dist = 0.1, repulsion_strength = 1)
    seen <- FALSE
    with_mocked_bindings({
        out <- fastEmbedR:::optimize_prepared_umap_metal(
            graph, init, cfg, 42L)
        expect_true(seen)
        expect_identical(dim(out), c(2L, 3L))
    },
    embedding_metal_available_cpp = function() TRUE,
    knn_embed_metal_csr_cpp = function(...) {
        seen <<- TRUE
        init
    }, .package = "fastEmbedR")
})

test_that("direct dense Metal CSR refuses truncated rows", {
    skip_if_not(fastEmbedR:::embedding_metal_available_cpp())
    skip_if_not_installed("float")
    offsets <- as.integer(c(0, 257, 257, 257,
        257, 257, 257))
    neighbors <- rep.int(2L, 257L)
    weights <- float::fl(rep(1, 257L))
    init <- cbind(seq_len(6L) / 20, seq_len(6L) / 30)
    expect_error(fastEmbedR:::knn_embed_metal_csr_cpp(
        offsets, neighbors, weights, init, 2L, 0L,
        0.1, 0.1, 1, 1, 42L, 1L,
        start_epoch = 0L, total_epochs = 2L),
        "exceeds 256 dense neighbors")
})
