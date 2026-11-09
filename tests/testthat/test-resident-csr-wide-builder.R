test_that("native fuzzy CSR wide construction matches narrow edges", {
    rows <- 48L
    width <- 8L
    indices <- outer(seq_len(rows), seq_len(width),
        function(i, j) (i + 3L * j - 1L) %% rows + 1L)
    distances <- matrix(as.double(rep(seq_len(width), each = rows)),
        rows, width)
    narrow <- fastEmbedR:::umap_graph_csr_float_cpp(
        indices, distances, 0L, width, width, 2L)
    wide <- fastEmbedR:::umap_graph_csr_float_cpp(
        indices, distances, 0L, width, width, 2L,
        force_wide = TRUE)
    expect_identical(narrow$offset_dtype, "int32")
    expect_identical(wide$offset_dtype, "uint64")
    expect_type(narrow$offsets, "integer")
    expect_type(wide$offsets, "double")
    expect_equal(as.double(narrow$offsets), wide$offsets)
    expect_identical(narrow$neighbors, wide$neighbors)
    expect_equal(narrow$weights, wide$weights, tolerance = 0)
    expect_equal(narrow$max_weight, wide$max_weight,
        tolerance = 0)
    expect_null(wide$epochs_per_sample)
    expect_equal(wide$nnz, as.double(narrow$nnz))
})

test_that("wide CSR tiled CPU updates match stateless COO updates", {
    rows <- 24L
    width <- 4L
    indices <- outer(seq_len(rows), seq_len(width),
        function(i, j) (i + 2L * j - 1L) %% rows + 1L)
    distances <- matrix(as.double(rep(seq_len(width), each = rows)),
        rows, width)
    graph <- fastEmbedR:::umap_graph_csr_float_cpp(
        indices, distances, 0L, width, width, 1L,
        force_wide = TRUE)
    init <- cbind(seq_len(rows) / rows,
        sin(seq_len(rows)) / 10)
    wide_fit <- getFromNamespace(
        "fast_knn_umap_csr_wide_init_cpp", "fastEmbedR")
    run <- function(tile) wide_fit(
            graph$offsets, graph$neighbors, graph$weights,
            init, 12L, 0.1, 5L, 1, 1, 1L, 42L, tile)
    one_tile <- run(65536L)
    many_tiles <- run(3L)
    expect_equal(as.numeric(one_tile), as.numeric(many_tiles),
        tolerance = 0)
    heads <- rep(seq_len(rows) - 1L, diff(graph$offsets))
    periods <- graph$max_weight / pmax(graph$weights, 1e-6)
    replay <- fastEmbedR:::fast_knn_umap_coo_replay_cpp(
        heads, graph$neighbors, graph$weights, periods,
        init, 12L, 0.1, 5L, 1, 1, 1L, 42L,
        FALSE, FALSE)
    expect_equal(as.numeric(one_tile), as.numeric(replay),
        tolerance = 1e-6)
    expect_identical(attr(one_tile, "csr_offset_dtype"),
        "uint64")
    bad <- graph$offsets
    bad[[2L]] <- 1.5
    expect_error(wide_fit(bad,
            graph$neighbors, graph$weights, init,
            12L, 0.1, 5L, 1, 1, 1L, 42L),
        "exact non-negative integers")
})

test_that("wide CSR diffusion preserves narrow initialization", {
    rows <- 32L
    width <- 5L
    indices <- outer(seq_len(rows), seq_len(width),
        function(i, j) (i + j - 1L) %% rows + 1L)
    distances <- matrix(as.double(rep(seq_len(width), each = rows)),
        rows, width)
    narrow <- fastEmbedR:::umap_graph_csr_float_cpp(
        indices, distances, 0L, width, width, 1L)
    wide <- fastEmbedR:::umap_graph_csr_float_cpp(
        indices, distances, 0L, width, width, 1L,
        force_wide = TRUE)
    narrow_init <- fastEmbedR:::umap_cuda_diffusion_init_csr_cpp(
        narrow$offsets, narrow$neighbors, narrow$weights,
        5L, 1L, 42L)
    wide_init <- fastEmbedR:::fast_knn_umap_csr_wide_diffusion_cpp(
        wide$offsets, wide$neighbors, wide$weights,
        5L, 1L, 42L)
    expect_equal(as.numeric(wide_init), as.numeric(narrow_init),
        tolerance = 0)
    expect_identical(attr(wide_init, "csr_offset_dtype"),
        "uint64")
    bad <- wide$offsets
    bad[[2L]] <- 0.5
    expect_error(
        fastEmbedR:::fast_knn_umap_csr_wide_diffusion_cpp(
            bad, wide$neighbors, wide$weights, 5L, 1L, 42L),
        "exact non-negative integers")
})

test_that("resident CPU 2D dispatch uses wide CSR graph metadata", {
    rows <- 28L
    width <- 5L
    indices <- outer(seq_len(rows), seq_len(width),
        function(i, j) (i + j - 1L) %% rows + 1L)
    distances <- matrix(as.double(rep(seq_len(width), each = rows)),
        rows, width)
    graph <- fastEmbedR:::umap_graph_csr_float_cpp(
        indices, distances, 0L, width, width, 1L,
        force_wide = TRUE)
    graph$graph_mode <- "fuzzy"
    cfg <- fastEmbedR:::fast_knn_umap_config(
        rows, width, backend = "cpu")
    cfg$n_threads <- 1L
    cfg$n_epochs <- 12L
    init <- fastEmbedR:::umap_init_from_csr_graph(
        graph, 2L, cfg, 42L)
    expect_identical(attr(init, "backend"),
        "cpu_wide_csr_diffusion")
    expect_error(
        fastEmbedR:::umap_init_from_csr_graph(
            graph, 3L, cfg, 42L),
        "supports 2D only")
    state <- list(graph = graph, init = init, cfg = cfg,
        return_float32 = FALSE)
    layout <- fastEmbedR:::run_umap_cpu_2d(state, 42L, FALSE)
    metadata <- attr(layout, "fastEmbedR_config")
    expect_equal(dim(layout), c(rows, 2L))
    expect_true(all(is.finite(layout)))
    expect_identical(metadata$optimizer_mode,
        "csr_wide_edge_tiles")
    expect_identical(metadata$optimizer_schedule,
        "stateless_edge_tiles")
    expect_identical(attr(layout, "csr_offset_dtype"),
        "uint64")
})

test_that("resident R graph builder preserves selected wide offsets", {
    rows <- 24L
    width <- 4L
    indices <- outer(seq_len(rows), seq_len(width),
        function(i, j) (i + j - 1L) %% rows + 1L)
    distances <- matrix(as.double(rep(seq_len(width), each = rows)),
        rows, width)
    build <- fastEmbedR:::umap_build_csr_graph
    expect_error(build(indices, distances, 0L, width, width,
            1L, force_wide = TRUE), "requires wide-route admission")
    expect_error(build(indices, distances, 0L, width, width,
            1L, graph_mode = "binary", allow_wide = TRUE),
        "requires the fuzzy graph mode")
    expect_error(build(indices, distances, 0L, width, width,
            1L, allow_wide = TRUE, force_wide = TRUE),
        "requires float32 KNN distances")
    skip_if_not_installed("float")
    graph <- build(indices, float::fl(distances), 0L, width, width,
        1L, allow_wide = TRUE, force_wide = TRUE)
    expect_identical(graph$offset_dtype, "uint64")
    expect_type(graph$offsets, "double")
    expect_equal(tail(graph$offsets, 1L), graph$nnz)
    expect_equal(length(graph$neighbors), graph$nnz)
})
