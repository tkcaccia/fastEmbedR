test_that("native fuzzy CSR wide construction matches narrow edges", {
    skip_if_not_installed("float")
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
    expect_s4_class(wide$weights, "float32")
    expect_equal(narrow$weights, as.numeric(wide$weights),
        tolerance = 0)
    expect_equal(narrow$max_weight, wide$max_weight,
        tolerance = 0)
    expect_null(wide$epochs_per_sample)
    expect_equal(wide$nnz, as.double(narrow$nnz))
})

test_that("wide CSR tiled CPU updates match stateless COO updates", {
    skip_if_not_installed("float")
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
    packed <- wide_fit(graph$offsets, graph$neighbors,
        graph$weights, init, 12L, 0.1, 5L, 1, 1,
        1L, 42L, 3L, return_float32 = TRUE)
    expect_s4_class(packed, "float32")
    expect_equal(as.numeric(float::dbl(packed)),
        as.numeric(many_tiles),
        tolerance = 0)
    heads <- rep(seq_len(rows) - 1L, diff(graph$offsets))
    periods <- graph$max_weight / pmax(
        as.numeric(graph$weights), 1e-6)
    replay <- fastEmbedR:::fast_knn_umap_coo_replay_cpp(
        heads, graph$neighbors, as.numeric(graph$weights), periods,
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
    skip_if_not_installed("float")
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
    skip_if_not_installed("float")
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
    init_3d <- fastEmbedR:::umap_init_from_csr_graph(
        graph, 3L, cfg, 42L)
    expect_identical(dim(init_3d), c(rows, 3L))
    expect_true(all(is.finite(init_3d)))
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
    skip_if_not_installed("float")
    rows <- 24L
    width <- 4L
    indices <- outer(seq_len(rows), seq_len(width),
        function(i, j) (i + j - 1L) %% rows + 1L)
    distances <- matrix(as.double(rep(seq_len(width), each = rows)),
        rows, width)
    build <- fastEmbedR:::umap_build_csr_graph
    expect_error(build(indices, distances, 0L, width, width,
            1L, force_wide = TRUE), "requires wide-route admission")
    binary <- build(indices, distances, 0L, width, width,
        1L, graph_mode = "binary", allow_wide = TRUE,
        force_wide = TRUE)
    expect_identical(binary$offset_dtype, "uint64")
    expect_true(all(as.numeric(binary$weights) == 1))
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

test_that("wide binary graph preserves narrow edges and weights", {
    skip_if_not_installed("float")
    rows <- 36L
    width <- 6L
    indices <- outer(seq_len(rows), seq_len(width),
        function(i, j) (i + 2L * j - 1L) %% rows + 1L)
    narrow <- fastEmbedR:::umap_graph_csr_binary_union_cpp(
        indices, 0L, width, 1L)
    wide <- fastEmbedR:::umap_graph_csr_binary_union_cpp(
        indices, 0L, width, 1L, force_wide = TRUE)
    expect_identical(wide$offset_dtype, "uint64")
    expect_equal(wide$offsets, as.double(narrow$offsets),
        tolerance = 0)
    expect_identical(wide$neighbors, narrow$neighbors)
    expect_s4_class(wide$weights, "float32")
    expect_equal(as.numeric(wide$weights), narrow$weights,
        tolerance = 0)
})

test_that("wide binary 3D optimization preserves edge-tile parity", {
    skip_if_not_installed("float")
    rows <- 36L
    width <- 6L
    indices <- outer(seq_len(rows), seq_len(width),
        function(i, j) (i + 2L * j - 1L) %% rows + 1L)
    graph <- fastEmbedR:::umap_graph_csr_binary_union_cpp(
        indices, 0L, width, 1L, force_wide = TRUE)
    init <- fastEmbedR:::fast_knn_umap_csr_wide_diffusion_cpp(
        graph$offsets, graph$neighbors, graph$weights,
        5L, 1L, 42L, n_components = 3L)
    expect_identical(dim(init), c(rows, 3L))
    run <- function(tile) fastEmbedR:::fast_knn_umap_csr_wide_init_cpp(
        graph$offsets, graph$neighbors, graph$weights,
        init, 12L, 0.1, 5L, 1, 1, 1L, 42L, tile)
    one <- run(65536L)
    tiled <- run(3L)
    expect_equal(one, tiled, tolerance = 0)
    expect_true(all(is.finite(one)))
    expect_identical(attr(one, "csr_offset_dtype"), "uint64")
})

test_that("prepared wide binary 3D stays on the CPU tile route", {
    skip_if_not_installed("float")
    rows <- 36L
    width <- 6L
    indices <- outer(seq_len(rows), seq_len(width),
        function(i, j) (i + 2L * j - 1L) %% rows + 1L)
    graph <- fastEmbedR:::umap_graph_csr_binary_union_cpp(
        indices, 0L, width, 1L, force_wide = TRUE)
    graph$offset_dtype <- "uint64"
    init <- fastEmbedR:::fast_knn_umap_csr_wide_diffusion_cpp(
        graph$offsets, graph$neighbors, graph$weights,
        5L, 1L, 42L, n_components = 3L)
    cfg <- fastEmbedR:::fast_knn_umap_config(
        rows, width, backend = "cpu")
    cfg$n_threads <- 1L
    cfg$n_epochs <- 12L
    cfg$graph_mode <- "binary"
    prepared <- list(graph = graph, knn = list(indices = indices),
        config = cfg, initialization = init,
        initialization_parameters = list(
            init_backend = "cpu_wide_csr_diffusion"))
    fit <- fastEmbedR:::fast_knn_umap_prepared_core(
        prepared, n_components = 3L, backend = "cpu")
    meta <- attr(fit, "fastEmbedR_config")
    expect_identical(dim(fit), c(rows, 3L))
    expect_true(all(is.finite(fit)))
    expect_identical(meta$optimizer_mode, "csr_wide_edge_tiles")
    expect_identical(meta$checkpoint_scope, "none")
    expect_false(meta$resume_supported)
    expect_error(fastEmbedR:::fast_knn_umap_prepared_core(
        prepared, n_components = 3L, backend = "cuda"),
        "qualified GPU tile optimizer")
})

test_that("public resident UMAP rejects checkpoint and resume", {
    data <- matrix(seq_len(80L) / 80, 20L, 4L)
    expect_error(umap(data, backend = "cpu", checkpoint = TRUE),
        "Resident embeddings cannot checkpoint or resume")
    expect_error(umap(data, backend = "cpu", checkpoint = TRUE,
            resume = TRUE),
        "Resident embeddings cannot checkpoint or resume")
    fit <- umap(data, n_neighbors = 5L, backend = "cpu")
    expect_identical(fit$parameters$checkpoint_scope, "none")
    expect_false(fit$parameters$resume_supported)
})

test_that("wide binary 3D Metal routing keeps typed graph arrays", {
    skip_if_not_installed("float")
    rows <- 36L
    width <- 6L
    indices <- outer(seq_len(rows), seq_len(width),
        function(i, j) (i + 2L * j - 1L) %% rows + 1L)
    graph <- fastEmbedR:::umap_build_csr_graph(
        indices, NULL, 0L, width, width, 1L,
        graph_mode = "binary", allow_wide = TRUE,
        force_wide = TRUE)
    init <- matrix(seq_len(rows * 3L) / 1000,
        rows, 3L)
    cfg <- fastEmbedR:::fast_knn_umap_config(
        rows, width, backend = "cpu")
    cfg$n_components <- 3L
    cfg$n_epochs <- 10L
    observed <- FALSE
    effective_rate <- NULL
    with_mocked_bindings({
        out <- fastEmbedR:::optimize_prepared_umap_metal(
            graph, init, cfg, 42L)
        expect_true(observed)
        expect_identical(dim(out), c(rows, 3L))
        expect_equal(effective_rate, cfg$learning_rate * 0.25)
        expect_equal(attr(out, "metal_effective_learning_rate"),
            effective_rate)
    },
    embedding_metal_available_cpp = function() TRUE,
    umap_metal_optimize_csr_wide_cpp = function(...) {
        args <- list(...)
        observed <<- is.double(args[[1L]]) &&
            inherits(args[[3L]], "float32") &&
            identical(dim(args[[4L]]), c(rows, 3L))
        effective_rate <<- args[[7L]]
        args[[4L]]
    }, .package = "fastEmbedR")
})
