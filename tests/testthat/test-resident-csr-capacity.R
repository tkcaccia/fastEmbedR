test_that("resident CSR rejects 32-bit overflow before graph building", {
    check <- fastEmbedR:::umap_check_resident_csr_capacity
    expect_no_error(check(1000, 30, "8GB", 8 * 1024^3))
    expect_error(check(100000000, 30),
        "Resident UMAP CSR may exceed signed 32-bit offsets")
    expect_error(check(100000000, 30),
        "file-backed full graph")
    wide <- check(50000000, 30, "512GB", 512 * 1024^3,
        16L, allow_wide = TRUE)
    expect_identical(wide$offset_dtype, "uint64")
    expect_equal(wide$max_edges, 3e9)
    expect_error(check(50000000, 30, "32GB",
            512 * 1024^3, 16L, allow_wide = TRUE),
        "exceeds the RAM budget")
    expect_error(check(1000000, 30, "1GB", 16 * 1024^3),
        "exceeds the RAM budget")
    expect_error(check(1000, 30, "8GB", NA_real_),
        "exceeds the RAM budget")
    plan <- check(1000000, 30, "8GB", 16 * 1024^3)
    expect_equal(plan$max_edges, 60000000)
    expect_gt(plan$peak_ram_estimate_bytes, 3 * 1024^3)
})

test_that("resident CSR edge count remains numeric metadata", {
    indices <- outer(seq_len(10L), seq_len(4L),
        function(i, j) (i + j - 1L) %% 10L + 1L)
    distances <- matrix(as.double(rep(seq_len(4L), each = 10L)),
        10L, 4L)
    graph <- fastEmbedR:::umap_build_csr_graph(indices, distances,
        0L, 4L, 4L, 1L)
    expect_type(graph$nnz, "integer")
    expect_identical(graph$offset_dtype, "int32")
    expect_true(graph$nnz > 0L)
    connectivity <- fastEmbedR:::knn_connectivity_range_cpp(
        indices, 0L, 4L)
    expect_type(connectivity$valid_edge_count, "double")
    expect_equal(connectivity$valid_edge_count, 40)
    expect_equal(connectivity$invalid_edge_count, 0)
})

test_that("resident CSR stage model keeps wide offsets and float32 edges", {
    plan <- fastEmbedR:::umap_resident_csr_stage_bytes(
        50000000, 30, 16L)
    expect_identical(plan$offset_dtype, "uint64")
    expect_equal(plan$max_edges, 3e9)
    expect_equal(plan$offset_bytes, (50000000 + 1) * 8)
    expect_equal(unname(plan$stages[["knn_input"]]), 12e9)
    expect_gt(plan$stages[["r_export"]],
        plan$stages[["symmetrize"]])
    expect_gt(plan$stages[["optimizer_spectral"]],
        plan$stages[["r_export"]])
    one_worker <- fastEmbedR:::umap_resident_csr_stage_bytes(
        50000000, 30, 1L)
    expect_equal(plan$stages[["reverse_build"]] -
        one_worker$stages[["reverse_build"]],
        50000000 * 8 * 15)
    expect_equal(plan$stages[["optimizer_spectral"]],
        one_worker$stages[["optimizer_spectral"]])
    expect_error(fastEmbedR:::umap_check_resident_csr_capacity(
        50000000, 30, "270GB", 269e9, 16L,
        allow_wide = TRUE), "RAM budget")
    admitted <- fastEmbedR:::umap_check_resident_csr_capacity(
        50000000, 30, "300GB", 300e9, 16L,
        allow_wide = TRUE)
    expect_lt(admitted$peak_ram_estimate_bytes, 0.7 * 300e9)
    small <- fastEmbedR:::umap_resident_csr_stage_bytes(1000, 30)
    expect_identical(small$offset_dtype, "int32")
    expect_equal(small$offset_bytes, (1000 + 1) * 4)
    hundred_million <- fastEmbedR:::umap_resident_csr_stage_bytes(
        100000000, 30, 16L)
    expect_gt(hundred_million$stages[["optimizer_spectral"]],
        256 * 1024^3)
})

test_that("resident CSR admission counts double KNN distances", {
    model <- fastEmbedR:::umap_resident_csr_stage_bytes
    single <- model(1000000, 30, distance_bytes = 4L)
    double <- model(1000000, 30, distance_bytes = 8L)
    expect_equal(double$stages[["knn_input"]] -
        single$stages[["knn_input"]], 120000000)
    expect_equal(double$stages[["optimizer_spectral"]] -
        single$stages[["optimizer_spectral"]], 120000000)
    expect_error(model(1000, 30, distance_bytes = 2L),
        "distance width")
})

test_that("wide resident Metal UMAP uses the bounded graph plan", {
    needs_wide <- fastEmbedR:::umap_csr_needs_wide
    check <- fastEmbedR:::umap_check_resident_csr_capacity
    expect_false(needs_wide(10000000, 30))
    for (rows in c(50000000, 100000000)) {
        expect_true(needs_wide(rows, 30))
        expect_error(check(rows, 30, "200GB", 256 * 1024^3,
            allow_wide = TRUE), "RAM budget")
        plan <- check(rows, 30, "200GB", 256 * 1024^3,
            allow_wide = TRUE, compact = TRUE)
        expect_identical(plan$offset_dtype, "uint64")
        expect_equal(plan$max_edges, 2 * rows * 30)
    }
})

test_that("wide Metal UMAP initializes from its float32 CSR graph", {
    skip_if_not_installed("float")
    rows <- 24L
    width <- 4L
    ids <- outer(seq_len(rows), seq_len(width),
        function(i, j) (i + j - 1L) %% rows + 1L)
    distances <- float::fl(matrix(rep(seq_len(width), each = rows), rows))
    cfg <- list(backend = "metal", n_threads = 1L, spectral_n_iter = 3L,
        init_scale = 1)
    with_mocked_bindings({
        pending <- fastEmbedR:::initialize_umap_gpu_host_knn(
            ids, distances, cfg, "fuzzy", 42L, 2L)
        expect_null(pending$init)
        state <- fastEmbedR:::build_umap_csr_state(
            ids, distances, 0L, width, pending$cfg, "fuzzy", 2L, 42L)
        expect_identical(state$graph$offset_dtype, "uint64")
        expect_s4_class(state$graph$weights, "float32")
        expect_identical(state$cfg$init_backend, "cpu_wide_csr_diffusion")
        expect_true(all(is.finite(state$init)))
    },
    umap_csr_needs_wide = function(rows, width) TRUE,
    spectral_knn_init = function(...) stop("Narrow initializer called"),
    .package = "fastEmbedR")
})

test_that("the compact resident CPU CSR fits the bounded float32 workload", {
    available <- 26 * 1024^3
    needs <- fastEmbedR:::umap_resident_cpu_needs_compact
    check <- fastEmbedR:::umap_check_resident_csr_capacity
    expect_false(needs(1000L, 30L, 8L, "26GB", available))
    expect_true(needs(10000000L, 30L, 8L, "26GB", available))
    expect_error(check(10000000L, 30L, "26GB", available, 8L), "RAM budget")
    plan <- check(10000000L, 30L, "26GB", available, 8L,
        allow_wide = TRUE, compact = TRUE)
    expect_identical(plan$offset_dtype, "uint64")
    expect_lt(plan$peak_ram_estimate_bytes, 0.7 * available)
    expect_error(check(10000000L, 30L, "12GB", available, 8L,
        allow_wide = TRUE, compact = TRUE), "RAM budget")
})

test_that("resident CPU UMAP uses the compact plan when RAM is tight", {
    skip_if_not_installed("float")
    rows <- 64L
    width <- 10L
    bytes <- fastEmbedR:::umap_resident_csr_stage_bytes
    limit <- mean(c(bytes(rows, width)$peak_ram_estimate_bytes,
        bytes(rows, width, compact = TRUE)$peak_ram_estimate_bytes)) / 0.7
    previous <- options(memory_limit = limit)
    on.exit(options(previous))
    idx <- outer(seq_len(rows), seq_len(width),
        function(i, j) (i + j - 1L) %% rows + 1L)
    dst <- float::fl(matrix(rep(seq_len(width), each = rows), rows))
    layout <- umap_knn(idx, dst, backend = "cpu", n.cores = 1L, seed = 19L)
    cfg <- attr(layout, "fastEmbedR_config")
    expect_identical(cfg$backend, "cpu")
    expect_identical(cfg$optimizer_schedule, "stateless_edge_tiles")
    expect_equal(dim(layout), c(rows, 2L))
    expect_true(all(is.finite(as.numeric(layout))))
})

test_that("file-backed CSR ownership counts exceed signed 32 bits", {
    path <- tempfile(fileext = ".offsets.u64")
    on.exit(unlink(path))
    offsets <- c(0, 2^31 + 3, 2^31 + 5,
        2^31 + 8, 2^31 + 10)
    encode <- function(value) {
        words <- c(value %% 2^32, floor(value / 2^32))
        signed <- ifelse(words >= 2^31, words - 2^32, words)
        writeBin(as.integer(signed), raw(), size = 4L,
            endian = "little")
    }
    writeBin(do.call(c, lapply(offsets, encode)), path)
    graph <- list(offsets_path = path, n_vertices = 4,
        n_edges = 2^31 + 10)
    counts <- fastEmbedR:::massive_graph_owner_edge_counts(graph,
        c(0L, 1L))
    expect_identical(counts, c(2^31 + 5, 5))
})

test_that("KNN index-base scan handles missing values without copies", {
    base <- fastEmbedR:::knn_index_base
    expect_identical(base(matrix(c(1L, NA_integer_, 3L), 1L), 3L),
        "one")
    expect_identical(base(matrix(c(0L, NA_integer_, 2L), 1L), 3L),
        "zero")
    expect_identical(base(matrix(NA_integer_, 2L, 2L), 2L),
        "zero")
})
