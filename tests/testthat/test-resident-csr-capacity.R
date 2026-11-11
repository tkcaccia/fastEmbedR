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
