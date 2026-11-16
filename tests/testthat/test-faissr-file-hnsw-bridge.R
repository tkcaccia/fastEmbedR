test_that("file-backed CPU HNSW uses faissR's native file builder", {
    skip_if_not_installed("faissR")
    skip_if_not_installed("float")
    skip_if_not(faissR::faiss_available(), "FAISS is required")
    set.seed(20261116)
    x <- matrix(rnorm(300L * 8L), nrow = 300L)
    path <- tempfile(fileext = ".f32")
    on.exit(unlink(path), add = TRUE)
    writeBin(as.numeric(t(x)), path, size = 4L, endian = "little")
    source <- massive_matrix(path, nrow = 300L, ncol = 8L)
    state <- fastEmbedR:::massive_faiss_hnsw_reference(
        source, k = 5L, n.cores = 2L,
        hnsw_build = list(m = 16L, ef_construction = 80L),
        target_recall = 0.95)
    expect_identical(state$reference_storage, "file_backed_float32")
    expect_identical(state$index$backend_used,
        "faiss_cpu_hnsw_file_index")
    expect_identical(attr(state$index$pointer, "residency"),
        "file_backed_reference_resident_index")
    rows <- c(2L, 31L, 116L, 251L)
    query <- float::fl(x[rows, , drop = FALSE])
    found <- fastEmbedR:::faissr_hnsw_index_search_cpp(
        state$index$pointer, query, 6L, 300L, 2L)
    expect_identical(found$backend_used,
        "faiss_cpu_hnsw_file_index")
    expect_identical(found$reference_residency, "file_backed")
    expect_identical(found$index_residency, "resident_cpu")
    for (i in seq_along(rows)) {
        ids <- found$indices[i, ]
        ids <- ids[ids != rows[[i]]][seq_len(5L)]
        distance <- sqrt(rowSums(sweep(x, 2L, x[rows[[i]], ])^2))
        expected <- order(distance, seq_len(nrow(x)))
        expected <- expected[expected != rows[[i]]][seq_len(5L)]
        expect_setequal(ids, expected)
    }
})
