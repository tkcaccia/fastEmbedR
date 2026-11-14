test_that("full KNN audit caps only exact reference reads", {
    source <- list(nrow = 1000000L, ncol = 96L)
    setup <- list(resources = list(chunk_rows = 128L,
        reference_chunk_rows = 1000000L,
        memory_limit_bytes = 256 * 1024^2), workers = 1L)
    checked <- integer()
    with_mocked_bindings({
        audited <- fastEmbedR:::massive_full_knn_audit(
            list(k = 1L), source, 128L, setup)
        expected <- as.integer(floor((seq_len(128L) - 0.5) *
            source$nrow / 128L) + 1)
        expect_identical(checked, as.double(expected))
        expect_identical(audited$audit_sample_rows, expected)
        expect_identical(audited$audit_reference_chunk_rows, 500000L)
        expect_identical(setup$resources$reference_chunk_rows,
            1000000L)
        expect_identical(audited$observed_recall, 1)
    },
    faissr_matrix_knn_batch_cpp = function(query, reference,
            rows, k, backend, method, reference_chunk_rows,
            memory_limit_bytes, workers, exclude_self) {
        expect_identical(reference_chunk_rows, 500000L)
        expect_identical(reference$nrow, 1000000L)
        expect_identical(backend, "cpu")
        expect_identical(method, "stream_exact")
        expect_true(exclude_self)
        checked <<- c(checked, rows)
        list(indices = matrix(1L, length(rows), 1L),
            distances = matrix(0, length(rows), 1L),
            backend_used = "faiss_cpu_stream_exact", exact = TRUE)
    },
    massive_read_graph_edges = function(...) list(to = 1L,
        distance = 0),
    .package = "fastEmbedR")
})

test_that("exact audit results are independent of reference read blocks", {
    set.seed(1093)
    values <- matrix(rnorm(37L * 5L), ncol = 5L)
    path <- tempfile(fileext = ".f32")
    on.exit(unlink(path))
    writeBin(as.numeric(t(values)), path, size = 4L,
        endian = "little")
    source <- massive_matrix(path, 37L, 5L)
    rows <- c(2L, 11L, 29L, 36L)
    exact <- fastEmbedR:::massive_exact_audit_batch
    small <- exact(source, source, rows, 4L, 7L, 1L, TRUE,
        256 * 1024^2)
    full <- exact(source, source, rows, 4L, 37L, 1L, TRUE,
        256 * 1024^2)
    expect_identical(small$indices, full$indices)
    expect_equal(small$distances, full$distances)
})

