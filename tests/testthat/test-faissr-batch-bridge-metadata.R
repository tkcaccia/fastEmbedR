test_that("faissR private batch bridge checks exact CPU result metadata", {
    x <- matrix(as.double(seq_len(40L)), nrow = 10L)
    resident <- massive_matrix(x)
    observed <- fastEmbedR:::faissr_matrix_knn_batch_cpp(
        resident, resident, as.double(1:3), 2L,
        "cpu", "stream_exact", 10L, 1e8, 1L, TRUE)
    expect_identical(dim(observed$indices), c(3L, 2L))
    expect_identical(dim(observed$distances), c(3L, 2L))
    expect_identical(observed$device_ids, integer())
    expect_identical(observed$residency, "resident")
    expect_true(all(is.finite(c(observed$index_build_seconds,
        observed$search_seconds, observed$total_seconds,
        observed$peak_ram_bytes_estimate,
        observed$peak_device_bytes))))
    expect_false(any(observed$indices == row(observed$indices)))

    path <- tempfile(fileext = ".f32")
    on.exit(unlink(path), add = TRUE)
    writeBin(as.vector(t(x)), path, size = 4L, endian = "little")
    backed <- massive_matrix(path, nrow = 10L, ncol = 4L)
    from_file <- fastEmbedR:::faissr_matrix_knn_batch_cpp(
        backed, backed, as.double(1:3), 2L,
        "cpu", "stream_exact", 10L, 1e8, 1L, TRUE)
    expect_identical(from_file$residency, "file-backed")
    expect_identical(from_file$indices, observed$indices)
    expect_equal(from_file$distances, observed$distances)
    mixed <- fastEmbedR:::faissr_matrix_knn_batch_cpp(
        resident, backed, as.double(1:3), 2L,
        "cpu", "stream_exact", 10L, 1e8, 1L, FALSE)
    expect_identical(mixed$residency, "mixed")
    expect_identical(dim(mixed$indices), c(3L, 2L))
    expect_error(fastEmbedR:::faissr_matrix_knn_batch_cpp(
        backed, backed, as.double(1:3), 2L,
        "unknown", "stream_exact", 10L, 1e8, 1L, TRUE),
        "Unsupported faissR KNN batch backend")
})
