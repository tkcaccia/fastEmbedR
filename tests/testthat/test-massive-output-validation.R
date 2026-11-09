test_that("full file-backed output is scanned with bounded reads", {
    path <- tempfile(fileext = ".f32")
    on.exit(unlink(path))
    layout <- matrix(seq_len(18L) / 10, 6L, 3L)
    writeBin(as.numeric(t(layout)), path, size = 4L,
        endian = "little")
    output <- massive_matrix(path, 6L, 3L)
    audit <- fastEmbedR:::massive_validate_matrix_output(output,
        max_bytes = 64)
    expect_equal(audit$rows, 6)
    expect_equal(audit$columns, 3)
    expect_equal(audit$output_bytes, 72)
    expect_true(audit$finite)
    expect_equal(audit$max_abs_coordinate, 1.8,
        tolerance = 1e-6)
})

test_that("full output audit rejects a non-finite coordinate", {
    path <- tempfile(fileext = ".f32")
    on.exit(unlink(path))
    values <- matrix(c(1, 2, 3, NaN, 5, 6), 3L, 2L)
    writeBin(as.numeric(t(values)), path, size = 4L,
        endian = "little")
    output <- massive_matrix(path, 3L, 2L)
    expect_error(fastEmbedR:::massive_validate_matrix_output(output,
        max_bytes = 64), "non-finite")
})
