test_that("native label compaction preserves sorted community IDs", {
    path <- tempfile(fileext = ".u32")
    on.exit(unlink(path), add = TRUE)
    writeBin(as.integer(c(1, 5, 5, 12, 3, 12, 8)), path,
        size = 4L, endian = "little")
    expect_identical(fastEmbedR:::massive_compact_u32_labels_cpp(
        path, 7), 5L)
    expect_identical(readBin(path, integer(), n = 7L,
        size = 4L, endian = "little"),
        as.integer(c(1, 3, 3, 5, 2, 5, 4)))
})

test_that("native compaction handles the maximum isolate label", {
    path <- tempfile(fileext = ".u32")
    on.exit(unlink(path), add = TRUE)
    writeBin(as.integer(c(1, 8, 1, 5)), path,
        size = 4L, endian = "little")
    expect_identical(fastEmbedR:::massive_compact_u32_labels_cpp(
        path, 4), 3L)
    expect_identical(readBin(path, integer(), n = 4L,
        size = 4L, endian = "little"),
        as.integer(c(1, 3, 1, 2)))
})

test_that("native compaction handles more than one bounded chunk", {
    path <- tempfile(fileext = ".u32")
    on.exit(unlink(path), add = TRUE)
    values <- seq_len(1000003L) %% 97L + 1L
    writeBin(as.integer(values), path, size = 4L,
        endian = "little")
    expect_identical(fastEmbedR:::massive_compact_u32_labels_cpp(
        path, length(values)), 97L)
    expect_identical(readBin(path, integer(), n = length(values),
        size = 4L, endian = "little"), as.integer(values))
})

test_that("native compaction validates before rewriting labels", {
    path <- tempfile(fileext = ".u32")
    on.exit(unlink(path), add = TRUE)
    original <- as.integer(c(1, 0, 3))
    writeBin(original, path, size = 4L, endian = "little")
    expect_error(fastEmbedR:::massive_compact_u32_labels_cpp(
        path, 3), "outside the validated range")
    expect_identical(readBin(path, integer(), n = 3L,
        size = 4L, endian = "little"), original)
    expect_error(fastEmbedR:::massive_compact_u32_labels_cpp(
        path, 4), "file size")
})
