test_that("bounded KNN row reads use the faissR private bridge", {
    indices <- rbind(c(2L, 3L), c(1L, 3L), c(2L, 1L))
    distances <- rbind(c(1, 2), c(1.5, 2.5), c(3, 4))
    ids_path <- tempfile(fileext = ".u32")
    distances_path <- tempfile(fileext = ".f32")
    on.exit(unlink(c(ids_path, distances_path)), add = TRUE)
    writeBin(as.integer(t(indices)), ids_path, size = 4L,
        endian = "little")
    writeBin(as.numeric(t(distances)), distances_path, size = 4L,
        endian = "little")
    found <- fastEmbedR:::massive_read_knn_rows_cpp(
        ids_path, distances_path,
        3, 2L, 3, 2, 2L, 1000)
    expect_identical(found$indices,
        indices[2:3, , drop = FALSE])
    expect_equal(found$distances,
        distances[2:3, , drop = FALSE], tolerance = 1e-6)
    expect_identical(names(found), c("indices", "distances"))
    expect_error(fastEmbedR:::massive_read_knn_rows_cpp(ids_path,
        distances_path, 3, 2L, 3, 3, 2L, 1000), "range")
})
