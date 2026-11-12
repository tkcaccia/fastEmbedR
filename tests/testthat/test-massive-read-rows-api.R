test_that("one reader covers matrix and full-graph cluster results", {
    matrix_path <- tempfile(fileext = ".f32")
    writeBin(as.numeric(1:12), matrix_path, size = 4L,
        endian = "little")
    matrix_source <- massive_matrix(matrix_path, nrow = 3L,
        ncol = 4L)
    expect_equal(massive_read_rows(matrix_source, 2L, 1L),
        matrix(5:8, nrow = 1L))

    labels_path <- tempfile(fileext = ".u32")
    writeBin(c(1L, 2L, 1L), labels_path, size = 4L,
        endian = "little")
    state <- list(n_vertices = 3L, membership_path = labels_path,
        membership_identity = fastEmbedR:::
            massive_checkpoint_file_identity(labels_path))
    for (kind in c("fastEmbedR_massive_louvain_level",
            "fastEmbedR_massive_leiden")) {
        result <- structure(state, class = kind)
        expect_identical(massive_read_rows(result, 2L, 2L),
            c(2L, 1L))
    }
    expect_error(massive_read_rows(list()), "supported")
})

test_that("one reader routes landmark cluster labels and confidence", {
    local_mocked_bindings(massive_read_cluster_rows = function(
            x, first, n) list(first = first, n = n),
        .package = "fastEmbedR")
    result <- structure(list(), class =
        "fastEmbedR_massive_clusters")
    expect_identical(massive_read_rows(result, 3L, 4L),
        list(first = 3L, n = 4L))
})
