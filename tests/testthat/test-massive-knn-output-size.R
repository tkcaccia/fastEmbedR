test_that("landmark KNN finalization reports short output files", {
    root <- tempfile("massive-knn-size-")
    dir.create(root)
    on.exit(unlink(root, recursive = TRUE), add = TRUE)
    paths <- c(indices = file.path(root, "index.u32"),
        distances = file.path(root, "distance.f32"))
    parts <- paste0(paths, ".part")
    writeBin(as.integer(1:3), parts[[1L]], size = 4L)
    writeBin(as.numeric(1:3), parts[[2L]], size = 4L)
    expect_error(fastEmbedR:::massive_knn_finish(
        NULL, NULL, parts, paths, 2, 3),
        "expected 24 bytes per file; observed 12, 12")
    expect_false(any(file.exists(paths)))
})
