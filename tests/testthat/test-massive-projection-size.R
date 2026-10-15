test_that("projection completion checks sizes beyond integer range", {
    layout <- list(path = "layout.f32", nrow = 100000000L, ncol = 2L)
    graph <- list(nrow = 100000000L, ncol = 30L,
        indices_path = "indices.u32", distances_path = "distances.f32")
    bytes <- c(800000000, 12000000000, 12000000000)
    result <- list(layout = layout, graph = graph)
    identity <- function(paths) list(bytes = tail(bytes, length(paths)))
    completed <- with_mocked_bindings(
        fastEmbedR:::massive_projection_complete(result),
        massive_checkpoint_file_identity = identity,
        massive_checkpoint_write = function(...) NULL)
    expect_identical(completed$file_identity$bytes, bytes)
})

test_that("projection reopening checks sizes beyond integer range", {
    directory <- tempfile("large-projection-")
    dir.create(directory)
    on.exit(unlink(directory, recursive = TRUE))
    path <- file.path(directory, "layout.f32")
    file.create(path)
    layout <- list(path = path, format = "f32",
        nrow = 100000000, ncol = 2L)
    graph <- list(nrow = 100000000L, ncol = 30L,
        indices_path = "indices.u32", distances_path = "distances.f32")
    class(layout) <- "fastEmbedR_massive_matrix"
    class(graph) <- "fastEmbedR_massive_knn"
    bytes <- c(800000000, 12000000000, 12000000000)
    identity <- list(bytes = bytes)
    result <- list(layout = layout, graph = graph,
        manifest_path = paste0(path, ".manifest.rds"),
        file_identity = identity, experimental = TRUE,
        backend = "cpu", method = "umap", row_start = 1L,
        row_end = 100000000L)
    class(result) <- "fastEmbedR_massive_projection"
    saved <- list(version = 1L, result = result)
    saveRDS(saved, result$manifest_path)
    reopened <- with_mocked_bindings(
        fastEmbedR::massive_open_projection(path),
        massive_checkpoint_file_identity = function(...) identity,
        .package = "fastEmbedR")
    expect_identical(reopened$file_identity$bytes, bytes)
})
