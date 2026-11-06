test_that("landmark HNSW effort is explicit and survives graph creation", {
    skip_if_not_installed("float")
    set.seed(20261102)
    values <- matrix(rnorm(80L * 8L), ncol = 8L)
    input <- tempfile(fileext = ".f32")
    writeBin(as.vector(t(values)), input, size = 4L,
        endian = "little")
    directory <- tempfile()
    dir.create(directory)
    on.exit(unlink(c(input, directory), recursive = TRUE))
    source <- massive_matrix(input, 80L, 8L)
    graph <- massive_landmark_knn(source, source, 5L,
        file.path(directory, "hnsw"), backend = "cpu",
        method = "hnsw", n.cores = 2L, ef_search = 20L,
        chunk_rows = 17L, memory_limit = "256MB", checkpoint = TRUE)
    expect_identical(graph$method, "hnsw")
    expect_identical(graph$hnsw_settings$efSearch, 20L)
    expect_identical(graph$hnsw_settings$tuning_rule,
        "explicit_ef_search")
    expect_false(graph$recall_audited)
    expect_equal(dim(massive_read_knn_rows(graph, 1L, 4L)$indices),
        c(4L, 5L))
    expect_error(massive_landmark_knn(source, source, 5L,
        file.path(directory, "exact"), method = "exact",
        ef_search = 20L), "requires CPU HNSW")
    expect_error(massive_landmark_knn(source, source, 5L,
        file.path(directory, "too_small"), method = "hnsw",
        ef_search = 5L), "reference row count")
    setup <- fastEmbedR:::massive_knn_setup(source, source, 5L,
        file.path(directory, "checkpoint"), "cpu", 1L,
        "hnsw", 17L, "256MB", FALSE, ef_search = 20L)
    saved <- fastEmbedR:::massive_knn_checkpoint(source, setup, FALSE)
    changed <- setup
    changed$ef_search <- 21L
    expect_error(fastEmbedR:::massive_knn_checkpoint(
        source, changed, TRUE), "does not match")
    unlink(saved$path)
})

test_that("massive UMAP forwards HNSW effort without fallback", {
    skip_if_not_installed("float")
    set.seed(20261103)
    values <- matrix(rnorm(80L * 8L), ncol = 8L)
    input <- tempfile(fileext = ".f32")
    writeBin(as.vector(t(values)), input, size = 4L,
        endian = "little")
    directory <- tempfile()
    dir.create(directory)
    on.exit(unlink(c(input, directory), recursive = TRUE))
    source <- massive_matrix(input, 80L, 8L)
    result <- with_mocked_bindings(
        fastembedr_cpu_knn_method = function(...) "hnsw",
        umap(source, massive = "landmark", landmarks = 30L,
            output = file.path(directory, "layout.f32"),
            backend = "cpu", n.cores = 2L, n_neighbors = 5L,
            transform_k = 5L, chunk_rows = 17L,
            memory_limit = "256MB", ef_search = 20L),
        .package = "fastEmbedR")
    expect_identical(result$graph$method, "hnsw")
    expect_identical(result$graph$hnsw_settings$efSearch, 20L)
    expect_identical(result$graph$hnsw_settings$tuning_rule,
        "explicit_ef_search")
    expect_true(all(is.finite(as.matrix(result$layout))))
    expect_error(umap(values, ef_search = 20L), "massive")
    expect_error(umap(source, massive = "landmark", landmarks = 30L,
        output = file.path(directory, "reused.f32"), nn = result,
        ef_search = 21L), "cannot change a reused")
    expect_error(fastEmbedR:::massive_embedding_ef_search(
        20L, NULL, "metal"), "no backend fallback")
})
