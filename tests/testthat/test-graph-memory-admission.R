test_that("file-backed graph and optimizers respect available RAM", {
    testthat::local_mocked_bindings(
        massive_available_ram_bytes = function() 16 * 1024^2,
        .package = "fastEmbedR")
    graph <- list(n_vertices = 100, n_edges = 1000,
        max_degree = 10)
    init <- list(ncol = 2L)
    expect_error(fastEmbedR:::massive_umap_fuzzy_preflight(
        graph, tempfile(), "1GB"), "at least 64 MB")
    expect_error(fastEmbedR:::massive_tsne_resource_plan(
        graph, init, tempfile(fileext = ".f32"), "1GB",
        "cpu", NULL), "exceeds `memory_limit`")
    expect_error(fastEmbedR:::massive_umap_plan_layout(
        graph, init, list(chunk_rows = 1L), "1GB",
        "cpu", "memory", "sgd"), "exceeds `memory_limit`")
})

test_that("unknown available RAM closes full-graph admission", {
    testthat::local_mocked_bindings(
        massive_available_ram_bytes = function() NA_real_,
        .package = "fastEmbedR")
    graph <- list(n_vertices = 100, n_edges = 1000,
        max_degree = 10)
    init <- list(ncol = 2L)
    expect_error(fastEmbedR:::massive_tsne_resource_plan(
        graph, init, tempfile(fileext = ".f32"), "1GB",
        "cpu", NULL), "cannot determine available RAM")
})
