test_that("large one-call Metal graph preparation is explicit", {
    source <- structure(list(nrow = 70000L, ncol = 96L),
        class = "fastEmbedR_massive_matrix")
    paths <- list(knn = tempfile("metal-one-call-knn-"))
    called <- NULL
    testthat::local_mocked_bindings(
        precompute_knn = function(..., backend, n.cores) {
            called <<- list(backend = backend, workers = n.cores)
            list(backend = backend, method = "hnsw_sharded",
                resources = list(peak_ram_bytes = 123))
        },
        .package = "fastEmbedR")
    stage <- fastEmbedR:::full_backed_knn_stage(source, 30L,
        "metal", 16L, paths, "8GB", NULL, NULL)
    expect_identical(called,
        list(backend = "cpu", workers = 16L))
    fit <- fastEmbedR:::full_backed_pipeline_record(
        list(resources = list(estimated_peak_ram_bytes = 456)),
        source, stage$object, list(resources = list()),
        list(layout = NULL, fit = NULL, backend = "cpu",
            resources = list()), 1, 2, 3, 4, 16L, 1L,
        "metal", FALSE, "8GB")
    expect_identical(fit$pipeline$stage_backend[["knn"]], "cpu")
    expect_identical(fit$pipeline$stage_backend[["optimize"]],
        "metal")
    expect_identical(fit$pipeline$knn_method, "hnsw_sharded")
    expect_identical(fit$pipeline$stage_resources$knn$peak_ram_bytes,
        123)
    fit$method <- "umap"
    class(fit) <- "fastEmbedR_massive_matrix"
    expect_identical(summary(fit)$memory_use$stage_resources$knn,
        list(peak_ram_bytes = 123))
})

test_that("small one-call Metal KNN uses one native worker", {
    source <- structure(list(nrow = 5000L),
        class = "fastEmbedR_massive_matrix")
    route <- fastEmbedR:::full_backed_knn_route(source,
        "metal", 16L)
    expect_identical(route, list(backend = "metal", workers = 1L))
})

test_that("Metal t-SNE reports graph-only streaming scope", {
    graph <- list(n_vertices = 1000000, n_edges = 60000000)
    plan <- list(backend = "metal", peak_ram = 2e9,
        peak_vram = NA_real_, graph_access = "stream",
        edge_chunk_capacity = 65536L, dimensions = 2L)
    resources <- fastEmbedR:::massive_tsne_result_resources(
        plan, graph, list(), list(every = 0L))
    expect_identical(resources$streaming_scope,
        "graph_upload_only")
    expect_identical(resources$optimizer_state_residency,
        "resident")
    expect_equal(resources$graph_upload_peak_bytes_estimate,
        65536 * 8 + (262144 + 1) * 8)
    expect_equal(resources$layout_state_bytes_estimate,
        1000000 * 2 * 24 + 1000000 * 8)
    expect_gt(resources$graph_bytes,
        resources$graph_upload_peak_bytes_estimate)
})

test_that("file-backed Metal UMAP retains its admission estimate", {
    plan <- fastEmbedR:::massive_umap_plan_layout(
        list(n_vertices = 1000000, max_degree = 30L),
        list(ncol = 2L), list(chunk_rows = 8192L),
        "8GB", "metal", "memory", "sgd")
    expected <- 128 * 1024^2 + 2 * 8000000 +
        65536 * 36 + 8000000
    expect_equal(plan$peak_ram_estimate, expected)
    expect_lt(plan$peak_ram_estimate, plan$limit)
})
