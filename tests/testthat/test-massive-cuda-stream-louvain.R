test_that("file-backed Louvain selects CUDA local moves explicitly", {
    graph <- structure(list(storage = "csr",
        weight_method = "umap_fuzzy_union",
        symmetrized = TRUE, n_vertices = 8L),
        class = "fastEmbedR_massive_graph")
    resources <- list(chunk_rows = 2L, edge_budget = 1000)
    calls <- character()
    with_mocked_bindings({
        fastEmbedR:::massive_louvain_graph_check(graph, "cuda",
            allow_cuda_level = TRUE)
        fit <- fastEmbedR:::massive_louvain_native_level(
            c("offsets", "indices", "weights"), graph,
            "labels", 3L, 1, resources, 4L, FALSE,
            list(path = "", count = 0L), "cuda")
        expect_identical(calls, "cuda")
        expect_identical(fit$backend, "cuda")
    },
    graph_clustering_cuda_available_cpp = function() TRUE,
    massive_louvain_cuda_level_cpp = function(...) {
        calls <<- c(calls, "cuda")
        list(backend = "cuda")
    },
    massive_louvain_level_cpp = function(...) {
        stop("CPU fallback")
    }, .package = "fastEmbedR")
})

test_that("file-backed CUDA Louvain rejects unavailable CUDA", {
    graph <- structure(list(storage = "csr",
        weight_method = "umap_fuzzy_union",
        symmetrized = TRUE),
        class = "fastEmbedR_massive_graph")
    with_mocked_bindings({
        expect_error(fastEmbedR:::massive_louvain_graph_check(
            graph, "cuda", allow_cuda_level = TRUE), "unavailable")
    }, graph_clustering_cuda_available_cpp = function() FALSE,
    .package = "fastEmbedR")
})

test_that("full-graph facade selects bounded CUDA Louvain", {
    calls <- 0L
    with_mocked_bindings({
        fit <- fastEmbedR:::massive_cluster_full_graph(
            list(), "louvain", "cuda", "labels", 1L, 2L,
            "128MB", 1, 3L, 4L, 7L, FALSE, FALSE, FALSE,
            NULL, NULL)
        expect_identical(calls, 1L)
        expect_identical(fit$backend, "cuda")
        expect_identical(fit$mode, "out_of_core_graph")
    },
    massive_louvain = function(...) {
        calls <<- calls + 1L
        list(backend = "cuda")
    },
    massive_cluster_full_cuda = function(...) stop("cuGraph fallback"),
    .package = "fastEmbedR")
})

test_that("file-backed Leiden refinement selects CUDA kernels", {
    calls <- 0L
    with_mocked_bindings({
        fit <- fastEmbedR:::massive_leiden_refine_native(
            c("offsets", "indices", "weights", "parent"),
            list(n_vertices = 8L, weight_dtype = "float32"),
            list(n_communities = 2L), "labels", 1,
            list(chunk_rows = 2L, edge_budget = 1000), 7L,
            "cuda")
        expect_identical(calls, 1L)
        expect_identical(fit$backend, "cuda")
    },
    massive_leiden_cuda_refine_cpp = function(...) {
        calls <<- calls + 1L
        list(backend = "cuda")
    },
    massive_leiden_refine_cpp = function(...) stop("CPU fallback"),
    .package = "fastEmbedR")
})

test_that("full-graph facade selects bounded CUDA Leiden", {
    calls <- 0L
    with_mocked_bindings({
        fit <- fastEmbedR:::massive_cluster_full_graph(
            list(), "leiden", "cuda", "labels", 1L, 2L,
            "128MB", 1, 3L, 4L, 7L, FALSE, FALSE, FALSE,
            NULL, NULL)
        expect_identical(calls, 1L)
        expect_identical(fit$backend, "cuda")
        expect_identical(fit$mode, "out_of_core_graph")
    },
    massive_leiden = function(...) {
        calls <<- calls + 1L
        list(backend = "cuda")
    },
    massive_cluster_full_cuda = function(...) stop("cuGraph fallback"),
    .package = "fastEmbedR")
})
