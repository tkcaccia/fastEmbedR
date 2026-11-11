test_that("native resident graph admission accounts for full CSR", {
    graph <- structure(list(n_vertices = 100L,
        n_edges = 300L, max_degree = 3L,
        storage = "csr", weight_method = "umap_fuzzy_union",
        symmetrized = TRUE),
        class = "fastEmbedR_massive_graph")
    with_mocked_bindings({
        plan <- fastEmbedR:::massive_cuda_resident_plan(
            graph, tempfile(), c(0L, 1L), 4L, "2GB")
        expect_identical(plan$devices, c(0L, 1L))
        expect_gt(plan$host_estimate,
            40 * graph$n_edges)
        graph$n_edges <- 1e9
        expect_error(fastEmbedR:::
            massive_cuda_resident_plan(graph, tempfile(),
                c(0L, 1L), 4L, "2GB"),
            "admitted RAM or VRAM")
    }, massive_validate_devices = function(devices, ...) devices,
    massive_auto_cuda_free = function(...) 8 * 1024^3,
    massive_cuda_device_uuids = function(...) c("GPU-A", "GPU-B"),
    massive_louvain_output = function(output, ...) output,
    .package = "fastEmbedR")
})

test_that("full-graph CUDA dispatches to the native resident route", {
    graph <- structure(list(),
        class = "fastEmbedR_massive_graph")
    with_mocked_bindings({
        result <- massive_cluster(graph, method = "leiden",
            massive = "out_of_core_graph", backend = "cuda",
            output = tempfile(), devices = c(0L, 1L))
        expect_identical(result$mode, "out_of_core_graph")
        expect_identical(result$backend, "cuda")
        expect_identical(result$devices, c(0L, 1L))
        expect_error(massive_cluster(graph, method = "walktrap",
            massive = "out_of_core_graph", backend = "cuda",
            output = tempfile(), devices = c(0L, 1L)),
            "Walktrap|supports Leiden/Louvain")
    }, massive_cuda_resident_cluster = function(graph,
            output, method, devices, ...) list(
        backend = "cuda", devices = devices),
    .package = "fastEmbedR")
})

test_that("native multi-GPU clustering preserves complete graph", {
    available <- tryCatch(
        fastEmbedR:::massive_cuda_device_count_cpp() >= 2L,
        error = function(e) FALSE)
    skip_if_not(available)
    prefix <- tempfile()
    source <- tempfile()
    neighbors <- t(vapply(seq_len(8L), function(row) {
        setdiff(if (row <= 4L) 1:4 else 5:8, row)
    }, integer(3L)))
    distances <- matrix(rep(c(0.1, 0.3, 0.7), 8L),
        nrow = 8L, byrow = TRUE)
    writeBin(as.integer(t(neighbors)),
        paste0(source, ".indices.u32"),
        size = 4L, endian = "little")
    writeBin(as.numeric(t(distances)),
        paste0(source, ".distances.f32"),
        size = 4L, endian = "little")
    graph <- massive_umap_fuzzy_graph(massive_umap_memberships(
        massive_knn_graph(source, 8L, 3L)), prefix,
        memory_limit = "64MB")
    for (storage in c("resident", "stream")) {
        for (method in c("louvain", "leiden")) {
            result <- graph_cluster(graph, method = method,
                backend = "cuda", devices = c(0L, 1L),
                gpu_storage = storage, output = tempfile(),
                chunk_rows = 2L, memory_limit = "2GB",
                n_iterations = 2L, seed = 42L)
            expect_equal(result$n_communities, 2L)
            expect_identical(result$backend, "cuda")
            expect_equal(file.info(result$membership_path)$size,
                4 * graph$n_vertices)
            expect_equal(result$modularity_final, 0.5,
                tolerance = 1e-5)
        }
    }
})
