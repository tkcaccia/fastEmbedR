test_that("streamed multi-GPU route is explicit and method-limited", {
    graph <- list(n_vertices = 12L, n_edges = 60L,
        max_degree = 6L, storage = "csr",
        weight_method = "umap_fuzzy_union", symmetrized = TRUE)
    class(graph) <- "fastEmbedR_massive_graph"
    local_mocked_bindings(
        massive_cuda_stream_cluster = function(graph, output,
                devices, chunk_rows, memory_limit, resolution,
                passes, seed) {
            expect_identical(devices, c(0L, 1L))
            expect_identical(passes, 3L)
            list(method = "louvain_multilevel_cuda_stream")
        }, .package = "fastEmbedR")
    out <- fastEmbedR:::massive_cluster_full_cuda(graph,
        "louvain", tempfile(), c(0L, 1L), 4L, "1GB",
        1, 3L, 1L, FALSE, FALSE, NULL, "stream")
    expect_identical(out$method, "louvain_multilevel_cuda_stream")
    local_mocked_bindings(
        massive_cuda_stream_leiden = function(graph, output,
                devices, chunk_rows, memory_limit, resolution,
                passes, seed) list(
            method = "leiden_multilevel_cuda_stream"),
        .package = "fastEmbedR")
    leiden <- fastEmbedR:::massive_cluster_full_cuda(graph,
        "leiden", tempfile(), c(0L, 1L), 4L, "1GB",
        1, 3L, 1L, FALSE, FALSE, NULL, "stream")
    expect_identical(leiden$method, "leiden_multilevel_cuda_stream")
    public <- massive_cluster(graph, method = "louvain",
        massive = "out_of_core_graph", backend = "cuda",
        devices = c(0L, 1L), gpu_storage = "stream",
        output = tempfile(), n_iterations = 3L)
    expect_identical(public$method, "louvain_multilevel_cuda_stream")
    expect_error(massive_cluster(graph, method = "louvain",
        massive = "out_of_core_graph", backend = "cpu",
        gpu_storage = "stream", output = tempfile()),
        "needs multi-GPU")
})

test_that("stream planner budgets host, disk, and GPU chunks", {
    graph <- list(n_vertices = 100L, n_edges = 300L,
        max_degree = 3L, storage = "csr",
        weight_method = "umap_fuzzy_union", symmetrized = TRUE)
    class(graph) <- "fastEmbedR_massive_graph"
    local_mocked_bindings(
        massive_validate_devices = function(devices, ...) devices,
        massive_auto_cuda_free = function(...) 8 * 1024^3,
        massive_cuda_device_uuids = function(...) c("GPU-A", "GPU-B"),
        massive_disk_available_cpp = function(...) 20 * 1024^3,
        .package = "fastEmbedR")
    path <- tempfile()
    plan <- fastEmbedR:::massive_cuda_stream_plan(graph,
        path, c(0L, 1L), 50L, "2GB")
    expect_identical(plan$devices, c(0L, 1L))
    expect_equal(plan$chunk_rows, 50L)
    expect_gt(plan$edge_budget, graph$max_degree)
    graph$max_degree <- 2e6 + 1
    graph$n_edges <- 2e6 + 1
    spanning <- fastEmbedR:::massive_cuda_stream_plan(graph,
        tempfile(), c(0L, 1L), 50L, "2GB")
    expect_lt(spanning$edge_budget, graph$max_degree)
})

test_that("stream planner retains eight GPU identities and bounded work", {
    graph <- structure(list(n_vertices = 80L, n_edges = 160L,
        max_degree = 2L, storage = "csr",
        weight_method = "umap_fuzzy_union", symmetrized = TRUE),
        class = "fastEmbedR_massive_graph")
    with_mocked_bindings({
        plan <- fastEmbedR:::massive_cuda_stream_plan(graph,
            tempfile(), 0:7, 10L, "2GB")
        expect_identical(plan$devices, 0:7)
        expect_identical(plan$uuids, paste0("GPU-", 0:7))
        expect_gte(plan$edge_budget, graph$max_degree)
    }, massive_validate_devices = function(devices, ...) devices,
    massive_auto_cuda_free = function(...) 8 * 1024^3,
    massive_louvain_output = function(output, ...) output,
    massive_cuda_device_uuids = function(devices)
        paste0("GPU-", devices), .package = "fastEmbedR")
})

test_that("streamed multi-GPU invokes native clustering", {
    graph <- list(offsets_path = "off", indices_path = "ids",
        weights_path = "weights", n_vertices = 12L,
        weight_dtype = "float64")
    plan <- list(path = tempfile(), devices = c(1L, 3L),
        uuids = c("GPU-1", "GPU-3"), chunk_rows = 4L,
        edge_budget = 100L)
    called <- NULL
    with_mocked_bindings({
        result <- fastEmbedR:::massive_cuda_stream_invoke(
            graph, plan, list(passes = 2L, resolution = 1,
                seed = 7L, initial_path = "", initial_count = 0L))
        expect_identical(called$devices, c(1L, 3L))
        expect_true(called$contracted)
        expect_identical(result$backend,
            "cuda_streamed_multigpu_native")
        expect_identical(result$gpu_identity, plan$uuids)
    }, massive_louvain_cuda_devices_level_cpp = function(
            offsets_path, indices_path, weights_path,
            n_vertices, output, devices, max_passes,
            resolution, chunk_rows, edge_budget, seed,
            contracted, initial_path, initial_count) {
        called <<- list(devices = devices,
            contracted = contracted)
        list(backend = "cuda_streamed_multigpu_native",
            n_vertices = n_vertices,
            gpu_devices_used = devices,
            initial_partition = FALSE, n_communities = 3L,
            modularity_final = 0.5,
            modularity_initial = 0.1, moves = 3,
            passes = 2L)
    }, .package = "fastEmbedR")
})

test_that("streamed Leiden checks the prior parent map", {
    path <- tempfile()
    writeBin(c(1L, 2L), path, size = 4L,
        endian = "little")
    initial <- structure(list(n_refined = 2L, n_parent = 2L,
        parent_mapping_path = path,
        parent_mapping_identity = fastEmbedR:::
            massive_checkpoint_file_identity(path)),
        class = "fastEmbedR_massive_leiden_refinement")
    graph <- list(n_vertices = 2L)
    checked <- fastEmbedR:::massive_cuda_stream_initial(
        initial, graph)
    expect_identical(checked$path, path)
    writeBin(c(2L, 1L, 3L), path, size = 4L,
        endian = "little")
    expect_error(fastEmbedR:::massive_cuda_stream_initial(
        initial, graph), "changed")
})

test_that("streamed hierarchy contracts and audits original graph", {
    graph <- list(n_vertices = 6L, n_edges = 12L)
    class(graph) <- "fastEmbedR_massive_graph"
    fit_count <- 0L
    local_mocked_bindings(
        massive_louvain_contract_resources = function(graph,
                output, memory_limit) list(
            memory_limit_bytes = 2 * 1024^3),
        massive_cuda_stream_graph_identity = function(...)
            list(id = 1L),
        massive_cuda_stream_level = function(graph, output, ...) {
            fit_count <<- fit_count + 1L
            labels <- if (fit_count == 1L)
                c(1L, 1L, 1L, 2L, 2L, 2L) else c(1L, 2L)
            writeBin(labels, output, size = 4L, endian = "little")
            list(membership_path = output, n_vertices = length(labels),
                n_communities = 2L, modularity_final = 0.5,
                method = "louvain_local_move_level")
        },
        massive_louvain_contract = function(...) list(n_vertices = 2L),
        massive_louvain_coarse_csr = function(...) list(
            n_vertices = 2L, n_edges = 2L),
        massive_remap_louvain_cpp = function(...) NULL,
        massive_graph_modularity = function(...) list(modularity = 0.5),
        massive_checkpoint_file_identity = function(...) list(id = 2L),
        .package = "fastEmbedR")
    output <- tempfile()
    result <- fastEmbedR:::massive_cuda_stream_cluster(graph,
        output, c(0L, 1L), 4L, "2GB", 1, 2L, 1L)
    expect_equal(fit_count, 2L)
    expect_identical(result$method, "louvain_multilevel_cuda_stream")
    expect_length(result$levels, 1L)
    expect_true(file.exists(output))
})

test_that("streamed Louvain levels use real CSR contraction", {
    skip_on_os("windows")
    prefix <- tempfile()
    source <- tempfile()
    output <- tempfile()
    neighbors <- t(vapply(seq_len(8L), function(row) {
        setdiff(if (row <= 4L) 1:4 else 5:8, row)
    }, integer(3L)))
    distances <- matrix(rep(c(0.1, 0.3, 0.7), 8L),
        nrow = 8L, byrow = TRUE)
    writeBin(as.integer(t(neighbors)), paste0(source,
        ".indices.u32"), size = 4L, endian = "little")
    writeBin(as.numeric(t(distances)), paste0(source,
        ".distances.f32"), size = 4L, endian = "little")
    graph <- massive_umap_fuzzy_graph(massive_umap_memberships(
        massive_knn_graph(source, 8L, 3L)), prefix,
        memory_limit = "64MB")
    local_mocked_bindings(massive_cuda_stream_level = function(
            graph, output, ...) {
        labels <- if (graph$n_vertices == 8L)
            rep(1:2, each = 4L) else seq_len(graph$n_vertices)
        writeBin(as.integer(labels), output, size = 4L,
            endian = "little")
        score <- massive_graph_modularity(graph, output,
            max(labels), 1, 2L, "128MB")
        list(membership_path = output,
            membership_identity = fastEmbedR:::
                massive_checkpoint_file_identity(output),
            n_vertices = graph$n_vertices,
            n_communities = max(labels),
            modularity_final = score$modularity,
            total_edge_weight = score$total_edge_weight,
            resources = list(chunk_rows = 2L, edge_budget = 100L))
    }, .package = "fastEmbedR")
    result <- fastEmbedR:::massive_cuda_stream_cluster(graph,
        output, c(0L, 1L), 2L, "128MB", 1, 2L, 1L)
    expect_equal(result$n_communities, 2L)
    expect_identical(result$contraction_backend, "cpu")
    expect_true(file.exists(output))
})

test_that("streamed Louvain rejects an unconverged 64-level result", {
    first_path <- tempfile()
    writeBin(c(1L, 1L, 2L), first_path, size = 4L,
        endian = "little")
    first <- list(membership_path = first_path,
        n_communities = 2L)
    graph <- list(n_vertices = 3L)
    local_mocked_bindings(
        massive_louvain_contract = function(...) list(
            n_vertices = 3L),
        massive_louvain_coarse_csr = function(...) list(
            n_vertices = 3L),
        massive_cuda_stream_level = function(graph, output, ...)
            list(n_communities = 2L, membership_path = output),
        massive_remap_louvain_cpp = function(...) NULL,
        .package = "fastEmbedR")
    expect_error(fastEmbedR:::massive_cuda_stream_hierarchy(
        graph, first, tempfile(), c(0L, 1L), 2L,
        "128MB", 1, 2L, 1L,
        list(memory_limit_bytes = 128 * 1024^2)),
        "exceeded 64 levels")
})

test_that("streamed Leiden uses real refinement and connectivity audit", {
    skip_on_os("windows")
    source <- tempfile()
    prefix <- tempfile()
    output <- tempfile()
    neighbors <- t(vapply(seq_len(8L), function(row) {
        setdiff(if (row <= 4L) 1:4 else 5:8, row)
    }, integer(3L)))
    distances <- matrix(rep(c(0.1, 0.3, 0.7), 8L),
        nrow = 8L, byrow = TRUE)
    writeBin(as.integer(t(neighbors)), paste0(source,
        ".indices.u32"), size = 4L, endian = "little")
    writeBin(as.numeric(t(distances)), paste0(source,
        ".distances.f32"), size = 4L, endian = "little")
    graph <- massive_umap_fuzzy_graph(massive_umap_memberships(
        massive_knn_graph(source, 8L, 3L)), prefix,
        memory_limit = "64MB")
    local_mocked_bindings(
        massive_cuda_stream_level = function(graph, output,
                devices, chunk_rows, memory_limit, resolution,
                ...) {
            labels <- if (graph$n_vertices == 8L)
                rep(1:2, each = 4L) else seq_len(graph$n_vertices)
            writeBin(as.integer(labels), output, size = 4L,
                endian = "little")
            score <- massive_graph_modularity(graph, output,
                max(labels), resolution, 2L, "128MB")
            structure(list(membership_path = output,
                membership_identity = fastEmbedR:::
                    massive_checkpoint_file_identity(output),
                n_vertices = graph$n_vertices,
                n_communities = max(labels),
                resolution = resolution,
                modularity_final = score$modularity,
                total_edge_weight = score$total_edge_weight,
                resources = list(chunk_rows = 2L,
                    edge_budget = 100L),
                graph_identity = graph$file_identity),
                class = "fastEmbedR_massive_louvain_level")
        },
        massive_cuda_device_uuids = function(...)
            c("GPU-A", "GPU-B"), .package = "fastEmbedR")
    result <- fastEmbedR:::massive_cuda_stream_leiden(graph,
        output, c(0L, 1L), 2L, "128MB", 1, 2L, 1L)
    expect_equal(result$n_communities, 2L)
    expect_equal(result$graph_audit$disconnected_communities, 0L)
    expect_identical(result$refinement_backend, "cpu")
})
