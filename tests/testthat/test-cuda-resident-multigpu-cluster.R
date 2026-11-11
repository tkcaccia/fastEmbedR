test_that("coordinated CUDA graph fit uses selected GPUs", {
    graph <- list(from = c(1L, 2L, 3L),
        to = c(2L, 3L, 4L), weight = rep(1, 3L),
        n_vertices = 4L)
    calls <- 0L
    with_mocked_bindings({
        fit <- graph_cluster(graph, method = "leiden",
            backend = "cuda", n_runs = 1L, devices = c(0L, 1L))
        expect_identical(calls, 1L)
        expect_identical(fit$implementation,
            "resident_cuda_coordinated_host_sync")
        expect_identical(fit$gpu_devices, c(0L, 1L))
        expect_identical(fit$gpu_identity, c("gpu0", "gpu1"))
        expect_identical(fit$parameters$n_runs, 1L)
    },
    massive_validate_devices = function(devices, backend, nrow)
        as.integer(devices),
    massive_cuda_device_uuids = function(devices)
        c("gpu0", "gpu1"),
    massive_auto_cuda_free = function(...) 8 * 1024^3,
    graph_clustering_cuda_available_cpp = function() TRUE,
    fastembedr_graph_cluster_cuda_devices_cpp = function(...) {
        calls <<- calls + 1L
        list(membership = c(1L, 1L, 2L, 2L),
            modularity = 0.2, n_communities = 2L,
            gpu_devices = c(0L, 1L),
            implementation =
                "resident_cuda_coordinated_host_sync")
    },
    fastembedr_graph_cluster_cuda_cpp = function(...) {
        stop("single-GPU fallback")
    }, .package = "fastEmbedR")
})

test_that("landmark clustering sends both GPUs to one shared fit", {
    graph <- list(from = c(1L, 2L, 3L),
        to = c(2L, 3L, 4L), weight = rep(1, 3L),
        n_vertices = 4L)
    controls <- list(method = "leiden", backend = "cuda",
        resolution = 1, n_iterations = 5L, n_runs = 1L,
        steps = 4L, seed = 1L, workers = 1L,
        devices = c(0L, 1L), parallel.mode = "restarts")
    calls <- 0L
    with_mocked_bindings({
        fit <- fastEmbedR:::massive_cluster_fit(graph, controls,
            checkpoint = NULL, resume = FALSE)
        expect_identical(calls, 1L)
        expect_identical(fit$gpu_devices, c(0L, 1L))
    }, graph_cluster = function(graph, ..., devices) {
        calls <<- calls + 1L
        expect_identical(devices, c(0L, 1L))
        list(gpu_devices = devices)
    }, .package = "fastEmbedR")
})

test_that("auto resident clustering builds KNN on a requested GPU", {
    source <- matrix(seq_len(64), nrow = 16L)
    selected <- 0L
    with_mocked_bindings({
        fit <- fastEmbedR:::massive_cluster_in_memory(source,
            3L, "leiden", "cuda", 1L, 1, 5L, 1L, 4L, 1L,
            1e8, devices = c(1L, 2L))
        expect_identical(fit$mode, "in_memory")
        expect_identical(fit$gpu_devices, c(1L, 2L))
        expect_identical(selected, 0L)
    }, massive_cuda_memory_cpp = function() list(device = 0L),
    massive_cuda_select_cpp = function(device) {
        selected <<- device
        invisible(NULL)
    }, knn_graph = function(...) {
        expect_identical(selected, 1L)
        list(n_vertices = 16L)
    }, graph_cluster = function(graph, ..., devices) {
        expect_identical(selected, 1L)
        expect_identical(devices, c(1L, 2L))
        list(gpu_devices = devices)
    }, .package = "fastEmbedR")
})

test_that("resident CUDA graph devices reject unsupported controls", {
    graph <- list(from = 1L, to = 2L, weight = 1,
        n_vertices = 2L)
    expect_error(graph_cluster(graph, backend = "cuda",
        devices = 0L), "at least two")
    with_mocked_bindings({
        expect_error(graph_cluster(graph, backend = "cuda",
            n.cores = 2L,
            devices = c(0L, 1L)), "CPU workers")
    }, massive_validate_devices = function(devices, backend, nrow)
        as.integer(devices), .package = "fastEmbedR")
})

test_that("resident graph clustering forwards eight selected GPUs", {
    graph <- list(from = seq.int(1L, 15L),
        to = seq.int(2L, 16L), weight = rep(1, 15L),
        n_vertices = 16L)
    selected <- 0:7
    calls <- 0L
    with_mocked_bindings({
        fit <- graph_cluster(graph, method = "louvain",
            backend = "cuda", devices = selected)
        expect_identical(calls, 1L)
        expect_identical(fit$gpu_devices, selected)
        expect_identical(fit$gpu_identity, paste0("gpu", selected))
        expect_length(fit$membership, 16L)
    },
    massive_validate_devices = function(devices, backend, nrow) {
        expect_identical(backend, "cuda")
        expect_identical(nrow, 16L)
        as.integer(devices)
    },
    massive_cuda_device_uuids = function(devices)
        paste0("gpu", devices),
    massive_auto_cuda_free = function(...) 8 * 1024^3,
    graph_clustering_cuda_available_cpp = function() TRUE,
    fastembedr_graph_cluster_cuda_devices_cpp = function(..., devices) {
        calls <<- calls + 1L
        expect_identical(devices, selected)
        list(membership = rep(1:2, each = 8L),
            modularity = 0.2, n_communities = 2L,
            gpu_devices = devices,
            implementation = "resident_cuda_coordinated_host_sync")
    }, .package = "fastEmbedR")
})

test_that("coordinated CUDA clustering accepts consensus controls", {
    graph <- list(from = c(1L, 2L, 3L),
        to = c(2L, 3L, 4L), weight = rep(1, 3L),
        n_vertices = 4L)
    calls <- 0L
    with_mocked_bindings({
        fit <- graph_cluster(graph, method = "leiden",
            backend = "cuda", resolution = c(0.5, 1),
            n_runs = 1L, devices = c(0L, 1L))
        expect_identical(calls, 3L)
        expect_identical(fit$parameters$consensus$partitions, 2L)
        expect_identical(fit$reweighted_graph$n_edges, 2L)
        expect_identical(fit$gpu_identity, c("gpu0", "gpu1"))
    },
    massive_validate_devices = function(devices, backend, nrow)
        as.integer(devices),
    massive_cuda_device_uuids = function(devices)
        c("gpu0", "gpu1"),
    massive_auto_cuda_free = function(...) 8 * 1024^3,
    graph_clustering_cuda_available_cpp = function() TRUE,
    fastembedr_graph_cluster_cuda_devices_cpp = function(...) {
        calls <<- calls + 1L
        list(membership = c(1L, 1L, 2L, 2L),
            modularity = 0.2, n_communities = 2L,
            implementation = "resident_cuda_coordinated_host_sync")
    }, .package = "fastEmbedR")
})
