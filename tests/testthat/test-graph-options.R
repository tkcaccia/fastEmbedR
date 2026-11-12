test_that("native consensus squares edge coassignment frequencies", {
    graph <- three_clique_graph()
    index <- order(graph$from, graph$to)
    for (field in c("from", "to", "weight")) {
        graph[[field]] <- graph[[field]][index]
    }
    grids <- list(list(resolution = c(0.001, 1), seed = 3L),
        list(resolution = 1, seed = c(3L, 4L)),
        list(resolution = c(0.001, 1), seed = c(3L, 4L)))
    for (method in c("leiden", "louvain")) for (controls in grids) {
        count <- length(controls$resolution) * length(controls$seed)
        frequency <- numeric(length(graph$weight))
        for (r in controls$resolution) for (seed in controls$seed) {
            labels <- graph_cluster(graph, method = method, resolution = r,
                backend = "cpu", seed = seed)$membership
            frequency <- frequency + (labels[graph$from] == labels[graph$to])
        }
        expected <- graph$weight * (frequency / count)^2
        if (length(controls$resolution) > 1L) {
            expect_true(any(frequency > 0 & frequency < count))
        }
        result <- graph_cluster(graph, method = method, backend = "cpu",
            resolution = controls$resolution, seed = controls$seed)
        expect_equal(result$reweighted_graph$weight, expected[expected > 0])
        expect_equal(result$parameters$consensus$partitions, count)
        expect_identical(result$parameters$consensus$method, method)
        expect_identical(result$parameters$consensus$implementation,
            "native_sparse_edge_consensus")
        expect_equal(result$parameters$final_resolution,
            controls$resolution[1L])
        expect_equal(result$parameters$final_seed, controls$seed[1L])
        reference <- graph_cluster(result$reweighted_graph,
            method = method, backend = "cpu",
            resolution = controls$resolution[1L], seed = controls$seed[1L])
        expect_identical(result$membership, reference$membership)
        expect_equal(result$modularity, reference$modularity)
    }
})

test_that("native projection weights use exact local double radii", {
    graph <- three_clique_graph()
    index <- order(graph$from, graph$to)
    for (field in c("from", "to", "weight")) {
        graph[[field]] <- graph[[field]][index]
    }
    layout <- cbind(c(seq(0, 0.25, length.out = 6L),
        seq(0.3, 0.55, length.out = 6L),
        seq(5, 5.25, length.out = 6L)), rep(0, 18L))
    distance <- as.matrix(stats::dist(layout))
    diag(distance) <- Inf
    radius <- apply(distance, 1L, function(x) sort(x)[3L])
    ratio <- distance[cbind(graph$from, graph$to)] /
        sqrt(radius[graph$from] * radius[graph$to])
    penalty <- exp(-pmax(ratio / 2 - 1, 0)^2)
    expected <- graph$weight * penalty
    keep <- penalty >= 0.001
    expect_true(any(!keep))
    for (method in c("leiden", "louvain")) {
        controls <- list(layout = layout, k = 3L, stretch = 2)
        result <- graph_cluster(graph, method = method, backend = "cpu",
            projection = controls, seed = 3L)
        expect_equal(result$reweighted_graph$weight, unname(expected[keep]),
            tolerance = 1e-12)
        expect_identical(result$parameters$projection$distance_precision,
            "double")
        expect_identical(result$parameters$projection$implementation,
            "native_exact_2d_edge_penalty")
        controls$layout <- cbind(-layout[, 2L], layout[, 1L]) * 3 + 50
        moved <- graph_cluster(graph, method = method, backend = "cpu",
            projection = controls, seed = 3L)
        expect_equal(moved$reweighted_graph$weight,
            result$reweighted_graph$weight, tolerance = 1e-12)
        expect_identical(moved$membership, result$membership)
    }
})

test_that("native small-group assignment preserves unresolved rows", {
    graph <- list(from = c(1L, 1L, 2L), to = c(2L, 3L, 3L),
        weight = rep(1, 3), n_vertices = 5L)
    nn <- list(indices = matrix(c(2L, 1L, 1L, 2L, 4L), ncol = 1L),
        distances = matrix(rep(1, 5), ncol = 1L))
    for (method in c("leiden", "louvain")) {
        result <- graph_cluster(graph, method = method, backend = "cpu",
            small_groups = list(knn = nn, min_size = 2L))
        expect_identical(result$membership, c(1L, 1L, 1L, 1L, 0L))
        expect_identical(result$reassigned, 4L)
        expect_identical(result$unassigned, 5L)
        expect_identical(result$n_communities, 1L)
        expect_true(is.na(result$modularity))
        expect_true(is.na(result$connected_communities))
        expect_true(is.finite(result$modularity_original))
        complete <- nn
        complete$indices[5L, 1L] <- 3L
        assigned <- graph_cluster(graph, method = method, backend = "cpu",
            small_groups = list(knn = complete, min_size = 2L))
        expect_true(all(assigned$membership == 1L))
        expect_true(is.finite(assigned$modularity))
        expect_identical(assigned$connected_communities, FALSE)
    }
})

test_that("host option wrappers preserve CPU graph-option semantics", {
    graph <- three_clique_graph()
    layout <- cbind(c(seq(0, 0.25, length.out = 6L),
        seq(0.3, 0.55, length.out = 6L),
        seq(5, 5.25, length.out = 6L)), rep(0, 18L))
    option <- list(layout = layout, k = 3L, stretch = 2)
    projection <- fastEmbedR:::fastembedr_graph_projection_cpp(
        graph$from, graph$to, graph$weight, graph$n_vertices, option)
    reference <- graph_cluster(graph, backend = "cpu",
        projection = option)$reweighted_graph
    expect_equal(projection$graph, reference)
    small <- list(from = c(1L, 1L, 2L), to = c(2L, 3L, 3L),
        weight = rep(1, 3), n_vertices = 5L)
    nn <- list(indices = matrix(c(2L, 1L, 1L, 2L, 4L), ncol = 1L),
        distances = matrix(rep(1, 5), ncol = 1L))
    raw <- graph_cluster(small, backend = "cpu")
    post <- fastEmbedR:::fastembedr_graph_small_groups_cpp(
        small$from, small$to, small$weight, small$n_vertices,
        raw$membership, 1, c(nn, min_size = 2L))
    expected <- graph_cluster(small, backend = "cpu",
        small_groups = list(knn = nn, min_size = 2L))
    expect_identical(post$membership, expected$membership)
    expect_identical(post$unassigned, expected$unassigned)
    expect_equal(post$modularity, expected$modularity)
})

test_that("graph options reject unsupported routes and invalid controls", {
    graph <- three_clique_graph()
    layout <- cbind(seq_len(18L), rep(0, 18L))
    if (!fastEmbedR:::graph_clustering_metal_available_cpp()) {
        expect_error(graph_cluster(graph, backend = "metal",
            resolution = c(0.1, 0.2)), "Metal.*unavailable")
        expect_error(graph_cluster(graph, backend = "metal",
            seed = c(1L, 2L)), "Metal.*unavailable")
        expect_error(graph_cluster(graph, backend = "metal",
            projection = list(layout = layout)), "Metal.*unavailable")
    }
    if (!fastEmbedR:::graph_clustering_cuda_available_cpp()) {
        expect_error(graph_cluster(graph, backend = "cuda",
            resolution = c(0.1, 0.2)), "CUDA.*unavailable")
        expect_error(graph_cluster(graph, backend = "cuda",
            projection = list(layout = layout)), "CUDA.*unavailable")
    }
    expect_error(graph_cluster(graph, resolution = numeric()),
        "positive finite")
    expect_error(graph_cluster(graph, resolution = c(1, NA_real_)),
        "positive finite")
    expect_error(graph_cluster(graph, resolution = c(1, -1)),
        "positive finite")
    expect_error(graph_cluster(graph, seed = integer()), "finite integers")
    expect_error(graph_cluster(graph, seed = c(1, NA_real_)), "finite integers")
    expect_error(graph_cluster(graph, seed = c(1, 2.5)), "finite integers")
    expect_error(graph_cluster(graph, seed = 1:2, n_runs = 2), "n_runs = 1")
    expect_error(graph_cluster(graph, projection = list()), "requires layout")
    expect_error(graph_cluster(graph, projection = list(layout = layout,
        k = 18L)), "k must be < n")
    expect_error(graph_cluster(graph, projection = list(
        layout = matrix(0, 18, 2), k = 3L)), "radii must be positive")
    expect_error(graph_cluster(graph, resolution = c(0.1, 0.2),
        layout_merge = list(layout = layout)), "cannot be combined")
    expect_error(graph_cluster(graph, consensus = list()),
        "File-backed controls require")
    defaults <- graph_cluster(graph, backend = "cpu")
    expect_null(defaults$parameters$consensus)
    expect_null(defaults$reweighted_graph)
    expect_null(formals(graph_cluster)$projection)
})

test_that("CUDA projection and nearest-core options retain CUDA clustering", {
    if (!fastEmbedR:::graph_clustering_cuda_available_cpp()) {
        skip("CUDA device unavailable")
    }
    graph <- three_clique_graph()
    layout <- cbind(c(seq(0, 0.25, length.out = 6L),
        seq(0.3, 0.55, length.out = 6L),
        seq(5, 5.25, length.out = 6L)), rep(0, 18L))
    projection <- list(layout = layout, k = 3L, stretch = 2)
    nn <- list(indices = matrix(c(2:18, 1L), ncol = 1L),
        distances = matrix(rep(1, 18L), ncol = 1L))
    for (method in c("leiden", "louvain")) {
        result <- graph_cluster(graph, method = method, backend = "cuda",
            projection = projection, small_groups = list(knn = nn,
                min_size = 1L))
        expected <- graph_cluster(graph, method = method, backend = "cpu",
            projection = projection)$reweighted_graph
        expect_equal(result$reweighted_graph$weight, expected$weight,
            tolerance = 1e-12)
        expect_identical(result$backend, "cuda")
        expect_identical(result$repeatability,
            "not_bitwise_deterministic")
        expect_identical(result$parameters$projection$execution,
            "host_native_cpp")
        expect_identical(result$parameters$small_groups$execution,
            "host_native_cpp")
        expect_identical(result$membership, result$membership_original)
        raw <- graph_cluster(result$reweighted_graph, method = method,
            backend = "cuda")
        expect_identical(raw$repeatability,
            "not_bitwise_deterministic")
        expect_identical(result$membership_original, raw$membership)
        combined <- graph_cluster(graph, method = method, backend = "cuda",
            resolution = c(0.001, 1), seed = c(3L, 4L),
            projection = projection, small_groups = list(knn = nn,
                min_size = 1L))
        expect_identical(combined$parameters$processing_order,
            c("consensus", "projection", "clustering", "small_groups"))
        expect_identical(combined$backend, "cuda")
        expect_true(combined$reweighted_graph$n_edges > 0)
    }
    small <- list(from = c(1L, 1L, 2L), to = c(2L, 3L, 3L),
        weight = rep(1, 3), n_vertices = 5L)
    nearest <- list(indices = matrix(c(2L, 1L, 1L, 2L, 4L), ncol = 1L),
        distances = matrix(rep(1, 5), ncol = 1L))
    reassigned <- graph_cluster(small, method = "leiden", backend = "cuda",
        small_groups = list(knn = nearest, min_size = 2L))
    expect_identical(reassigned$membership, c(1L, 1L, 1L, 1L, 0L))
    expect_identical(reassigned$reassigned, 4L)
    expect_identical(reassigned$unassigned, 5L)
})

test_that("CPU graph restarts use requested workers and preserve selection", {
    graph <- three_clique_graph()
    positional <- graph_cluster(graph, "leiden", "cpu", 1, 10L, 1L,
        4L, 17L, NULL, NULL, NULL)
    named <- graph_cluster(graph, method = "leiden", backend = "cpu",
        seed = 17L)
    expect_identical(positional$membership, named$membership)
    for (method in c("louvain", "leiden")) {
        serial <- graph_cluster(graph, method = method, backend = "cpu",
            n_runs = 5L, n.cores = 1L, seed = 17L)
        parallel <- graph_cluster(graph, method = method, backend = "cpu",
            n_runs = 5L, n.cores = 3L, seed = 17L)
        expect_identical(parallel$membership, serial$membership)
        expect_identical(parallel$modularity, serial$modularity)
        expect_identical(parallel$effective_cpu_workers, 3L)
        expect_identical(parallel$parameters$n.cores, 3L)
    }
    expect_error(graph_cluster(graph, n_runs = 2L, n.cores = 3L),
        "n_runs >= n.cores")
    expect_error(graph_cluster(graph, n.cores = 0L), "n.cores")
    expect_error(graph_cluster(graph, n.cores = 1.5), "n.cores")
    expect_error(graph_cluster(graph, backend = "cuda", n_runs = 3L,
        n.cores = 2L), "CPU restart workers")
    expect_error(graph_cluster(graph, resolution = c(0.5, 1),
        n.cores = 2L), "CPU restart workers")
})

test_that("CPU graph threads coordinate one partition in fit mode", {
    graph <- three_clique_graph()
    for (method in c("louvain", "leiden")) {
        for (cores in c(2L, 4L)) {
            fit <- graph_cluster(graph, method = method,
                backend = "cpu", n_runs = 1L, n.cores = cores,
                parallel.mode = "fit", seed = 17L)
            expect_identical(fit$parallel_mode, "fit")
            expect_identical(fit$effective_cpu_workers, cores)
            expect_identical(fit$parameters$parallel.mode, "fit")
            expect_identical(fit$n_communities, 3L)
            expect_gt(fit$modularity, 0.5)
            expect_true(fit$connected_communities)
        }
    }
    expect_error(graph_cluster(graph, backend = "cuda",
        parallel.mode = "fit"), "requires CPU")
    expect_error(graph_cluster(graph, n.cores = 33L,
        parallel.mode = "fit"), "at most 32")
})

test_that("CUDA consensus uses CUDA partitions and exact edge agreement", {
    if (!fastEmbedR:::graph_clustering_cuda_available_cpp()) {
        skip("CUDA device unavailable")
    }
    graph <- three_clique_graph()
    index <- order(graph$from, graph$to)
    for (field in c("from", "to", "weight")) {
        graph[[field]] <- graph[[field]][index]
    }
    resolutions <- c(0.001, 1)
    seeds <- c(3L, 4L)
    for (method in c("leiden", "louvain")) {
        agreement <- numeric(length(graph$weight))
        for (resolution in resolutions) for (seed in seeds) {
            labels <- graph_cluster(graph, method = method, backend = "cuda",
                resolution = resolution, seed = seed)$membership
            agreement <- agreement +
                (labels[graph$from] == labels[graph$to])
        }
        result <- graph_cluster(graph, method = method, backend = "cuda",
            resolution = resolutions, seed = seeds)
        expected <- graph$weight * (agreement / 4)^2
        expect_equal(result$reweighted_graph$weight,
            expected[expected > 0], tolerance = 1e-6)
        expect_identical(result$backend, "cuda")
        expect_identical(result$parameters$consensus$implementation,
            "cuda_partitions_host_sparse_edge_consensus")
        reference <- graph_cluster(result$reweighted_graph,
            method = method, backend = "cuda", resolution = resolutions[1L],
            seed = seeds[1L])
        expect_identical(result$membership, reference$membership)
    }
})

test_that("Metal graph options retain native final partitions", {
    skip_if_not(fastEmbedR:::graph_clustering_metal_available_cpp())
    graph <- three_clique_graph()
    layout <- cbind(rep(c(0, 0.5, 5), each = 6L) +
        rep(seq(0, 0.25, length.out = 6L), 3L), rep(0, 18L))
    neighbor <- c(2:6, 1L, 8:12, 7L, 14:18, 13L)
    knn <- list(indices = matrix(neighbor, ncol = 1L),
        distances = matrix(1, nrow = 18L, ncol = 1L))
    for (method in c("leiden", "louvain")) {
        result <- with_mocked_bindings(
            fastembedr_graph_cluster_cpp = function(...)
                stop("CPU clustering fallback"),
            graph_cluster(graph, method = method, backend = "metal",
                resolution = c(0.001, 1), seed = c(3L, 4L),
                projection = list(layout = layout, k = 3L,
                    stretch = 2),
                small_groups = list(knn = knn, min_size = 2L,
                    k = 1L)),
            .package = "fastEmbedR")
        expect_identical(result$backend, "metal")
        expect_equal(length(result$membership), 18L)
        expect_identical(result$parameters$consensus$partitions, 4L)
        expect_identical(result$parameters$consensus$implementation,
            "metal_partitions_host_sparse_edge_consensus")
        expect_identical(result$parameters$projection$execution,
            "host_native_cpp")
        expect_identical(result$parameters$small_groups$execution,
            "host_native_cpp")
        expect_identical(result$parameters$processing_order,
            c("consensus", "projection", "clustering", "small_groups"))
    }
})

test_that("projection is applied after vector-triggered consensus", {
    graph <- three_clique_graph()
    layout <- cbind(c(seq(0, 0.25, length.out = 6L),
        seq(0.3, 0.55, length.out = 6L),
        seq(5, 5.25, length.out = 6L)), rep(0, 18L))
    projection <- list(layout = layout, k = 3L, stretch = 2)
    for (method in c("leiden", "louvain")) {
        consensus <- graph_cluster(graph, method = method, backend = "cpu",
            resolution = c(0.001, 1), seed = c(3L, 4L))
        sequential <- graph_cluster(consensus$reweighted_graph,
            method = method, backend = "cpu", resolution = 0.001, seed = 3L,
            projection = projection)
        combined <- graph_cluster(graph, method = method, backend = "cpu",
            resolution = c(0.001, 1), seed = c(3L, 4L),
            projection = projection)
        expect_equal(combined$reweighted_graph, sequential$reweighted_graph)
        expect_identical(combined$membership, sequential$membership)
        expect_equal(combined$modularity, sequential$modularity)
        expect_identical(combined$parameters$processing_order,
            c("consensus", "projection", "clustering"))
        expect_equal(combined$parameters$final_resolution, 0.001)
        expect_equal(combined$parameters$final_seed, 3)
    }
})

test_that("nearest-core ties follow supplied neighbor-column order", {
    graph <- list(from = c(1L, 3L), to = c(2L, 4L),
        weight = c(1, 1), n_vertices = 6L)
    nn <- list(indices = cbind(c(2L, 1L, 4L, 3L, 3L, 5L),
        c(3L, 3L, 1L, 1L, 1L, 5L)), distances = matrix(1, 6, 2))
    for (method in c("leiden", "louvain")) {
        first <- graph_cluster(graph, method = method, backend = "cpu",
            small_groups = list(knn = nn, min_size = 2L))
        expect_identical(first$membership[5L], first$membership[3L])
        nn$distances[5L, 1L] <- 2
        second <- graph_cluster(graph, method = method, backend = "cpu",
            small_groups = list(knn = nn, min_size = 2L))
        expect_identical(second$membership[5L], second$membership[1L])
        nn$distances[5L, 1L] <- 1
    }
})

test_that("projection explicitly decodes float32 layouts to double", {
    skip_if_not_installed("float")
    graph <- three_clique_graph()
    layout <- float::fl(cbind(seq_len(18L) / 7, rep(0, 18L)))
    converted <- graph_cluster(graph, backend = "cpu",
        projection = list(layout = layout, k = 3L))
    decoded <- graph_cluster(graph, backend = "cpu",
        projection = list(layout = float::dbl(layout), k = 3L))
    expect_identical(converted$parameters$projection$layout_input_type,
        "float32")
    expect_identical(converted$parameters$projection$distance_precision,
        "double")
    expect_identical(converted$membership, decoded$membership)
    expect_equal(converted$reweighted_graph, decoded$reweighted_graph)
})
