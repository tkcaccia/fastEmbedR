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

test_that("graph options reject unsupported routes and invalid controls", {
    graph <- three_clique_graph()
    layout <- cbind(seq_len(18L), rep(0, 18L))
    for (backend in c("cuda", "metal")) {
        expect_error(graph_cluster(graph, backend = backend,
            resolution = c(0.1, 0.2)), "require CPU")
        expect_error(graph_cluster(graph, backend = backend,
            seed = c(1L, 2L)), "require CPU")
        expect_error(graph_cluster(graph, backend = backend,
            projection = list(layout = layout)), "require CPU")
    }
    expect_error(graph_cluster(graph, method = "walktrap", backend = "cpu",
        seed = c(1L, 2L)), "require CPU")
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
    expect_error(graph_cluster(graph, consensus = list()), "unused argument")
    defaults <- graph_cluster(graph, backend = "cpu")
    expect_null(defaults$parameters$consensus)
    expect_null(defaults$reweighted_graph)
    expect_null(formals(graph_cluster)$projection)
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
