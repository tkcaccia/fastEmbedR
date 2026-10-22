adjusted_rand_index <- function(left, right) {
    contingency <- table(left, right)
    choose2 <- function(x) x * (x - 1) / 2
    n <- sum(contingency)
    same <- sum(choose2(contingency))
    row_pairs <- sum(choose2(rowSums(contingency)))
    col_pairs <- sum(choose2(colSums(contingency)))
    expected <- row_pairs * col_pairs / choose2(n)
    maximum <- (row_pairs + col_pairs) / 2
    if (maximum == expected) {
        return(1)
    }
    (same - expected) / (maximum - expected)
}

three_clique_graph <- function() {
    groups <- split(seq_len(18L), rep(seq_len(3L), each = 6L))
    from <- to <- integer()
    weight <- numeric()
    for (group in groups) {
        pairs <- utils::combn(group, 2L)
        from <- c(from, pairs[1L, ])
        to <- c(to, pairs[2L, ])
        weight <- c(weight, rep(1, ncol(pairs)))
    }
    from <- c(from, 6L, 12L)
    to <- c(to, 7L, 13L)
    weight <- c(weight, 0.15, 0.15)
    structure(
        list(
            from = from, to = to, weight = weight,
            n_vertices = 18L, n_edges = length(from)
        ),
        class = c("fastEmbedR_graph", "list")
    )
}

test_that("knn_graph builds a compact native graph", {
    set.seed(17)
    x <- rbind(
        matrix(rnorm(80, 0, 0.2), ncol = 4),
        matrix(rnorm(80, 3, 0.2), ncol = 4)
    )
    graph <- knn_graph(x, k = 8L, weight = "snn", n.cores = 2L)
    expect_s3_class(graph, "fastEmbedR_graph")
    expect_identical(graph$n_vertices, nrow(x))
    expect_equal(length(graph$from), graph$n_edges)
    expect_true(all(graph$from < graph$to))
    expect_true(all(graph$weight > 0 & graph$weight <= 1))
    expect_identical(graph$parameters$weight, "snn")
    expect_identical(graph$parameters$n.cores, 2L)
})

test_that("native graph methods recover canonical communities", {
    graph <- three_clique_graph()
    truth <- rep(seq_len(3L), each = 6L)
    for (method in c("louvain", "leiden", "walktrap")) {
        result <- graph_cluster(
            graph,
            method = method,
            n_iterations = 20L,
            n_runs = 2L,
            seed = 3L
        )
        expect_s3_class(result, "fastEmbedR_graph_cluster")
        expect_identical(result$backend, "cpu")
        expect_identical(result$n_communities, 3L)
        expect_equal(adjusted_rand_index(result$membership, truth), 1)
        expect_gt(result$modularity, 0.6)
        if (identical(method, "leiden")) {
            expect_true(result$connected_communities)
        }
    }
})

test_that("layout merging needs graph support and local continuity", {
    graph <- three_clique_graph()
    layout <- cbind(c(seq(0, 0.25, length.out = 6L),
        seq(0.3, 0.55, length.out = 6L),
        seq(5, 5.25, length.out = 6L)), rep(0, 18L))
    base <- graph_cluster(graph, method = "leiden", seed = 3L)
    merged <- graph_cluster(graph, method = "leiden", seed = 3L,
        layout_merge = list(layout = layout, min_connectivity = 0.05,
            min_cross_edges = 1L))
    expect_identical(merged$membership_original, base$membership)
    expect_identical(merged$n_communities, 2L)
    expect_identical(length(unique(merged$membership[1:12])), 1L)
    expect_false(merged$membership[1L] == merged$membership[13L])
    expect_true(is.finite(merged$modularity))
    expect_identical(sum(merged$layout_merge_pairs$selected), 1L)
    expect_identical(base$parameters$layout_merge, NULL)
    strict <- graph_cluster(graph, method = "leiden", seed = 3L,
        layout_merge = list(layout = layout, min_cross_edges = 2L))
    expect_identical(strict$n_communities, base$n_communities)
    for (method in c("louvain", "walktrap")) {
        other <- graph_cluster(graph, method = method, seed = 3L,
            layout_merge = list(layout = layout,
                min_connectivity = 0.05, min_cross_edges = 1L))
        expect_identical(other$n_communities, 2L)
        expect_equal(other$modularity_original,
            graph_cluster(graph, method = method, seed = 3L)$modularity)
    }
    if (requireNamespace("float", quietly = TRUE)) {
        float_layout <- float::fl(layout)
        from_float <- graph_cluster(graph, method = "leiden", seed = 3L,
            layout_merge = list(layout = float_layout,
                min_connectivity = 0.05, min_cross_edges = 1L))
        expect_identical(from_float$n_communities, 2L)
    }
    near <- layout
    near[13:18, 1L] <- near[13:18, 1L] - 5
    guarded <- graph_cluster(graph, method = "leiden", seed = 3L,
        layout_merge = list(layout = near, min_cross_edges = 1L))
    expect_false(guarded$membership[1L] == guarded$membership[13L])
    expect_error(graph_cluster(graph, layout_merge = list(
        layout = layout[-1L, , drop = FALSE])), "n-by-2")
})

test_that("accelerator clustering never falls back silently", {
    graph <- three_clique_graph()
    expect_error(
        graph_cluster(graph, method = "walktrap", backend = "metal"),
        "supported only.*cpu"
    )
    expect_error(
        graph_cluster(graph, method = "walktrap", backend = "cuda"),
        "supported only.*cpu"
    )

    if (!fastEmbedR:::graph_clustering_cuda_available_cpp()) {
        expect_error(
            graph_cluster(graph, method = "louvain", backend = "cuda"),
            "CUDA graph clustering is unavailable"
        )
    }
    if (!fastEmbedR:::graph_clustering_metal_available_cpp()) {
        expect_error(
            graph_cluster(graph, method = "leiden", backend = "metal"),
            "Metal graph clustering is unavailable"
        )
    }
})

available_clustering_accelerators <- function() {
    candidates <- c(
        metal = isTRUE(fastEmbedR:::graph_clustering_metal_available_cpp()),
        cuda = isTRUE(fastEmbedR:::graph_clustering_cuda_available_cpp())
    )
    names(candidates)[candidates]
}

test_that("accelerated Louvain and Leiden preserve the CPU graph objective", {
    backends <- available_clustering_accelerators()
    skip_if(!length(backends))
    graph <- three_clique_graph()
    truth <- rep(seq_len(3L), each = 6L)
    for (method in c("louvain", "leiden")) {
        for (backend in backends) {
            accelerated <- graph_cluster(
                graph,
                method = method,
                backend = backend,
                n_iterations = 20L,
                n_runs = 2L,
                seed = 3L
            )
            expect_identical(accelerated$backend_requested, backend)
            expect_identical(accelerated$backend, backend)
            expect_identical(accelerated$n_communities, 3L)
            expect_equal(adjusted_rand_index(accelerated$membership, truth), 1)
            expect_gt(accelerated$modularity, 0.6)
            expect_identical(accelerated$graph_storage, "float32_csr")
            expect_identical(accelerated$coarsening_backend, "native_cpp")
            if (identical(method, "leiden")) {
                expect_true(accelerated$connected_communities)
            }
        }
    }
})

test_that("accelerated Leiden splits disconnected communities", {
    backends <- available_clustering_accelerators()
    skip_if(!length(backends))
    graph <- structure(
        list(
            from = c(1L, 2L, 4L, 5L),
            to = c(2L, 3L, 5L, 6L),
            weight = rep(1, 4L),
            n_vertices = 6L,
            n_edges = 4L
        ),
        class = c("fastEmbedR_graph", "list")
    )
    for (backend in backends) {
        result <- graph_cluster(
            graph,
            method = "leiden",
            backend = backend,
            n_iterations = 20L,
            seed = 19L
        )
        expect_true(result$connected_communities)
        expect_gte(result$n_communities, 2L)
    }
})

test_that("resolution and isolated vertices are handled consistently", {
    graph <- three_clique_graph()
    graph$n_vertices <- 20L
    low <- graph_cluster(graph, "louvain", resolution = 0.5, seed = 8L)
    high <- graph_cluster(graph, "louvain", resolution = 2, seed = 8L)
    expect_gte(high$n_communities, low$n_communities)
    expect_length(low$membership, 20L)
    expect_false(anyNA(low$membership))

    for (backend in available_clustering_accelerators()) {
        accelerated <- graph_cluster(
            graph, "louvain",
            backend = backend, resolution = 1, seed = 8L
        )
        expect_length(accelerated$membership, 20L)
        expect_false(anyNA(accelerated$membership))
        expect_true(accelerated$connected_communities)
    }
})

test_that(paste(
    "accelerators handle vertices with more than 64 candidate",
    "communities"
), {
    backends <- available_clustering_accelerators()
    skip_if(!length(backends))

    n_leaves <- 96L
    graph <- structure(
        list(
            from = rep.int(1L, n_leaves),
            to = seq.int(2L, n_leaves + 1L),
            weight = seq(0.25, 1.25, length.out = n_leaves),
            n_vertices = n_leaves + 1L,
            n_edges = n_leaves
        ),
        class = c("fastEmbedR_graph", "list")
    )

    for (method in c("louvain", "leiden")) {
        cpu <- graph_cluster(
            graph,
            method = method,
            backend = "cpu",
            n_iterations = 20L,
            n_runs = 2L,
            seed = 31L
        )
        for (backend in backends) {
            accelerated <- graph_cluster(
                graph,
                method = method,
                backend = backend,
                n_iterations = 20L,
                n_runs = 2L,
                seed = 31L
            )
            expect_false(anyNA(accelerated$membership))
            expect_true(is.finite(accelerated$modularity))
            expect_gte(accelerated$modularity, cpu$modularity - 1e-5)
            expect_true(accelerated$connected_communities)
        }
    }
})

test_that("Walktrap agrees with the igraph oracle", {
    skip_if_not_installed("igraph")
    graph <- three_clique_graph()
    igraph_graph <- igraph::graph_from_data_frame(
        data.frame(from = graph$from, to = graph$to, weight = graph$weight),
        directed = FALSE,
        vertices = seq_len(graph$n_vertices)
    )
    reference <- igraph::cluster_walktrap(
        igraph_graph,
        weights = igraph::E(igraph_graph)$weight,
        steps = 4L
    )
    native <- graph_cluster(graph, method = "walktrap", steps = 4L)
    expect_equal(
        adjusted_rand_index(native$membership, igraph::membership(reference)),
        1
    )
    expect_equal(
        native$modularity, igraph::modularity(reference),
        tolerance = 1e-10
    )
})

test_that("Walktrap preserves sparse-walk partitions and modularity", {
    skip_if_not_installed("igraph")
    n <- 600L
    group <- (seq_len(n) - 1L) %/% 100L
    from <- rep(seq_len(n), each = 8L)
    offset <- rep(seq_len(8L), times = n)
    to <- group[from] * 100L + ((from - 1L + offset) %% 100L) + 1L
    graph <- structure(list(
        from = from, to = to, weight = rep(1, length(from)),
        n_vertices = n, n_edges = length(from)
    ), class = c("fastEmbedR_graph", "list"))
    reference_graph <- igraph::graph_from_edgelist(
        cbind(from, to), directed = FALSE
    )
    reference <- igraph::cluster_walktrap(
        reference_graph, weights = graph$weight, steps = 4L
    )
    native <- graph_cluster(graph, method = "walktrap", steps = 4L)
    expect_equal(
        adjusted_rand_index(
            native$membership, igraph::membership(reference)
        ), 1
    )
    expect_equal(
        native$modularity,
        igraph::modularity(
            reference_graph, native$membership,
            weights = graph$weight
        ),
        tolerance = 1e-8
    )
})

test_that("Louvain and Leiden remain competitive with igraph", {
    skip_if_not_installed("igraph")
    set.seed(103)
    sizes <- c(45L, 55L, 65L, 75L)
    preference <- matrix(0.006, nrow = 4L, ncol = 4L)
    diag(preference) <- 0.07
    igraph_graph <- igraph::sample_sbm(
        sum(sizes),
        pref.matrix = preference,
        block.sizes = sizes,
        directed = FALSE,
        loops = FALSE
    )
    igraph::E(igraph_graph)$weight <- runif(
        igraph::ecount(igraph_graph), 0.5, 1.5
    )
    edges <- igraph::as_data_frame(igraph_graph, what = "edges")
    graph <- structure(
        list(
            from = as.integer(edges$from),
            to = as.integer(edges$to),
            weight = edges$weight,
            n_vertices = igraph::vcount(igraph_graph),
            n_edges = igraph::ecount(igraph_graph)
        ),
        class = c("fastEmbedR_graph", "list")
    )

    native_louvain <- graph_cluster(
        graph, "louvain",
        n_iterations = 20L, n_runs = 3L, seed = 7L
    )
    set.seed(7)
    reference_louvain <- igraph::cluster_louvain(
        igraph_graph,
        weights = igraph::E(igraph_graph)$weight,
        resolution = 1
    )
    expect_gte(
        native_louvain$modularity,
        igraph::modularity(reference_louvain) - 0.03
    )

    native_leiden <- graph_cluster(
        graph, "leiden",
        n_iterations = 10L, n_runs = 3L, seed = 7L
    )
    set.seed(7)
    reference_leiden <- igraph::cluster_leiden(
        igraph_graph,
        objective_function = "modularity",
        weights = igraph::E(igraph_graph)$weight,
        resolution = 1,
        n_iterations = 10L
    )
    reference_modularity <- igraph::modularity(
        igraph_graph,
        igraph::membership(reference_leiden),
        weights = igraph::E(igraph_graph)$weight,
        resolution = 1
    )
    expect_true(native_leiden$connected_communities)
    expect_gte(native_leiden$modularity, reference_modularity - 0.03)

    for (backend in available_clustering_accelerators()) {
        accelerated_louvain <- graph_cluster(
            graph,
            "louvain",
            backend = backend,
            n_iterations = 20L,
            n_runs = 3L,
            seed = 7L
        )
        accelerated_leiden <- graph_cluster(
            graph,
            "leiden",
            backend = backend,
            n_iterations = 10L,
            n_runs = 3L,
            seed = 7L
        )
        expect_gte(
            accelerated_louvain$modularity,
            igraph::modularity(reference_louvain) - 0.03
        )
        expect_gte(accelerated_leiden$modularity, reference_modularity - 0.03)
        expect_true(accelerated_leiden$connected_communities)
    }
})

test_that("Walktrap fails before unsafe dense allocation", {
    graph <- structure(
        list(
            from = 1L,
            to = 2L,
            weight = 1,
            n_vertices = 4001L,
            n_edges = 1L
        ),
        class = c("fastEmbedR_graph", "list")
    )
    expect_error(
        graph_cluster(graph, method = "walktrap"),
        "16-million-entry"
    )
})
