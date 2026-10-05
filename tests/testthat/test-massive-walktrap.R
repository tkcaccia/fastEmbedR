make_massive_walktrap_graph <- function(directory, neighbors) {
    n <- nrow(neighbors)
    prefix <- file.path(directory, "knn")
    writeBin(as.integer(t(neighbors)), paste0(prefix,
        ".indices.u32"), size = 4L, endian = "little")
    writeBin(rep(0.1, length(neighbors)), paste0(prefix,
        ".distances.f32"), size = 4L, endian = "little")
    knn <- massive_knn_graph(prefix, n, ncol(neighbors))
    directed <- massive_umap_memberships(knn)
    massive_umap_fuzzy_graph(directed,
        file.path(directory, "fuzzy"), memory_limit = "64MB")
}

test_that("partitioned Walktrap preserves disconnected communities", {
    skip_on_os("windows")
    directory <- tempfile()
    dir.create(directory)
    on.exit(unlink(directory, recursive = TRUE))
    neighbors <- t(vapply(seq_len(8L), function(row) {
        setdiff(if (row <= 4L) 1:4 else 5:8, row)
    }, integer(3L)))
    graph <- make_massive_walktrap_graph(directory, neighbors)
    output <- file.path(directory, "walktrap.clusters.u32")
    fit <- massive_cluster(graph, method = "walktrap",
        massive = "out_of_core_graph", output = output,
        backend = "cpu", steps = 4L,
        chunk_rows = 4L, memory_limit = "128MB")
    edges <- massive_read_graph_edges(graph, 1L, 8L)
    exact <- graph_cluster(list(from = edges$from,
        to = edges$to, weight = edges$weight,
        n_vertices = 8L), method = "walktrap")
    labels <- massive_read_walktrap_rows(fit, 1L, 8L)
    expect_identical(outer(labels, labels, "=="),
        outer(exact$membership, exact$membership, "=="))
    expect_identical(head(fit, 4L), labels[1:4])
    expect_identical(fit$backend, "cpu")
    expect_true(fit$approximate)
    expect_equal(fit$modularity_final,
        massive_graph_modularity(graph, output,
            fit$n_communities, memory_limit = "128MB")$modularity)
    expect_error(massive_walktrap(graph,
        file.path(directory, "cuda.u32"), backend = "cuda"),
        "requires CPU")
})

test_that("partitioned Walktrap rejects non-contracting graph", {
    skip_on_os("windows")
    directory <- tempfile()
    dir.create(directory)
    on.exit(unlink(directory, recursive = TRUE))
    neighbors <- matrix(c(3L, 4L, 1L, 2L), ncol = 1L)
    graph <- make_massive_walktrap_graph(directory, neighbors)
    output <- file.path(directory, "walktrap.clusters.u32")
    expect_error(massive_walktrap(graph, output,
        chunk_rows = 2L, memory_limit = "128MB"),
        "no contraction progress")
    expect_false(file.exists(output))
    expect_true(file.exists(paste0(output, ".part")))
    expect_error(massive_cluster(graph, method = "walktrap",
        massive = "out_of_core_graph", output = output,
        checkpoint = TRUE), "no initialization or checkpoint")
})

test_that("partitioned Walktrap labels a connected fuzzy graph", {
    skip_on_os("windows")
    directory <- tempfile()
    dir.create(directory)
    on.exit(unlink(directory, recursive = TRUE))
    set.seed(141)
    truth <- rep(seq_len(3L), each = 30L)
    x <- matrix(rnorm(90L * 8L, sd = 0.7), ncol = 8L)
    x[, 1L] <- x[, 1L] + (truth - 1L) * 3
    x <- x[sample.int(90L), , drop = FALSE]
    knn <- massive_full_knn_graph(massive_matrix(x), k = 8L,
        output = file.path(directory, "knn"),
        chunk_rows = 45L, memory_limit = "256MB")
    graph <- massive_umap_fuzzy_graph(
        massive_umap_memberships(knn),
        file.path(directory, "fuzzy"), memory_limit = "256MB")
    edges <- massive_read_graph_edges(graph, 1L, 90L)
    expect_true(any(ceiling(edges$from / 30) !=
        ceiling(edges$to / 30)))
    keep <- edges$from < edges$to
    exact <- graph_cluster(list(from = edges$from[keep],
        to = edges$to[keep], weight = edges$weight[keep],
        n_vertices = 90L), method = "walktrap")
    one <- massive_walktrap(graph, file.path(directory, "one.u32"),
        chunk_rows = 90L, memory_limit = "256MB")
    expect_false(one$approximate)
    expect_equal(one$initial_retained_weight_fraction, 1)
    expect_identical(outer(massive_read_walktrap_rows(one, 1L, 90L),
        massive_read_walktrap_rows(one, 1L, 90L), "=="),
        outer(exact$membership, exact$membership, "=="))
    part <- massive_walktrap(graph, file.path(directory, "part.u32"),
        chunk_rows = 30L, memory_limit = "256MB")
    expect_true(part$approximate)
    expect_gt(part$initial_retained_weight_fraction, 0)
    expect_lt(part$initial_retained_weight_fraction, 1)
    expect_true(is.finite(part$modularity_final))
    expect_equal(part$graph_identity, graph$file_identity)
    expect_equal(part$block_rows, 30L)
    expect_error(massive_walktrap(graph,
        file.path(directory, "small.u32"), memory_limit = "64MB"),
        "memory budget")
})
