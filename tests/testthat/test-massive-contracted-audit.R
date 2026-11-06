test_that("contracted float64 graph passes the modularity audit", {
    skip_on_os("windows")
    edges <- tempfile(fileext = ".coarse.edges.bin")
    prefix <- tempfile()
    labels <- tempfile(fileext = ".clusters.u32")
    on.exit(unlink(c(edges, labels, Sys.glob(paste0(prefix, "*"))),
        recursive = TRUE))
    con <- file(edges, "wb")
    pairs <- list(c(1L, 1L), c(1L, 2L),
        c(3L, 3L), c(3L, 4L))
    weights <- c(4, 1, 4, 1)
    for (i in seq_along(pairs)) {
        writeBin(pairs[[i]], con, size = 4L, endian = "little")
        writeBin(weights[[i]], con, size = 8L,
            endian = "little")
    }
    close(con)
    contracted <- list(path = edges, n_vertices = 4L,
        n_edges = 4L, total_edge_weight = 10)
    graph <- fastEmbedR:::massive_louvain_coarse_csr(
        contracted, prefix, 128 * 1024^2)
    writeBin(c(1L, 1L, 2L, 2L), labels,
        size = 4L, endian = "little")
    score <- massive_graph_modularity(graph, labels,
        n_communities = 2L, chunk_rows = 2L,
        memory_limit = "128MB", connectivity = TRUE)
    expect_equal(score$modularity, 0.5, tolerance = 1e-7)
    expect_equal(score$total_edge_weight, 10)
    expect_equal(score$n_edge_pairs, 4)
    expect_identical(score$disconnected_communities, 0L)
})
