test_that("massive modularity audits within-community connectivity", {
    skip_on_os("windows")
    prefix <- tempfile()
    paths <- paste0(prefix, c(".offsets.u64", ".indices.u32",
        ".weights.f32"))
    labels <- paste0(prefix, ".clusters.u32")
    on.exit(unlink(c(paths, labels)))
    offsets <- 0:4
    writeBin(as.integer(rbind(offsets, rep(0L, 5L))), paths[[1L]],
        size = 4L, endian = "little")
    writeBin(c(2L, 1L, 4L, 3L), paths[[2L]], size = 4L,
        endian = "little")
    writeBin(rep(1, 4L), paths[[3L]], size = 4L,
        endian = "little")
    graph <- structure(list(n_vertices = 4L, n_edges = 4L,
        max_degree = 1L, offsets_path = paths[[1L]],
        indices_path = paths[[2L]], weights_path = paths[[3L]],
        storage = "csr", weight_method = "umap_fuzzy_union",
        symmetrized = TRUE,
        file_identity = fastEmbedR:::
            massive_checkpoint_file_identity(paths)),
        class = "fastEmbedR_massive_graph")
    writeBin(rep(1L, 4L), labels, size = 4L, endian = "little")
    split <- massive_graph_modularity(graph, labels, 1L,
        memory_limit = "64MB", connectivity = TRUE)
    expect_true(split$connectivity_audited)
    expect_identical(split$disconnected_communities, 1L)
    expect_equal(split$n_components, 2)
    writeBin(c(1L, 1L, 2L, 2L), labels, size = 4L,
        endian = "little")
    joined <- massive_graph_modularity(graph, labels, 2L,
        memory_limit = "64MB", connectivity = TRUE)
    expect_identical(joined$disconnected_communities, 0L)
    expect_equal(joined$n_components, 2)
    expect_error(massive_graph_modularity(graph, labels, 2L,
        memory_limit = "32MB", connectivity = TRUE),
        "buffers exceed")
})

test_that("isolated rows require singleton Leiden communities", {
    skip_on_os("windows")
    prefix <- tempfile()
    paths <- paste0(prefix, c(".offsets.u64", ".indices.u32",
        ".weights.f32"))
    labels <- paste0(prefix, ".clusters.u32")
    on.exit(unlink(c(paths, labels)))
    offsets <- c(0L, 1L, 2L, 2L, 3L, 4L)
    writeBin(as.integer(rbind(offsets, rep(0L, 6L))), paths[[1L]],
        size = 4L, endian = "little")
    writeBin(c(2L, 1L, 5L, 4L), paths[[2L]], size = 4L,
        endian = "little")
    writeBin(rep(1, 4L), paths[[3L]], size = 4L,
        endian = "little")
    graph <- structure(list(n_vertices = 5L, n_edges = 4L,
        max_degree = 1L, offsets_path = paths[[1L]],
        indices_path = paths[[2L]], weights_path = paths[[3L]],
        storage = "csr", weight_method = "umap_fuzzy_union",
        symmetrized = TRUE,
        file_identity = fastEmbedR:::
            massive_checkpoint_file_identity(paths)),
        class = "fastEmbedR_massive_graph")
    writeBin(c(1L, 1L, 3L, 2L, 2L), labels, size = 4L,
        endian = "little")
    score <- massive_graph_modularity(graph, labels, 3L,
        memory_limit = "64MB", connectivity = TRUE)
    expect_identical(score$disconnected_communities, 0L)
    expect_equal(score$n_components, 3)
    writeBin(c(1L, 1L, 1L, 2L, 2L), labels, size = 4L,
        endian = "little")
    joined <- massive_graph_modularity(graph, labels, 2L,
        memory_limit = "64MB", connectivity = TRUE)
    expect_identical(joined$disconnected_communities, 1L)
})
