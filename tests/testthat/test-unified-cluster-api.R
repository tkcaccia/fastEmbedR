test_that("graph_cluster accepts resident and file-backed matrices", {
    set.seed(905)
    x <- rbind(matrix(rnorm(48L, 0, 0.1), ncol = 2L),
        matrix(rnorm(48L, 4, 0.1), ncol = 2L),
        matrix(rnorm(48L, 8, 0.1), ncol = 2L))
    graph <- knn_graph(x, k = 4L)
    resident <- graph_cluster(graph, method = "louvain", seed = 19L)
    expect_length(resident$membership, nrow(x))
    direct <- graph_cluster(x, method = "louvain", k = 4L,
        memory_limit = "1GB", seed = 19L)
    expect_length(direct$membership, nrow(x))
    landmark <- graph_cluster(x, method = "louvain",
        landmarks = 24L, k = 4L, vote_k = 4L, seed = 19L)
    reference <- cluster_landmarks(x, landmarks = 24L,
        method = "louvain", k = 4L, vote_k = 4L, seed = 19L)
    expect_s3_class(landmark, "fastEmbedR_landmark_clusters")
    expect_identical(landmark$membership, reference$membership)
    default <- graph_cluster(x, landmarks = 24L,
        k = 4L, vote_k = 4L)
    expect_identical(default$method, "leiden")
    expect_length(default$membership, nrow(x))
    expect_error(graph_cluster(x, landmarks = 24L,
        method = "walktrap"), "supports Leiden or Louvain")
    expect_error(graph_cluster(x, method = "louvain",
        landmarks = 24L, massive = "landmark"),
        "Resident landmark clustering")
    expect_error(graph_cluster(graph, massive = "landmark"),
        "File-backed controls")
    expect_error(graph_cluster(graph, landmarks = 24L),
        "data matrix, not a prepared graph")
    expect_error(graph_cluster(graph, vote_k = 4L),
        "data matrix, not a prepared graph")
    prepared <- structure(list(n_vertices = 8L),
        class = "fastEmbedR_massive_graph")
    expect_error(graph_cluster(prepared, landmarks = 4L),
        "data matrix, not a prepared graph")
    expect_error(graph_cluster(x, vote_k = 4L),
        "requires `landmarks`")
    work <- tempfile("unified-cluster-")
    dir.create(work)
    on.exit(unlink(work, recursive = TRUE))
    path <- file.path(work, "input.f32")
    writeBin(as.numeric(t(x)), path, size = 4L,
        endian = "little")
    input <- massive_matrix(path, nrow(x), ncol(x))
    expect_error(graph_cluster(input, vote_k = 4L),
        "requires `landmarks`")
    fit <- graph_cluster(input, method = "louvain",
        landmarks = 24L, k = 4L,
        output = file.path(work, "fit"),
        memory_limit = "512MB", seed = 19L)
    labels <- massive_read_rows(fit, 1L, nrow(x))
    expect_length(labels$membership, nrow(x))
})

test_that("graph_cluster accepts a file-backed full graph", {
    skip_if(.Platform$OS.type != "unix")
    work <- tempfile("unified-full-")
    dir.create(work)
    on.exit(unlink(work, recursive = TRUE))
    neighbors <- t(vapply(seq_len(8L), function(row) {
        setdiff(if (row <= 4L) 1:4 else 5:8, row)
    }, integer(3L)))
    distances <- matrix(rep(c(0.1, 0.3, 0.7), 8L),
        nrow = 8L, byrow = TRUE)
    prefix <- file.path(work, "knn")
    writeBin(as.integer(t(neighbors)),
        paste0(prefix, ".indices.u32"), size = 4L,
        endian = "little")
    writeBin(as.numeric(t(distances)),
        paste0(prefix, ".distances.f32"), size = 4L,
        endian = "little")
    knn <- massive_knn_graph(prefix, 8L, 3L)
    directed <- massive_umap_memberships(knn)
    graph <- massive_umap_fuzzy_graph(directed,
        file.path(work, "fuzzy"), memory_limit = "64MB")
    fit <- graph_cluster(graph, method = "louvain",
        output = file.path(work, "full.u32"),
        chunk_rows = 2L, memory_limit = "128MB", seed = 4L)
    expect_length(massive_read_rows(fit, 1L, 8L), 8L)
})
