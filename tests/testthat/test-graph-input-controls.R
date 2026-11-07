test_that("graph_cluster distinguishes data from prepared graphs", {
    set.seed(71)
    x <- matrix(rnorm(80L), ncol = 2L)
    graph <- knn_graph(x, k = 4L)
    expect_length(graph_cluster(x, method = "louvain",
        k = 4L)$membership, nrow(x))
    expect_length(graph_cluster(graph, method = "louvain")$membership,
        nrow(x))
    expect_error(graph_cluster(graph, landmarks = 10L),
        "data matrix, not a prepared graph")
    expect_error(graph_cluster(graph, k = 4L),
        "data matrix, not a prepared graph")
    expect_error(graph_cluster(x, vote_k = 4L),
        "requires `landmarks`")
    prepared <- structure(list(n_vertices = 40L),
        class = "fastEmbedR_massive_graph")
    expect_error(graph_cluster(prepared, landmarks = 10L),
        "data matrix, not a prepared graph")
    input <- structure(list(format = "f32", nrow = 40L, ncol = 2L),
        class = "fastEmbedR_massive_matrix")
    expect_error(graph_cluster(input, vote_k = 4L),
        "requires `landmarks`")
})
