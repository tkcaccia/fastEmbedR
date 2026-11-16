test_that("graph_cluster distinguishes data from prepared graphs", {
    set.seed(71)
    x <- matrix(rnorm(80L), ncol = 2L)
    graph <- knn_graph(x, k = 4L)
    expect_length(graph_cluster(x, method = "louvain",
        k = 4L)$membership, nrow(x))
    expect_length(graph_cluster(graph, method = "louvain")$membership,
        nrow(x))
    tiered <- graph_cluster(x, method = "louvain", k = 4L,
        target_recall = 0.9)
    expect_identical(tiered$target_recall, 0.9)
    expect_identical(tiered$recall_status, "exact_by_construction")
    expect_error(graph_cluster(graph, target_recall = 0.9),
        "Prepared graphs")
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

test_that("knn_graph uses the requested tier only for computed KNN", {
    set.seed(72)
    x <- matrix(rnorm(80L), ncol = 2L)
    withr::local_options(fastEmbedR.target_recall = 0.90)
    option <- knn_graph(x, k = 4L, backend = "cpu")
    explicit <- knn_graph(x, k = 4L, backend = "cpu",
        target_recall = 0.95)
    expect_identical(option$parameters$target_recall, 0.90)
    expect_identical(explicit$parameters$target_recall, 0.95)
    expect_true(option$parameters$knn_exact)
    expect_true(is.na(option$parameters$measured_recall))
    expect_identical(option$parameters$recall_status,
        "exact_by_construction")
    expect_identical(option$from, explicit$from)
    expect_identical(option$to, explicit$to)
    expect_error(knn_graph(x, k = 4L, target_recall = 0.97),
        "target_recall")
    supplied <- list(indices = matrix(c(2L, 3L, 1L, 3L,
        1L, 4L, 3L, 2L), 4L, byrow = TRUE),
        distances = matrix(1:8, 4L, byrow = TRUE),
        target_recall = 0.99)
    graph <- knn_graph(supplied, k = 2L)
    expect_identical(graph$parameters$target_recall, 0.99)
    expect_error(knn_graph(supplied, k = 2L,
        target_recall = 0.90), "Precomputed KNN")
})

test_that("landmark clustering uses one tier for graph and vote KNN", {
    set.seed(74)
    x <- matrix(rnorm(120L * 4L), ncol = 4L)
    fit <- graph_cluster(x, method = "louvain", landmarks = 24L,
        k = 5L, vote_k = 4L, target_recall = 0.9, seed = 4L)
    expect_identical(fit$target_recall, 0.9)
    expect_true(is.na(fit$reference_knn_recall))
    expect_true(is.na(fit$vote_knn_recall))
    expect_length(fit$membership, nrow(x))
    expect_error(graph_cluster(x, method = "louvain", landmarks = 24L,
        target_recall = 0.8), "target_recall")
})

test_that("file-backed landmark clustering preserves one recall tier", {
    set.seed(75)
    x <- matrix(rnorm(96L * 4L), ncol = 4L)
    directory <- tempfile()
    dir.create(directory)
    on.exit(unlink(directory, recursive = TRUE))
    input <- file.path(directory, "input.f32")
    writeBin(as.vector(t(x)), input, size = 4L, endian = "little")
    source <- massive_matrix(input, nrow(x), ncol(x))
    fit <- graph_cluster(source, method = "louvain", massive = "landmark",
        landmarks = 24L, k = 5L, output = file.path(directory, "fit"),
        backend = "cpu", target_recall = 0.9, memory_limit = "256MB")
    expect_identical(fit$target_recall, 0.9)
    expect_identical(fit$knn$target_recall, 0.9)
    expect_identical(fit$reference_graph_parameters$target_recall, 0.9)
    expect_identical(fit$recall_status, "exact_by_construction")
    expect_equal(file.info(fit$membership_path)$size, 4 * nrow(x))
})
