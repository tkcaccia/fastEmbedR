test_that("knn_graph builds an igraph graph from a KNN object", {
  skip_if_not_installed("igraph")
  set.seed(501)
  x <- rbind(
    matrix(rnorm(60, -2, 0.2), ncol = 4),
    matrix(rnorm(60, 2, 0.2), ncol = 4)
  )
  knn <- nn(x, k = 6L, backend = "cpu")

  g <- knn_graph(knn, k = 5L, weight = "snn")

  expect_s3_class(g, "igraph")
  expect_equal(igraph::vcount(g), nrow(x))
  expect_gt(igraph::ecount(g), 0L)
  expect_true("weight" %in% igraph::edge_attr_names(g))
  expect_equal(attr(g, "fastEmbedR_graph")$weight, "snn")
})

test_that("knn_graph accepts embedding objects as layout graphs", {
  skip_if_not_installed("igraph")
  set.seed(503)
  x <- rbind(
    matrix(rnorm(80, -2, 0.2), ncol = 4),
    matrix(rnorm(80, 2, 0.2), ncol = 4)
  )
  layout <- x[, 1:2, drop = FALSE]
  knn <- nn(x, k = 6L, backend = "cpu")
  fit <- list(
    layout = layout,
    method = "opentsne"
  )
  class(fit) <- "fastEmbedR_embedding"

  g_embedding <- knn_graph(fit, k = 5L, backend = "cpu")
  expect_s3_class(g_embedding, "igraph")
  expect_equal(attr(g_embedding, "fastEmbedR_graph")$space, "embedding")
  expect_equal(attr(g_embedding, "fastEmbedR_graph")$weight, "distance")
  expect_equal(attr(g_embedding, "fastEmbedR_graph")$input_method, "opentsne")
})

test_that("knn_graph supports adaptive weights and mutual edges", {
  skip_if_not_installed("igraph")
  set.seed(504)
  x <- rbind(
    matrix(rnorm(80, -2, 0.2), ncol = 2),
    matrix(rnorm(80, 2, 0.2), ncol = 2)
  )
  g_union <- knn_graph(x, k = 8L, backend = "cpu", weight = "adaptive", mutual = FALSE)
  g_mutual <- knn_graph(x, k = 8L, backend = "cpu", weight = "adaptive", mutual = TRUE)

  expect_s3_class(g_union, "igraph")
  expect_s3_class(g_mutual, "igraph")
  expect_equal(attr(g_union, "fastEmbedR_graph")$weight, "adaptive")
  expect_true(attr(g_mutual, "fastEmbedR_graph")$mutual)
  expect_lte(igraph::ecount(g_mutual), igraph::ecount(g_union))
  expect_true(all(igraph::E(g_mutual)$weight > 0))
})

test_that("knn_graph output works with standard igraph clustering", {
  skip_if_not_installed("igraph")
  set.seed(502)
  x <- rbind(
    matrix(rnorm(80, -2, 0.2), ncol = 4),
    matrix(rnorm(80, 2, 0.2), ncol = 4)
  )
  g <- knn_graph(x, k = 8L, backend = "cpu", weight = "snn")
  weights <- igraph::E(g)$weight

  louvain <- igraph::cluster_louvain(g, weights = weights)
  expect_length(igraph::membership(louvain), nrow(x))
  expect_gte(length(unique(igraph::membership(louvain))), 2L)

  if ("cluster_leiden" %in% getNamespaceExports("igraph")) {
    leiden <- igraph::cluster_leiden(
      g,
      objective_function = "modularity",
      weights = weights,
      resolution = 1,
      n_iterations = 2
    )
    expect_length(igraph::membership(leiden), nrow(x))
  }
})
