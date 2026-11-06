test_that("automatic clustering uses CPU fit workers in one run", {
    set.seed(41)
    x <- rbind(matrix(rnorm(120, -3, 0.2), ncol = 4L),
        matrix(rnorm(120, 3, 0.2), ncol = 4L))
    fit <- massive_cluster(x, method = "leiden", massive = "auto",
        k = 8L, backend = "cpu", n.cores = 2L,
        n_runs = 1L, parallel.mode = "fit",
        memory_limit = "2GB")
    expect_identical(fit$mode, "in_memory")
    expect_identical(fit$parameters$parallel.mode, "fit")
    expect_identical(fit$effective_cpu_workers, 2L)
    expect_length(fit$membership, nrow(x))
    expect_true(all(fit$membership >= 1L))
})

test_that("CPU fit mode rejects incompatible massive routes", {
    x <- matrix(seq_len(80), ncol = 4L)
    expect_error(massive_cluster(x, massive = "auto",
        method = "walktrap", parallel.mode = "fit"),
        "requires CPU Louvain or Leiden")
    expect_error(massive_cluster(x, massive = "auto",
        backend = "cuda", parallel.mode = "fit"),
        "requires CPU Louvain or Leiden")
})

test_that("CPU fit memory planning shares the graph across workers", {
    fit <- fastEmbedR:::massive_cluster_auto_bytes(
        100000L, 32L, 30L, 16L, "leiden", "cpu", 16L,
        parallel.mode = "fit")
    restarts <- fastEmbedR:::massive_cluster_auto_bytes(
        100000L, 32L, 30L, 16L, "leiden", "cpu", 16L,
        parallel.mode = "restarts")
    expect_lt(fit, restarts)
    knn <- list(nrow = 100000L, n_reference = 1000L, ncol = 30L)
    graph <- list(n_vertices = 1000L, weight = rep(1, 3000L))
    shared <- fastEmbedR:::massive_cluster_resources(knn, graph,
        16L, NULL, "2GB", "cpu", "leiden", 16L,
        parallel.mode = "fit")
    repeated <- fastEmbedR:::massive_cluster_resources(knn, graph,
        16L, NULL, "2GB", "cpu", "leiden", 16L,
        parallel.mode = "restarts")
    expect_lt(shared$reference_graph_estimate_bytes,
        repeated$reference_graph_estimate_bytes)
    expect_identical(shared$fitting_cpu_workers_budgeted, 16L)
})
