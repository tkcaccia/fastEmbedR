test_that("resident clustering checks CSR width and measured RAM", {
    check <- fastEmbedR:::graph_check_resident_counts
    plan <- check(100, 20, "cpu", "1GB", 2 * 1024^3)
    expect_equal(plan$max_csr_edges, 200)
    expect_error(check(100, 20, "cpu", "1MB", 2 * 1024^3),
        "exceeds the RAM budget")
    expect_error(check(100, 20, "cpu", "1GB", NA_real_),
        "exceeds the RAM budget")
    expect_error(check(1.1e9, 100000000, "cuda", 1e12, 2 * 1024^4),
        "signed-32-bit CSR offsets")
    expect_error(check(1e9, 100000000, "cpu", 1e12, 2 * 1024^3),
        "exceeds the RAM budget")
})

test_that("KNN graph admission happens before neighbor search", {
    check <- fastEmbedR:::graph_check_knn_capacity
    x <- matrix(0, 10L, 3L)
    plan <- check(x, 3L, "cpu", "1GB", 2 * 1024^3)
    expect_equal(plan$max_csr_edges, 60)
    expect_error(check(x, 3L, "cpu", "1MB", 2 * 1024^3),
        "exceeds the RAM budget")
    withr::local_options(memory_limit = "1MB")
    expect_error(knn_graph(x, k = 3L, backend = "cpu"),
        "exceeds the RAM budget")
})

test_that("resident CUDA graph checks free VRAM", {
    check <- fastEmbedR:::graph_check_resident_vram
    graph <- list(from = seq_len(100L), n_vertices = 20L)
    expect_no_error(check(graph, free = 2 * 1024^3))
    expect_error(check(graph, free = 16 * 1024^2),
        "exceeds the free-VRAM budget")
    expect_error(check(graph, free = NA_real_),
        "exceeds the free-VRAM budget")
})
