test_that("resident t-SNE affinity memory is checked before fitting", {
    check <- fastEmbedR:::tsne_check_resident_affinity_capacity
    plan <- check(1000, 30, "1GB", 2 * 1024^3)
    expect_equal(plan$max_directed_edges, 60000)
    fifty <- check(50000000, 30, "256GB", 256 * 1024^3)
    expect_gt(fifty$max_directed_edges, .Machine$integer.max)
    expect_lt(fifty$peak_ram_estimate_bytes,
        0.7 * 256 * 1024^3)
    expect_error(check(100000000, 30, 1e12, 256 * 1024^3),
        "exceed the RAM budget")
    expect_error(check(1000, 30, "1GB", NA_real_),
        "exceed the RAM budget")
    expect_error(check(1000, 30, "1MB", 2 * 1024^3),
        "exceed the RAM budget")
})

test_that("compact CPU affinity admission accounts for KNN precision", {
    check <- fastEmbedR:::tsne_check_resident_affinity_capacity
    available <- 26 * 1024^3
    single <- check(10000000L, 30L, "26GB", available, "cpu", 4L)
    double <- check(10000000L, 30L, "26GB", available, "cpu", 8L)
    expect_lt(single$peak_ram_estimate_bytes, 0.7 * available)
    expect_equal(double$peak_ram_estimate_bytes -
        single$peak_ram_estimate_bytes, 2400000000)
    expect_error(check(10000000L, 30L, "16GB", available, "cpu", 4L),
        "exceed the RAM budget")
})

test_that("single CUDA affinity admission budgets host KNN copies", {
    check <- fastEmbedR:::tsne_check_resident_affinity_capacity
    available <- 26 * 1024^3
    single <- check(10000000L, 30L, "26GB", available, "cuda", 4L)
    double <- check(10000000L, 30L, "26GB", available, "cuda", 8L, 0L)
    expect_equal(single$peak_ram_estimate_bytes, 6348435456)
    expect_equal(double$peak_ram_estimate_bytes -
        single$peak_ram_estimate_bytes, 2400000000)
    expect_error(check(10000000L, 30L, "8GB", available, "cuda", 4L),
        "exceed the RAM budget")
    expect_error(check(10000000L, 30L, "26GB", available, "cuda", 4L,
        0:1), "exceed the RAM budget")
})
