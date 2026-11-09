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
