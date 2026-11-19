test_that("CUDA IVF uses one index when its budget fits", {
    source <- list(format = "file", nrow = 1000021L, ncol = 11L)
    plan <- with_mocked_bindings(
        massive_cuda_sharded_resources(source, 30L, NULL, NULL,
            "4GB", method = "ivf_sharded"),
        massive_cuda_shard_info = function()
            list(free_bytes = 12 * 1024^3, device = 0L),
        .package = "fastEmbedR")
    expect_identical(plan$n_shards, 1L)
    expect_identical(plan$reference_chunk_rows, source$nrow)
    expect_lte(plan$peak_vram_bytes, plan$vram_budget_bytes)
    expect_equal(plan$vram_budget_bytes, 0.7 * 4 * 1024^3)
    capped <- with_mocked_bindings(
        massive_cuda_sharded_resources(source, 30L, NULL, 200000L,
            "4GB", method = "ivf_sharded"),
        massive_cuda_shard_info = function()
            list(free_bytes = 12 * 1024^3, device = 0L),
        .package = "fastEmbedR")
    expect_identical(capped$n_shards, 6L)
})
