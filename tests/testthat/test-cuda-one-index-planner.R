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

test_that("CUDA shard uses requested k for IVF policy", {
    observed <- with_mocked_bindings(
        massive_knn_reference(list(format = "file"), 31L,
            "ivf", 1L, "cuda", policy_k = 30L),
        massive_knn_reference_input = function(...)
            list(data = matrix(0, 40L, 3L), storage = "R_float32"),
        massive_faissr_cuda_reference = function(data, k, method,
                started, storage, target_recall, policy_k)
            c(build_k = k, policy_k = policy_k),
        .package = "fastEmbedR")
    expect_equal(observed, c(build_k = 31L, policy_k = 30L))
})
