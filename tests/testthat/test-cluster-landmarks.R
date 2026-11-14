test_that("resident landmark communities label every observation", {
    set.seed(121L)
    x <- rbind(matrix(rnorm(400L, 0, 0.2), ncol = 5L),
        matrix(rnorm(400L, 4, 0.2), ncol = 5L))
    ids <- seq.int(1L, nrow(x), by = 2L)
    for (method in c("leiden", "louvain")) {
        fit <- cluster_landmarks(x, ids, method = method,
            k = 10L, vote_k = 5L, backend = "cpu",
            n.cores = 2L, parallel.mode = "fit", seed = 121L)
        expect_s3_class(fit, "fastEmbedR_landmark_clusters")
        expect_identical(length(fit$membership), nrow(x))
        expect_true(all(fit$membership > 0L))
        expect_true(all(fit$confidence >= 0 & fit$confidence <= 1))
        expect_identical(fit$membership[ids],
            fit$reference_fit$membership)
        expect_true(all(fit$confidence[ids] == 1))
        expect_identical(fit$n_reference, length(ids))
        expect_identical(fit$n_query, nrow(x) - length(ids))
        expect_identical(fit$backend, "cpu")
        expect_identical(fit$graph_assembly_backend, "cpu")
        expect_identical(fit$vote_backend, "cpu")
        expect_true(all(is.finite(fit$stage_seconds)))
        purity <- table(rep(1:2, each = 80L), fit$membership)
        expect_true(all(colSums(purity > 0L) == 1L))
    }
})

test_that("landmark clustering rejects invalid reference and device routes", {
    x <- matrix(seq_len(80L), ncol = 4L)
    expect_error(cluster_landmarks(x, seq_len(nrow(x)), k = 5L),
        "fewer reference")
    expect_error(cluster_landmarks(x, 1:4, k = 5L),
        "nrow\\(reference\\) - 1")
    if (!fastEmbedR:::graph_clustering_cuda_available_cpp()) {
        expect_error(cluster_landmarks(x, 1:10, k = 5L,
            backend = "cuda"), "GPU backend|CUDA")
    }
    if (!fastEmbedR:::graph_clustering_metal_available_cpp()) {
        expect_error(cluster_landmarks(x, 1:10, k = 5L,
            backend = "metal"), "Metal")
    }
})
