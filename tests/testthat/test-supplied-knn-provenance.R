test_that("supplied neighbor provenance survives matrix and KNN input", {
    set.seed(20261116)
    x <- matrix(rnorm(96L), 24L, 4L)
    nn <- precompute_knn(x, k = 6L, backend = "cpu",
        n.cores = 1L, target_recall = 0.95)
    expect_equal(nn$target_recall, 0.95)
    selection <- fastEmbedR:::fastembedr_knn_selection(
        "cpu", 24L, 4L, 6L, "euclidean", 0.95)
    expect_identical(nn$knn_selection, selection)
    fits <- list(
        umap(x, nn = nn, n_neighbors = 6L, backend = "cpu", seed = 1L),
        umap(nn, backend = "cpu", seed = 1L),
        tsne(x, nn = nn, perplexity = 2, affinity_k = 6L,
            backend = "cpu", seed = 1L,
            early_exaggeration_iter = 2L, n_iter = 3L),
        tsne(nn, perplexity = 2, affinity_k = 6L,
            Y_init = matrix(0, 24L, 2L), backend = "cpu", seed = 1L,
            early_exaggeration_iter = 2L, n_iter = 3L)
    )
    for (fit in fits) {
        provenance <- fit$parameters$knn_input_provenance
        expect_identical(provenance$provider, "faissR")
        expect_identical(provenance$selection, selection)
        expect_equal(provenance$target_recall, 0.95)
        expect_true(provenance$exact_recall_by_construction)
        expect_identical(dim(fit$layout), c(24L, 2L))
    }
    expect_true(is.na(fits[[1L]]$parameters$target_recall))
    expect_true(is.na(fits[[3L]]$parameters$target_recall))
    expect_error(umap(x, nn = nn, target_recall = 0.90),
        "cannot be retuned")
})

test_that("precomputed query KNN retains the native selection", {
    x <- matrix(seq_len(48L), 12L, 4L)
    nn <- precompute_query_knn(x[1:9, ], x[10:12, ], k = 3L,
        backend = "cpu", n.cores = 1L, target_recall = 0.90)
    selection <- fastEmbedR:::fastembedr_knn_selection(
        "cpu", 9L, 4L, 3L, "euclidean", 0.90, n_query = 3L,
        self_query = FALSE)
    expect_identical(nn$knn_selection, selection)
    expect_identical(nn$execution_backend, "cpu")
})
