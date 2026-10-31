test_that("UMAP rejects supplied KNN with the wrong row count", {
    x <- as.matrix(iris[1:20, 1:4])
    nn <- precompute_knn(x[1:10, ], k = 4L, backend = "cpu")
    expect_error(umap(x, nn = nn, n_neighbors = 3L, backend = "cpu"),
        "row count must match", fixed = TRUE)
})

test_that("UMAP honors and reports supplied host KNN width", {
    x <- as.matrix(iris[1:20, 1:4])
    nn <- precompute_knn(x, k = 5L, backend = "cpu")
    fit <- umap(x, nn = nn, n_neighbors = 3L, backend = "cpu",
        keep_knn = TRUE, n.cores = 1L)
    expect_equal(dim(fit$layout), c(20L, 2L))
    expect_true(all(is.finite(fit$layout)))
    expect_equal(ncol(fit$knn$indices), 3L)
    expect_equal(fit$knn$indices, nn$indices[, 1:3, drop = FALSE])
    expect_equal(fit$metrics$n_neighbors, 3L)
    expect_equal(fit$parameters$knn_n_neighbors, 3L)
    expect_error(umap(x, nn = nn, n_neighbors = 6L, backend = "cpu"),
        "larger than the supplied KNN width", fixed = TRUE)

    default <- umap(x, nn = nn, backend = "cpu", n.cores = 1L)
    expect_equal(default$metrics$n_neighbors, 5L)
    expect_equal(default$parameters$knn_n_neighbors, 5L)
})

test_that("UMAP trims non-self neighbors while preserving float32 distances", {
    skip_if_not_installed("float")
    x <- as.matrix(iris[1:20, 1:4])
    nn <- precompute_knn(x, k = 5L, backend = "cpu")
    distances <- fastEmbedR:::embedding_dense_double_matrix(nn$distances)
    supplied <- list(indices = cbind(seq_len(nrow(x)), nn$indices),
        distances = float::fl(cbind(0, distances)))
    fit <- umap(x, nn = supplied, n_neighbors = 3L, backend = "cpu",
        keep_knn = TRUE, n.cores = 1L)
    expect_equal(fit$knn$indices, nn$indices[, 1:3, drop = FALSE])
    expect_s4_class(fit$knn$distances, "float32")
    expect_equal(fit$parameters$knn_n_neighbors, 3L)
    expect_true(all(is.finite(fit$layout)))
})

test_that("UMAP rejects incompatible GPU KNN widths without materializing", {
    nn <- structure(list(n_query = 8L, k = 3L, exclude_self = TRUE),
        class = "fastEmbedR_gpu_knn")
    expect_error(fastEmbedR:::validate_umap_supplied_knn(nn, "cuda", 8L, 2L),
        "GPU KNN width must match", fixed = TRUE)
    expect_identical(
        fastEmbedR:::validate_umap_supplied_knn(nn, "cuda", 8L, 3L), nn)
    expect_error(fastEmbedR:::validate_umap_supplied_knn(nn, "cuda", 9L, 3L),
        "incompatible with UMAP", fixed = TRUE)
})
