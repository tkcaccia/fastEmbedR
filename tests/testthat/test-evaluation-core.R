test_that("standard rank metrics are exact for an unchanged embedding", {
    set.seed(10)
    x <- matrix(rnorm(80L), nrow = 20L)
    metrics <- fastEmbedR:::exact_structure_metrics_cpp(x, x, c(3L, 5L), 2L)

    expect_equal(metrics[, "trustworthiness"], c(1, 1), tolerance = 1e-12)
    expect_equal(metrics[, "continuity"], c(1, 1), tolerance = 1e-12)
    expect_equal(metrics[, "knn_preservation"], c(1, 1), tolerance = 1e-12)
    expect_equal(
        metrics[, "mean_neighbor_rank_error"],
        c(0, 0), tolerance = 1e-12
    )
})

test_that("evaluation samples before local quadratic work", {
    set.seed(11)
    x <- matrix(rnorm(120L * 6L), nrow = 120L)
    layout <- x[, 1:2, drop = FALSE] + matrix(rnorm(240L, sd = 0.05), 120L, 2L)
    labels <- factor(rep(letters[1:3], length.out = 120L))

    result <- fastEmbedR::evaluate_embedding(
        x,
        layout,
        labels = labels,
        k = c(5L, 10L),
        sample_size_for_local_metrics = 60L,
        sample_size_for_global_metrics = 50L,
        use_cache = FALSE,
        seed = 11L,
        dataset = "core-test"
    )

    expect_equal(result$local_sample_size, 60L)
    expect_identical(result$rank_metric_backend, "cpu_exact")
    expect_true(all(is.finite(unlist(result[c(
        "trustworthiness", "continuity", "knn_preservation", "silhouette"
    )], use.names = FALSE))))
})

test_that("self neighbors are removed by row identity rather than position", {
    indices <- rbind(
        c(2L, 1L, 3L),
        c(1L, 3L, 2L),
        c(4L, 3L, 2L),
        c(3L, 1L, 4L)
    )
    distances <- matrix(
        as.numeric(seq_len(length(indices))),
        nrow(indices), ncol(indices)
    )
    distances[cbind(seq_len(nrow(indices)), c(2L, 3L, 2L, 3L))] <- 0
    normalized <- fastEmbedR:::normalize_supplied_knn(
        list(indices = indices, distances = distances),
        n = 4L,
        n_neighbors = 2L
    )

    expect_false(any(normalized$indices == row(normalized$indices)))
})

test_that("evaluation cache keys include a data fingerprint", {
    set.seed(12)
    x1 <- matrix(rnorm(200L), 50L, 4L)
    x2 <- x1
    x2[1L, 1L] <- x2[1L, 1L] + 1

    expect_false(identical(
        fastEmbedR:::evaluation_data_fingerprint(x1),
        fastEmbedR:::evaluation_data_fingerprint(x2)
    ))
})

test_that("one-call and KNN openTSNE agree with a shared initialization", {
    set.seed(13)
    x <- matrix(rnorm(60L * 5L), 60L, 5L)
    knn <- test_exact_knn(x, x, k = 16L)
    y_init <- fastEmbedR::pca(
        x, backend = "cpu", seed = 13L, tsne_init = TRUE
    )$tsne_init

    full <- fastEmbedR::tsne(
        x,
        nn = knn,
        perplexity = 5,
        Y_init = y_init,
        backend = "cpu",
        n.cores = 2L,
        early_exaggeration_iter = 2L,
        n_iter = 3L,
        auto_config = FALSE,
        seed = 13L
    )
    from_knn <- fastEmbedR::tsne_knn(
        knn,
        perplexity = 5,
        Y_init = y_init,
        backend = "cpu",
        n.cores = 2L,
        early_exaggeration_iter = 2L,
        n_iter = 3L,
        auto_config = FALSE,
        seed = 13L
    )

    expect_equal(
        as.vector(full$layout),
        as.vector(from_knn),
        tolerance = 1e-7
    )
})

test_that("evaluate_embedding accepts float32 data and layouts", {
    skip_if_not_installed("float")
    set.seed(104)
    x <- matrix(rnorm(240L), 40L, 6L)
    layout <- x[, 1:2, drop = FALSE]

    metrics <- evaluate_embedding(
        float::fl(x),
        float::fl(layout),
        k = 5L,
        sample_size_for_local_metrics = 30L,
        sample_size_for_global_metrics = 30L,
        n.cores = 2L,
        seed = 104L
    )

    expect_true(is.finite(metrics$trustworthiness))
    expect_true(is.finite(metrics$distance_spearman))
    expect_equal(metrics$local_sample_size, 30L)
})

test_that("evaluate_embedding rejects invalid values and integer controls", {
    set.seed(106)
    x <- matrix(rnorm(120L), 30L, 4L)
    layout <- x[, 1:2, drop = FALSE]
    nonfinite_x <- x
    nonfinite_x[1L, 1L] <- Inf
    nonfinite_layout <- layout
    nonfinite_layout[2L, 2L] <- NA_real_

    expect_error(
        evaluate_embedding(nonfinite_x, layout),
        "`x_high` must contain only finite values"
    )
    expect_error(
        evaluate_embedding(x, nonfinite_layout),
        "`embedding` must contain only finite values"
    )
    expect_error(
        evaluate_embedding(matrix(letters[1:120], 30L), layout),
        "`x_high` must be a numeric matrix"
    )
    expect_error(evaluate_embedding(x, layout, k = 3.5), "positive integers")
    expect_error(
        evaluate_embedding(x, layout, primary_k = 4.5),
        "positive integer"
    )
    expect_error(
        evaluate_embedding(
            x, layout, sample_size_for_local_metrics = 10.5
        ),
        "Sample sizes must be positive integers"
    )
})

test_that("seeded helpers preserve the caller RNG state", {
    set.seed(105)
    before <- .Random.seed

    fastEmbedR:::sample_indices(100L, 10L, seed = 7L)
    fastEmbedR:::make_opentsne_random_init(20L, 2L, seed = 7L)

    expect_identical(.Random.seed, before)
})
