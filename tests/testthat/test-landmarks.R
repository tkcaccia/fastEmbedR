test_that("landmark selection is deterministic and exhaustive", {
    set.seed(610)
    x <- rbind(
        matrix(rnorm(120, 0, 0.2), ncol = 4),
        matrix(rnorm(120, 3, 0.2), ncol = 4)
    )
    first <- select_landmarks(x, landmarks = 0.5, seed = 7, n.cores = 2)
    second <- select_landmarks(x, landmarks = 0.5, seed = 7, n.cores = 2)
    expect_s3_class(first, "fastEmbedR_landmark_selection")
    expect_identical(first$indices, second$indices)
    expect_equal(length(first$indices), nrow(x) / 2)
    expect_setequal(
        c(first$indices, first$query_indices),
        seq_len(nrow(x))
    )
    expect_length(intersect(first$indices, first$query_indices), 0L)
})

test_that("landmark UMAP preserves its requested graph mode", {
    set.seed(614)
    x <- matrix(rnorm(80 * 5), nrow = 80)
    old_refine <- getOption("fastEmbedR.landmark_umap_refine_epochs", NULL)
    on.exit(options(fastEmbedR.landmark_umap_refine_epochs = old_refine))
    options(fastEmbedR.landmark_umap_refine_epochs = 1L)
    for (mode in c("fuzzy", "binary")) {
        fit <- umap(
            x,
            landmarks = 40L,
            n_neighbors = 8L,
            standardize = FALSE,
            backend = "cpu",
            n.cores = 2L,
            graph_mode = mode,
            seed = 10L
        )
        expect_identical(fit$parameters$graph_mode, mode)
        expect_identical(
            fit$landmarks$reference_fit$parameters$graph_mode,
            mode
        )
    }
})

test_that("primary embedding functions disable landmarking by default", {
    expect_identical(formals(tsne)$landmarks, FALSE)
    expect_identical(formals(umap)$landmarks, FALSE)
    expect_error(tsne(matrix(rnorm(40), 10), landmarks = TRUE),
        "fraction"
    )
    expect_error(umap(matrix(rnorm(40), 10), landmarks = TRUE),
        "fraction"
    )
})

test_that("umap accepts explicit landmark row indices", {
    set.seed(615)
    x <- matrix(rnorm(60 * 4), nrow = 60)
    rows <- seq.int(1L, 59L, by = 2L)
    old_refine <- getOption("fastEmbedR.landmark_umap_refine_epochs", NULL)
    on.exit(options(fastEmbedR.landmark_umap_refine_epochs = old_refine))
    options(fastEmbedR.landmark_umap_refine_epochs = 0L)
    fit <- umap(
        x, landmarks = rows, n_neighbors = 6L,
        backend = "cpu", n.cores = 2L, seed = 11L
    )
    expect_identical(fit$landmarks$indices, rows)
    expect_identical(fit$parameters$n_landmarks, length(rows))
    expect_true(isTRUE(fit$parameters$landmark))
    expect_error(
        umap(x, landmarks = c(1.5, 2.5), n_neighbors = 6L),
        "valid row numbers"
    )
})

test_that("query KNN searches only the fixed reference", {
    set.seed(613)
    reference <- matrix(rnorm(80 * 5), nrow = 80)
    query <- matrix(rnorm(20 * 5), nrow = 20)
    found <- precompute_query_knn(
        reference,
        query,
        k = 10L,
        backend = "cpu",
        n.cores = 2L
    )
    expect_s3_class(found, "fastEmbedR_knn")
    expect_equal(dim(found$indices), c(20L, 10L))
    expect_true(all(found$indices >= 1L & found$indices <= 80L))
    expect_false(isTRUE(found$exclude_self))
    expect_identical(found$n_reference, 80L)
})

test_that("integrated landmark UMAP returns a reusable model", {
    set.seed(616)
    x <- matrix(rnorm(60 * 6), nrow = 60)
    old_refine <- getOption("fastEmbedR.landmark_umap_refine_epochs", NULL)
    on.exit(options(fastEmbedR.landmark_umap_refine_epochs = old_refine))
    options(fastEmbedR.landmark_umap_refine_epochs = 0L)
    fit <- umap(
        x, landmarks = 30L, n_neighbors = 6L,
        standardize = TRUE, pca_dims = 3L,
        backend = "cpu", n.cores = 2L, seed = 12L
    )
    projected <- project_landmark_model(
        fit$model, x[1:5, , drop = FALSE],
        transform_k = 6L, refinement_epochs = 0L
    )
    expect_s3_class(fit$model, "fastEmbedR_landmark_model")
    expect_null(fit$model$preprocess_transform$pca$scores)
    expect_equal(dim(projected$layout), c(5L, 2L))
    expect_true(all(is.finite(projected$layout)))
    expect_true(fit$metrics$landmark_reference_memory_saving > 0)
    expect_gt(fit$metrics$landmark_projection_knn_bytes, 0)
    expect_identical(projected$parameters$projection_scope, "held_out_query")
})

test_that("same-sized new data are not mistaken for training data", {
    set.seed(617)
    x <- matrix(rnorm(48 * 4), nrow = 48)
    old_refine <- getOption("fastEmbedR.landmark_umap_refine_epochs", NULL)
    on.exit(options(fastEmbedR.landmark_umap_refine_epochs = old_refine))
    options(fastEmbedR.landmark_umap_refine_epochs = 0L)
    fit <- umap(
        x, landmarks = 24L, n_neighbors = 5L,
        backend = "cpu", seed = 13L
    )
    query <- x + 0.01
    projected <- project_landmark_model(
        fit$model, query, transform_k = 5L, refinement_epochs = 0L
    )
    expect_equal(dim(projected$layout), c(nrow(query), 2L))
    expect_identical(projected$parameters$projection_scope, "held_out_query")
})

test_that("integrated landmark t-SNE returns a reusable model", {
    set.seed(619)
    x <- matrix(rnorm(36 * 4), nrow = 36)
    fit <- tsne(
        x, landmarks = 18L, perplexity = 3,
        early_exaggeration_iter = 1L, n_iter = 1L,
        transform_iter = 0L, negative_gradient_method = "exact",
        backend = "cpu", n.cores = 2L, seed = 15L
    )
    projected <- project_landmark_model(
        fit$model, x[1:4, , drop = FALSE],
        transform_k = 4L, transform_perplexity = 2,
        transform_iter = 1L
    )
    expect_s3_class(fit$model, "fastEmbedR_landmark_model")
    expect_identical(fit$model$method, "tsne")
    expect_s3_class(fit$model$fit, "fastEmbedR_embedding")
    expect_equal(dim(projected$layout), c(4L, 2L))
    expect_true(all(is.finite(projected$layout)))
    expect_true(fit$metrics$landmark_reference_memory_saving > 0)
    expect_gt(fit$metrics$landmark_projection_knn_bytes, 0)
})

test_that("original reconstruction requires explicit query indices", {
    set.seed(618)
    x <- matrix(rnorm(44 * 4), nrow = 44)
    old_refine <- getOption("fastEmbedR.landmark_umap_refine_epochs", NULL)
    on.exit(options(fastEmbedR.landmark_umap_refine_epochs = old_refine))
    options(fastEmbedR.landmark_umap_refine_epochs = 0L)
    fit <- umap(
        x, landmarks = 22L, n_neighbors = 5L,
        backend = "cpu", seed = 14L
    )
    model <- fit$model
    rebuilt <- project_landmark_model(
        model, x, query_indices = model$selection$query_indices,
        transform_k = 5L, refinement_epochs = 0L
    )
    expect_equal(dim(rebuilt$layout), c(nrow(x), 2L))
    expect_identical(
        rebuilt$parameters$projection_scope,
        "original_reconstruction"
    )
    expect_equal(
        as.numeric(rebuilt$layout[model$selection$indices, ]),
        as.numeric(model$fit$layout), tolerance = 1e-7
    )
})

test_that("landmark integer controls reject fractional values", {
    x <- matrix(rnorm(40 * 3), nrow = 40)
    expect_error(
        select_landmarks(x, landmarks = 10.5),
        "whole number"
    )
    expect_error(
        umap(x, landmarks = 20L, n_neighbors = 5.5),
        "positive and smaller"
    )
    expect_error(
        select_landmark_query_rows(x, c(1, 2.5), 1L),
        "invalid rows"
    )
    request <- list(
        initialization = "median", transform_iter = 2.5,
        transform_early_exaggeration_iter = 0L,
        transform_k = NULL, transform_n_negatives = NULL
    )
    expect_error(
        normalize_landmark_tsne_request(request),
        "transform_iter"
    )
})

test_that("landmark projection rejects fractional iteration controls", {
    set.seed(620)
    x <- matrix(rnorm(36 * 4), nrow = 36)
    old_refine <- getOption("fastEmbedR.landmark_umap_refine_epochs", NULL)
    on.exit(options(fastEmbedR.landmark_umap_refine_epochs = old_refine))
    options(fastEmbedR.landmark_umap_refine_epochs = 0L)
    fit <- umap(
        x, landmarks = 18L, n_neighbors = 5L,
        backend = "cpu", n.cores = 1L, seed = 16L
    )
    expect_error(
        project_landmark_model(
            fit$model, x[1:3, , drop = FALSE],
            transform_k = 5L, refinement_epochs = 1.5
        ),
        "refinement_epochs"
    )
})

test_that("landmark memory metrics separate fit and projection stages", {
    memory <- fastEmbedR:::landmark_memory_estimate(
        n = 1000, n_landmarks = 250, n_neighbors = 15,
        transform_k = 15, n_components = 2
    )
    expect_lt(
        memory$landmark_reference_optimizer_bytes,
        memory$full_optimizer_bytes
    )
    expect_equal(
        memory$landmark_reference_memory_saving,
        1 - memory$landmark_reference_memory_ratio
    )
    expect_gt(memory$landmark_projection_knn_bytes, 0)
})
