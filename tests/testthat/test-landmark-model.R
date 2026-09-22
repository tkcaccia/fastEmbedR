test_that("landmark selection is reusable across embedding methods", {
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

test_that("staged UMAP model preserves graph mode and original row order", {
    set.seed(611)
    x <- rbind(
        matrix(rnorm(120, 0, 0.2), ncol = 4),
        matrix(rnorm(120, 3, 0.2), ncol = 4)
    )
    selection <- select_landmarks(x, landmarks = 30L, seed = 8)
    model <- fit_landmark_model(
        x,
        selection,
        method = "umap",
        n_neighbors = 8L,
        graph_mode = "fuzzy",
        backend = "cpu",
        n.cores = 2L,
        seed = 8L
    )
    fit <- project_landmark_model(
        model,
        x,
        transform_k = 8L,
        refinement_epochs = 2L,
        n.cores = 2L,
        keep_knn = TRUE
    )

    expect_s3_class(model, "fastEmbedR_landmark_model")
    expect_identical(model$graph_mode, "fuzzy")
    expect_identical(model$fit$parameters$graph_mode, "fuzzy")
    expect_s3_class(fit, "fastEmbedR_embedding")
    expect_equal(dim(fit$layout), c(nrow(x), 2L))
    expect_true(all(is.finite(fit$layout)))
    expect_equal(
        unname(fit$layout[selection$indices, ]),
        matrix(
            as.numeric(model$fit$layout),
            nrow = nrow(model$fit$layout),
            ncol = ncol(model$fit$layout)
        ),
        tolerance = 1e-7
    )
    expect_identical(fit$parameters$projection_nn_backend, "cpu")
    expect_true(is.list(fit$knn))
})

test_that("staged openTSNE uses the ordinary reference optimizer", {
    set.seed(612)
    x <- rbind(
        matrix(rnorm(90, 0, 0.2), ncol = 3),
        matrix(rnorm(90, 2, 0.2), ncol = 3)
    )
    selection <- select_landmarks(x, landmarks = 30L, seed = 9)
    model <- fit_landmark_model(
        x,
        selection,
        method = "tsne",
        n_neighbors = 4L,
        perplexity = 4,
        backend = "cpu",
        n.cores = 2L,
        seed = 9L,
        early_exaggeration_iter = 2L,
        n_iter = 3L,
        negative_gradient_method = "exact"
    )
    fit <- project_landmark_model(
        model,
        x,
        transform_k = 10L,
        transform_perplexity = 2,
        transform_iter = 3L,
        transform_n_negatives = 8L,
        n.cores = 2L
    )

    expect_identical(model$fit$method, "tsne")
    expect_identical(model$n_neighbors, 4L)
    expect_identical(model$affinity_support, "compact")
    expect_identical(model$fit$parameters$affinity_support_k, 4L)
    expect_equal(dim(fit$layout), c(nrow(x), 2L))
    expect_true(all(is.finite(fit$layout)))
    expect_equal(
        unname(fit$layout[selection$indices, ]),
        matrix(
            as.numeric(model$fit$layout),
            nrow = nrow(model$fit$layout),
            ncol = ncol(model$fit$layout)
        ),
        tolerance = 1e-7
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

test_that(paste(
    "one-call landmark UMAP passes both graph modes to its",
    "reference fit"
), {
    set.seed(614)
    x <- matrix(rnorm(80 * 5), nrow = 80)
    old_refine <- getOption("fastEmbedR.landmark_umap_refine_epochs", NULL)
    on.exit(options(fastEmbedR.landmark_umap_refine_epochs = old_refine))
    options(fastEmbedR.landmark_umap_refine_epochs = 1L)
    for (mode in c("fuzzy", "binary")) {
        fit <- landmark_umap(
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

    default_fit <- landmark_umap(
        x,
        landmarks = 40L,
        n_neighbors = 8L,
        standardize = FALSE,
        backend = "cpu",
        n.cores = 2L,
        seed = 10L
    )
    expect_identical(default_fit$parameters$graph_mode, "fuzzy")
})

test_that("staged landmark projection preserves float32 layout output", {
    skip_if_not_installed("float")
    set.seed(615)
    x <- float::fl(matrix(rnorm(60 * 4), nrow = 60))
    selection <- select_landmarks(x, landmarks = 30L, seed = 11L)
    model <- fit_landmark_model(
        x,
        selection,
        method = "umap",
        n_neighbors = 6L,
        backend = "cpu",
        n.cores = 2L,
        seed = 11L
    )
    fit <- project_landmark_model(
        model,
        x,
        transform_k = 6L,
        refinement_epochs = 0L,
        n.cores = 2L
    )

    expect_true(inherits(model$reference_data, "float32"))
    expect_true(inherits(fit$layout, "float32"))
    expect_identical(dim(fit$layout), c(60L, 2L))
})

test_that(paste(
    "all-reference models treat same-sized matrices as",
    "held-out queries"
), {
    set.seed(616)
    reference <- rbind(
        matrix(rnorm(48, 0, 0.2), ncol = 4),
        matrix(rnorm(48, 3, 0.2), ncol = 4)
    )
    query <- reference + matrix(rnorm(length(reference), sd = 0.05), ncol = 4)
    selection <- select_landmarks(
        reference,
        landmarks = seq_len(nrow(reference)),
        seed = 12L
    )
    model <- fit_landmark_model(
        reference,
        selection,
        method = "umap",
        n_neighbors = 6L,
        graph_mode = "fuzzy",
        backend = "cpu",
        n.cores = 2L,
        seed = 12L
    )
    reference_before <- unname(as.matrix(model$fit$layout))
    projected <- project_landmark_model(
        model,
        query,
        transform_k = 6L,
        refinement_epochs = 1L,
        n.cores = 2L
    )

    expect_s3_class(projected, "fastEmbedR_embedding")
    expect_identical(dim(projected$layout), c(nrow(query), 2L))
    expect_identical(projected$parameters$projection_scope, "held_out_query")
    expect_false(identical(
        unname(as.matrix(projected$layout)), reference_before
    ))
    expect_equal(unname(as.matrix(model$fit$layout)), reference_before)
})
