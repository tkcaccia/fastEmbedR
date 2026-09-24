test_that("tsne_knn runs native t-SNE from supplied neighbours", {
    set.seed(321)
    x <- matrix(rnorm(50L * 5L), 50L, 5L)
    knn <- test_exact_knn(x, k = 16L, backend = "cpu")

    layout <- tsne_knn(
        knn,
        perplexity = 5,
        early_exaggeration_iter = 3L,
        n_iter = 4L,
        learning_rate = "auto",
        negative_gradient_method = "fft",
        n.cores = 2L,
        seed = 321L
    )

    expect_equal(dim(layout), c(nrow(x), 2L))
    expect_true(all(is.finite(layout)))
    cfg <- attr(layout, "fastEmbedR_config")
    expect_equal(cfg$method, "tsne")
    expect_match(cfg$optimizer, "^opentsne_fitsne_fft_grid_sparse_knn")
    expect_equal(cfg$repulsion, "fft_grid")
    expect_equal(cfg$early_exaggeration_iter, 3L)
    expect_equal(cfg$n_iter, 4L)
    expect_equal(cfg$learning_rate, "auto_opt_sne_n_over_early_exaggeration")
    expect_equal(cfg$learning_rate_early, nrow(x) / cfg$early_exaggeration)
    expect_equal(cfg$learning_rate_normal, nrow(x) / cfg$early_exaggeration)
})

test_that("t-SNE integer controls reject fractional values", {
    set.seed(320)
    x <- matrix(rnorm(30L * 4L), 30L, 4L)
    knn <- test_exact_knn(x, k = 8L, backend = "cpu")

    expect_error(
        tsne_knn(knn, perplexity = 3, n_neighbors = 3.5),
        "positive integer"
    )
    expect_error(
        tsne_knn(
            knn, perplexity = 3,
            early_exaggeration_iter = 1L, n_iter = 2.5
        ),
        "n_iter"
    )
    expect_error(
        transform_tsne(
            matrix(rnorm(20), nrow = 10),
            knn = list(
                indices = matrix(1:6, nrow = 3L),
                distances = matrix(0.1, nrow = 3L, ncol = 2L)
            ),
            k = 1.5, perplexity = 1
        ),
        "positive integer"
    )
})

test_that("t-SNE auto configuration uses compact affinity support", {
    policy <- fastEmbedR:::tsne_auto_parameters_cpp(
        150L,
        30L,
        NA_real_,
        TRUE,
        "cpu",
        "exact"
    )
    expect_equal(policy$perplexity, 30)
    expect_equal(policy$n_neighbors, 30L)
    expect_equal(policy$learning_rate, 150 / policy$early_exaggeration)
    expect_false(policy$auto_kld_stop)
    expect_equal(policy$auto_iter_end, 5000)
    expect_equal(
        policy$rule,
        "opt_sne_learning_rate_fixed_iterations_no_expensive_kld_polling"
    )

    large_fft <- fastEmbedR:::tsne_auto_parameters_cpp(
        70000L,
        90L,
        NA_real_,
        TRUE,
        "cpu",
        "fft"
    )
    expect_equal(large_fft$perplexity, 30)
    expect_equal(large_fft$n_neighbors, 30L)
    expect_false(large_fft$auto_kld_stop)
})

test_that("t-SNE exposes FFT and exact negative-gradient choices", {
    set.seed(312)
    x <- matrix(rnorm(42L * 4L), 42L, 4L)
    knn <- test_exact_knn(x, k = 13L, backend = "cpu")

    expect_error(
        tsne_knn(
            knn,
            perplexity = 4,
            negative_gradient_method = "bh",
            early_exaggeration_iter = 2L,
            n_iter = 3L,
            n.cores = 2L,
            seed = 312L
        ),
        "removed"
    )
    expect_error(
        tsne_knn(
            knn,
            perplexity = 4,
            negative_gradient_method = "sampled",
            early_exaggeration_iter = 2L,
            n_iter = 3L,
            n.cores = 2L,
            seed = 312L
        ),
        "changes the optimization mathematics"
    )

    exact <- tsne_knn(
        knn,
        perplexity = 4,
        negative_gradient_method = "exact",
        early_exaggeration_iter = 2L,
        n_iter = 3L,
        n.cores = 2L,
        seed = 312L
    )
    expect_equal(attr(exact, "fastEmbedR_config")$repulsion, "pair_symmetric")

    fft <- tsne_knn(
        knn,
        perplexity = 4,
        negative_gradient_method = "fft",
        early_exaggeration_iter = 2L,
        n_iter = 3L,
        n.cores = 2L,
        seed = 312L
    )
    expect_equal(attr(fft, "fastEmbedR_config")$repulsion, "fft_grid")
    expect_match(
        attr(fft, "fastEmbedR_config")$optimizer,
        "^opentsne_fitsne_fft_grid_sparse_knn"
    )
})

test_that("tsne has direct KNN input functions", {
    set.seed(322)
    x <- matrix(rnorm(54L * 5L), 54L, 5L)
    labels <- rep(1:3, length.out = nrow(x))
    knn <- test_exact_knn(x, k = 19L, backend = "cpu")

    layout <- tsne_knn(
        knn$indices,
        knn$distances,
        perplexity = 3,
        early_exaggeration_iter = 2L,
        n_iter = 3L,
        n.cores = 2L,
        seed = 322L
    )
    expect_equal(dim(layout), c(nrow(x), 2L))
    expect_true(all(is.finite(layout)))
    cfg <- attr(layout, "fastEmbedR_config")
    expect_equal(cfg$method, "tsne")
    expect_equal(cfg$n_neighbors, 3L)
    expect_equal(cfg$perplexity, 3)
    expect_equal(cfg$affinity_support, "compact")
    expect_equal(cfg$affinity_support_multiplier, 1)
    expect_true(cfg$compact_affinity_support)
    expect_type(layout, "double")
    expect_identical(attr(layout, "precision"), "double")
    expect_identical(cfg$output_precision, "double")

    fit <- tsne(
        knn,
        perplexity = 3,
        early_exaggeration_iter = 2L,
        n_iter = 3L,
        seed = 322L
    )
    expect_s3_class(fit, "fastEmbedR_embedding")
    expect_equal(dim(fit$layout), c(nrow(x), 2L))
    expect_equal(fit$parameters$input, "knn")
    expect_equal(fit$metrics$n_neighbors, 3L)
    expect_equal(fit$parameters$affinity_support, "compact")
    expect_equal(fit$parameters$affinity_support_k, 3L)
    expect_equal(fit$parameters$affinity_support_multiplier, 1)
    expect_true(fit$parameters$compact_affinity_support)
    expect_equal(fit$metrics$preprocess_elapsed, 0)
    expect_equal(fit$metrics$knn_elapsed, 0)

    expect_error(
        tsne(knn, perplexity = 3, affinity_support = "standard"),
        "not configurable"
    )
})

test_that("t-SNE defaults to compact support for precomputed KNN", {
    set.seed(323)
    x <- matrix(rnorm(60L * 5L), 60L, 5L)
    compact_knn <- test_exact_knn(x, k = 6L, backend = "cpu")

    layout <- tsne_knn(
        compact_knn,
        perplexity = 5,
        early_exaggeration_iter = 1L,
        n_iter = 2L
    )
    cfg <- attr(layout, "fastEmbedR_config")
    expect_equal(cfg$affinity_support, "compact")
    expect_true(cfg$compact_affinity_support)

    expect_error(
        tsne_knn(
            compact_knn,
            perplexity = 5,
            affinity_support = "standard",
            early_exaggeration_iter = 1L,
            n_iter = 2L
        ),
        "not configurable"
    )
})

test_that("post-fit KL diagnostic matches the optimizer's exact final KL", {
    set.seed(325)
    x <- matrix(rnorm(80L * 6L), 80L, 6L)
    knn <- test_exact_knn(x, k = 16L, backend = "cpu")
    layout <- tsne_knn(
        knn,
        perplexity = 5,
        early_exaggeration_iter = 1L,
        n_iter = 2L,
        record_costs = TRUE,
        n.cores = 2L,
        seed = 325L
    )
    recorded <- tail(attr(layout, "itercosts"), 1L)
    normalized <- fastEmbedR:::normalize_opentsne_knn_input(knn, NULL, 5L)
    diagnostic <- fastEmbedR:::opentsne_kl_diagnostic_cpp(
        normalized$indices,
        normalized$distances,
        layout,
        5,
        2L
    )
    expect_equal(diagnostic, recorded, tolerance = 1e-6)
})

test_that("tsne_knn preserves the documented float32 return contract", {
    skip_if_not_installed("float")
    set.seed(324)
    x <- float::fl(matrix(rnorm(48L * 5L), 48L, 5L))
    knn <- precompute_knn(x, k = 12L, backend = "cpu", n.cores = 2L)

    layout <- tsne_knn(
        knn,
        perplexity = 3,
        early_exaggeration_iter = 1L,
        n_iter = 2L,
        n.cores = 2L,
        seed = 324L
    )

    expect_s4_class(layout, "float32")
    expect_identical(attr(layout, "precision"), "float32")
    expect_identical(
        attr(layout, "fastEmbedR_config")$output_precision,
        "float32"
    )
})

test_that("native Metal t-SNE runs FFT-grid without CPU fallback", {
    skip_if_not(fastEmbedR:::embedding_metal_available_cpp())
    skip_if_not(fastEmbedR:::metal_opentsne_native_available())

    old_grid <- Sys.getenv("FASTEMBEDR_TSNE_FFT_GRID", unset = NA_character_)
    Sys.setenv(FASTEMBEDR_TSNE_FFT_GRID = "32")
    on.exit(
        {
            if (is.na(old_grid)) {
                Sys.unsetenv("FASTEMBEDR_TSNE_FFT_GRID")
            } else {
                Sys.setenv(FASTEMBEDR_TSNE_FFT_GRID = old_grid)
            }
        },
        add = TRUE
    )

    set.seed(323)
    x <- matrix(rnorm(96L * 5L), 96L, 5L)
    knn <- test_exact_knn(x, k = 16L, backend = "cpu")
    metal <- tsne_knn(
        knn,
        perplexity = 5,
        early_exaggeration_iter = 1L,
        n_iter = 2L,
        negative_gradient_method = "fft",
        backend = "metal",
        seed = 323L
    )
    expect_equal(dim(metal), c(nrow(x), 2L))
    expect_true(all(is.finite(metal)))
    cfg <- attr(metal, "fastEmbedR_config")
    expect_equal(cfg$backend, "metal")
    expect_equal(cfg$optimizer, "opentsne_fitsne_fft_grid_native_metal")
    expect_equal(cfg$repulsion, "fft_grid_metal")
})

test_that(paste(
    "t-SNE GPU optimizers are native and fail clearly when",
    "unavailable"
), {
    set.seed(319)
    x <- matrix(rnorm(32L * 4L), 32L, 4L)
    knn <- test_exact_knn(x, k = 10L, backend = "cpu")

    if (isTRUE(fastEmbedR:::embedding_metal_available_cpp()) &&
        isTRUE(fastEmbedR:::metal_opentsne_native_available())) {
        metal <- tsne_knn(
            knn,
            perplexity = 3,
            early_exaggeration_iter = 1L,
            n_iter = 1L,
            negative_gradient_method = "exact",
            backend = "metal"
        )
        cfg <- attr(metal, "fastEmbedR_config")
        expect_equal(cfg$backend, "metal")
        expect_equal(cfg$optimizer, "opentsne_exact_sparse_native_metal")
        expect_equal(cfg$repulsion, "exact_metal")
        expect_equal(
            cfg$probabilities,
            "symmetric_sparse_knn_cpu_prepared_for_metal"
        )
    } else {
        expect_error(
            tsne_knn(
                knn,
                perplexity = 3,
                early_exaggeration_iter = 1L,
                n_iter = 1L,
                backend = "metal"
            ),
            "Native Metal t-SNE optimizer was requested",
            fixed = TRUE
        )
    }
    if (isTRUE(fastEmbedR:::embedding_cuda_available_cpp()) &&
        isTRUE(fastEmbedR:::cuda_opentsne_native_available())) {
        cuda <- tsne_knn(
            knn,
            perplexity = 3,
            early_exaggeration_iter = 1L,
            n_iter = 1L,
            negative_gradient_method = "fft",
            backend = "cuda"
        )
        cfg <- attr(cuda, "fastEmbedR_config")
        expect_equal(cfg$backend, "cuda")
        expect_equal(cfg$optimizer, "opentsne_fitsne_fft_grid_native_cuda")
        expect_equal(cfg$repulsion, "fft_grid_cuda_cufft")
        expect_true(all(is.finite(cuda)))
    } else {
        expect_error(
            tsne(
                knn,
                perplexity = 3,
                early_exaggeration_iter = 1L,
                n_iter = 1L,
                backend = "cuda"
            ),
            "CUDA"
        )
    }
})

test_that("tsne rejects low-level KNN backend names", {
    set.seed(313)
    x <- matrix(rnorm(36L * 4L), 36L, 4L)
    expect_error(
        tsne(
            x,
            perplexity = 2,
            early_exaggeration_iter = 2L,
            n_iter = 3L,
            backend = "cuda_cuvs_bruteforce"
        ),
        "must be one of"
    )
})

test_that("CPU openTSNE reports native affinity and optimization timings", {
    set.seed(910)
    x <- matrix(rnorm(180 * 6), nrow = 180)
    d <- as.matrix(stats::dist(x))
    diag(d) <- Inf
    k <- 15L
    indices <- t(apply(d, 1L, order))[, seq_len(k), drop = FALSE]
    distances <- matrix(
        d[cbind(rep(seq_len(nrow(d)), each = k), as.vector(t(indices)))],
        nrow = nrow(d), byrow = TRUE
    )
    layout <- tsne_knn(
        indices, distances,
        perplexity = 5,
        Y_init = matrix(rnorm(360, sd = 1e-4), ncol = 2),
        backend = "cpu", n.cores = 2,
        early_exaggeration_iter = 2, n_iter = 3, auto_config = FALSE
    )
    cfg <- attr(layout, "fastEmbedR_config")
    expect_identical(cfg$n.cores_requested, 2L)
    expect_identical(cfg$n.cores, 2L)
    expect_true(is.finite(cfg$affinity_elapsed_sec))
    expect_true(is.finite(cfg$optimization_elapsed_sec))
    expect_gt(cfg$affinity_elapsed_sec, 0)
    expect_gt(cfg$optimization_elapsed_sec, 0)
    expect_equal(
        cfg$native_total_elapsed_sec,
        cfg$affinity_elapsed_sec + cfg$optimization_elapsed_sec,
        tolerance = 1e-12
    )
})
