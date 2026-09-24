test_that(paste(
    "Metal public paths stay native and do not depend on",
    "Python bridges"
), {
    desc <- utils::packageDescription("fastEmbedR")
    dependency_fields <- paste(
        unlist(desc[c("Depends", "Imports", "LinkingTo")], use.names = FALSE),
        collapse = " "
    )
    expect_false(grepl(
        "\\b(reticulate|torch|mlx)\\b", dependency_fields,
        ignore.case = TRUE
    ))

    prepare_body <- paste(
        deparse(body(fastEmbedR:::prepare_embedding_data)),
        deparse(body(fastEmbedR:::standardize_embedding_data)),
        deparse(body(fastEmbedR:::try_accelerated_standardization)),
        collapse = "\n"
    )
    transform_body <- paste(
        deparse(body(fastEmbedR::transform_tsne)),
        deparse(body(fastEmbedR:::run_tsne_transform_optimizer)),
        deparse(body(fastEmbedR:::run_tsne_transform_gpu)),
        collapse = "\n"
    )
    opentsne_body <- paste(
        deparse(body(fastEmbedR:::fast_knn_opentsne_materialized)),
        deparse(body(fastEmbedR:::run_opentsne_native_optimizer)),
        deparse(body(fastEmbedR:::run_opentsne_metal_optimizer)),
        collapse = "\n"
    )

    expect_match(prepare_body, "standardize_metal_cpp", fixed = TRUE)
    expect_match(transform_body, "transform_tsne_metal_cpp", fixed = TRUE)
    expect_match(opentsne_body, "knn_tsne_opentsne_metal_cpp", fixed = TRUE)
    native_bodies <- paste(
        prepare_body, transform_body, opentsne_body
    )
    expect_false(grepl(
        "reticulate|py_|python|torch|mlx", native_bodies,
        ignore.case = TRUE
    ))
})

test_that("GPU UMAP exposes one CUDA fused entry shape", {
    expect_length(formals(fastEmbedR:::knn_umap_cuda_fused_cpp), 10L)
})

test_that("Metal t-SNE FFT-grid exposes opt-in per-stage timing", {
    skip_if_not(fastEmbedR:::embedding_metal_available_cpp())

    old_timing <- Sys.getenv(
        "FASTEMBEDR_METAL_STAGE_TIMING", unset = NA_character_
    )
    on.exit(
        {
            if (is.na(old_timing)) {
                Sys.unsetenv("FASTEMBEDR_METAL_STAGE_TIMING")
            } else {
                Sys.setenv(FASTEMBEDR_METAL_STAGE_TIMING = old_timing)
            }
        },
        add = TRUE
    )

    Sys.setenv(FASTEMBEDR_METAL_STAGE_TIMING = "1")
    set.seed(93)
    x <- matrix(rnorm(120L * 6L), nrow = 120L, ncol = 6L)
    knn <- test_exact_knn(x, k = 12L, backend = "cpu")
    layout <- fastEmbedR::tsne_knn(
        knn,
        perplexity = 3,
        early_exaggeration_iter = 1L,
        n_iter = 1L,
        backend = "metal",
        negative_gradient_method = "fft",
        seed = 93L
    )
    timing <- attr(layout, "metal_stage_timing")

    expect_s3_class(timing, "data.frame")
    expect_named(
        timing,
        c(
            "stage", "command_count", "wall_sec", "gpu_sec",
            "gpu_timestamps_available"
        )
    )
    expect_equal(
        timing$stage,
        c(
            "layout_stats", "clear_scatter_load", "fft_forward",
            "fft_convolution", "sum_q", "epoch_update", "center"
        )
    )
    expect_true(all(timing$command_count == 2L))
    expect_true(all(is.finite(timing$wall_sec)))
    expect_true(all(timing$wall_sec >= 0))
    expect_equal(attr(layout, "fastEmbedR_config")$metal_stage_timing, timing)
})

test_that("GPU UMAP config records only the validated atomic paths", {
    cfg <- fastEmbedR:::fast_knn_umap_config(
        n = 70000L,
        k = 50L,
        backend = "metal"
    )
    cfg$gpu_optimizer_mode <- "atomic_inplace"
    cfg$gpu_umap_path <- "metal_atomic_inplace"
    expect_equal(cfg$gpu_optimizer_mode, "atomic_inplace")
    expect_equal(cfg$gpu_umap_path, "metal_atomic_inplace")

    cuda_cfg <- cfg
    cuda_cfg$backend <- "cuda"
    cuda_cfg$gpu_optimizer_mode <- "atomic_coo"
    cuda_cfg$gpu_umap_path <- "cuda_pure_atomic"
    expect_equal(cuda_cfg$gpu_optimizer_mode, "atomic_coo")
    expect_equal(cuda_cfg$gpu_umap_path, "cuda_pure_atomic")
})

test_that(paste(
    "Metal UMAP landmark refinement stays native and reports",
    "its backend"
), {
    skip_if_not(fastEmbedR:::embedding_metal_available_cpp())

    old_refine <- getOption("fastEmbedR.landmark_umap_refine_epochs", NULL)
    on.exit(
        {
            if (is.null(old_refine)) {
                options(fastEmbedR.landmark_umap_refine_epochs = NULL)
            } else {
                options(fastEmbedR.landmark_umap_refine_epochs = old_refine)
            }
        },
        add = TRUE
    )

    set.seed(91)
    x <- matrix(rnorm(360L), nrow = 60L, ncol = 6L)
    options(fastEmbedR.landmark_umap_refine_epochs = 2L)
    fit <- fastEmbedR::umap(
        x,
        landmarks = 0.5,
        n_neighbors = 8L,
        backend = "metal",
        n.cores = 2L,
        seed = 91L,
        standardize = FALSE,
        verbose = FALSE
    )

    expect_equal(fit$parameters$landmark_refinement_backend, "metal")
    expect_equal(
        attr(fit$layout, "metal_refinement"),
        "landmark_rows_atomic_inplace"
    )
    expect_equal(dim(fit$layout), c(60L, 2L))
    expect_false(anyNA(fit$layout))
})

test_that(paste(
    "Metal affine landmark projection matches CPU local",
    "affine correction"
), {
    skip_if_not(fastEmbedR:::embedding_metal_available_cpp())

    set.seed(92)
    reference_data <- matrix(rnorm(50L * 6L), nrow = 50L, ncol = 6L)
    query_data <- matrix(rnorm(12L * 6L), nrow = 12L, ncol = 6L)
    reference_layout <- cbind(rnorm(50L), rnorm(50L))
    projection <- test_exact_knn(
        reference_data, query_data, k = 14L, backend = "cpu"
    )

    cpu <- fastEmbedR:::project_embedding_affine_parallel_cpp(
        reference_data,
        query_data,
        reference_layout,
        projection$indices,
        projection$distances,
        12L,
        1e-3,
        2.5,
        2L
    )
    metal <- fastEmbedR:::project_embedding_affine_metal_cpp(
        reference_data,
        query_data,
        reference_layout,
        projection$indices,
        projection$distances,
        12L,
        1e-3,
        2.5
    )

    expect_equal(metal$method, "local_affine_knn_projection_metal")
    expect_equal(metal$backend, "metal")
    expect_false(anyNA(metal$layout))
    metal_layout <- metal$layout
    cpu_layout <- cpu$layout
    shared_attributes <- list(dim = dim(cpu$layout))
    attributes(metal_layout) <- shared_attributes
    attributes(cpu_layout) <- shared_attributes
    expect_equal(metal_layout, cpu_layout, tolerance = 1e-4)
    expect_equal(metal$fallback, cpu$fallback)
})

test_that(paste(
    "Metal preprocessing, projection, interpolation, and",
    "scoring match CPU"
), {
    skip_if_not(fastEmbedR:::embedding_metal_available_cpp())

    set.seed(71)
    x <- matrix(rnorm(160), nrow = 40L, ncol = 4L)
    cpu_pre <- fastEmbedR:::prepare_embedding_data(
        x,
        standardize = TRUE,
        pca_dims = NULL,
        seed = 71L,
        backend = "cpu"
    )
    metal_pre <- fastEmbedR:::prepare_embedding_data(
        x,
        standardize = TRUE,
        pca_dims = NULL,
        seed = 71L,
        backend = "metal"
    )
    expect_equal(metal_pre$preprocess$standardize_backend, "metal")
    expect_equal(metal_pre$data, cpu_pre$data, tolerance = 1e-5)

    x_pca <- matrix(rnorm(60L * 30L), nrow = 60L, ncol = 30L)
    metal_pca <- fastEmbedR:::prepare_embedding_data(
        x_pca,
        standardize = TRUE,
        pca_dims = 4L,
        seed = 71L,
        backend = "metal"
    )
    expect_equal(dim(metal_pca$data), c(60L, 4L))
    expect_equal(metal_pca$preprocess$pca_backend, "metal_mps_rsvd")
    expect_equal(metal_pca$preprocess$pca_method, "rsvd")

    reference_layout <- cbind(rnorm(8L), rnorm(8L))
    projection_indices <- matrix(
        c(
            1L, 2L, 3L,
            3L, 4L, 5L,
            5L, 6L, 7L,
            2L, 4L, 8L,
            1L, 7L, 8L
        ),
        nrow = 5L,
        byrow = TRUE
    )
    projection_distances <- matrix(
        c(
            0.0, 0.4, 0.7,
            0.2, 0.3, 0.9,
            0.1, 0.6, 0.8,
            0.5, 0.6, 0.7,
            0.3, 0.4, 0.5
        ),
        nrow = 5L,
        byrow = TRUE
    )
    cpu_project <- fastEmbedR:::project_embedding_knn_cpp(
        reference_layout,
        projection_indices,
        projection_distances
    )
    metal_project <- fastEmbedR:::project_embedding_knn_metal_cpp(
        reference_layout,
        projection_indices,
        projection_distances
    )
    expect_equal(metal_project, cpu_project, tolerance = 1e-5)

    landmark_indices <- c(2L, 5L, 8L, 11L, 14L, 17L, 20L, 23L)
    landmark_projection_indices <- matrix(
        rep(c(1L, 2L, 3L, 4L), length.out = 24L * 4L),
        nrow = 24L,
        ncol = 4L
    )
    landmark_projection_distances <- matrix(
        abs(rnorm(24L * 4L)) + 0.01,
        nrow = 24L,
        ncol = 4L
    )
    for (i in seq_along(landmark_indices)) {
        landmark_projection_indices[landmark_indices[i], 1L] <- i
        landmark_projection_distances[landmark_indices[i], 1L] <- 0
    }
    cpu_interp <- fastEmbedR:::interpolate_landmark_layout_cpp(
        reference_layout,
        as.integer(landmark_indices),
        landmark_projection_indices,
        landmark_projection_distances,
        24L
    )
    metal_interp <- fastEmbedR:::interpolate_landmark_layout_metal_cpp(
        reference_layout,
        as.integer(landmark_indices),
        landmark_projection_indices,
        landmark_projection_distances,
        24L
    )
    expect_equal(metal_interp, cpu_interp, tolerance = 1e-5)

    x_query <- matrix(rnorm(36L * 5L), nrow = 36L, ncol = 5L)
    fused_landmark_indices <- c(1L, 4L, 9L, 13L, 18L, 22L, 29L, 35L)
    x_landmarks <- x_query[fused_landmark_indices, , drop = FALSE]
    fused_layout_reference <- cbind(
        rnorm(length(fused_landmark_indices)),
        rnorm(length(fused_landmark_indices))
    )
    fused_projection <- test_exact_knn(
        x_landmarks, x_query, k = 4L, backend = "cpu"
    )
    cpu_fused_reference <- fastEmbedR:::interpolate_landmark_layout_cpp(
        fused_layout_reference,
        as.integer(fused_landmark_indices),
        fused_projection$indices,
        fused_projection$distances,
        nrow(x_query)
    )
    metal_fused <- fastEmbedR:::landmark_project_interpolate_metal_cpp(
        x_landmarks,
        x_query,
        fused_layout_reference,
        as.integer(fused_landmark_indices),
        4L
    )
    expect_equal(metal_fused, cpu_fused_reference, tolerance = 1e-4)

    fused_name <- paste0(
        "landmark_project_interpolate_",
        "knn_confidence_metal_cpp"
    )
    metal_fused_fun <- getFromNamespace(fused_name, "fastEmbedR")
    metal_fused_knn <- metal_fused_fun(
        x_landmarks,
        x_query,
        fused_layout_reference,
        as.integer(fused_landmark_indices),
        4L
    )
    expect_equal(metal_fused_knn$layout, cpu_fused_reference, tolerance = 1e-4)
    expect_equal(metal_fused_knn$indices, fused_projection$indices)
    expect_equal(
        metal_fused_knn$distances, fused_projection$distances,
        tolerance = 1e-4
    )
    expect_equal(length(metal_fused_knn$confidence), nrow(x_query))
    expect_true(all(is.finite(metal_fused_knn$confidence)))
    expect_true(all(
        metal_fused_knn$confidence >= 0 &
            metal_fused_knn$confidence <= 1
    ))
    expect_true(all(metal_fused_knn$confidence[fused_landmark_indices] > 0.99))

    layout <- cbind(rnorm(30L), rnorm(30L))
    labels <- rep(1:3, each = 10L)
    knn <- test_exact_knn(layout, layout, k = 7L, backend = "cpu")
    high_indices <- knn$indices[, -1L, drop = FALSE]
    keep <- seq_len(nrow(layout))
    cpu_score <- fastEmbedR:::knn_structure_score_cpp(
        layout,
        high_indices,
        as.integer(keep),
        6L,
        as.integer(labels),
        3L
    )
    metal_score <- fastEmbedR:::knn_structure_score_metal_cpp(
        layout,
        high_indices,
        as.integer(keep),
        6L,
        as.integer(labels),
        3L
    )
    expect_equal(metal_score, cpu_score, tolerance = 1e-5)

    cpu_sil <- fastEmbedR:::silhouette_score_cpp(layout, as.integer(labels))
    metal_sil <- fastEmbedR:::silhouette_score_metal_cpp(
        layout, as.integer(labels), 3L
    )
    expect_equal(metal_sil, cpu_sil, tolerance = 1e-5)
})

test_that("native Metal MPS rSVD matches reference PCA", {
    skip_if_not(fastEmbedR:::embedding_metal_available_cpp())

    set.seed(72)
    x <- matrix(rnorm(240L * 20L), nrow = 240L, ncol = 20L)
    reference <- stats::prcomp(x, center = TRUE, scale. = FALSE, rank. = 2L)$x
    fit <- fastEmbedR::pca(
        x,
        ncomp = 2L,
        center = TRUE,
        scale = FALSE,
        backend = "metal",
        seed = 72L
    )

    expect_s3_class(fit, "fastEmbedR_pca")
    expect_identical(fit$backend, "metal_mps_rsvd")
    expect_identical(fit$method, "rsvd")
    expect_identical(fit$precision, "float32")
    expect_equal(dim(fit$scores), c(240L, 2L))
    expect_equal(dim(fit$loadings), c(20L, 2L))
    agreement <- abs(stats::cor(reference, fit$scores))
    expect_gte(max(agreement[1L, ]), 0.99)
    expect_gte(max(agreement[2L, ]), 0.99)

    if (requireNamespace("float", quietly = TRUE)) {
        float_x <- float::fl(x)
        float_fit <- fastEmbedR::pca(
            float_x,
            ncomp = 2L,
            xtest = float_x[seq_len(12L), , drop = FALSE],
            center = TRUE,
            scale = FALSE,
            backend = "metal",
            seed = 72L,
            tsne_init = TRUE
        )
        expect_identical(float_fit$precision, "float32")
        expect_s4_class(float_fit$scores, "float32")
        expect_s4_class(float_fit$loadings, "float32")
        expect_s4_class(float_fit$tsne_init, "float32")
        expect_s4_class(float_fit$scores_test, "float32")
        expect_equal(dim(float_fit$scores_test), c(12L, 2L))
        expect_equal(float::dbl(float_fit$scores), fit$scores, tolerance = 1e-5)
        expect_equal(
            float::dbl(float_fit$loadings), fit$loadings,
            tolerance = 1e-5
        )
        expect_lt(
            as.numeric(object.size(float_fit$scores)),
            as.numeric(object.size(fit$scores)) * 0.7
        )
        expect_lt(
            abs(
                max(apply(
                    float::dbl(float_fit$tsne_init), 2L, stats::sd
                )) - 1e-4
            ),
            2e-8
        )

        float_init <- fastEmbedR::pca(
            float_x,
            ncomp = 2L,
            backend = "metal",
            seed = 72L,
            tsne_init = TRUE
        )$tsne_init
        expect_s4_class(float_init, "float32")
        expect_lt(
            abs(max(apply(float::dbl(float_init), 2L, stats::sd)) - 1e-4),
            2e-8
        )

        cache_file <- tempfile(fileext = ".rds")
        on.exit(unlink(cache_file), add = TRUE)
        saveRDS(float_init, cache_file)
        cached_init <- readRDS(cache_file)
        expect_s4_class(cached_init, "float32")
        expect_equal(
            float::dbl(cached_init), float::dbl(float_init),
            tolerance = 0
        )

        bad_x <- x
        bad_x[4L, 3L] <- Inf
        expect_error(
            fastEmbedR::pca(
                float::fl(bad_x),
                ncomp = 2L,
                backend = "metal",
                seed = 72L
            ),
            "finite"
        )
    }

    init <- fastEmbedR::pca(
        x,
        ncomp = 2L,
        backend = "metal",
        seed = 72L,
        tsne_init = TRUE
    )$tsne_init
    expect_identical(attr(init, "fastEmbedR_init_backend"), "metal_mps_rsvd")
    expect_identical(attr(init, "fastEmbedR_init_method"), "pca_rsvd")
    expect_equal(max(apply(init, 2L, stats::sd)), 1e-4, tolerance = 1e-8)
})
