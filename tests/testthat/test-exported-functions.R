make_cluster_data <- function(n = 18L, p = 5L) {
    labels <- rep(seq_len(3L), length.out = n)
    x <- matrix(rnorm(n * p, sd = 0.25), nrow = n, ncol = p)
    x <- x + labels
    list(x = x)
}

expect_embedding <- function(layout, n) {
    expect_equal(dim(layout), c(n, 2L))
    expect_true(all(is.finite(layout)))
}

test_that("public API is KNN and t-SNE focused", {
    exports <- getNamespaceExports("fastEmbedR")
    expect_true(all(c(
        "umap", "umap_knn", "tsne", "tsne_knn",
        "evaluate_embedding", "transform_tsne",
        "umap_init", "prepare_umap_knn", "prepare_tsne_knn", "precompute_knn",
        "precompute_query_knn", "select_landmarks",
        "project_landmark_model", "pca",
        "knn_graph", "graph_cluster"
    ) %in% exports))

    pca_args <- names(formals(pca))
    expect_identical(
        pca_args[seq_len(6L)],
        c("x", "ncomp", "xtest", "center", "scale", "backend")
    )
    expect_true(all(c("n.cores", "seed", "tsne_init") %in% pca_args))
    expect_false("method" %in% pca_args)
    expect_identical(formals(pca)$tsne_init, FALSE)
    removed_exports <- c(
        "embed_knn", "fastEmbedR_api", "fastEmbedR_capabilities",
        "fastEmbedR_backend", "fit_landmark_model", "tsne_pca_init",
        "landmark_tsne", "landmark_umap"
    )
    expect_false(any(removed_exports %in% exports))
    expect_false(any(c(
        "supervised_umap", "infotsne", "pacmap", "trimap",
        "localmap", "transform_embedding",
        "nn", "nn_without_self", "candidate_knn", "fast_kmeans",
        "knn_fit", "predict_proba", "knn_recall", "faiss_available",
        "cuvs_available", "cuda_available", "metal_available", "backend_info"
    ) %in% exports))
    removed_private <- c(
        "backend_info", "fastembedr_cuda_tsvd_pca",
        "fastembedr_metal_tsvd_pca", "raft_tsvd_init_cuda_cpp"
    )
    expect_false(any(vapply(
        removed_private,
        exists,
        logical(1L),
        envir = asNamespace("fastEmbedR"),
        inherits = FALSE
    )))

    expect_identical(
        names(formals(precompute_knn)),
        c("data", "k", "metric", "backend", "n.cores")
    )

    expect_true("n.cores" %in% names(formals(tsne)))
    expect_true("n.cores" %in% names(formals(tsne_knn)))
    expect_true("n.cores" %in% names(formals(umap)))
    expect_true("n.cores" %in% names(formals(umap_knn)))
    expect_true("landmarks" %in% names(formals(tsne)))
    expect_true("landmarks" %in% names(formals(umap)))
    expect_true("n.cores" %in% names(formals(transform_tsne)))
    expect_true("n.cores" %in% names(formals(evaluate_embedding)))
    expect_true("n.cores" %in% names(formals(knn_graph)))
    expect_true("backend" %in% names(formals(graph_cluster)))
})

test_that("core exported functions have tiny t-SNE smoke tests", {
    set.seed(101)
    fixture <- make_cluster_data()
    x <- fixture$x
    labels <- fixture$labels
    n <- nrow(x)

    expect_type(fastEmbedR:::embedding_metal_available_cpp(), "logical")
    expect_length(fastEmbedR:::embedding_metal_available_cpp(), 1L)
    expect_type(fastEmbedR:::embedding_cuda_available_cpp(), "logical")
    expect_length(fastEmbedR:::embedding_cuda_available_cpp(), 1L)
    knn <- test_exact_knn(x, backend = "cpu")
    expect_type(knn, "list")
    expected_dim <- c(
        n, fastEmbedR:::auto_k(x, include_self = TRUE)
    )
    expect_equal(dim(knn$indices), expected_dim)
    expect_equal(dim(knn$distances), expected_dim)
    expect_equal(attr(knn, "backend"), "test_exact")
    layout <- tsne_knn(
        knn, perplexity = 1,
        early_exaggeration_iter = 2L, n_iter = 3L
    )
    expect_embedding(layout, n)
    expect_equal(attr(layout, "fastEmbedR_config")$method, "tsne")

    layout_umap <- umap_knn(knn)
    expect_embedding(layout_umap, n)
    expect_equal(attr(layout_umap, "fastEmbedR_config")$method, "umap")

    layout_umap_knn <- umap_knn(knn)
    expect_embedding(layout_umap_knn, n)
    expect_equal(attr(layout_umap_knn, "fastEmbedR_config")$method, "umap")
    expect_equal(
        attr(layout_umap_knn, "fastEmbedR_config")$auto_parameter_backend,
        "cpp_knn_distance_profile"
    )

    layout_knn <- tsne_knn(
        knn, perplexity = 1,
        early_exaggeration_iter = 2L, n_iter = 3L
    )
    expect_embedding(layout_knn, n)
    expect_equal(attr(layout_knn, "fastEmbedR_config")$method, "tsne")

    prep_tsne <- prepare_tsne_knn(knn, perplexity = 1)
    expect_s3_class(prep_tsne, "fastEmbedR_tsne_prepared")
    layout_prepared_tsne <- tsne_knn(prep_tsne,
        early_exaggeration_iter = 2L, n_iter = 3L
    )
    expect_embedding(layout_prepared_tsne, n)
    expect_equal(attr(layout_prepared_tsne, "fastEmbedR_config")$method, "tsne")

    prep_umap <- prepare_umap_knn(knn, graph_mode = "binary")
    expect_s3_class(prep_umap, "fastEmbedR_umap_prepared")
    graph_fields <- c(
        "offsets", "neighbors", "weights", "epochs_per_sample"
    )
    expect_true(all(graph_fields %in% names(prep_umap$graph)))
    layout_prepared_umap <- umap_knn(prep_umap)
    expect_embedding(layout_prepared_umap, n)
    prepared_cfg <- attr(layout_prepared_umap, "fastEmbedR_config")
    expect_true(isTRUE(prepared_cfg$prepared_reuse_hit))

    init_umap <- umap_init(prep_umap, seed = 101L)
    expect_s3_class(init_umap, "fastEmbedR_umap_initialization")
    expect_embedding(init_umap$layout, n)
    expect_true(isTRUE(init_umap$parameters$initialization_reusable))
    layout_initialized_umap <- umap_knn(init_umap, seed = 101L)
    expect_embedding(layout_initialized_umap, n)
    expect_true(isTRUE(
        attr(
            layout_initialized_umap,
            "fastEmbedR_config"
        )$initialization_reuse_hit
    ))

    fit <- tsne(x,
        perplexity = 1,
        early_exaggeration_iter = 2L, n_iter = 3L,
        n.cores = 2L
    )
    expect_s3_class(fit, "fastEmbedR_embedding")
    expect_embedding(fit$layout, n)
    expect_equal(fit$parameters$method, "tsne")
    expect_equal(fit$parameters$n.cores, 2L)
    expect_null(fit$knn)
    expect_false("n_neighbors" %in% names(formals(tsne)))

    fit_knn <- tsne(knn,
        perplexity = 1,
        early_exaggeration_iter = 2L, n_iter = 3L
    )
    expect_s3_class(fit_knn, "fastEmbedR_embedding")
    expect_embedding(fit_knn$layout, n)
    expect_equal(fit_knn$parameters$input, "knn")
    expect_equal(fit_knn$metrics$knn_elapsed, 0)

    scores <- evaluate_embedding(
        x,
        fit$layout,
        k = c(4L, 5L),
        sample_size_for_global_metrics = n,
        sample_size_for_local_metrics = n,
        use_cache = FALSE,
        method = "tsne",
        backend = "cpu",
        dataset = "toy"
    )
    expect_true(all(c(
        "trustworthiness", "knn_preservation", "silhouette",
        "density_spearman", "density_log_radius_rmse", "rare_class_recall"
    ) %in% names(scores)))

    gpu_scores <- evaluate_embedding(
        x,
        fit$layout,
        k = c(4L, 5L),
        sample_size_for_global_metrics = n,
        sample_size_for_local_metrics = n,
        use_cache = FALSE,
        method = "tsne",
        backend = "gpu",
        n.cores = 2L,
        dataset = "toy"
    )
    if (isTRUE(fastEmbedR:::cuda_metric_available())) {
        expect_equal(gpu_scores$metric_backend, "cuda")
    } else {
        expect_equal(gpu_scores$metric_backend, "cpu")
        expect_match(
            gpu_scores$metric_backend_reason,
            "metric_backend_unavailable"
        )
    }
})
