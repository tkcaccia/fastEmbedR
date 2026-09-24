test_that("backend precedence is explicit, option, environment, CPU", {
    old_options <- options(backend = NULL)
    old_env <- Sys.getenv("FASTEMBEDR_BACKEND", unset = NA_character_)
    on.exit({
        options(old_options)
        if (is.na(old_env)) {
            Sys.unsetenv("FASTEMBEDR_BACKEND")
        } else {
            Sys.setenv(FASTEMBEDR_BACKEND = old_env)
        }
    })
    Sys.unsetenv("FASTEMBEDR_BACKEND")
    expect_identical(fastEmbedR:::resolve_embedding_backend(NULL), "cpu")
    Sys.setenv(FASTEMBEDR_BACKEND = "metal")
    expect_identical(fastEmbedR:::resolve_embedding_backend(NULL), "metal")
    options(backend = "cuda")
    expect_identical(fastEmbedR:::resolve_embedding_backend(NULL), "cuda")
    expect_identical(fastEmbedR:::resolve_embedding_backend("cpu"), "cpu")
    expect_error(
        fastEmbedR:::resolve_embedding_backend("auto"),
        "must be one of"
    )
})

test_that("n.cores option configures omitted CPU core arguments", {
    old_options <- options(n.cores = 3L)
    on.exit(options(old_options))
    expect_identical(fastEmbedR:::resolve_n_cores(), 3L)
    expect_identical(fastEmbedR:::normalize_nn_threads(NULL), 3L)
    expect_identical(fastEmbedR:::default_tsne_threads(), 3L)
    expect_identical(fastEmbedR:::normalize_pca_threads(NULL), 3L)
    expect_identical(fastEmbedR:::normalize_nn_threads(2L), 2L)
    cfg <- fastEmbedR:::fast_knn_umap_config(100L, 15L, "cpu")
    cfg <- fastEmbedR:::apply_umap_thread_override(cfg, NULL)
    expect_identical(cfg$n.cores_requested, 3L)
    expect_identical(cfg$n.cores_effective, 3L)
})

test_that("principal public functions defer omitted session settings", {
    functions <- list(
        umap, tsne, pca, knn_graph, graph_cluster,
        transform_tsne, precompute_knn, precompute_query_knn
    )
    expect_true(all(vapply(
        functions,
        function(fn) is.null(formals(fn)$backend),
        logical(1L)
    )))
    threaded <- list(
        umap, tsne, pca, knn_graph, transform_tsne,
        precompute_knn, precompute_query_knn
    )
    expect_true(all(vapply(
        threaded,
        function(fn) is.null(formals(fn)$n.cores),
        logical(1L)
    )))
})
