testthat::test_that("analysis uses four public entry points", {
    exports <- getNamespaceExports("fastEmbedR")
    testthat::expect_true(all(c(
        "pca", "tsne", "umap", "graph_cluster",
        "massive_matrix", "massive_read_rows",
        "massive_full_knn_graph", "massive_umap_fuzzy_graph",
        "massive_tsne_affinities"
    ) %in% exports))
    testthat::expect_false(any(c(
        "massive_cluster", "massive_leiden", "massive_louvain",
        "massive_tsne_optimize", "massive_umap_optimize",
        "massive_select_landmarks", "massive_landmark_knn",
        "cluster_landmarks"
    ) %in% exports))
})
