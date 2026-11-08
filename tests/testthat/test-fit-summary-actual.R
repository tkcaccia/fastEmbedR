test_that("four resident entry points return a common summary schema", {
    x <- as.matrix(iris[, 1:4])
    fitted <- list(
        pca = pca(x, ncomp = 2L, backend = "cpu"),
        tsne = tsne(x, perplexity = 5, affinity_k = 15L,
            n_iter = 20L, early_exaggeration_iter = 5L,
            backend = "cpu"),
        umap = umap(x, n_neighbors = 10L, backend = "cpu"),
        cluster = graph_cluster(x, method = "louvain",
            k = 10L, backend = "cpu"))
    required <- c("backend", "backend_engine", "devices",
        "device_identity", "scope", "stage_backend",
        "stage_seconds", "memory_use", "output_rows")
    for (fit in fitted) {
        info <- summary(fit)
        expect_identical(names(info), required)
        expect_match(info$backend, "cpu")
        expect_identical(info$scope, "full")
        expect_true(is.list(info$stage_seconds))
        expect_true(is.list(info$memory_use))
    }
})

test_that("file-backed full-graph cluster summary records fit time", {
    skip_if(.Platform$OS.type != "unix")
    x <- as.matrix(iris[, 1:4])
    work <- tempfile("summary-cluster-")
    dir.create(work)
    on.exit(unlink(work, recursive = TRUE), add = TRUE)
    source <- file.path(work, "input.f32")
    writeBin(as.numeric(t(x)), source, size = 4L,
        endian = "little")
    input <- massive_matrix(source, nrow(x), ncol(x))
    knn <- precompute_knn(input, k = 10L, backend = "cpu",
        output = file.path(work, "knn"), memory_limit = "512MB")
    graph <- massive_umap_fuzzy_graph(knn,
        output = file.path(work, "fuzzy"),
        memory_limit = "512MB", sort_method = "bucketed")
    for (method in c("leiden", "louvain")) {
        fit <- graph_cluster(graph, method = method,
            backend = "cpu", n.cores = 2L,
            parallel.mode = "fit", storage = "file_backed",
            output = file.path(work,
                paste0(method, ".clusters.u32")),
            memory_limit = "512MB")
        info <- summary(fit)
        expect_identical(info$backend, "cpu")
        expect_identical(info$scope, "full")
        expect_true(is.finite(info$stage_seconds$fit))
        expect_gt(info$stage_seconds$fit, 0)
        expect_length(massive_read_rows(fit, 1L, 5L), 5L)
    }
})
