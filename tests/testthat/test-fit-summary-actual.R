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

test_that("file-backed landmark fit has bounded output and scope metadata", {
    skip_if(.Platform$OS.type != "unix")
    work <- tempfile("summary-projection-")
    dir.create(work)
    on.exit(unlink(work, recursive = TRUE), add = TRUE)
    source <- file.path(work, "layout.f32")
    values <- matrix(seq_len(20L), nrow = 10L, ncol = 2L)
    writeBin(as.numeric(t(values)), source, size = 4L,
        endian = "little")
    fit <- structure(list(layout = massive_matrix(source, 10L, 2L),
        method = "umap", backend = "cpu", mode = "landmark",
        stage_seconds = list(projection = 0.2)),
        class = "fastEmbedR_massive_projection")
    info <- summary(fit)
    expect_identical(info$backend, "cpu")
    expect_identical(info$scope, "landmark")
    expect_identical(info$stage_seconds$projection, 0.2)
    expect_match(info$output_rows, "massive_read_rows")
    rows <- massive_read_rows(fit, first = 3L, n = 4L)
    expect_identical(dim(rows), c(4L, 2L))
    expect_equal(rows, values[3:6, , drop = FALSE])
})

test_that("file-backed landmark entry points expose workflow stages", {
    skip_if(.Platform$OS.type != "unix")
    work <- tempfile("summary-landmark-")
    dir.create(work)
    on.exit(unlink(work, recursive = TRUE), add = TRUE)
    data <- as.matrix(iris[, 1:4])
    source <- file.path(work, "iris.f32")
    writeBin(as.numeric(t(data)), source, size = 4L,
        endian = "little")
    input <- massive_matrix(source, nrow(data), ncol(data))
    embedding <- umap(input, landmarks = 40L,
        n_neighbors = 10L, n_components = 2L,
        backend = "cpu", n.cores = 2L,
        storage = "file_backed",
        output = file.path(work, "umap.f32"),
        memory_limit = "512MB", seed = 1L)
    info <- summary(embedding)
    expect_identical(info$scope, "landmark")
    expect_identical(names(info$stage_seconds),
        c("selection", "reference_fit", "query_knn",
            "projection"))
    expect_identical(dim(massive_read_rows(embedding, 1L, 5L)),
        c(5L, 2L))
    clusters <- graph_cluster(input, method = "leiden",
        landmarks = 40L, k = 10L, backend = "cpu",
        n.cores = 2L, storage = "file_backed",
        output = file.path(work, "clusters"),
        memory_limit = "512MB", seed = 1L)
    info <- summary(clusters)
    expect_identical(info$scope, "landmark")
    expect_identical(names(info$stage_seconds),
        c("precompute", "cluster_and_assign"))
    expect_length(massive_read_rows(clusters, 1L, 5L)$membership,
        5L)
})
