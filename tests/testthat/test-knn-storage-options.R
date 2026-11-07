test_that("file-backed KNN uses the public route and session budget", {
    x <- matrix(c(0, 0, 1, 0, 0, 2, 3, 0, 3, 3, 4, 4), ncol = 2L,
        byrow = TRUE)
    path <- tempfile(fileext = ".f32")
    directory <- tempfile()
    dir.create(directory)
    on.exit(unlink(c(path, directory), recursive = TRUE))
    con <- file(path, "wb")
    writeBin(as.numeric(t(x)), con, size = 4L, endian = "little")
    close(con)
    input <- massive_matrix(path, nrow = nrow(x), ncol = ncol(x))
    previous <- options(backend = "cpu", storage = "file_backed",
        memory_limit = "128MB")
    on.exit(options(previous), add = TRUE)
    graph <- precompute_knn(input, k = 2L,
        output = file.path(directory, "knn"))
    expect_s3_class(graph, "fastEmbedR_massive_graph")
    expect_equal(graph$n_vertices, nrow(x))
    expect_equal(graph$k, 2L)
    expect_true(isTRUE(graph$exact))
    expect_true(file.exists(file.path(directory, "knn.indices.u32")))
    expect_true(file.exists(file.path(directory, "knn.distances.f32")))
    indices <- matrix(readBin(graph$indices_path, integer(),
        n = nrow(x) * 2L, size = 4L, endian = "little"),
        nrow = nrow(x), byrow = TRUE)
    distances <- matrix(readBin(graph$values_path, numeric(),
        n = nrow(x) * 2L, size = 4L, endian = "little"),
        nrow = nrow(x), byrow = TRUE)
    for (row in seq_len(nrow(x))) {
        exact <- sqrt(rowSums(sweep(x, 2L, x[row, ], "-")^2))
        exact[[row]] <- Inf
        expected <- order(exact)[seq_len(2L)]
        expect_equal(indices[row, ], expected)
        expect_equal(distances[row, ], exact[expected], tolerance = 1e-5)
    }
})

test_that("storage options fail before hidden conversion", {
    x <- matrix(seq_len(24), ncol = 3L)
    previous <- options(storage = "file_backed")
    on.exit(options(previous))
    expect_error(precompute_knn(x, k = 2L, backend = "cpu"),
        "not silently written to disk")
    expect_error(pca(x, backend = "cpu"),
        "not silently written to disk")
    expect_error(tsne(x, backend = "cpu"),
        "not silently written to disk")
    expect_error(umap(x, backend = "cpu"),
        "not silently written to disk")
    expect_error(graph_cluster(x, backend = "cpu"),
        "not silently written to disk")
    resident <- precompute_knn(x, k = 2L, backend = "cpu",
        storage = "resident")
    expect_s3_class(resident, "fastEmbedR_knn")
    expect_error(pca(x, storage = "typo"), "Storage must")
    expect_error(tsne(x, storage = "typo"), "Storage must")
    expect_error(umap(x, storage = "typo"), "Storage must")
    expect_error(graph_cluster(x, storage = "typo"), "Storage must")
    options(storage = "auto")
    expect_error(precompute_knn(x, k = 2L, backend = "cpu",
        output = tempfile()), "File-backed KNN controls")
    expect_error(precompute_knn(x, k = 2L, backend = "cpu",
        storage = "typo"), "Storage must")
})

test_that("file-backed KNN requires output and supported metric", {
    x <- matrix(seq_len(24), ncol = 3L)
    path <- tempfile(fileext = ".f32")
    on.exit(unlink(path))
    con <- file(path, "wb")
    writeBin(as.numeric(t(x)), con, size = 4L, endian = "little")
    close(con)
    input <- massive_matrix(path, nrow = nrow(x), ncol = ncol(x))
    expect_error(precompute_knn(input, k = 2L, backend = "cpu"),
        "requires a new `output`")
    expect_error(precompute_knn(input, k = 2L, backend = "cpu",
        output = tempfile(), metric = "cosine"), "Euclidean")
    expect_error(knn_graph(input, k = 2L), "Use precompute_knn")
    graph <- structure(list(), class = "fastEmbedR_massive_graph")
    expect_error(precompute_knn(graph, output = tempfile()),
        "needs a data matrix")
    previous <- options(memory_limit = "invalid")
    on.exit(options(previous), add = TRUE)
    expect_error(precompute_knn(input, k = 2L, backend = "cpu",
        output = tempfile()), "memory_limit")
})

test_that("file-backed KNN routing respects work and session options", {
    testthat::local_mocked_bindings(
        massive_full_knn_graph = function(...) list(...),
        .package = "fastEmbedR")
    previous <- options(n.cores = 1L, storage = "file_backed",
        memory_limit = "128MB")
    on.exit(options(previous))
    source <- function(rows, columns) structure(
        list(format = "f32", nrow = rows, ncol = columns),
        class = "fastEmbedR_massive_matrix")
    small <- precompute_knn(source(1000, 10), k = 30L,
        backend = "cpu", output = "small")
    expect_identical(small$method, "exact")
    expect_identical(small$memory_limit, "128MB")
    expect_identical(small$n.cores, 1L)
    expect_identical(precompute_knn(source(6000, 10), k = 30L,
        backend = "cpu", output = "large")$method, "hnsw_sharded")
    expect_identical(precompute_knn(source(2000, 50000), k = 30L,
        backend = "cpu", output = "wide")$method, "hnsw_sharded")
    expect_identical(precompute_knn(source(70000, 784), k = 30L,
        backend = "cuda", output = "cuda")$method, "ivf_sharded")
    expect_identical(precompute_knn(source(1000, 10), k = 30L,
        backend = "metal", output = "metal")$method, "exact")
})
