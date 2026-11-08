test_that("resident materialization preserves float32 across tiles", {
    skip_if_not_installed("float")
    rows <- 11000L
    columns <- 96L
    values <- matrix(seq_len(rows * columns) / 1000,
        nrow = rows, ncol = columns)
    path <- tempfile(fileext = ".fbin")
    on.exit(unlink(path))
    stream <- file(path, "wb")
    writeBin(as.integer(c(rows, columns)), stream,
        size = 4L, endian = "little")
    writeBin(as.vector(t(values)), stream,
        size = 4L, endian = "little")
    close(stream)
    source <- massive_matrix(path)
    resident <- fastEmbedR:::massive_auto_materialize(source)
    expect_s4_class(resident, "float32")
    expect_equal(nrow(resident), rows)
    expect_equal(ncol(resident), columns)
    expect_equal(float::dbl(resident), values, tolerance = 1e-4)
    block <- fastEmbedR:::resident_float32_row_major_block_cpp(
        resident, 10920L, 50L)
    packed <- matrix(readBin(block, numeric(), n = 50L * columns,
        size = 4L, endian = "little"), nrow = 50L, byrow = TRUE)
    expect_equal(packed, values[10920:10969, ], tolerance = 1e-4)
    tiled <- fastEmbedR:::resident_float32_row_major_block_cpp(
        resident, 10700L, 300L)
    packed_tiles <- matrix(readBin(tiled, numeric(),
        n = 300L * columns, size = 4L, endian = "little"),
        nrow = 300L, byrow = TRUE)
    expect_equal(packed_tiles, values[10700:10999, ],
        tolerance = 1e-4)
    column_block <- fastEmbedR:::resident_float32_column_major_block_cpp(
        resident, 10700L, 300L)
    column_values <- matrix(readBin(column_block, numeric(),
        n = 300L * columns, size = 4L, endian = "little"),
        nrow = 300L)
    expect_equal(column_values, values[10700:10999, ],
        tolerance = 1e-4)
    expect_error(fastEmbedR:::resident_float32_row_major_block_cpp(
        resident, rows, 2L), "row range")
    scores <- fastEmbedR:::resident_float32_score_matrix_cpp(rows, 2L)
    score_block <- values[10920:10969, 1:2]
    scores <- fastEmbedR:::resident_float32_score_assign_cpp(
        scores, 10920L, score_block)
    expect_s4_class(scores, "float32")
    expect_equal(float::dbl(scores)[10920:10969, ],
        score_block, tolerance = 1e-4)
    expect_error(fastEmbedR:::resident_float32_score_assign_cpp(
        scores, rows, score_block), "dimensions")
    expect_error(fastEmbedR:::fastembedr_explicit_resident_data(
        source, "resident", "1MB"), "memory limit")
})

test_that("four entry points explicitly load float32 file input", {
    skip_if_not_installed("float")
    set.seed(87L)
    values <- matrix(rnorm(160L), 40L, 4L)
    path <- tempfile(fileext = ".fbin")
    on.exit(unlink(path))
    stream <- file(path, "wb")
    writeBin(as.integer(c(40L, 4L)), stream,
        size = 4L, endian = "little")
    writeBin(as.vector(t(values)), stream,
        size = 4L, endian = "little")
    close(stream)
    source <- massive_matrix(path)
    expect_error(pca(source, massive = "out_of_core",
        storage = "resident", memory_limit = "64MB"),
        "conflicts")
    expect_error(tsne(source, massive = "landmark",
        storage = "resident", memory_limit = "64MB"),
        "conflicts")
    expect_error(umap(source, massive = "landmark",
        storage = "resident", memory_limit = "64MB"),
        "conflicts")
    expect_error(pca(source, storage = "resident",
        output = tempfile(), memory_limit = "64MB"),
        "conflicts")
    expect_error(tsne(source, storage = "resident",
        output = tempfile(), memory_limit = "64MB"),
        "conflicts")
    expect_error(umap(source, storage = "resident",
        output = tempfile(), memory_limit = "64MB"),
        "conflicts")
    expect_error(graph_cluster(source, storage = "resident",
        output = tempfile(), memory_limit = "64MB"),
        "conflicts")
    previous <- options(memory_limit = 100)
    on.exit(options(previous), add = TRUE)
    expect_error(pca(source, storage = "resident"),
        "memory limit")
    expect_error(tsne(source, storage = "resident"),
        "memory limit")
    expect_error(umap(source, storage = "resident"),
        "memory limit")
    expect_error(graph_cluster(source, storage = "resident"),
        "memory limit")
    options(previous)
    fit_pca <- pca(source, backend = "cpu", n.cores = 2L,
        storage = "resident", memory_limit = "64MB")
    fit_tsne <- tsne(source, backend = "cpu", n.cores = 2L,
        perplexity = 5, n_iter = 10L, seed = 87L,
        storage = "resident", memory_limit = "64MB")
    fit_umap <- umap(source, backend = "cpu", n.cores = 2L,
        n_neighbors = 5L, seed = 87L,
        storage = "resident", memory_limit = "64MB")
    fit_cluster <- graph_cluster(source, backend = "cpu",
        method = "louvain", n.cores = 2L, k = 5L,
        storage = "resident", memory_limit = "64MB")
    expect_equal(dim(fit_pca$scores), c(40L, 2L))
    expect_equal(dim(fit_tsne$layout), c(40L, 2L))
    expect_equal(dim(fit_umap$layout), c(40L, 2L))
    expect_length(fit_cluster$membership, 40L)
})

test_that("resident embedding preserves its admitted memory budget", {
    source <- structure(list(format = "fbin", nrow = 40L,
        ncol = 4L), class = "fastEmbedR_massive_matrix")
    with_mocked_bindings({
        explicit <- fastEmbedR:::fastembedr_embedding_storage(source,
            NULL, "off", FALSE, "64MB", FALSE, "t-SNE",
            storage = "resident")
        expect_identical(explicit$memory_limit, "64MB")
        expect_identical(explicit$control_memory_limit, "8GB")
        previous <- options(memory_limit = "96MB")
        on.exit(options(previous))
        defaulted <- fastEmbedR:::fastembedr_embedding_storage(source,
            NULL, "off", FALSE, "8GB", TRUE, "UMAP",
            storage = "resident")
        expect_identical(defaulted$memory_limit, "96MB")
        expect_identical(defaulted$control_memory_limit, "8GB")
    }, fastembedr_explicit_resident_data = function(...) {
        matrix(0, 40L, 4L)
    }, .package = "fastEmbedR")
})
