test_that("file-backed full UMAP runs through the main API", {
    events <- character()
    input <- structure(list(format = "f32", nrow = 100L, ncol = 4L),
        class = "fastEmbedR_massive_matrix")
    coordinates <- structure(list(format = "f32", nrow = 100L,
        ncol = 2L), class = "fastEmbedR_massive_matrix")
    graph <- structure(list(n_vertices = 100L),
        class = "fastEmbedR_massive_graph")
    testthat::local_mocked_bindings(
        massive_float_output = function(output, bytes, ...) output,
        massive_memory_bytes = function(value) 8 * 1024^3,
        precompute_knn = function(...) {
            events <<- c(events, "knn")
            graph
        },
        massive_umap_fuzzy_graph = function(...) {
            events <<- c(events, "fuzzy")
            graph
        },
        massive_spectral_init_stream_cpp = function(...) {
            events <<- c(events, "spectral")
            list(backend_used = "native_cpu_streamed_spectral")
        },
        massive_matrix = function(...) coordinates,
        run_massive_umap_graph = function(...) {
            events <<- c(events, "fit")
            list(layout = coordinates)
        },
        .package = "fastEmbedR")
    fit <- umap(input, n_neighbors = 5L, output = "umap.f32",
        backend = "cpu", storage = "file_backed")
    expect_identical(events, c("knn", "fuzzy", "spectral", "fit"))
    expect_identical(fit$pipeline$fit_scope, "all_rows_optimized")
    expect_identical(fit$pipeline$stage_backend[["init"]], "cpu")
    expect_identical(fit$pipeline$initialization$method,
        "streamed_graph_spectral")
    expect_identical(names(fit$pipeline$stage_seconds),
        c("knn", "graph", "init", "optimize"))
    events <- character()
    expect_error(umap(input, n_neighbors = 5L, output = "bad.f32",
        backend = "cpu", metric = "cosine", storage = "file_backed"),
        "Euclidean")
    expect_length(events, 0L)
    expect_error(umap(input, n_neighbors = 5L, output = "bad.f32",
        backend = "cpu", storage = "file_backed",
        layout_storage = "managed"), "requires CUDA")
    expect_length(events, 0L)
    path <- tempfile(fileext = ".f32")
    occupied <- paste0(sub("\\.f32$", "", path),
        ".knn.indices.u32")
    file.create(occupied)
    on.exit(unlink(occupied), add = TRUE)
    expect_error(umap(input, n_neighbors = 5L, output = path,
        backend = "cpu", storage = "file_backed"),
        "intermediate output already exists")
    expect_length(events, 0L)
})

test_that("file-backed full t-SNE runs through the main API", {
    events <- character()
    input <- structure(list(format = "f32", nrow = 100L, ncol = 4L),
        class = "fastEmbedR_massive_matrix")
    coordinates <- structure(list(format = "f32", nrow = 100L,
        ncol = 3L), class = "fastEmbedR_massive_matrix")
    graph <- structure(list(n_vertices = 100L, perplexity = 5),
        class = "fastEmbedR_massive_graph")
    testthat::local_mocked_bindings(
        massive_float_output = function(output, bytes, ...) output,
        massive_memory_bytes = function(value) 8 * 1024^3,
        precompute_knn = function(...) {
            events <<- c(events, "knn")
            graph
        },
        massive_tsne_affinities = function(...) {
            events <<- c(events, "affinity")
            graph
        },
        pca = function(...) {
            events <<- c(events, "pca")
            list(tsne_init = coordinates)
        },
        run_massive_tsne_graph = function(...) {
            events <<- c(events, "fit")
            list(layout = coordinates)
        },
        .package = "fastEmbedR")
    fit <- tsne(input, perplexity = 5, affinity_k = 6L,
        n_components = 3L, output = "tsne.f32",
        backend = "cpu", storage = "file_backed")
    expect_identical(events, c("knn", "affinity", "pca", "fit"))
    expect_identical(fit$pipeline$fit_scope, "all_rows_optimized")
    events <- character()
    expect_error(tsne(input, perplexity = 5, n_components = 3L,
        output = "bad.f32", backend = "cpu",
        storage = "file_backed", resume = TRUE),
        "requires `checkpoint = TRUE`")
    expect_length(events, 0L)
})
