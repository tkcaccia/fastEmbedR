test_that("file-backed Metal t-SNE runs a 3D graph without fallback", {
    if (!isTRUE(fastEmbedR:::embedding_metal_available_cpp()))
        skip("Metal is unavailable")
    prefix <- tempfile()
    affinity <- tempfile()
    init_path <- tempfile(fileext = ".f32")
    output <- tempfile(fileext = ".f32")
    streamed_path <- tempfile(fileext = ".f32")
    auto_path <- tempfile(fileext = ".f32")
    stream_resume_path <- tempfile(fileext = ".f32")
    bad_path <- tempfile(fileext = ".f32")
    resumed_path <- tempfile(fileext = ".f32")
    paths <- c(paste0(prefix,
        c(".indices.u32", ".distances.f32")),
        paste0(affinity,
            c(".offsets.u64", ".indices.u32", ".weights.f32")),
        paste0(affinity, c(".affinity.rds", ".work")),
        init_path, output, streamed_path, auto_path,
        stream_resume_path, bad_path, resumed_path,
        paste0(c(resumed_path, stream_resume_path),
            ".checkpoint.rds"))
    on.exit(unlink(paths, recursive = TRUE), add = TRUE)
    indices <- outer(1:16, 1:3,
        function(i, j) (i + j - 1L) %% 16L + 1L)
    distances <- matrix(rep(c(0.2, 0.5, 0.9), 16L),
        nrow = 16L, byrow = TRUE)
    writeBin(as.integer(t(indices)), paths[[1L]],
        size = 4L, endian = "little")
    writeBin(as.vector(t(distances)), paths[[2L]],
        size = 4L, endian = "little")
    knn <- massive_knn_graph(prefix, 16L, 3L)
    graph <- massive_tsne_affinities(knn, 3, affinity,
        memory_limit = "64MB")
    initial <- matrix(seq_len(48L) / 10000, 16L, 3L)
    writeBin(as.numeric(t(initial)), init_path,
        size = 4L, endian = "little")
    start <- massive_matrix(init_path, 16L, 3L)
    result <- massive_tsne_optimize(graph, start, output,
        early_exaggeration_iter = 2L, n_iter = 3L,
        learning_rate = 1, memory_limit = "2GB",
        backend = "metal", checkpoint = TRUE,
        checkpoint_every = 2L)
    expect_identical(result$backend, "native_metal_fft_graph_mmap")
    expect_identical(result$resources$graph_access,
        "read_only_mmap")
    expect_identical(result$fft_grid_size, 16L)
    expect_true(all(is.finite(massive_read_rows(result, 1L, 16L))))
    expect_false(file.exists(paste0(output, ".checkpoint.rds")))
    streamed <- massive_tsne_optimize(graph, start, streamed_path,
        early_exaggeration_iter = 2L, n_iter = 3L,
        learning_rate = 1, memory_limit = "2GB",
        backend = "metal", graph_access = "stream",
        edge_chunk_capacity = 8L)
    expect_identical(streamed$backend,
        "native_metal_fft_graph_stream")
    expect_identical(streamed$resources$graph_access,
        "bounded_stream")
    expect_identical(massive_read_rows(streamed, 1L, 16L),
        massive_read_rows(result, 1L, 16L))
    automatic <- massive_tsne_optimize(graph, start, auto_path,
        early_exaggeration_iter = 2L, n_iter = 3L,
        learning_rate = 1, memory_limit = "2GB",
        backend = "metal", edge_chunk_capacity = 8L)
    expect_identical(automatic$backend,
        "native_metal_fft_graph_stream")
    expect_identical(massive_read_rows(automatic, 1L, 16L),
        massive_read_rows(result, 1L, 16L))
    split <- massive_tsne_optimize(graph, start, bad_path,
        early_exaggeration_iter = 2L, n_iter = 3L,
        learning_rate = 1, memory_limit = "2GB",
        backend = "metal", graph_access = "stream",
        edge_chunk_capacity = 1L)
    expect_identical(split$backend,
        "native_metal_fft_graph_stream")
    expect_identical(massive_read_rows(split, 1L, 16L),
        massive_read_rows(result, 1L, 16L))
    commit <- fastEmbedR:::massive_tsne_checkpoint_commit
    pause_after_commit <- function(...) {
        commit(...)
        stop("intentional pause")
    }
    with_mocked_bindings(
        expect_error(massive_tsne_optimize(graph, start,
            resumed_path, early_exaggeration_iter = 2L,
            n_iter = 3L, learning_rate = 1,
            memory_limit = "2GB", backend = "metal",
            checkpoint = TRUE, checkpoint_every = 2L),
            "intentional pause"),
        massive_tsne_checkpoint_commit = pause_after_commit,
        .package = "fastEmbedR")
    expect_true(file.exists(paste0(resumed_path,
        ".checkpoint.rds")))
    resumed <- massive_tsne_optimize(graph, start, resumed_path,
        early_exaggeration_iter = 2L, n_iter = 3L,
        learning_rate = 1, memory_limit = "2GB",
        backend = "metal", checkpoint = TRUE,
        resume = TRUE, checkpoint_every = 2L)
    expect_equal(massive_read_rows(resumed, 1L, 16L),
        massive_read_rows(result, 1L, 16L), tolerance = 1e-3)
    expect_identical(resumed$parameters$resumed_from_iteration, 2L)
    expect_false(file.exists(paste0(resumed_path,
        ".checkpoint.rds")))
    with_mocked_bindings(
        expect_error(massive_tsne_optimize(graph, start,
            stream_resume_path, early_exaggeration_iter = 2L,
            n_iter = 3L, learning_rate = 1,
            memory_limit = "2GB", backend = "metal",
            graph_access = "stream", edge_chunk_capacity = 1L,
            checkpoint = TRUE, checkpoint_every = 2L),
            "intentional pause"),
        massive_tsne_checkpoint_commit = pause_after_commit,
        .package = "fastEmbedR")
    stream_resumed <- massive_tsne_optimize(graph, start,
        stream_resume_path, early_exaggeration_iter = 2L,
        n_iter = 3L, learning_rate = 1,
        memory_limit = "2GB", backend = "metal",
        graph_access = "stream", edge_chunk_capacity = 1L,
        checkpoint = TRUE, resume = TRUE,
        checkpoint_every = 2L)
    expect_identical(massive_read_rows(stream_resumed, 1L, 16L),
        massive_read_rows(split, 1L, 16L))
    expect_identical(stream_resumed$parameters$resumed_from_iteration,
        2L)
})

test_that("file-backed Metal t-SNE maps and streams a 2D graph", {
    if (!isTRUE(fastEmbedR:::embedding_metal_available_cpp()))
        skip("Metal is unavailable")
    root <- tempfile()
    dir.create(root)
    on.exit(unlink(root, recursive = TRUE), add = TRUE)
    prefix <- file.path(root, "knn")
    affinity <- file.path(root, "affinity")
    indices <- outer(1:16, 1:3,
        function(i, j) (i + j - 1L) %% 16L + 1L)
    distances <- matrix(rep(c(0.2, 0.5, 0.9), 16L),
        nrow = 16L, byrow = TRUE)
    writeBin(as.integer(t(indices)),
        paste0(prefix, ".indices.u32"), size = 4L,
        endian = "little")
    writeBin(as.vector(t(distances)),
        paste0(prefix, ".distances.f32"), size = 4L,
        endian = "little")
    knn <- massive_knn_graph(prefix, 16L, 3L)
    graph <- massive_tsne_affinities(knn, 3, affinity,
        memory_limit = "64MB")
    init_path <- file.path(root, "init.f32")
    writeBin(as.numeric(t(matrix(seq_len(32L) / 10000,
        16L, 2L))), init_path, size = 4L,
        endian = "little")
    start <- massive_matrix(init_path, 16L, 2L)
    expect_error(massive_tsne_optimize(graph, start,
        file.path(root, "invalid-cpu.f32"),
        early_exaggeration_iter = 2L, n_iter = 3L,
        backend = "cpu", graph_access = "stream"),
        "Metal graph controls require")
    mapped <- massive_tsne_optimize(graph, start,
        file.path(root, "mapped.f32"),
        early_exaggeration_iter = 2L, n_iter = 3L,
        learning_rate = 1, memory_limit = "2GB",
        backend = "metal", graph_access = "mmap")
    streamed <- massive_tsne_optimize(graph, start,
        file.path(root, "streamed.f32"),
        early_exaggeration_iter = 2L, n_iter = 3L,
        learning_rate = 1, memory_limit = "2GB",
        backend = "metal", graph_access = "stream",
        edge_chunk_capacity = 8L)
    y <- massive_read_rows(mapped, 1L, 16L)
    z <- massive_read_rows(streamed, 1L, 16L)
    expect_identical(dim(y), c(16L, 2L))
    expect_true(all(is.finite(y)))
    expect_true(all(is.finite(z)))
    expect_identical(mapped$backend, "native_metal_fft_graph_mmap")
    expect_identical(streamed$backend,
        "native_metal_fft_graph_stream")
    expect_identical(mapped$fft_grid_size, 128L)
    expect_identical(streamed$resources$graph_access,
        "bounded_stream")
    expect_identical(z, y)
    commit <- fastEmbedR:::massive_tsne_checkpoint_commit
    pause <- function(...) {
        commit(...)
        stop("intentional pause")
    }
    resume_path <- file.path(root, "resume.f32")
    with_mocked_bindings(
        expect_error(massive_tsne_optimize(graph, start,
            resume_path, early_exaggeration_iter = 2L,
            n_iter = 3L, learning_rate = 1,
            memory_limit = "2GB", backend = "metal",
            graph_access = "stream", edge_chunk_capacity = 8L,
            checkpoint = TRUE, checkpoint_every = 2L),
            "intentional pause"),
        massive_tsne_checkpoint_commit = pause,
        .package = "fastEmbedR")
    resumed <- massive_tsne_optimize(graph, start, resume_path,
        early_exaggeration_iter = 2L, n_iter = 3L,
        learning_rate = 1, memory_limit = "2GB",
        backend = "metal", graph_access = "stream",
        edge_chunk_capacity = 8L, checkpoint = TRUE,
        resume = TRUE, checkpoint_every = 2L)
    expect_identical(massive_read_rows(resumed, 1L, 16L), z)
    expect_identical(resumed$parameters$resumed_from_iteration,
        2L)
})
