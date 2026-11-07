test_that("one-call full UMAP resumes after graph-stage failure", {
    skip_on_os("windows")
    x <- as.matrix(iris[, 1:4])
    work <- tempfile("full-umap-recovery-")
    dir.create(work)
    on.exit(unlink(work, recursive = TRUE), add = TRUE)
    input_path <- file.path(work, "input.f32")
    writeBin(as.numeric(t(x)), input_path, size = 4L,
        endian = "little")
    input <- massive_matrix(input_path, nrow(x), ncol(x))
    output <- file.path(work, "resumed.f32")
    run <- function(path, resume = FALSE) umap(input,
        n_neighbors = 10L, n_components = 2L,
        backend = "cpu", n.cores = 1L, seed = 11L,
        storage = "file_backed", output = path,
        memory_limit = "512MB", checkpoint = TRUE,
        resume = resume)
    expect_error(testthat::with_mocked_bindings(
        run(output),
        massive_umap_fuzzy_graph = function(...) stop(
            "injected graph interruption"),
        .package = "fastEmbedR"),
        "injected graph interruption")
    sidecar <- sub("\\.f32$", ".pipeline.rds", output)
    expect_true(file.exists(sidecar))
    expect_error(umap(input, n_neighbors = 11L,
        n_components = 2L, backend = "cpu", n.cores = 1L,
        seed = 11L, storage = "file_backed", output = output,
        memory_limit = "512MB", checkpoint = TRUE,
        resume = TRUE), "checkpoint does not match")
    resumed <- run(output, resume = TRUE)
    reference <- run(file.path(work, "reference.f32"))
    expect_false(file.exists(sidecar))
    expect_false(file.exists(sub("\\.f32$",
        ".knn.indices.u32", output)))
    expect_identical(resumed$pipeline$fit_scope,
        "all_rows_optimized")
    expect_identical(resumed$pipeline$initialization$method,
        "streamed_graph_spectral")
    expect_identical(resumed$pipeline$stage_backend[["init"]], "cpu")
    expect_equal(massive_read_rows(resumed, 1L, nrow(x)),
        massive_read_rows(reference, 1L, nrow(x)),
        tolerance = 1e-4)
})

test_that("streamed graph spectral initialization is bounded and stable", {
    skip_on_os("windows")
    x <- as.matrix(iris[, 1:4])
    work <- tempfile("streamed-spectral-")
    dir.create(work)
    on.exit(unlink(work, recursive = TRUE), add = TRUE)
    input_path <- file.path(work, "input.f32")
    writeBin(as.numeric(t(x)), input_path, size = 4L,
        endian = "little")
    input <- massive_matrix(input_path, nrow(x), ncol(x))
    fit <- umap(input, n_neighbors = 10L, backend = "cpu",
        output = file.path(work, "fit.f32"),
        storage = "file_backed", memory_limit = "512MB")
    graph <- fit$pipeline$graph
    native <- fastEmbedR:::massive_spectral_init_stream_cpp
    args <- list(graph$offsets_path, graph$indices_path,
        graph$weights_path, nrow(x), file.path(work, "low.f32"),
        2L, 10L, 11L, 64L, 1024)
    expect_error(do.call(native, args), "RAM budget")
    expect_false(file.exists(args[[5L]]))
    args[[5L]] <- file.path(work, "first.f32")
    args[[10L]] <- 512 * 1024^2
    first <- do.call(native, args)
    args[[5L]] <- file.path(work, "second.f32")
    do.call(native, args)
    expect_identical(first$backend_used,
        "native_cpu_streamed_spectral")
    expect_identical(first$graph_passes, 12L)
    read_all <- function(path) readBin(path, "raw",
        n = file.info(path)$size)
    expect_identical(read_all(file.path(work, "first.f32")),
        read_all(file.path(work, "second.f32")))
})

test_that("one-call spectral initialization rejects a changing graph", {
    skip_on_os("windows")
    x <- as.matrix(iris[, 1:4])
    work <- tempfile("spectral-source-change-")
    dir.create(work)
    on.exit(unlink(work, recursive = TRUE), add = TRUE)
    input_path <- file.path(work, "input.f32")
    writeBin(as.numeric(t(x)), input_path, size = 4L,
        endian = "little")
    input <- massive_matrix(input_path, nrow(x), ncol(x))
    fit <- umap(input, n_neighbors = 10L, backend = "cpu",
        output = file.path(work, "fit.f32"),
        storage = "file_backed", memory_limit = "512MB")
    paths <- list(spectral = file.path(work, "changed.f32"))
    expect_error(testthat::with_mocked_bindings(
        fastEmbedR:::full_backed_pipeline_init(input, NULL,
            paths, 2L, "cpu", 1L, 11L, "512MB", NULL,
            graph = fit$pipeline$graph),
        massive_spectral_init_stream_cpp = function(...) {
            weights <- list(...)[[3L]]
            Sys.setFileTime(weights, Sys.time() + 10)
            list(backend_used = "cpu")
        }, .package = "fastEmbedR"),
        "graph changed during spectral initialization")
})

test_that("one-call UMAP retires only its incomplete spectral part", {
    skip_on_os("windows")
    x <- as.matrix(iris[, 1:4])
    work <- tempfile("spectral-part-recovery-")
    dir.create(work)
    on.exit(unlink(work, recursive = TRUE), add = TRUE)
    input_path <- file.path(work, "input.f32")
    writeBin(as.numeric(t(x)), input_path, size = 4L,
        endian = "little")
    input <- massive_matrix(input_path, nrow(x), ncol(x))
    output <- file.path(work, "resumed.f32")
    run <- function(path, resume = FALSE) umap(input,
        n_neighbors = 10L, backend = "cpu", seed = 11L,
        storage = "file_backed", output = path,
        memory_limit = "512MB", checkpoint = TRUE,
        resume = resume)
    expect_error(testthat::with_mocked_bindings(
        run(output),
        massive_spectral_init_stream_cpp = function(...) {
            path <- list(...)[[5L]]
            file.create(paste0(path, ".part"))
            stop("injected spectral interruption")
        }, .package = "fastEmbedR"),
        "injected spectral interruption")
    partial <- sub("\\.f32$", ".spectral.f32.part", output)
    expect_true(file.exists(partial))
    resumed <- run(output, resume = TRUE)
    reference <- run(file.path(work, "reference.f32"))
    expect_false(file.exists(partial))
    expect_equal(massive_read_rows(resumed, 1L, nrow(x)),
        massive_read_rows(reference, 1L, nrow(x)),
        tolerance = 1e-4)
})

test_that("one-call full UMAP rejects occupied membership artifacts", {
    skip_on_os("windows")
    work <- tempfile("full-umap-occupied-")
    dir.create(work)
    on.exit(unlink(work, recursive = TRUE), add = TRUE)
    input_path <- file.path(work, "input.f32")
    writeBin(as.numeric(t(as.matrix(iris[, 1:4]))),
        input_path, size = 4L, endian = "little")
    input <- massive_matrix(input_path, nrow(iris), 4L)
    output <- file.path(work, "fit.f32")
    for (suffix in c(".weights.f32", ".weights.f32.part",
            ".weights.f32.checkpoint.rds.part")) {
        occupied <- file.path(work, paste0("fit.knn", suffix))
        file.create(occupied)
        expect_error(fastEmbedR:::full_backed_pipeline_paths(
            input, output, 2L, "512MB"),
            "intermediate output already exists")
        unlink(occupied)
    }
})

test_that("one-call UMAP preserves completed graph-stage timing", {
    skip_on_os("windows")
    work <- tempfile("full-umap-graph-time-")
    dir.create(work)
    on.exit(unlink(work, recursive = TRUE), add = TRUE)
    input_path <- file.path(work, "input.f32")
    x <- as.matrix(iris[, 1:4])
    writeBin(as.numeric(t(x)), input_path, size = 4L,
        endian = "little")
    input <- massive_matrix(input_path, nrow(x), ncol(x))
    output <- file.path(work, "fit.f32")
    run <- function(resume = FALSE) umap(input,
        n_neighbors = 10L, n_components = 2L,
        backend = "cpu", n.cores = 1L, seed = 11L,
        storage = "file_backed", output = output,
        memory_limit = "512MB", checkpoint = TRUE,
        resume = resume)
    expect_error(testthat::with_mocked_bindings(
        run(), full_backed_pipeline_init = function(...) stop(
            "injected init interruption"), .package = "fastEmbedR"),
        "injected init interruption")
    state <- readRDS(sub("\\.f32$", ".pipeline.rds", output))
    saved <- state$stages$membership$seconds +
        state$stages$graph$seconds
    expect_true(is.finite(saved) && saved > 0)
    fit <- run(resume = TRUE)
    expect_equal(unname(fit$pipeline$stage_seconds[["graph"]]),
        saved, tolerance = 1e-8)
})

test_that("one-call full 3D t-SNE resumes after graph failure", {
    skip_on_os("windows")
    x <- as.matrix(iris[, 1:4])
    work <- tempfile("full-tsne-recovery-")
    dir.create(work)
    on.exit(unlink(work, recursive = TRUE), add = TRUE)
    input_path <- file.path(work, "input.f32")
    writeBin(as.numeric(t(x)), input_path, size = 4L,
        endian = "little")
    input <- massive_matrix(input_path, nrow(x), ncol(x))
    output <- file.path(work, "resumed.f32")
    run <- function(path, resume = FALSE) tsne(input,
        perplexity = 5, affinity_k = 10L,
        n_components = 3L, n_iter = 20L,
        early_exaggeration_iter = 5L, backend = "cpu",
        n.cores = 1L, seed = 11L, storage = "file_backed",
        output = path, memory_limit = "2GB",
        checkpoint = TRUE, resume = resume)
    expect_error(testthat::with_mocked_bindings(
        run(output),
        massive_tsne_affinities = function(...) stop(
            "injected affinity interruption"),
        .package = "fastEmbedR"),
        "injected affinity interruption")
    sidecar <- sub("\\.f32$", ".pipeline.rds", output)
    expect_true(file.exists(sidecar))
    resumed <- run(output, resume = TRUE)
    reference <- run(file.path(work, "reference.f32"))
    expect_false(file.exists(sidecar))
    expect_false(file.exists(sub("\\.f32$",
        ".knn.indices.u32", output)))
    expect_identical(resumed$pipeline$fit_scope,
        "all_rows_optimized")
    expect_equal(massive_read_rows(resumed, 1L, nrow(x)),
        massive_read_rows(reference, 1L, nrow(x)),
        tolerance = 1e-4)
})

test_that("one-call full UMAP resumes a saved optimizer epoch", {
    skip_on_os("windows")
    x <- as.matrix(iris[, 1:4])
    work <- tempfile("full-umap-epoch-")
    dir.create(work)
    on.exit(unlink(work, recursive = TRUE), add = TRUE)
    input_path <- file.path(work, "input.f32")
    writeBin(as.numeric(t(x)), input_path, size = 4L,
        endian = "little")
    input <- massive_matrix(input_path, nrow(x), ncol(x))
    output <- file.path(work, "resumed.f32")
    run <- function(path, resume = FALSE) umap(input,
        n_neighbors = 10L, n_components = 2L,
        backend = "cpu", n.cores = 1L, seed = 11L,
        storage = "file_backed", output = path,
        memory_limit = "512MB", checkpoint = TRUE,
        resume = resume)
    native <- fastEmbedR:::massive_umap_checkpoint_commit
    expect_error(testthat::with_mocked_bindings(
        run(output),
        massive_umap_checkpoint_commit = function(...) {
            native(...)
            stop("injected optimizer interruption")
        }, .package = "fastEmbedR"),
        "injected optimizer interruption")
    expect_true(file.exists(paste0(output, ".checkpoint.rds")))
    resumed <- run(output, resume = TRUE)
    reference <- run(file.path(work, "reference.f32"))
    expect_equal(massive_read_rows(resumed, 1L, nrow(x)),
        massive_read_rows(reference, 1L, nrow(x)),
        tolerance = 1e-4)
})

test_that("one-call full 3D t-SNE resumes saved optimizer state", {
    skip_on_os("windows")
    x <- as.matrix(iris[, 1:4])
    work <- tempfile("full-tsne-iteration-")
    dir.create(work)
    on.exit(unlink(work, recursive = TRUE), add = TRUE)
    input_path <- file.path(work, "input.f32")
    writeBin(as.numeric(t(x)), input_path, size = 4L,
        endian = "little")
    input <- massive_matrix(input_path, nrow(x), ncol(x))
    output <- file.path(work, "resumed.f32")
    run <- function(path, resume = FALSE) tsne(input,
        perplexity = 5, affinity_k = 10L,
        n_components = 3L, n_iter = 110L,
        early_exaggeration_iter = 5L, backend = "cpu",
        n.cores = 1L, seed = 11L, storage = "file_backed",
        output = path, memory_limit = "2GB",
        checkpoint = TRUE, resume = resume)
    native <- fastEmbedR:::massive_tsne_checkpoint_commit
    expect_error(testthat::with_mocked_bindings(
        run(output),
        massive_tsne_checkpoint_commit = function(...) {
            native(...)
            stop("injected t-SNE interruption")
        }, .package = "fastEmbedR"),
        "injected t-SNE interruption")
    expect_true(file.exists(paste0(output, ".checkpoint.rds")))
    resumed <- run(output, resume = TRUE)
    reference <- run(file.path(work, "reference.f32"))
    expect_equal(massive_read_rows(resumed, 1L, nrow(x)),
        massive_read_rows(reference, 1L, nrow(x)),
        tolerance = 1e-4)
})

test_that("one-call full UMAP finishes cleanup after interruption", {
    skip_on_os("windows")
    x <- as.matrix(iris[, 1:4])
    work <- tempfile("full-umap-cleanup-")
    dir.create(work)
    on.exit(unlink(work, recursive = TRUE), add = TRUE)
    input_path <- file.path(work, "input.f32")
    writeBin(as.numeric(t(x)), input_path, size = 4L,
        endian = "little")
    input <- massive_matrix(input_path, nrow(x), ncol(x))
    output <- file.path(work, "resumed.f32")
    run <- function(resume = FALSE) umap(input,
        n_neighbors = 10L, n_components = 2L,
        backend = "cpu", n.cores = 1L, seed = 11L,
        storage = "file_backed", output = output,
        memory_limit = "512MB", checkpoint = TRUE,
        resume = resume)
    expect_error(testthat::with_mocked_bindings(
        run(),
        full_backed_checkpoint_finish = function(...) stop(
            "injected cleanup interruption"),
        .package = "fastEmbedR"),
        "injected cleanup interruption")
    sidecar <- sub("\\.f32$", ".pipeline.rds", output)
    expect_true(file.exists(sidecar))
    expect_true(file.exists(output))
    fit <- run(resume = TRUE)
    expect_false(file.exists(sidecar))
    expect_identical(dim(massive_read_rows(fit, 1L, 5L)),
        c(5L, 2L))
})

test_that("one-call 3D Metal UMAP resumes without CPU fallback", {
    skip_if_not(fastEmbedR:::native_metal_knn_available_cpp())
    skip_if_not(fastEmbedR:::embedding_metal_available_cpp())
    x <- as.matrix(iris[, 1:4])
    work <- tempfile("full-metal-recovery-")
    dir.create(work)
    on.exit(unlink(work, recursive = TRUE), add = TRUE)
    input_path <- file.path(work, "input.f32")
    writeBin(as.numeric(t(x)), input_path, size = 4L,
        endian = "little")
    input <- massive_matrix(input_path, nrow(x), ncol(x))
    output <- file.path(work, "resumed.f32")
    run <- function(path, resume = FALSE) umap(input,
        n_neighbors = 10L, n_components = 3L,
        backend = "metal", n.cores = 1L, seed = 11L,
        storage = "file_backed", output = path,
        memory_limit = "2GB", checkpoint = TRUE,
        resume = resume)
    expect_error(testthat::with_mocked_bindings(
        run(output),
        massive_umap_fuzzy_graph = function(...) stop(
            "injected Metal graph interruption"),
        .package = "fastEmbedR"),
        "injected Metal graph interruption")
    resumed <- run(output, resume = TRUE)
    reference <- run(file.path(work, "reference.f32"))
    expect_match(resumed$backend, "metal")
    expect_equal(massive_read_rows(resumed, 1L, nrow(x)),
        massive_read_rows(reference, 1L, nrow(x)),
        tolerance = 1e-4)
})
