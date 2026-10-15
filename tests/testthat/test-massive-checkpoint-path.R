test_that("checkpoint paths use one cross-platform identity", {
    first <- tempfile("fastembedr_path_a_")
    second <- tempfile("fastembedr_path_b_")
    on.exit(unlink(c(first, second)))
    writeBin(as.raw(1L), first)
    writeBin(as.raw(2L), second)
    forward <- normalizePath(first, winslash = "/")
    native <- normalizePath(first, winslash = "\\")
    expect_true(fastEmbedR:::massive_checkpoint_same_paths(
        forward, native))
    expect_false(fastEmbedR:::massive_checkpoint_same_paths(
        first, second))
})

test_that("checkpoint identity detects edits with preserved mtime", {
    skip_on_os("windows")
    path <- tempfile(fileext = ".f32")
    on.exit(unlink(path))
    writeBin(1, path, size = 4L)
    source <- massive_matrix(path, 1L, 1L)
    before <- fastEmbedR:::massive_checkpoint_source_identity(source)
    stamp <- file.info(path)$mtime
    Sys.sleep(1.1)
    writeBin(2, path, size = 4L)
    expect_true(Sys.setFileTime(path, stamp))
    expect_identical(as.numeric(file.info(path)$mtime),
        as.numeric(stamp))
    expect_false(identical(before,
        fastEmbedR:::massive_checkpoint_source_identity(source)))
})

test_that("optimizer input identity detects preserved-mtime edits", {
    skip_on_os("windows")
    paths <- c(tempfile(fileext = ".u32"),
        tempfile(fileext = ".f32"))
    on.exit(unlink(paths))
    writeBin(1L, paths[[1L]], size = 4L)
    writeBin(1, paths[[2L]], size = 4L)
    before <- fastEmbedR:::massive_checkpoint_input_files_identity(
        paths)
    ordinary <- fastEmbedR:::massive_checkpoint_file_identity(paths)
    graph <- list(indices_path = paths[[1L]],
        distances_path = paths[[2L]], nrow = 2L,
        ncol = 1L, n_reference = 2L, metric = "euclidean")
    fit <- list(method = "tsne", layout = matrix(0, 2L, 2L))
    signature <- function() fastEmbedR:::massive_projection_signature(
        fit, graph,
            list(output_bytes = 16), list(), 1:2)
    signed_before <- signature()
    stamp <- file.info(paths[[2L]])$mtime
    Sys.sleep(1.1)
    writeBin(2, paths[[2L]], size = 4L)
    expect_true(Sys.setFileTime(paths[[2L]], stamp))
    expect_identical(ordinary,
        fastEmbedR:::massive_checkpoint_file_identity(paths))
    expect_false(identical(before,
        fastEmbedR:::massive_checkpoint_input_files_identity(paths)))
    expect_false(identical(signed_before, signature()))
})

test_that("PCA checkpoints include the loaded native implementation", {
    path <- tempfile(fileext = ".f32")
    on.exit(unlink(path))
    writeBin(as.numeric(1:8), path, size = 4L)
    source <- massive_matrix(path, 4L, 2L)
    resources <- list(chunk_rows = 2L)
    identity <- fastEmbedR:::massive_checkpoint_implementation_identity()
    dll <- getLoadedDLLs()[["fastEmbedR"]]
    expect_identical(identity$native,
        unname(tools::md5sum(dll[["path"]])))
    expect_gt(length(identity$r_code), 0L)
    expect_false(anyNA(identity$r_code))
    signature <- function(native) with_mocked_bindings(
        fastEmbedR:::massive_pca_checkpoint_signature(source,
            path, 1L, TRUE, FALSE, "cpu", 1L, 1L, resources),
        massive_checkpoint_implementation_identity = function()
            list(package_version = "0.1", native = native),
        .package = "fastEmbedR")
    expect_false(identical(signature("first"), signature("second")))
})

test_that("multi-GPU workers reject changed implementation", {
    path <- normalizePath(find.package("fastEmbedR"))
    identity <- fastEmbedR:::massive_checkpoint_implementation_identity()
    expect_silent(fastEmbedR:::massive_cuda_worker_identity(
        path, identity))
    altered <- identity
    altered$native <- "changed"
    expect_error(fastEmbedR:::massive_cuda_worker_identity(
        path, altered), "different fastEmbedR implementation")
})

test_that("multi-GPU projection signs its implementation", {
    path <- tempfile(fileext = ".f32")
    arguments <- list(fit = list(layout = matrix(0, 2L, 2L)),
        graph = list(nrow = 4L), selection = NULL)
    tasks <- list(list(device = 0L, row_range = c(1, 4),
        bytes = 32))
    signature <- function(native) with_mocked_bindings(
        fastEmbedR:::massive_multigpu_signature(
            arguments, 0L, tasks, path),
        massive_checkpoint_implementation_identity = function()
            list(native = native),
        massive_cuda_device_uuids = function(...) "GPU-A",
        massive_reference_indices = function(...) 1L,
        massive_projection_signature = function(fit, graph,
                resources, settings, reference_rows) {
            settings$implementation
        },
        .package = "fastEmbedR")
    expect_false(identical(signature("first"), signature("second")))
})

test_that("KNN checkpoints reject a changed implementation", {
    directory <- tempfile()
    dir.create(directory)
    on.exit(unlink(directory, recursive = TRUE))
    path <- file.path(directory, "input.f32")
    writeBin(as.numeric(1:8), path, size = 4L)
    source <- massive_matrix(path, 4L, 2L)
    paths <- function(name) stats::setNames(
        paste0(file.path(directory, name),
            c(".indices.u32", ".distances.f32")),
        c("indices", "distances"))
    changed <- function(expr) with_mocked_bindings(
        force(expr),
        massive_checkpoint_implementation_identity = function()
            list(native = "changed"),
        .package = "fastEmbedR")
    landmark <- list(paths = paths("landmark"), reference = source,
        k = 1L, method = "exact", backend = "cpu", workers = 1L,
        resources = list(chunk_rows = 2L))
    fastEmbedR:::massive_knn_checkpoint(source, landmark, FALSE)
    changed(expect_error(fastEmbedR:::massive_knn_checkpoint(
        source, landmark, TRUE), "not match"))
    full <- list(output = file.path(directory, "full"),
        paths = paths("full"), k = 1L, workers = 1L,
        method = "exact", checkpoint_every = 1L,
        resources = list(chunk_rows = 2L, reference_chunk_rows = 2L))
    fastEmbedR:::massive_full_knn_checkpoint(source, full, FALSE)
    changed(expect_error(fastEmbedR:::massive_full_knn_checkpoint(
        source, full, TRUE), "not match"))
    sharded <- list(k = 1L, method = "hnsw_sharded",
        workers = 1L, checkpoint_every = 1L,
        resources = list(chunk_rows = 2L, reference_chunk_rows = 2L,
            n_shards = 2L, memory_limit_bytes = 1e8))
    fastEmbedR:::massive_sharded_checkpoint(source, sharded,
        paths("sharded"), "cpu", FALSE)
    changed(expect_error(fastEmbedR:::massive_sharded_checkpoint(
        source, sharded, paths("sharded"), "cpu", TRUE),
        "not match"))
})

test_that("HNSW shard-boundary resume pins the faissR library", {
    directory <- tempfile()
    dir.create(directory)
    on.exit(unlink(directory, recursive = TRUE))
    input <- file.path(directory, "input.f32")
    writeBin(as.numeric(1:8), input, size = 4L)
    source <- massive_matrix(input, 4L, 2L)
    identity <- fastEmbedR:::massive_faissr_identity()
    expect_identical(identity$native, unname(tools::md5sum(
        getLoadedDLLs()[["faissR"]][["path"]])))
    paths <- stats::setNames(paste0(file.path(directory, "graph"),
        c(".indices.u32", ".distances.f32")),
        c("indices", "distances"))
    setup <- list(k = 1L, method = "hnsw_sharded",
        workers = 1L, build_workers = 1L,
        checkpoint_every = 1L,
        resources = list(chunk_rows = 2L, reference_chunk_rows = 2L,
            n_shards = 2L, memory_limit_bytes = 1e8))
    saved <- fastEmbedR:::massive_sharded_checkpoint(source,
        setup, paths, "cpu", FALSE)
    writeBin(rep(0L, 4L), paste0(paths[[1L]], ".part"), size = 4L)
    writeBin(rep(0, 4L), paste0(paths[[2L]], ".part"), size = 4L)
    saved$state$shard <- 1L
    saveRDS(saved$state, saved$path)
    changed <- identity
    changed$native <- "different"
    expect_error(with_mocked_bindings(
        fastEmbedR:::massive_sharded_checkpoint(source,
            setup, paths, "cpu", TRUE),
        massive_faissr_identity = function() changed,
        .package = "fastEmbedR"), "checkpoint")
})
