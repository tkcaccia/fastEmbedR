make_massive_fixture <- function(x, format = "f32") {
    path <- tempfile(fileext = paste0(".", format))
    connection <- file(path, "wb")
    on.exit(close(connection))
    if (format == "fbin") {
        writeBin(as.integer(dim(x)), connection, size = 4L,
            endian = "little")
    }
    writeBin(as.vector(t(x)), connection, size = 4L,
        endian = "little")
    path
}

test_that("public 3D CUDA landmark t-SNE writes all query rows", {
    skip_if_not(fastEmbedR:::embedding_cuda_available_cpp())
    set.seed(514)
    x <- matrix(rnorm(64L * 4L), ncol = 4L)
    input <- make_massive_fixture(x)
    directory <- tempfile()
    dir.create(directory)
    output <- file.path(directory, "layout.f32")
    on.exit(unlink(c(input, directory), recursive = TRUE))
    source <- massive_matrix(input, nrow = 64L, ncol = 4L)
    result <- tsne(source, massive = "landmark", landmarks = 32L,
        n_components = 3L, output = output, backend = "cuda",
        perplexity = 3, n_iter = 3L,
        early_exaggeration_iter = 0L,
        transform_perplexity = 3, transform_iter = 3L,
        chunk_rows = 11L, memory_limit = "512MB")
    expect_identical(result$backend, "cuda")
    expect_s3_class(result$layout, "fastEmbedR_massive_matrix")
    expect_equal(dim(as.matrix(result$layout)), c(64L, 3L))
    expect_true(all(is.finite(as.matrix(result$layout))))
    expect_true(result$resources$peak_vram_bytes > 0)
})

test_that("public UMAP and t-SNE landmark modes stay file-backed", {
    skip_if_not_installed("float")
    set.seed(293)
    x <- matrix(rnorm(57L * 4L), ncol = 4L)
    source_path <- make_massive_fixture(x)
    directory <- tempfile()
    dir.create(directory)
    on.exit(unlink(c(source_path, directory), recursive = TRUE))
    source <- massive_matrix(source_path, nrow = 57L, ncol = 4L)
    umap_path <- file.path(directory, "umap.f32")
    tsne_path <- file.path(directory, "tsne.f32")
    observed_umap <- umap(source, massive = "landmark",
        landmarks = 20L, output = umap_path, n_neighbors = 5L,
        transform_k = 5L,
        chunk_rows = 9L, memory_limit = "512MB", backend = "cpu")
    reference_umap <- readRDS(observed_umap$reference_fit_path)
    printed <- paste(capture.output(print(observed_umap)),
        collapse = "\n")
    expect_match(printed, "landmark projection")
    expect_match(printed, "reference rows: 20")
    expect_identical(observed_umap$mode, "landmark")
    expect_s3_class(observed_umap$layout,
        "fastEmbedR_massive_matrix")
    expect_gte(observed_umap$resources$peak_ram_bytes,
        observed_umap$resources$reference_ram_bytes)
    expect_gte(observed_umap$resources$peak_ram_bytes,
        observed_umap$graph$resources$peak_ram_bytes)
    expect_equal(dim(as.matrix(observed_umap$layout)), c(57L, 2L))
    expect_equal(as.matrix(observed_umap$layout)[
        observed_umap$selection$indices, ],
        matrix(as.numeric(reference_umap$layout), ncol = 2L),
        tolerance = 1e-5)
    observed_tsne <- tsne(source, massive = "landmark",
        landmarks = 20L, output = tsne_path, perplexity = 3,
        early_exaggeration_iter = 5L, n_iter = 5L,
        transform_perplexity = 3, transform_iter = 5L,
        chunk_rows = 9L, memory_limit = "512MB", backend = "cpu")
    reference_tsne <- readRDS(observed_tsne$reference_fit_path)
    expect_s3_class(observed_tsne$layout,
        "fastEmbedR_massive_matrix")
    expect_gte(observed_tsne$resources$peak_ram_bytes,
        observed_tsne$resources$reference_ram_bytes)
    expect_gte(observed_tsne$resources$peak_ram_bytes,
        observed_tsne$graph$resources$peak_ram_bytes)
    expect_equal(dim(as.matrix(observed_tsne$layout)), c(57L, 2L))
    expect_true(all(is.finite(as.matrix(observed_tsne$layout))))
    expect_equal(as.matrix(observed_tsne$layout)[
        observed_tsne$selection$indices, ],
        matrix(as.numeric(reference_tsne$layout), ncol = 2L),
        tolerance = 1e-5)
    expect_error(umap(source, massive = "landmark", landmarks = 20L,
        output = file.path(directory, "bad.f32"),
        standardize = TRUE), "without preprocessing")
    expect_error(umap(source, massive = "landmark", landmarks = 20L,
        output = file.path(directory, "metal.f32"),
        backend = "metal"), "no backend fallback")
    expect_error(umap(source, massive = "landmark", landmarks = 20L,
        output = file.path(directory, "cuda.f32"),
        refinement_epochs = 10L, backend = "cuda"),
        "no CPU fallback")
    expect_false(file.exists(file.path(directory, "bad.f32")))
    expect_false(file.exists(file.path(directory, "metal.f32")))
    expect_false(file.exists(file.path(directory, "cuda.f32")))
    expect_error(umap(x, devices = 0L),
        "Experimental output controls")
    expect_error(umap(source, massive = "landmark",
        landmarks = 20L, output = file.path(directory, "cpu_device.f32"),
        backend = "cpu", devices = 0L), "requires CUDA")
    if (isTRUE(fastEmbedR:::embedding_cuda_available_cpp()) &&
        isTRUE(fastEmbedR:::native_cuda_knn_available_cpp())) {
        gpu_umap <- umap(source, massive = "landmark",
            landmarks = 20L, output = file.path(directory, "gpu_umap.f32"),
            n_neighbors = 5L, transform_k = 5L, chunk_rows = 9L,
            memory_limit = "512MB", backend = "cuda", devices = 0L)
        expect_identical(gpu_umap$gpu_devices, 0L)
        expect_identical(gpu_umap$resources$per_device[[1L]]$gpu_device,
            0L)
        expect_gte(gpu_umap$resources$peak_vram_bytes,
            gpu_umap$graph$resources$peak_vram_bytes)
        expect_true(all(is.finite(as.matrix(gpu_umap$layout))))
        gpu_tsne <- tsne(source, massive = "landmark",
            landmarks = 20L, output = file.path(directory, "gpu_tsne.f32"),
            perplexity = 3, early_exaggeration_iter = 5L, n_iter = 5L,
            transform_perplexity = 3, transform_iter = 5L,
            chunk_rows = 9L, memory_limit = "512MB",
            backend = "cuda", devices = 0L)
        expect_identical(gpu_tsne$gpu_devices, 0L)
        expect_true(all(is.finite(as.matrix(gpu_tsne$layout))))
        gpu_reused <- tsne(source, massive = "landmark",
            landmarks = 20L, nn = gpu_umap,
            output = file.path(directory, "gpu_reused.f32"),
            perplexity = 3, early_exaggeration_iter = 5L, n_iter = 5L,
            transform_k = 5L, transform_perplexity = 3,
            transform_iter = 5L, chunk_rows = 9L,
            memory_limit = "512MB", backend = "cuda", devices = 0L)
        expect_true(gpu_reused$graph_reused)
        expect_identical(gpu_reused$graph$indices_path,
            gpu_umap$graph$indices_path)
        expect_true(all(is.finite(as.matrix(gpu_reused$layout))))
    }
})

test_that("public landmark UMAP resumes without repeating completed stages", {
    skip_if_not_installed("float")
    set.seed(612)
    x <- matrix(rnorm(57L * 4L), ncol = 4L)
    input <- make_massive_fixture(x)
    directory <- tempfile()
    dir.create(directory)
    on.exit(unlink(c(input, directory), recursive = TRUE))
    source <- massive_matrix(input, 57L, 4L)
    output <- file.path(directory, "umap.f32")
    run <- function(resume = FALSE, source_data = source,
                    neighbors = 5L) {
        umap(source_data, massive = "landmark", landmarks = 20L,
            output = output, n_neighbors = neighbors, transform_k = 5L,
            chunk_rows = 9L, memory_limit = "512MB", backend = "cpu",
            checkpoint = TRUE, resume = resume)
    }
    original <- fastEmbedR:::massive_project_batch
    calls <- 0L
    with_mocked_bindings(
        massive_project_batch = function(...) {
            calls <<- calls + 1L
            if (calls == 2L) stop("interrupted projection")
            original(...)
        },
        expect_error(run(), "interrupted projection"),
        .package = "fastEmbedR"
    )
    stem <- sub("\\.f32$", "", output)
    sidecar <- paste0(stem, ".workflow.checkpoint.rds")
    saved <- readRDS(sidecar)
    expect_identical(saved$stage, "graph")
    expect_true(file.exists(paste0(output, ".part")))
    expect_error(run(resume = TRUE, neighbors = 6L),
        "checkpoint does not match")
    altered <- make_massive_fixture(x)
    on.exit(unlink(altered), add = TRUE)
    expect_error(run(resume = TRUE,
        source_data = massive_matrix(altered, 57L, 4L)),
        "checkpoint does not match")
    model_identity <- file.info(paste0(stem, ".reference.rds"))$mtime
    with_mocked_bindings(
        massive_select_landmarks = function(...) stop("resampled"),
        massive_landmark_knn = function(...) stop("KNN repeated"),
        resumed <- run(resume = TRUE),
        .package = "fastEmbedR"
    )
    expect_equal(dim(as.matrix(resumed$layout)), c(57L, 2L))
    expect_true(all(is.finite(as.matrix(resumed$layout))))
    expect_identical(file.info(paste0(stem, ".reference.rds"))$mtime,
        model_identity)
    expect_false(file.exists(sidecar))
    expect_false(file.exists(paste0(output, ".part")))
})

test_that("landmark fit resumes from saved selection", {
    skip_if_not_installed("float")
    withr::local_options(n.cores = 2L)
    set.seed(614)
    input <- make_massive_fixture(matrix(rnorm(45L * 4L), ncol = 4L))
    directory <- tempfile()
    dir.create(directory)
    on.exit(unlink(c(input, directory), recursive = TRUE))
    source <- massive_matrix(input, 45L, 4L)
    output <- file.path(directory, "umap.f32")
    run <- function(resume = FALSE) umap(source, massive = "landmark",
        landmarks = 18L, output = output, n_neighbors = 5L,
        transform_k = 5L, chunk_rows = 8L, memory_limit = "512MB",
        backend = "cpu", checkpoint = TRUE, resume = resume)
    with_mocked_bindings(
        massive_read_rows_cpp = function(...) stop("fit interrupted"),
        expect_error(run(), "fit interrupted"),
        .package = "fastEmbedR"
    )
    stem <- sub("\\.f32$", "", output)
    saved <- readRDS(paste0(stem, ".workflow.checkpoint.rds"))
    expect_identical(saved$stage, "selected")
    with_mocked_bindings(
        massive_select_landmarks = function(...) stop("resampled"),
        resumed <- run(resume = TRUE),
        .package = "fastEmbedR"
    )
    expect_identical(resumed$selection$indices, saved$selection$indices)
    expect_identical(resumed$n.cores, 2L)
    expect_true(all(is.finite(as.matrix(resumed$layout))))
})

test_that("public landmark t-SNE accepts checkpointed completion", {
    skip_if_not_installed("float")
    set.seed(613)
    input <- make_massive_fixture(matrix(rnorm(48L * 4L), ncol = 4L))
    output <- tempfile(fileext = ".f32")
    stem <- sub("\\.f32$", "", output)
    on.exit(unlink(c(input, output, paste0(stem, ".reference.rds"),
        paste0(stem, ".landmarks.f32"),
        paste0(stem, ".knn.indices.u32"),
        paste0(stem, ".knn.distances.f32"))))
    source <- massive_matrix(input, 48L, 4L)
    fit <- tsne(source, massive = "landmark", landmarks = 18L,
        output = output, perplexity = 3, transform_perplexity = 3,
        transform_iter = 3L, early_exaggeration_iter = 3L,
        n_iter = 3L, chunk_rows = 8L, memory_limit = "512MB",
        backend = "cpu", checkpoint = TRUE)
    expect_equal(dim(as.matrix(fit$layout)), c(48L, 2L))
    expect_false(file.exists(paste0(stem, ".workflow.checkpoint.rds")))
    expect_error(tsne(source, massive = "off", checkpoint = TRUE),
        "requires massive mode")
})

test_that("landmark t-SNE reuses a checked UMAP neighbor graph", {
    skip_if_not_installed("float")
    set.seed(804)
    x <- matrix(rnorm(52L * 4L), ncol = 4L)
    path <- make_massive_fixture(x)
    other <- make_massive_fixture(x)
    directory <- tempfile()
    dir.create(directory)
    on.exit(unlink(c(path, other, directory), recursive = TRUE))
    source <- massive_matrix(path, nrow = 52L, ncol = 4L)
    fit <- umap(source, massive = "landmark", landmarks = 20L,
        n_neighbors = 5L, transform_k = 5L,
        output = file.path(directory, "umap.f32"),
        chunk_rows = 8L, memory_limit = "512MB", backend = "cpu")
    expect_false(fit$graph_reused)
    reused <- function(name, nn = fit, data = source, k = 5L) {
        tsne(data, massive = "landmark", landmarks = 20L,
            nn = nn, perplexity = 3, transform_k = k,
            transform_perplexity = 3, transform_iter = 3L,
            early_exaggeration_iter = 3L, n_iter = 3L,
            output = file.path(directory, paste0(name, ".f32")),
            chunk_rows = 8L, memory_limit = "512MB", backend = "cpu")
    }
    second <- reused("tsne")
    expect_true(second$graph_reused)
    expect_identical(second$graph$indices_path, fit$graph$indices_path)
    expect_identical(second$selection$indices, fit$selection$indices)
    expect_true(all(is.finite(as.matrix(second$layout))))
    expect_false(file.exists(file.path(directory, "tsne.knn.indices.u32")))
    expect_error(reused("bad_k", k = 6L), "not recomputed")
    expect_error(reused("bad_source", data = massive_matrix(other,
        nrow = 52L, ncol = 4L)), "not recomputed")
    changed <- fit
    changed$selection$indices[1L] <-
        if (changed$selection$indices[1L] == 1L) 2L else 1L
    expect_error(reused("bad_rows", nn = changed),
        "row indices changed|sorted query rows")
    changed <- fit
    changed$graph$output_identity <- NULL
    expect_error(reused("bad_graph", nn = changed), "not recomputed")
    expect_false(file.exists(file.path(directory, "bad_k.f32")))
    expect_false(file.exists(file.path(directory, "bad_source.f32")))
})

test_that("public landmark embeddings honor random-access sampling", {
    skip_if_not_installed("float")
    set.seed(912)
    x <- matrix(rnorm(42L * 4L), ncol = 4L)
    raw <- make_massive_fixture(x, "fbin")
    directory <- tempfile()
    dir.create(directory)
    on.exit(unlink(c(raw, directory), recursive = TRUE))
    source <- massive_matrix(raw)
    output <- function(name) file.path(directory, paste0(name, ".f32"))
    fit_umap <- umap(source, massive = "landmark", landmarks = 18L,
        landmark_method = "random", output = output("umap"),
        n_neighbors = 5L, transform_k = 5L, backend = "cpu",
        chunk_rows = 6L, memory_limit = "512MB", seed = 23L)
    fit_tsne <- tsne(source, massive = "landmark", landmarks = 18L,
        landmark_method = "random", output = output("tsne"),
        perplexity = 3, transform_perplexity = 3,
        early_exaggeration_iter = 3L, n_iter = 3L,
        transform_iter = 3L, backend = "cpu", chunk_rows = 6L,
        memory_limit = "512MB", seed = 23L)
    expect_identical(fit_umap$selection$method, "random")
    expect_identical(fit_tsne$selection$method, "random")
    expect_identical(fit_umap$selection$indices,
        fit_tsne$selection$indices)
    expect_equal(as.matrix(fit_umap$selection$data),
        unname(x[fit_umap$selection$indices, ]), tolerance = 1e-5)
    expect_true(all(is.finite(as.matrix(fit_umap$layout))))
    expect_true(all(is.finite(as.matrix(fit_tsne$layout))))
    expect_error(umap(x, landmark_method = "random"),
        "Experimental output controls")
    expect_error(tsne(x, landmark_method = "random"),
        "Experimental output controls")
})

test_that("embedding approximation requires explicit opt-in", {
    expect_error(umap(matrix(1, 10L, 2L), massive = "auto"),
        "use.*landmark.*explicitly")
    expect_error(tsne(matrix(1, 10L, 2L), massive = "auto"),
        "use.*landmark.*explicitly")
})

test_that("auto RAM probe respects host and container reports", {
    directory <- tempfile()
    dir.create(directory)
    on.exit(unlink(directory, recursive = TRUE))
    meminfo <- file.path(directory, "meminfo")
    v2 <- file.path(directory, "v2")
    v1 <- file.path(directory, "v1")
    dir.create(v2)
    dir.create(v1)
    writeLines("MemAvailable: 4000000 kB", meminfo)
    writeLines("1073741824", file.path(v2, "memory.max"))
    writeLines("268435456", file.path(v2, "memory.current"))
    available <- fastEmbedR:::massive_linux_available_ram(
        meminfo, v2, v1)
    expect_equal(available, 805306368)
    expect_equal(fastEmbedR:::massive_auto_memory_limit(
        "8GB", available), 0.8 * available)
    writeLines("max", file.path(v2, "memory.max"))
    writeLines("2147483648", file.path(v1,
        "memory.limit_in_bytes"))
    writeLines("536870912", file.path(v1,
        "memory.usage_in_bytes"))
    expect_equal(fastEmbedR:::massive_linux_available_ram(
        meminfo, v2, v1), 1610612736)
    expect_equal(fastEmbedR:::massive_macos_available_ram(c(
        "The system has 8589934592 (524288 pages).",
        "System-wide memory free percentage: 35%")),
        8589934592 * 0.35)
    expect_equal(fastEmbedR:::massive_windows_available_ram(
        " 1048576 "), 1024^3)
    expect_true(is.na(fastEmbedR:::massive_windows_available_ram(
        "unavailable")))
    expect_error(fastEmbedR:::massive_auto_memory_limit(
        "8GB", NA_real_), "cannot determine available RAM")
})

test_that("auto PCA cannot materialize beyond detected free RAM", {
    testthat::local_mocked_bindings(
        massive_auto_memory_limit = function(...) 128 * 1024^2,
        .package = "fastEmbedR")
    source <- fastEmbedR:::massive_synthetic_matrix(100L, 4L)
    plan <- fastEmbedR:::massive_pca_auto_route(source, "8GB",
        tempfile(fileext = ".f32"), backend = "cpu")
    expect_identical(plan$mode, "out_of_core")
})

test_that("auto clustering honors available RAM", {
    testthat::local_mocked_bindings(
        massive_auto_memory_limit = function(...) 512 * 1024^2,
        .package = "fastEmbedR")
    source <- fastEmbedR:::massive_synthetic_matrix(1000L, 4L)
    clustering <- fastEmbedR:::massive_cluster_auto_plan(source,
        NULL, 5L, "cpu", 1L, "8GB", "auto", tempfile(),
        NULL, "leiden")
    expect_identical(clustering$mode, "landmark")
})

test_that("experimental file readers preserve row order", {
    x <- matrix(seq_len(42) / 7, nrow = 7L)
    raw <- make_massive_fixture(x)
    fbin <- make_massive_fixture(x, "fbin")
    on.exit(unlink(c(raw, fbin)))

    streamed <- massive_matrix(raw, nrow = 7, ncol = 6)
    mapped <- massive_matrix(raw, nrow = 7, ncol = 6,
        access = "mmap")
    headered <- massive_matrix(fbin)
    expect_equal(massive_read_rows(streamed, 3, 4), x[3:6, ],
        tolerance = 1e-6)
    expect_equal(massive_read_rows(mapped, 3, 4), x[3:6, ],
        tolerance = 1e-6)
    expect_equal(massive_read_rows(headered, 3, 4), x[3:6, ],
        tolerance = 1e-6)
    if (requireNamespace("float", quietly = TRUE)) {
        f32 <- fastEmbedR:::massive_read_rows_cpp(
            streamed, 3, 4L, 4 * 4 * 6, TRUE)
        expect_s4_class(f32, "float32")
        expect_equal(float::dbl(f32), x[3:6, ], tolerance = 1e-6)
        expect_error(fastEmbedR:::massive_read_rows_cpp(
            streamed, 3, 4L, 4 * 4 * 6 - 1, TRUE), "byte limit")
    }
    expect_equal(head(streamed, 2), x[1:2, ], tolerance = 1e-6)
    expect_equal(as.matrix(headered), x, tolerance = 1e-6)
    expect_error(massive_read_rows(streamed, 1, 2.5), "counts")
    expect_error(massive_read_rows(streamed, 0, 1), "counts")
    expect_error(massive_read_rows(streamed, 7, 2), "range")
    expect_error(massive_matrix(raw, nrow = 8, ncol = 6), "size")
    expect_error(massive_matrix(fbin, nrow = 7), "header")
})

test_that("file-backed sources reject same-size input changes", {
    x <- matrix(seq_len(24) / 5, nrow = 6L)
    raw <- make_massive_fixture(x)
    on.exit(unlink(raw))
    source <- massive_matrix(raw, nrow = 6L, ncol = 4L)
    view <- fastEmbedR:::massive_matrix_rows(source, 2L, 3L)
    mapped <- if (.Platform$OS.type == "unix") {
        massive_matrix(raw, nrow = 6L, ncol = 4L,
            access = "mmap")
    } else NULL
    previous <- file.info(raw)$mtime
    connection <- file(raw, "r+b")
    writeBin(123.25, connection, size = 4L, endian = "little")
    close(connection)
    Sys.setFileTime(raw, previous + 10)
    expect_equal(file.info(raw)$size, 6L * 4L * 4L)
    expect_error(massive_read_rows(source, 1L, 1L), "changed")
    expect_error(massive_read_rows(view, 1L, 1L), "changed")
    if (!is.null(mapped)) {
        expect_error(massive_read_rows(mapped, 1L, 1L), "changed")
    }
    fresh <- massive_matrix(raw, nrow = 6L, ncol = 4L)
    expect_equal(massive_read_rows(fresh, 1L, 1L)[1L, 1L],
        123.25)
})

test_that("row views preserve source offsets without materialization", {
    x <- matrix(seq_len(84) / 11, nrow = 14L)
    raw <- make_massive_fixture(x)
    on.exit(unlink(raw))
    source <- massive_matrix(raw, nrow = 14L, ncol = 6L)
    view <- fastEmbedR:::massive_matrix_rows(source, 4, 8)
    expect_equal(massive_read_rows(view, 2, 3),
        x[5:7, ], tolerance = 1e-6)
    expect_equal(as.matrix(view), x[4:11, ], tolerance = 1e-6)
    expect_error(fastEmbedR:::massive_matrix_rows(source, 12, 4),
        "file-backed range")
    huge <- fastEmbedR:::massive_synthetic_matrix(1e12, 3L)
    large <- fastEmbedR:::massive_matrix_rows(huge, 2^31 + 7, 11)
    expected <- massive_read_rows(huge, 2^31 + 15, 3)
    expect_equal(massive_read_rows(large, 9, 3), expected)
    expect_false(identical(
        fastEmbedR:::massive_checkpoint_source_identity(view),
        fastEmbedR:::massive_checkpoint_source_identity(
            fastEmbedR:::massive_matrix_rows(source, 5, 8))))
})

test_that("streamed PCA agrees with an independent dense SVD", {
    rows <- seq_len(41)
    x <- cbind(rows / 9, sin(rows / 3), cos(rows / 5),
        sin(rows / 7) + rows / 90)
    raw <- make_massive_fixture(x)
    output <- tempfile(fileext = ".f32")
    on.exit(unlink(c(raw, output, paste0(output, ".part"))))

    source <- massive_matrix(raw, nrow = nrow(x), ncol = ncol(x))
    fit <- pca(source, ncomp = 3L, backend = "cpu", n.cores = 2L,
        massive = "out_of_core", output = output,
        chunk_rows = 7L, memory_limit = "256KB")
    dense <- stats::prcomp(x, center = TRUE, scale. = FALSE)
    scores <- as.matrix(fit$scores)
    expect_equal(dim(scores), c(nrow(x), 3L))
    expect_equal(fit$center, colMeans(x), tolerance = 1e-6)
    expect_equal(fit$singular_values, dense$sdev[1:3] *
        sqrt(nrow(x) - 1), tolerance = 1e-4)
    for (component in 1:3) {
        expect_equal(abs(cor(scores[, component],
            dense$x[, component])), 1, tolerance = 1e-4)
    }
    expect_equal(fit$backend, "cpu")
    expect_equal(fit$n_threads, 2L)
    expect_true(fit$experimental)
    expect_error(pca(source, ncomp = 2L, backend = "cpu",
        massive = "out_of_core", output = output), "already exists")
})

test_that("completed streamed PCA reopens for file-backed graph work", {
    rows <- seq_len(37)
    x <- cbind(rows / 11, sin(rows / 4), cos(rows / 6))
    raw <- make_massive_fixture(x)
    output <- tempfile(fileext = ".f32")
    graph_prefix <- tempfile()
    on.exit(unlink(c(raw, output, paste0(output, ".manifest.rds"),
        paste0(graph_prefix, c(".indices.u32", ".distances.f32")))))
    source <- massive_matrix(raw, nrow = nrow(x), ncol = ncol(x))
    fit <- pca(source, ncomp = 2L, backend = "cpu",
        massive = "out_of_core", output = output,
        chunk_rows = 8L, memory_limit = "256MB")
    reopened <- massive_open_pca(output)
    expect_s3_class(reopened, "fastEmbedR_massive_pca")
    expect_equal(reopened$loadings, fit$loadings)
    expect_equal(massive_read_rows(reopened$scores, 1, 5),
        massive_read_rows(fit$scores, 1, 5))
    graph <- massive_full_knn_graph(reopened$scores, k = 3L,
        output = graph_prefix, backend = "cpu", method = "exact",
        chunk_rows = 8L, reference_chunk_rows = 15L,
        memory_limit = "128MB")
    edges <- massive_read_graph_edges(graph, 1, 5)
    expect_length(edges$to, 15L)
    expect_error(massive_open_pca(tempfile(fileext = ".f32")),
        "manifest is missing")
    saved <- readRDS(fit$manifest_path)
    saveRDS(1L, fit$manifest_path)
    expect_error(massive_open_pca(output), "manifest is invalid")
    saveRDS(saved, fit$manifest_path)
    con <- file(output, "ab")
    writeBin(as.raw(1L), con)
    close(con)
    expect_error(massive_open_pca(output), "changed")
})

test_that("wide streamed PCA agrees with dense PCA", {
    set.seed(885)
    latent <- matrix(rnorm(96L * 3L), ncol = 3L)
    weights <- matrix(rnorm(3L * 1200L), nrow = 3L)
    x <- latent %*% weights +
        matrix(rnorm(96L * 1200L, sd = 0.01), nrow = 96L)
    raw <- make_massive_fixture(x)
    output <- tempfile(fileext = ".f32")
    on.exit(unlink(c(raw, output, paste0(output, ".part"))))
    source <- massive_matrix(raw, nrow = nrow(x), ncol = ncol(x))
    fit <- pca(source, ncomp = 3L, backend = "cpu",
        n.cores = 2L, seed = 15L, massive = "out_of_core",
        output = output, chunk_rows = 13L, memory_limit = "256MB")
    dense <- stats::prcomp(x, center = TRUE, scale. = FALSE)
    scores <- as.matrix(fit$scores)
    expect_identical(fit$method, "streamed_randomized")
    expect_equal(fit$sketch_size, 19L)
    expect_equal(fit$center, colMeans(x), tolerance = 1e-6)
    expect_equal(fit$singular_values,
        dense$sdev[1:3] * sqrt(nrow(x) - 1), tolerance = 0.01)
    for (component in 1:3) {
        expect_equal(abs(cor(scores[, component],
            dense$x[, component])), 1, tolerance = 0.01)
    }
    if (!isTRUE(embedding_cuda_available_cpp())) {
        expect_error(pca(source, ncomp = 3L, backend = "cuda",
            massive = "out_of_core",
            output = tempfile(fileext = ".f32")),
            "requires native CUDA")
    }
    scaled_path <- tempfile(fileext = ".f32")
    on.exit(unlink(c(scaled_path, paste0(scaled_path, ".part"))),
        add = TRUE)
    scaled <- pca(source, ncomp = 3L, backend = "cpu",
        n.cores = 2L, seed = 15L, center = TRUE, scale = TRUE,
        massive = "out_of_core", output = scaled_path,
        chunk_rows = 13L, memory_limit = "256MB")
    dense_scaled <- stats::prcomp(x, center = TRUE, scale. = TRUE)
    expect_equal(scaled$scale, apply(x, 2L, stats::sd),
        tolerance = 1e-5)
    expect_equal(scaled$singular_values,
        dense_scaled$sdev[1:3] * sqrt(nrow(x) - 1),
        tolerance = 0.02)
    expect_error(pca(source, ncomp = 3L, backend = "cpu",
        massive = "out_of_core", memory_limit = "1MB",
        output = tempfile(fileext = ".f32")), "PCA buffers exceed")
})

test_that("wide streamed CUDA PCA uses GPU sketch and projection", {
    skip_if_not(embedding_cuda_available_cpp())
    set.seed(887)
    latent <- matrix(rnorm(96L * 3L), ncol = 3L)
    weights <- matrix(rnorm(3L * 2100L), nrow = 3L)
    x <- latent %*% weights +
        matrix(rnorm(96L * 2100L, sd = 0.01), nrow = 96L)
    raw <- make_massive_fixture(x)
    output <- tempfile(fileext = ".f32")
    on.exit(unlink(c(raw, output, paste0(output, ".part"))))
    source <- massive_matrix(raw, nrow = nrow(x), ncol = ncol(x))
    fit <- pca(source, ncomp = 3L, backend = "cuda",
        n.cores = 2L, seed = 15L, massive = "out_of_core",
        output = output, chunk_rows = 13L, memory_limit = "256MB")
    dense <- stats::prcomp(x, center = TRUE, scale. = FALSE)
    expect_identical(fit$backend, "cuda")
    expect_identical(fit$moments_backend, "cpu")
    expect_identical(fit$sketch_backend, "cuda")
    expect_identical(fit$method, "streamed_randomized")
    expect_equal(fit$singular_values,
        dense$sdev[1:3] * sqrt(nrow(x) - 1), tolerance = 0.02)
    scores <- as.matrix(fit$scores)
    for (component in 1:3) {
        expect_equal(abs(cor(scores[, component],
            dense$x[, component])), 1, tolerance = 0.02)
    }
})

test_that("moderately wide CUDA PCA uses exact streamed covariance", {
    expect_false(fastEmbedR:::massive_pca_covariance_route(
        1200L, "cpu"))
    expect_true(fastEmbedR:::massive_pca_covariance_route(
        1200L, "cuda"))
    expect_false(fastEmbedR:::massive_pca_covariance_route(
        2100L, "cuda"))
    skip_if_not(embedding_cuda_available_cpp())
    set.seed(893)
    x <- matrix(rnorm(48L * 1200L), nrow = 48L)
    raw <- make_massive_fixture(x)
    output <- tempfile(fileext = ".f32")
    on.exit(unlink(c(raw, output, paste0(output, ".part"))))
    source <- massive_matrix(raw, nrow = 48L, ncol = 1200L)
    fit <- pca(source, ncomp = 2L, backend = "cuda",
        massive = "out_of_core", output = output,
        chunk_rows = 13L, memory_limit = "256MB")
    dense <- stats::prcomp(x, center = TRUE, scale. = FALSE)
    expect_identical(fit$method, "streamed_covariance")
    expect_identical(fit$backend, "cuda")
    expect_null(fit$sketch_backend)
    expect_equal(fit$singular_values,
        dense$sdev[1:2] * sqrt(nrow(x) - 1), tolerance = 1e-3)
})

test_that("wide streamed PCA resumes committed score rows", {
    set.seed(113)
    latent <- matrix(rnorm(67L * 3L), ncol = 3L)
    weights <- matrix(rnorm(3L * 1100L), nrow = 3L)
    x <- latent %*% weights
    raw <- make_massive_fixture(x)
    output <- tempfile(fileext = ".f32")
    reference_path <- tempfile(fileext = ".f32")
    sidecar <- paste0(output, ".checkpoint.rds")
    on.exit(unlink(c(raw, output, reference_path,
        paste0(output, ".part"), sidecar)))
    source <- massive_matrix(raw, nrow = nrow(x), ncol = ncol(x))
    resources <- fastEmbedR:::massive_pca_resources(source,
        3L, 2L, 11L, "256MB", "cpu")
    path <- fastEmbedR:::massive_float_output(output,
        resources$output_bytes)
    prepared <- fastEmbedR:::massive_pca_prepare(source, path,
        3L, TRUE, FALSE, "cpu", 2L, 4L, resources,
        checkpoint = TRUE, resume = FALSE)
    interrupt <- function(rows) {
        prepared$progress(rows)
        stop("simulated interruption")
    }
    expect_error(fastEmbedR:::massive_pca_project_cpp(source,
        path, prepared$fit$loadings, prepared$fit$center,
        prepared$fit$scale, 11L, 2L, 0, interrupt, FALSE),
        "simulated interruption")
    expect_equal(readRDS(sidecar)$completed_rows, 11)
    resumed <- pca(source, ncomp = 3L, backend = "cpu",
        n.cores = 2L, seed = 4L, massive = "out_of_core",
        output = output, chunk_rows = 11L, memory_limit = "256MB",
        checkpoint = TRUE, resume = TRUE)
    reference <- pca(source, ncomp = 3L, backend = "cpu",
        n.cores = 2L, seed = 4L, massive = "out_of_core",
        output = reference_path, chunk_rows = 11L,
        memory_limit = "256MB")
    expect_identical(readBin(output, "raw", n = 67L * 3L * 4L),
        readBin(reference_path, "raw", n = 67L * 3L * 4L))
    expect_identical(resumed$method, "streamed_randomized")
    expect_false(file.exists(sidecar))
})

test_that("wide PCA resumes from validated column moments", {
    set.seed(188)
    values <- matrix(rnorm(43L * 1030L), nrow = 43L)
    input <- make_massive_fixture(values)
    output <- tempfile(fileext = ".f32")
    reference_path <- tempfile(fileext = ".f32")
    sidecar <- paste0(output, ".checkpoint.rds")
    on.exit(unlink(c(input, output, reference_path,
        paste0(output, ".part"), sidecar)))
    source <- massive_matrix(input, 43L, 1030L)
    run <- function(path, resume = FALSE, checkpoint = FALSE) {
        pca(source, ncomp = 2L, backend = "cpu", n.cores = 2L,
            seed = 19L, center = TRUE, scale = TRUE,
            massive = "out_of_core", output = path,
            chunk_rows = 11L, memory_limit = "256MB",
            checkpoint = checkpoint, resume = resume)
    }
    with_mocked_bindings(
        massive_pca_wide_action_cpp = function(...) {
            stop("forced wide sketch interruption")
        },
        expect_error(run(output, checkpoint = TRUE),
            "forced wide sketch interruption"),
        .package = "fastEmbedR"
    )
    state <- readRDS(sidecar)
    expect_identical(state$stage, "moments")
    expect_equal(state$moments$center, colMeans(values),
        tolerance = 1e-6)
    expect_equal(state$moments$scale, apply(values, 2L, sd),
        tolerance = 1e-5)
    changed <- state
    changed$moments$scale[[1L]] <- 0
    saveRDS(changed, sidecar)
    expect_error(run(output, resume = TRUE, checkpoint = TRUE),
        "statistics checkpoint is invalid")
    saveRDS(state, sidecar)
    with_mocked_bindings(
        massive_pca_wide_moments_cpp = function(...) {
            stop("column moments repeated")
        },
        resumed <- run(output, resume = TRUE, checkpoint = TRUE),
        .package = "fastEmbedR"
    )
    reference <- run(reference_path)
    expect_equal(as.matrix(resumed$scores),
        as.matrix(reference$scores), tolerance = 1e-6)
    expect_false(file.exists(sidecar))
})

test_that("wide CUDA PCA resumes moments without a CPU rescan", {
    skip_if_not(embedding_cuda_available_cpp())
    set.seed(189)
    values <- matrix(rnorm(37L * 2100L), nrow = 37L)
    input <- make_massive_fixture(values)
    output <- tempfile(fileext = ".f32")
    reference_path <- tempfile(fileext = ".f32")
    sidecar <- paste0(output, ".checkpoint.rds")
    on.exit(unlink(c(input, output, reference_path,
        paste0(output, ".part"), sidecar)))
    source <- massive_matrix(input, 37L, 2100L)
    run <- function(path, resume = FALSE, checkpoint = FALSE) {
        pca(source, ncomp = 2L, backend = "cuda", n.cores = 2L,
            seed = 19L, center = TRUE, scale = FALSE,
            massive = "out_of_core", output = path,
            chunk_rows = 11L, memory_limit = "256MB",
            checkpoint = checkpoint, resume = resume)
    }
    with_mocked_bindings(
        massive_pca_wide_action_cuda_cpp = function(...) {
            stop("forced CUDA sketch interruption")
        },
        expect_error(run(output, checkpoint = TRUE),
            "forced CUDA sketch interruption"),
        .package = "fastEmbedR"
    )
    expect_identical(readRDS(sidecar)$stage, "moments")
    with_mocked_bindings(
        massive_pca_wide_moments_cpp = function(...) {
            stop("column moments repeated")
        },
        resumed <- run(output, resume = TRUE, checkpoint = TRUE),
        .package = "fastEmbedR"
    )
    reference <- run(reference_path)
    expect_identical(resumed$sketch_backend, "cuda")
    expect_equal(as.matrix(resumed$scores),
        as.matrix(reference$scores), tolerance = 1e-5)
    expect_false(file.exists(sidecar))
})

test_that("auto PCA reports its route and preserves explicit limits", {
    testthat::local_mocked_bindings(
        massive_available_ram_bytes = function() 4 * 1024^3,
        .package = "fastEmbedR")
    x <- matrix(sin(seq_len(120) / 7), nrow = 30L)
    raw <- make_massive_fixture(x)
    output <- tempfile(fileext = ".f32")
    on.exit(unlink(c(raw, output, paste0(output, ".part"))))
    source <- massive_matrix(raw, nrow = 30L, ncol = 4L)
    expect_message(dense <- pca(source, ncomp = 2L,
        backend = "cpu", massive = "auto",
        memory_limit = "512MB"), "selected in-memory")
    expect_false(isTRUE(dense$experimental))
    expect_identical(dense$massive_auto, "in_memory")
    expect_equal(dim(dense$scores), c(30L, 2L))
    expect_error(pca(source, ncomp = 2L, massive = "auto",
        memory_limit = "256KB"), "persistent.*output")
    expect_error(pca(x, ncomp = 2L, massive = "auto",
        memory_limit = "256KB", output = output),
        "file-backed")
    expect_error(pca(source, ncomp = 2L, massive = "auto",
        memory_limit = "512MB", output = output),
        "fits in memory")
    expect_message(streamed <- pca(source, ncomp = 2L,
        backend = "cpu", massive = "auto", output = output,
        memory_limit = "256KB", chunk_rows = 8L),
        "selected out-of-core")
    expect_true(streamed$experimental)
    expect_identical(streamed$massive_auto, "out_of_core")
    expect_s3_class(streamed$scores, "fastEmbedR_massive_matrix")
    expect_equal(dim(as.matrix(streamed$scores)), c(30L, 2L))
    huge <- fastEmbedR:::massive_synthetic_matrix(1e9, 96L)
    plan <- fastEmbedR:::massive_pca_auto_route(
        huge, "16GB", tempfile(fileext = ".f32"))
    expect_identical(plan$mode, "out_of_core")
    expect_identical(plan$data$nrow, 1e9)
    if (!isTRUE(fastEmbedR:::embedding_cuda_available_cpp())) {
        expect_error(pca(source, ncomp = 2L, backend = "cuda",
            massive = "auto", memory_limit = "16GB",
            output = tempfile(fileext = ".f32")),
            "no CPU fallback")
    }
})

test_that("streamed PCA resumes only committed score batches", {
    set.seed(654)
    x <- matrix(rnorm(51L * 5L), ncol = 5L)
    raw <- make_massive_fixture(x)
    output <- tempfile(fileext = ".f32")
    reference_path <- tempfile(fileext = ".f32")
    sidecar <- paste0(output, ".checkpoint.rds")
    on.exit(unlink(c(raw, output, reference_path,
        paste0(output, ".part"), sidecar)))
    source <- massive_matrix(raw, nrow = 51L, ncol = 5L)
    resources <- fastEmbedR:::massive_pca_resources(source,
        3L, 2L, 11L, "512MB", "cpu")
    path <- fastEmbedR:::massive_float_output(output,
        resources$output_bytes)
    prepared <- fastEmbedR:::massive_pca_prepare(source, path,
        3L, TRUE, FALSE, "cpu", 2L, 4L, resources,
        checkpoint = TRUE, resume = FALSE)
    interrupt <- function(rows) {
        prepared$progress(rows)
        stop("simulated interruption")
    }
    expect_error(fastEmbedR:::massive_pca_project_cpp(source,
        path, prepared$fit$loadings, prepared$fit$center,
        prepared$fit$scale, 11L, 2L, 0, interrupt, FALSE),
        "simulated interruption")
    expect_equal(readRDS(sidecar)$completed_rows, 11)
    connection <- file(paste0(output, ".part"), "ab")
    writeBin(as.raw(255), connection)
    close(connection)
    expect_error(pca(source, ncomp = 3L, backend = "cpu",
        n.cores = 2L, massive = "out_of_core", output = output,
        chunk_rows = 10L, memory_limit = "512MB",
        checkpoint = TRUE, resume = TRUE), "does not match")
    resumed <- pca(source, ncomp = 3L, backend = "cpu",
        n.cores = 2L, massive = "out_of_core", output = output,
        chunk_rows = 11L, memory_limit = "512MB",
        checkpoint = TRUE, resume = TRUE)
    reference <- pca(source, ncomp = 3L, backend = "cpu",
        n.cores = 2L, massive = "out_of_core",
        output = reference_path, chunk_rows = 11L,
        memory_limit = "512MB")
    expect_equal(as.matrix(resumed$scores),
        as.matrix(reference$scores), tolerance = 1e-6)
    expect_false(file.exists(sidecar))
    expect_false(file.exists(paste0(output, ".part")))
    expect_error(pca(source, massive = "off", checkpoint = TRUE),
        "require")
})

test_that("covariance PCA resumes from a saved mean pass", {
    set.seed(185)
    values <- matrix(rnorm(275L), nrow = 55L)
    input <- make_massive_fixture(values)
    output <- tempfile(fileext = ".f32")
    reference_path <- tempfile(fileext = ".f32")
    sidecar <- paste0(output, ".checkpoint.rds")
    on.exit(unlink(c(input, output, reference_path,
        paste0(output, ".part"), sidecar)))
    source <- massive_matrix(input, 55L, 5L)
    run <- function(path, resume = FALSE, checkpoint = FALSE) {
        pca(source, ncomp = 3L, backend = "cpu", n.cores = 2L,
            massive = "out_of_core", output = path,
            chunk_rows = 11L, memory_limit = "256MB",
            checkpoint = checkpoint, resume = resume)
    }
    with_mocked_bindings(
        massive_pca_covariance_cpp = function(...) {
            stop("forced covariance interruption")
        },
        expect_error(run(output, checkpoint = TRUE),
            "forced covariance interruption"),
        .package = "fastEmbedR"
    )
    state <- readRDS(sidecar)
    expect_identical(state$stage, "covariance_partial")
    expect_equal(state$cross_rows, 0)
    expect_equal(state$means, colMeans(values), tolerance = 1e-6)
    changed <- state
    changed$means[[1L]] <- changed$means[[1L]] + 1
    saveRDS(changed, sidecar)
    expect_error(run(output, resume = TRUE, checkpoint = TRUE),
        "statistics checkpoint is invalid")
    saveRDS(state, sidecar)
    with_mocked_bindings(
        massive_pca_means_cpp = function(...) stop("mean pass repeated"),
        resumed <- run(output, resume = TRUE, checkpoint = TRUE),
        .package = "fastEmbedR"
    )
    reference <- run(reference_path)
    expect_equal(as.matrix(resumed$scores),
        as.matrix(reference$scores), tolerance = 1e-6)
    expect_false(file.exists(sidecar))
})

test_that("covariance PCA resumes within a mean pass", {
    set.seed(190)
    values <- matrix(rnorm(330L), nrow = 66L)
    input <- make_massive_fixture(values)
    output <- tempfile(fileext = ".f32")
    reference_path <- tempfile(fileext = ".f32")
    sidecar <- paste0(output, ".checkpoint.rds")
    on.exit(unlink(c(input, output, reference_path,
        paste0(output, ".part"), sidecar)))
    source <- massive_matrix(input, 66L, 5L)
    run <- function(path, resume = FALSE, checkpoint = FALSE) {
        pca(source, ncomp = 3L, backend = "cpu", n.cores = 2L,
            massive = "out_of_core", output = path,
            chunk_rows = 11L, memory_limit = "256MB",
            checkpoint = checkpoint, resume = resume)
    }
    native <- fastEmbedR:::massive_pca_mean_chunk_cpp
    calls <- new.env(parent = emptyenv())
    calls$count <- 0L
    with_mocked_bindings(
        massive_pca_mean_chunk_cpp = function(...) {
            calls$count <- calls$count + 1L
            if (calls$count == 3L) stop("forced mean interruption")
            native(...)
        },
        expect_error(run(output, checkpoint = TRUE),
            "forced mean interruption"),
        .package = "fastEmbedR"
    )
    state <- readRDS(sidecar)
    expect_identical(state$stage, "mean_partial")
    expect_equal(state$mean_rows, 22)
    expect_equal(colSums(state$mean_sums), colSums(values[1:22, ]),
        tolerance = 1e-6)
    changed <- state
    changed$mean_sums[1L, 1L] <- changed$mean_sums[1L, 1L] + 1
    saveRDS(changed, sidecar)
    expect_error(run(output, resume = TRUE, checkpoint = TRUE),
        "statistics checkpoint is invalid")
    saveRDS(state, sidecar)
    with_mocked_bindings(
        massive_pca_mean_chunk_cpp = function(spec, first, rows,
                n_threads, previous) {
            if (first <= 22) stop("completed mean rows were reread")
            native(spec, first, rows, n_threads, previous)
        },
        resumed <- run(output, resume = TRUE, checkpoint = TRUE),
        .package = "fastEmbedR"
    )
    reference <- run(reference_path)
    expect_equal(as.matrix(resumed$scores),
        as.matrix(reference$scores), tolerance = 1e-6)
    expect_false(file.exists(sidecar))
})

test_that("covariance PCA resumes completed cross-product blocks", {
    set.seed(191)
    values <- matrix(rnorm(374L * 5L), ncol = 5L)
    input <- make_massive_fixture(values)
    output <- tempfile(fileext = ".f32")
    reference_path <- tempfile(fileext = ".f32")
    sidecar <- paste0(output, ".checkpoint.rds")
    on.exit(unlink(c(input, output, reference_path,
        paste0(output, ".part"), sidecar)))
    source <- massive_matrix(input, 374L, 5L)
    run <- function(path, resume = FALSE, checkpoint = FALSE) {
        pca(source, ncomp = 3L, backend = "cpu", n.cores = 2L,
            massive = "out_of_core", output = path,
            chunk_rows = 11L, memory_limit = "256MB",
            checkpoint = checkpoint, resume = resume)
    }
    native <- fastEmbedR:::massive_pca_covariance_cpp
    with_mocked_bindings(
        massive_pca_covariance_cpp = function(spec, chunk_rows,
                n_threads, center, means, rows, sums, progress, every) {
            interrupt <- function(done, current) {
                progress(done, current)
                stop("forced covariance block interruption")
            }
            native(spec, chunk_rows, n_threads, center, means,
                rows, sums, interrupt, every)
        },
        expect_error(run(output, checkpoint = TRUE),
            "forced covariance block interruption"),
        .package = "fastEmbedR"
    )
    state <- readRDS(sidecar)
    expect_identical(state$stage, "covariance_partial")
    expect_equal(state$cross_rows, 176)
    changed <- state
    changed$cross_sums[1L, 1L] <- changed$cross_sums[1L, 1L] + 1
    saveRDS(changed, sidecar)
    expect_error(run(output, resume = TRUE, checkpoint = TRUE),
        "statistics checkpoint is invalid")
    saveRDS(state, sidecar)
    with_mocked_bindings(
        massive_pca_covariance_cpp = function(spec, chunk_rows,
                n_threads, center, means, rows, sums, progress, every) {
            if (rows < 176) stop("completed covariance rows repeated")
            native(spec, chunk_rows, n_threads, center, means,
                rows, sums, progress, every)
        },
        resumed <- run(output, resume = TRUE, checkpoint = TRUE),
        .package = "fastEmbedR"
    )
    reference <- run(reference_path)
    expect_equal(as.matrix(resumed$scores),
        as.matrix(reference$scores), tolerance = 1e-6)
    expect_false(file.exists(sidecar))
})

test_that("wide covariance checkpoints bound snapshot traffic", {
    seen <- new.env(parent = emptyenv())
    with_mocked_bindings(
        massive_checkpoint_write = function(...) invisible(NULL),
        massive_pca_covariance_cuda_cpp = function(spec, chunk_rows,
                n_threads, center, means, rows, sums, progress, every) {
            seen$cuda <- every
            list(cross_sums = sums)
        },
        massive_pca_covariance_cpp = function(spec, chunk_rows,
                n_threads, center, means, rows, sums, progress, every) {
            seen$cpu <- every
            list(cross_sums = sums)
        },
        {
            fastEmbedR:::massive_pca_checkpoint_covariance(
                list(ncol = 1200L), TRUE, "cuda", 1L,
                list(chunk_rows = 2048L),
                list(stage = "means", means = numeric(1200L)),
                tempfile())
            fastEmbedR:::massive_pca_checkpoint_covariance(
                list(ncol = 1200L), TRUE, "cpu", 2L,
                list(chunk_rows = 2048L),
                list(stage = "means", means = numeric(1200L)),
                tempfile())
        },
        .package = "fastEmbedR"
    )
    expect_identical(seen$cuda, 118L)
    expect_identical(seen$cpu, 235L)
})

test_that("streamed CUDA PCA resumes without a CPU projection", {
    skip_if_not(embedding_cuda_available_cpp())
    set.seed(754)
    x <- matrix(rnorm(47L * 5L), ncol = 5L)
    raw <- make_massive_fixture(x)
    output <- tempfile(fileext = ".f32")
    reference_path <- tempfile(fileext = ".f32")
    sidecar <- paste0(output, ".checkpoint.rds")
    on.exit(unlink(c(raw, output, reference_path,
        paste0(output, ".part"), sidecar)))
    source <- massive_matrix(raw, nrow = 47L, ncol = 5L)
    resources <- fastEmbedR:::massive_pca_resources(source,
        2L, 1L, 13L, "512MB", "cuda")
    path <- fastEmbedR:::massive_float_output(output,
        resources$output_bytes)
    prepared <- fastEmbedR:::massive_pca_prepare(source, path,
        2L, TRUE, FALSE, "cuda", 1L, 4L, resources,
        checkpoint = TRUE, resume = FALSE)
    interrupt <- function(rows) {
        prepared$progress(rows)
        stop("simulated interruption")
    }
    expect_error(fastEmbedR:::massive_pca_project_cuda_cpp(source,
        path, prepared$fit$loadings, prepared$fit$center,
        prepared$fit$scale, 13L, 0, interrupt, FALSE),
        "simulated interruption")
    resumed <- pca(source, ncomp = 2L, backend = "cuda",
        n.cores = 1L, massive = "out_of_core", output = output,
        chunk_rows = 13L, memory_limit = "512MB",
        checkpoint = TRUE, resume = TRUE)
    reference <- pca(source, ncomp = 2L, backend = "cuda",
        n.cores = 1L, massive = "out_of_core",
        output = reference_path, chunk_rows = 13L,
        memory_limit = "512MB")
    expect_equal(as.matrix(resumed$scores),
        as.matrix(reference$scores), tolerance = 1e-5)
    expect_identical(resumed$backend, "cuda")
})

test_that("CUDA covariance PCA reuses a saved CPU mean pass", {
    skip_if_not(embedding_cuda_available_cpp())
    set.seed(186)
    values <- matrix(rnorm(61L * 1100L), nrow = 61L)
    input <- make_massive_fixture(values)
    output <- tempfile(fileext = ".f32")
    reference_path <- tempfile(fileext = ".f32")
    sidecar <- paste0(output, ".checkpoint.rds")
    on.exit(unlink(c(input, output, reference_path,
        paste0(output, ".part"), sidecar)))
    source <- massive_matrix(input, 61L, 1100L)
    run <- function(path, resume = FALSE, checkpoint = FALSE) {
        pca(source, ncomp = 2L, backend = "cuda", n.cores = 1L,
            massive = "out_of_core", output = path,
            chunk_rows = 13L, memory_limit = "512MB",
            checkpoint = checkpoint, resume = resume)
    }
    with_mocked_bindings(
        massive_pca_covariance_cuda_cpp = function(...) {
            stop("forced CUDA covariance interruption")
        },
        expect_error(run(output, checkpoint = TRUE),
            "forced CUDA covariance interruption"),
        .package = "fastEmbedR"
    )
    expect_identical(readRDS(sidecar)$stage,
        "covariance_partial")
    with_mocked_bindings(
        massive_pca_means_cpp = function(...) stop("mean pass repeated"),
        resumed <- run(output, resume = TRUE, checkpoint = TRUE),
        .package = "fastEmbedR"
    )
    reference <- run(reference_path)
    expect_identical(resumed$backend, "cuda")
    expect_equal(as.matrix(resumed$scores),
        as.matrix(reference$scores), tolerance = 1e-5)
    expect_false(file.exists(sidecar))
})

test_that("PCA memory budget includes resident R-matrix input", {
    x <- matrix(seq_len(100000L) / 100, ncol = 10L)
    source <- massive_matrix(x)
    resources <- fastEmbedR:::massive_pca_resources(source, 2L,
        1L, 100L, "4MB", "cpu")
    expect_equal(resources$input_resident_bytes, length(x) * 8)
    expect_gt(resources$peak_ram_bytes,
        resources$input_resident_bytes)
    output <- tempfile(fileext = ".f32")
    expect_error(pca(source, ncomp = 2L, backend = "cpu",
        massive = "out_of_core", output = output,
        memory_limit = "256KB"), "PCA buffers exceed")
    expect_false(file.exists(output))
})

test_that("massive PCA refuses an unavailable GPU route", {
    skip_if(embedding_cuda_available_cpp())
    x <- matrix(seq_len(60) / 9, ncol = 3L)
    source <- massive_matrix(x)
    output <- tempfile(fileext = ".f32")
    on.exit(unlink(output))
    expect_error(pca(source, ncomp = 2L, backend = "cuda",
        massive = "out_of_core", output = output),
        "requires native CUDA")
    expect_false(file.exists(output))
})

test_that("streamed CUDA PCA agrees with dense and CPU PCA", {
    skip_if_not(embedding_cuda_available_cpp())
    set.seed(281)
    x <- matrix(rnorm(89L * 6L), ncol = 6L)
    raw <- make_massive_fixture(x)
    cpu_path <- tempfile(fileext = ".f32")
    cuda_path <- tempfile(fileext = ".f32")
    on.exit(unlink(c(raw, cpu_path, cuda_path)))
    source <- massive_matrix(raw, nrow = nrow(x), ncol = ncol(x))
    cpu <- pca(source, ncomp = 3L, backend = "cpu",
        n.cores = 2L, massive = "out_of_core", center = TRUE,
        scale = TRUE, output = cpu_path, chunk_rows = 11L)
    cuda <- pca(source, ncomp = 3L, backend = "cuda",
        n.cores = 2L, massive = "out_of_core", center = TRUE,
        scale = TRUE, output = cuda_path, chunk_rows = 13L)
    dense <- stats::prcomp(x, center = TRUE, scale. = TRUE)
    expect_identical(cuda$backend, "cuda")
    expect_true(cuda$resources$peak_vram_bytes > 0)
    expect_equal(cuda$singular_values, cpu$singular_values,
        tolerance = 2e-4)
    expect_equal(cuda$singular_values, dense$sdev[1:3] *
        sqrt(nrow(x) - 1), tolerance = 2e-4)
    expect_equal(unname(abs(cor(as.matrix(cuda$scores),
        dense$x[, 1:3]))),
        diag(3), tolerance = 2e-3)
})

test_that("multi-GPU PCA shards preserve row order and output size", {
    x <- matrix(seq_len(55L), ncol = 5L)
    raw <- make_massive_fixture(x)
    output <- tempfile(fileext = ".f32")
    source <- massive_matrix(raw, nrow = 11L, ncol = 5L)
    tasks <- fastEmbedR:::massive_pca_shards_check(source,
        output, 2L, 11L * 2L * 4L, c(0L, 1L))
    paths <- vapply(tasks, `[[`, "", "output")
    on.exit(unlink(c(raw, output, paths,
        paste0(paths, ".part"), paste0(output, ".part"))))
    expect_equal(lapply(tasks, `[[`, "row_range"),
        list(c(1, 5), c(6, 11)))
    values <- matrix(seq_len(22L), ncol = 2L, byrow = TRUE)
    for (task in tasks) {
        rows <- seq.int(task$row_range[[1L]], task$row_range[[2L]])
        writeBin(as.numeric(t(values[rows, , drop = FALSE])),
            task$output, size = 4L, endian = "little")
    }
    fastEmbedR:::massive_concat_word_files_cpp(paths, output,
        vapply(tasks, `[[`, 0, "bytes"))
    expect_equal(massive_read_rows(
        massive_matrix(output, 11L, 2L), 1L, 11L), values)
    expect_error(fastEmbedR:::massive_pca_shards_check(source,
        output, 2L, 88, c(0L, 1L)), "already exists")
    upper <- tempfile(fileext = ".F32")
    upper_tasks <- fastEmbedR:::massive_pca_shards_check(source,
        upper, 2L, 88, c(0L, 1L))
    expect_false(any(vapply(upper_tasks, `[[`, "", "output") == upper))
    expect_error(fastEmbedR:::massive_pca_shards_check(
        fastEmbedR:::massive_synthetic_matrix(11L, 5L),
        upper, 2L, 88, c(0L, 1L)), "file-backed")
})

test_that("PCA device selection never falls back to another backend", {
    x <- matrix(seq_len(55L), ncol = 5L)
    raw <- make_massive_fixture(x)
    source <- massive_matrix(raw, nrow = 11L, ncol = 5L)
    on.exit(unlink(raw))
    expect_error(pca(x, devices = 0L), "out-of-core")
    expect_error(pca(source, massive = "out_of_core",
        backend = "cpu", devices = 0L,
        output = tempfile(fileext = ".f32")), "requires CUDA")
    if (!embedding_cuda_available_cpp()) expect_error(pca(source,
        massive = "out_of_core", backend = "cuda", devices = 0L,
        output = tempfile(fileext = ".f32")), "no CPU fallback")
})

test_that("multi-GPU PCA resumes partial shards and checks the model", {
    x <- matrix(seq_len(55L), ncol = 5L)
    raw <- make_massive_fixture(x)
    output <- tempfile(fileext = ".f32")
    source <- massive_matrix(raw, 11L, 5L)
    tasks <- fastEmbedR:::massive_pca_shards_check(source,
        output, 2L, 88, c(0L, 1L))
    paths <- vapply(tasks, `[[`, "", "output")
    on.exit(unlink(c(raw, output, paste0(output, ".part"),
        paths, paste0(paths, ".part"),
        paste0(paths, ".checkpoint.rds"),
        paste0(paths, ".result.rds"))))
    fit <- list(loadings = matrix(1, 5L, 2L),
        center = rep(0, 5L), scale = rep(1, 5L))
    view <- fastEmbedR:::massive_matrix_rows(source, 6, 6)
    resources <- list(chunk_rows = 3L, output_bytes = 48)
    saved <- fastEmbedR:::massive_pca_shard_checkpoint(
        tasks[[2L]], view, fit, resources, TRUE)
    saved$progress(3L)
    writeBin(rep(0, 6), paste0(paths[[2L]], ".part"), size = 4L)
    writeBin(rep(0, 10), paths[[1L]], size = 4L)
    expect_error(fastEmbedR:::massive_pca_shards_check(source,
        output, 2L, 88, c(0L, 1L), resume = TRUE),
        "Completed CUDA shard is inconsistent")
    fastEmbedR:::massive_multigpu_record_shard(tasks[[1L]])
    recovered <- fastEmbedR:::massive_pca_shards_check(source,
        output, 2L, 88, c(0L, 1L), resume = TRUE)
    expect_equal(vapply(recovered, `[[`, "", "status"),
        c("complete", "partial"))
    expect_equal(vapply(recovered, `[[`, 0, "remaining"), c(0, 24))
    Sys.setFileTime(paths[[1L]], Sys.time() + 5)
    expect_error(fastEmbedR:::massive_pca_shards_check(source,
        output, 2L, 88, c(0L, 1L), resume = TRUE),
        "Completed CUDA shard is inconsistent")
    fastEmbedR:::massive_multigpu_record_shard(tasks[[1L]])
    resumed <- fastEmbedR:::massive_pca_shard_checkpoint(
        recovered[[2L]], view, fit, resources, TRUE)
    expect_equal(resumed$rows, 3L)
    fit$loadings[1L, 1L] <- 2
    expect_error(fastEmbedR:::massive_pca_shard_checkpoint(
        recovered[[2L]], view, fit, resources, TRUE),
        "checkpoint does not match")
    resources$gpu_devices <- c(0L, 1L)
    left <- fastEmbedR:::massive_pca_checkpoint_signature(
        source, output, 2L, TRUE, FALSE, "cuda", 1L, 4L,
        resources)
    resources$gpu_devices <- c(0L, 2L)
    right <- fastEmbedR:::massive_pca_checkpoint_signature(
        source, output, 2L, TRUE, FALSE, "cuda", 1L, 4L,
        resources)
    expect_false(identical(left, right))
})

test_that("streamed PCA uses every requested CUDA device", {
    skip_if_not(embedding_cuda_available_cpp())
    set.seed(4403)
    x <- matrix(rnorm(41L * 5L), ncol = 5L)
    raw <- make_massive_fixture(x)
    directory <- tempfile()
    dir.create(directory)
    on.exit(unlink(c(raw, directory), recursive = TRUE))
    source <- massive_matrix(raw, nrow = 41L, ncol = 5L)
    single <- pca(source, ncomp = 2L, backend = "cuda",
        massive = "out_of_core", devices = 0L,
        output = file.path(directory, "single.f32"),
        chunk_rows = 9L, memory_limit = "1GB")
    expect_identical(single$gpu_devices, 0L)
    expect_identical(single$resources$gpu_device, 0L)
    shard <- fastEmbedR:::massive_row_shards(
        file.path(directory, "shard.f32"), 41L, 2L, 0L)[[1L]]
    shard$status <- "new"
    worker <- fastEmbedR:::massive_pca_device_worker(shard,
        list(source = source, fit = single, rank = 2L,
            chunk_rows = 9L, memory_limit = 1e9,
            checkpoint = TRUE),
        normalizePath(find.package("fastEmbedR")))
    expect_identical(worker$gpu_device, 0L)
    expect_equal(as.matrix(massive_matrix(shard$output, 41L, 2L)),
        as.matrix(single$scores), tolerance = 1e-5)
    expect_identical(fastEmbedR:::massive_multigpu_task_status(
        shard, TRUE)$status, "complete")
    skip_if_not(fastEmbedR:::massive_cuda_device_count_cpp() >= 2L)
    multi <- pca(source, ncomp = 2L, backend = "cuda",
        massive = "out_of_core", devices = c(0L, 1L),
        output = file.path(directory, "multi.f32"),
        chunk_rows = 9L, memory_limit = "1GB")
    expect_identical(multi$gpu_devices, c(0L, 1L))
    expect_identical(vapply(multi$resources$per_device,
        `[[`, 0L, "gpu_device"), c(0L, 1L))
    expect_equal(as.matrix(multi$scores),
        as.matrix(single$scores), tolerance = 1e-5)
    checkpointed <- pca(source, ncomp = 2L, backend = "cuda",
        massive = "out_of_core", devices = c(0L, 1L),
        output = file.path(directory, "checkpointed.f32"),
        chunk_rows = 9L, memory_limit = "1GB", checkpoint = TRUE)
    expect_equal(as.matrix(checkpointed$scores),
        as.matrix(single$scores), tolerance = 1e-5)
    expect_false(file.exists(file.path(directory,
        "checkpointed.f32.checkpoint.rds")))
})

test_that("scaled PCA agrees across workers and file access modes", {
    set.seed(219)
    x <- matrix(rnorm(73L * 5L), ncol = 5L)
    raw <- make_massive_fixture(x)
    single_path <- tempfile(fileext = ".f32")
    parallel_path <- tempfile(fileext = ".f32")
    on.exit(unlink(c(raw, single_path, parallel_path)))
    streamed <- massive_matrix(raw, nrow = 73, ncol = 5)
    mapped <- massive_matrix(raw, nrow = 73, ncol = 5,
        access = "mmap")
    single <- pca(streamed, ncomp = 3L, backend = "cpu",
        n.cores = 1L, massive = "out_of_core", center = TRUE,
        scale = TRUE, output = single_path, chunk_rows = 11L)
    parallel <- pca(mapped, ncomp = 3L, backend = "cpu",
        n.cores = 3L, massive = "out_of_core", center = TRUE,
        scale = TRUE, output = parallel_path, chunk_rows = 13L)
    dense <- stats::prcomp(x, center = TRUE, scale. = TRUE)
    expect_equal(single$singular_values, parallel$singular_values,
        tolerance = 1e-5)
    expect_equal(single$singular_values,
        dense$sdev[1:3] * sqrt(nrow(x) - 1), tolerance = 1e-4)
    scores <- as.matrix(parallel$scores)
    for (component in 1:3) {
        expect_equal(abs(cor(scores[, component],
            dense$x[, component])), 1, tolerance = 1e-4)
    }
})

test_that("reservoir landmarks are reproducible and retain source rows", {
    rows <- seq_len(103)
    x <- cbind(rows, rows^2 / 10, sin(rows))
    raw <- make_massive_fixture(x)
    first_path <- tempfile(fileext = ".f32")
    second_path <- tempfile(fileext = ".f32")
    on.exit(unlink(c(raw, first_path, second_path)))
    source <- massive_matrix(raw, nrow = nrow(x), ncol = ncol(x))
    first <- massive_select_landmarks(
        source, landmarks = 12L, output = first_path,
        seed = 77L, chunk_rows = 13L
    )
    second <- massive_select_landmarks(
        source, landmarks = 12L, output = second_path,
        seed = 77L, chunk_rows = 17L
    )
    expect_identical(first$indices, second$indices)
    expect_true(all(diff(first$indices) > 0))
    expect_equal(as.matrix(first$data), unname(x[first$indices, ]),
        tolerance = 1e-5)
    expect_equal(as.matrix(first$data), as.matrix(second$data))
    expect_equal(file.info(first_path)$size, 12 * ncol(x) * 4)
    expect_equal(first$method, "reservoir")
    expect_error(massive_select_landmarks(source, 4.5, second_path),
        "valid integers")
    expect_error(massive_select_landmarks(
        source, 12L, tempfile(fileext = ".f32"),
        memory_limit = "1MB"
    ), "memory_limit")
})

test_that("random landmarks read sparse rows in source order", {
    rows <- seq_len(103)
    x <- cbind(rows, rows^2 / 10, sin(rows))
    raw <- make_massive_fixture(x, format = "fbin")
    first_path <- tempfile(fileext = ".f32")
    second_path <- tempfile(fileext = ".f32")
    on.exit(unlink(c(raw, first_path, second_path)))
    source <- massive_matrix(raw, access = "stream")
    mapped <- if (.Platform$OS.type == "windows") source else
        massive_matrix(raw, access = "mmap")
    first <- massive_select_landmarks(source, 12L, first_path,
        seed = 77L, chunk_rows = 3L,
        landmark_method = "random")
    second <- massive_select_landmarks(mapped, 12L, second_path,
        seed = 77L, chunk_rows = 7L,
        landmark_method = "random")
    expect_identical(first$indices, second$indices)
    expect_true(all(diff(first$indices) > 0))
    expect_identical(first$method, "random")
    expect_equal(as.matrix(first$data), unname(x[first$indices, ]),
        tolerance = 1e-5)
    expect_equal(as.matrix(first$data), as.matrix(second$data))
})

test_that("random sampling handles virtual trillion-row sources", {
    source <- fastEmbedR:::massive_synthetic_matrix(1e12, 3L)
    output <- tempfile(fileext = ".f32")
    on.exit(unlink(output))
    selected <- massive_select_landmarks(source, 6L, output,
        seed = 91L, chunk_rows = 2L,
        landmark_method = "random")
    expected <- outer(selected$indices - 1, 0:2,
        function(row, col) (row %% 10007) / 10007 + col * 0.01)
    expect_true(all(selected$indices > 2^31))
    expect_true(all(diff(selected$indices) > 0))
    expect_equal(as.matrix(selected$data), expected,
        tolerance = 1e-6)
})

test_that("virtual rows beyond signed 32-bit offsets remain addressable", {
    first <- 2^31 + 7
    source <- fastEmbedR:::massive_synthetic_matrix(2^31 + 20, 3)
    observed <- massive_read_rows(source, first = first, n = 3L)
    expected <- outer(first + 0:2 - 1, 0:2, function(row, col) {
        (row %% 10007) / 10007 + col * 0.01
    })
    expect_equal(observed, expected, tolerance = 1e-6)
    expect_error(massive_read_rows(source, first = 2^31 + 20,
        n = 2L), "range")
})

test_that("checkpointed PCA means use 64-bit source rows", {
    first <- 2^31 + 7
    source <- fastEmbedR:::massive_synthetic_matrix(2^31 + 20, 3)
    saved <- matrix(0, 2L, 3L)
    observed <- fastEmbedR:::massive_pca_mean_chunk_cpp(
        source, first, 3L, 2L, saved)
    expected <- massive_read_rows(source, first, 3L)
    expect_equal(colSums(observed), colSums(expected),
        tolerance = 1e-6)
    expect_equal(saved, matrix(0, 2L, 3L))
    expect_error(fastEmbedR:::massive_pca_mean_chunk_cpp(
        source, source$nrow, 2L, 2L, saved), "row range")
})

test_that("streamed and mapped files address billion-row offsets", {
    skip_on_os("windows")
    skip_if(.Machine$sizeof.pointer < 8L)
    path <- tempfile(fileext = ".f32")
    on.exit(unlink(path))
    con <- file(path, "w+b")
    on.exit(try(close(con), silent = TRUE), add = TRUE)
    last_row <- 1e9
    first <- last_row - 2
    seek(con, where = (first - 1) * 4, origin = "start",
        rw = "write")
    values <- c(1.25, -2.5, 3.75)
    writeBin(values, con, size = 4L, endian = "little")
    close(con)
    expect_equal(file.info(path)$size, 4 * last_row)
    for (mode in c("stream", "mmap")) {
        source <- massive_matrix(path, last_row, 1L, access = mode)
        observed <- massive_read_rows(source, first, 3L)
        expect_equal(drop(observed), values)
        view <- fastEmbedR:::massive_matrix_rows(source, first, 3L)
        expect_equal(drop(massive_read_rows(view, 1L, 3L)), values)
    }
})

read_massive_knn <- function(graph) {
    n <- graph$nrow * graph$ncol
    indices <- readBin(graph$indices_path, integer(), n = n,
        size = 4L, endian = "little")
    distances <- readBin(graph$distances_path, numeric(), n = n,
        size = 4L, endian = "little")
    list(indices = matrix(indices, ncol = graph$ncol, byrow = TRUE),
        distances = matrix(distances, ncol = graph$ncol, byrow = TRUE))
}

test_that("shard merge keeps same-numbered separate query neighbors", {
    prefix <- tempfile()
    paths <- paste0(prefix, c(".indices.u32", ".distances.f32"))
    on.exit(unlink(paths))
    first_ids <- matrix(c(1L, 2L, 2L, 1L, 1L, 2L),
        nrow = 3L, byrow = TRUE)
    first_dist <- matrix(c(0, 5, 0, 4, 1, 2),
        nrow = 3L, byrow = TRUE)
    second_ids <- matrix(c(1L, 2L, 1L, 2L, 1L, 2L),
        nrow = 3L, byrow = TRUE)
    second_dist <- matrix(c(2, 3, 2, 3, 0, 4),
        nrow = 3L, byrow = TRUE)
    fastEmbedR:::massive_merge_knn_shard_cpp(paths[[1L]],
        paths[[2L]], first_ids, first_dist, 1, 1, 2, 4,
        2L, TRUE, 3, FALSE)
    fastEmbedR:::massive_merge_knn_shard_cpp(paths[[1L]],
        paths[[2L]], second_ids, second_dist, 1, 3, 2, 4,
        2L, FALSE, 3, FALSE)
    observed <- matrix(readBin(paths[[1L]], integer(), n = 6L,
        size = 4L, endian = "little"), ncol = 2L, byrow = TRUE)
    expect_equal(observed, rbind(c(1L, 3L), c(2L, 3L),
        c(3L, 1L)))
    expect_equal(file.info(paths)$size, rep(24, 2L))
})

test_that("row-sharded KNN merges into exact source order", {
    set.seed(741)
    query <- matrix(rnorm(29L * 4L), ncol = 4L)
    reference <- matrix(rnorm(13L * 4L), ncol = 4L)
    raw <- make_massive_fixture(query)
    reference_file <- make_massive_fixture(reference)
    directory <- tempfile()
    dir.create(directory)
    on.exit(unlink(c(raw, reference_file, directory),
        recursive = TRUE))
    source <- massive_matrix(raw, nrow = 29L, ncol = 4L)
    landmarks <- massive_matrix(reference_file, nrow = 13L, ncol = 4L)
    prefix <- file.path(directory, "joined")
    tasks <- fastEmbedR:::massive_knn_device_tasks(source, 3L,
        prefix, c(0L, 1L))
    expect_equal(lapply(tasks, `[[`, "row_range"),
        list(c(1, 14), c(15, 29)))
    baseline <- massive_landmark_knn(source, landmarks, 3L,
        file.path(directory, "baseline"), chunk_rows = 5L,
        memory_limit = "256MB")
    results <- lapply(tasks, function(task) {
        view <- fastEmbedR:::massive_matrix_rows(source,
            task$row_range[1L], diff(task$row_range) + 1)
        massive_landmark_knn(view, landmarks, 3L, task$output,
            chunk_rows = 4L, memory_limit = "256MB")
    })
    paths <- fastEmbedR:::massive_knn_paths(prefix, 29 * 3 * 8)
    fastEmbedR:::massive_knn_merge_shards(results, paths, tasks)
    setup <- list(paths = paths,
        resources = list(output_bytes = 29 * 3 * 8))
    joined <- fastEmbedR:::massive_knn_devices_result(results,
        setup, source, c(0L, 1L), 0)
    expect_equal(read_massive_knn(joined), read_massive_knn(baseline))
    expect_identical(joined$gpu_devices, c(0L, 1L))
    expect_equal(joined$nrow, 29)
    expect_length(joined$per_device, 2L)
    mixed <- results
    mixed[[2L]]$reference_identity <- list(changed = TRUE)
    expect_error(fastEmbedR:::massive_knn_devices_result(
        mixed, setup, source, c(0L, 1L), 0),
        "different references")
    results[[2L]]$nprobe <- 3L
    heterogeneous <- fastEmbedR:::massive_knn_devices_result(
        results, setup, source, c(0L, 1L), 0)
    expect_true(is.na(heterogeneous$nprobe))
    expect_identical(heterogeneous$per_device[[2L]]$nprobe, 3L)
    expect_equal(file.info(paths)$size, rep(29 * 3 * 4, 2L))
    expect_error(massive_landmark_knn(source, landmarks, 3L,
        tempfile(), backend = "cpu", devices = 0L), "requires CUDA")
})

test_that("fresh multi-GPU KNN shards reject fallback and wrong method", {
    set.seed(753)
    query <- matrix(rnorm(9L * 3L), ncol = 3L)
    reference <- matrix(rnorm(6L * 3L), ncol = 3L)
    raw <- make_massive_fixture(query)
    ref <- make_massive_fixture(reference)
    directory <- tempfile()
    dir.create(directory)
    on.exit(unlink(c(raw, ref, directory), recursive = TRUE))
    source <- massive_matrix(raw, 9L, 3L)
    landmarks <- massive_matrix(ref, 6L, 3L)
    reported_backend <- "cpu"
    reported_method <- "exact"
    mock_worker <- function(tasks, worker, arguments) {
        lapply(tasks, function(task) {
            view <- fastEmbedR:::massive_matrix_rows(source,
                task$row_range[[1L]], diff(task$row_range) + 1)
            result <- massive_landmark_knn(view, landmarks, 2L,
                task$output, chunk_rows = 3L, memory_limit = "256MB")
            result$backend <- reported_backend
            result$method <- reported_method
            result$resources$gpu_device <- task$device
            result
        })
    }
    testthat::local_mocked_bindings(
        massive_cuda_parallel_tasks = mock_worker,
        .package = "fastEmbedR")
    controls <- list(x = source, reference = landmarks, k = 2L)
    run <- function(name) {
        paths <- fastEmbedR:::massive_knn_paths(
            file.path(directory, name), 9 * 2 * 8)
        setup <- list(paths = paths, reference = landmarks,
            k = 2L, method = "exact",
            resources = list(output_bytes = 9 * 2 * 8))
        list(paths = paths, result = fastEmbedR:::
            massive_knn_devices_execute(source, setup, c(0L, 1L),
                controls, FALSE, FALSE))
    }
    expect_error(run("cpu"), "inconsistent")
    expect_false(file.exists(file.path(directory,
        "cpu.indices.u32")))
    reported_backend <- "cuda"
    reported_method <- "ivf"
    expect_error(run("method"), "inconsistent")
    reported_method <- "exact"
    observed <- run("valid")
    baseline <- massive_landmark_knn(source, landmarks, 2L,
        file.path(directory, "baseline"), memory_limit = "256MB")
    expect_equal(read_massive_knn(observed$result),
        read_massive_knn(baseline))
})

test_that("multi-GPU KNN resumes verified shards and interrupted merges", {
    set.seed(751)
    query <- matrix(rnorm(19L * 4L), ncol = 4L)
    reference <- matrix(rnorm(11L * 4L), ncol = 4L)
    raw <- make_massive_fixture(query)
    ref <- make_massive_fixture(reference)
    directory <- tempfile()
    dir.create(directory)
    on.exit(unlink(c(raw, ref, directory), recursive = TRUE))
    source <- massive_matrix(raw, 19L, 4L)
    landmarks <- massive_matrix(ref, 11L, 4L)
    prefix <- file.path(directory, "joined")
    tasks <- fastEmbedR:::massive_knn_device_tasks(source, 3L,
        prefix, c(0L, 1L))
    results <- lapply(tasks, function(task) {
        view <- fastEmbedR:::massive_matrix_rows(source,
            task$row_range[1L], diff(task$row_range) + 1)
        result <- massive_landmark_knn(view, landmarks, 3L,
            task$output, chunk_rows = 4L, memory_limit = "256MB")
        result$backend <- "cuda"
        result$resources$gpu_device <- task$device
        fastEmbedR:::massive_checkpoint_write(result,
            paste0(task$output, ".result.rds"))
        result
    })
    status <- lapply(tasks, fastEmbedR:::massive_knn_device_status,
        x = source, reference = landmarks, k = 3L,
        method = "exact", resume = TRUE)
    expect_identical(vapply(status, `[[`, "", "status"),
        c("complete", "complete"))
    expect_equal(vapply(status, `[[`, 0, "remaining"), c(0, 0))
    changed <- results[[1L]]
    changed$backend <- "cpu"
    fastEmbedR:::massive_checkpoint_write(changed,
        paste0(tasks[[1L]]$output, ".result.rds"))
    expect_error(fastEmbedR:::massive_knn_device_status(
        tasks[[1L]], source, landmarks, 3L, "exact", TRUE),
        "inconsistent")
    fastEmbedR:::massive_checkpoint_write(results[[1L]],
        paste0(tasks[[1L]]$output, ".result.rds"))
    paths <- fastEmbedR:::massive_knn_paths(prefix, 19 * 3 * 8)
    plan <- list(paths = fastEmbedR:::massive_knn_paths(
        file.path(directory, "plan"), 19 * 3 * 8),
        reference = landmarks)
    controls <- list(x = source, reference = landmarks, k = 3L,
        method = "exact", chunk_rows = 4L)
    invalid <- list(paths = paths, reference = landmarks,
        k = 3L, method = "exact",
        resources = list(output_bytes = 19 * 3 * 8))
    expect_error(fastEmbedR:::massive_knn_devices_execute(
        source, invalid, c(0L, 1L), controls, TRUE, FALSE),
        "already exists")
    parent <- sub("\\.indices\\.u32$", "", paths[["indices"]])
    expect_false(file.exists(paste0(parent,
        ".multigpu.checkpoint.rds")))
    reference_view <- fastEmbedR:::massive_matrix_rows(
        landmarks, 1L, landmarks$nrow)
    view_plan <- list(paths = fastEmbedR:::massive_knn_paths(
        file.path(directory, "viewplan"), 19 * 3 * 8),
        reference = reference_view)
    expect_true(file.exists(fastEmbedR:::massive_knn_devices_checkpoint(
        source, view_plan, tasks, c(0L, 1L), controls, FALSE)))
    sidecar <- fastEmbedR:::massive_knn_devices_checkpoint(
        source, plan, tasks, c(0L, 1L), controls, FALSE)
    expect_identical(sidecar,
        fastEmbedR:::massive_knn_devices_checkpoint(source,
            plan, tasks, c(0L, 1L), controls, TRUE))
    controls$method <- "ivf"
    expect_error(fastEmbedR:::massive_knn_devices_checkpoint(
        source, plan, tasks, c(0L, 1L), controls, TRUE),
        "does not match")
    controls$method <- "exact"
    partial <- tasks[[1L]]
    partial$output <- file.path(directory, "partial")
    file.create(paste0(partial$output, c(
        ".indices.u32.part", ".distances.f32.part")))
    fastEmbedR:::massive_checkpoint_write(list(completed_rows = 0),
        paste0(partial$output, ".checkpoint.rds"))
    expect_identical(fastEmbedR:::massive_knn_device_status(
        partial, source, landmarks, 3L, "exact", TRUE)$status,
        "partial")
    file.create(paste0(paths[["indices"]], ".merge.part"))
    fastEmbedR:::massive_knn_merge_one(
        vapply(results, `[[`, "", "indices_path"),
        paths[["indices"]], vapply(tasks, `[[`, 0, "bytes"), TRUE)
    expect_silent(fastEmbedR:::massive_knn_paths(prefix,
        19 * 3 * 8, TRUE, TRUE))
    merged_setup <- list(paths = paths, reference = landmarks,
        k = 3L, method = "exact",
        resources = list(output_bytes = 19 * 3 * 8))
    merged_sidecar <- fastEmbedR:::massive_knn_devices_checkpoint(
        source, merged_setup, tasks, c(0L, 1L), controls, FALSE)
    joined <- fastEmbedR:::massive_knn_devices_execute(
        source, merged_setup, c(0L, 1L), controls, TRUE, TRUE)
    expect_false(file.exists(merged_sidecar))
    expect_equal(read_massive_knn(joined)$indices,
        do.call(rbind, lapply(results, function(result) {
            read_massive_knn(result)$indices
        })))
    expect_equal(file.info(paths)$size, rep(19 * 3 * 4, 2L))
    expect_error(fastEmbedR:::massive_knn_merge_shards(
        results, paths, tasks), "staging files|inconsistent")
    connection <- file(paths[["indices"]], "r+b")
    writeBin(0L, connection, size = 4L, endian = "little")
    close(connection)
    expect_error(fastEmbedR:::massive_knn_merge_one(
        vapply(results, `[[`, "", "indices_path"),
        paths[["indices"]], vapply(tasks, `[[`, 0, "bytes"), TRUE),
        "differs from its shards")
})

test_that("CUDA KNN coordinator resumes a verified device shard", {
    skip_if_not(fastEmbedR:::native_cuda_knn_available_cpp())
    set.seed(752)
    query <- matrix(rnorm(19L * 4L), ncol = 4L)
    reference <- matrix(rnorm(11L * 4L), ncol = 4L)
    raw <- make_massive_fixture(query)
    ref <- make_massive_fixture(reference)
    directory <- tempfile()
    dir.create(directory)
    on.exit(unlink(c(raw, ref, directory), recursive = TRUE))
    source <- massive_matrix(raw, 19L, 4L)
    landmarks <- massive_matrix(ref, 11L, 4L)
    output <- file.path(directory, "joined")
    setup <- fastEmbedR:::massive_knn_setup(source, landmarks, 3L,
        output, "cuda", 1L, "exact", 5L, "2GB", FALSE,
        cuda_memory = FALSE)
    arguments <- list(x = source, reference = setup$reference,
        k = setup$k, backend = "cuda", n.cores = 1L,
        method = setup$method, chunk_rows = 5L,
        memory_limit = "2GB")
    first <- fastEmbedR:::massive_knn_devices_execute(source, setup,
        0L, arguments, TRUE, FALSE)
    expected <- t(vapply(seq_len(nrow(query)), function(i) {
        distances <- rowSums((reference - matrix(query[i, ],
            nrow(reference), 4L, byrow = TRUE))^2)
        order(distances)[1:3]
    }, integer(3)))
    expect_identical(first$backend, "cuda")
    expect_identical(first$per_device[[1L]]$resources$gpu_device, 0L)
    expect_equal(read_massive_knn(first)$indices, expected)
    before <- tools::md5sum(unname(setup$paths))
    tasks <- fastEmbedR:::massive_knn_device_tasks(source, 3L,
        output, 0L)
    sidecar <- fastEmbedR:::massive_knn_devices_checkpoint(
        source, setup, tasks, 0L, arguments, FALSE)
    second <- fastEmbedR:::massive_knn_devices_execute(source, setup,
        0L, arguments, TRUE, TRUE)
    expect_identical(second$gpu_devices, 0L)
    expect_identical(unname(tools::md5sum(unname(setup$paths))),
        unname(before))
    expect_false(file.exists(sidecar))
})

test_that("multi-GPU KNN uses both requested devices without fallback", {
    skip_if_not(fastEmbedR:::native_cuda_knn_available_cpp())
    skip_if_not(fastEmbedR:::massive_cuda_device_count_cpp() >= 2L)
    set.seed(742)
    query <- matrix(rnorm(31L * 4L), ncol = 4L)
    reference <- matrix(rnorm(15L * 4L), ncol = 4L)
    raw <- make_massive_fixture(query)
    reference_file <- make_massive_fixture(reference)
    directory <- tempfile()
    dir.create(directory)
    on.exit(unlink(c(raw, reference_file, directory),
        recursive = TRUE))
    source <- massive_matrix(raw, nrow = 31L, ncol = 4L)
    landmarks <- massive_matrix(reference_file, nrow = 15L, ncol = 4L)
    single <- massive_landmark_knn(source, landmarks, 4L,
        file.path(directory, "single"), backend = "cuda",
        method = "exact", devices = 0L, memory_limit = "1GB")
    multi <- massive_landmark_knn(source, landmarks, 4L,
        file.path(directory, "multi"), backend = "cuda",
        method = "exact", devices = c(0L, 1L), memory_limit = "2GB",
        checkpoint = TRUE)
    expect_identical(multi$gpu_devices, c(0L, 1L))
    expect_identical(vapply(multi$per_device, function(value) {
        value$resources$gpu_device
    }, 0L), c(0L, 1L))
    expect_equal(read_massive_knn(multi), read_massive_knn(single),
        tolerance = 1e-5)
    sharded <- massive_landmark_knn(source, landmarks, 4L,
        file.path(directory, "sharded_multi"), backend = "cuda",
        method = "sharded_exact", devices = c(0L, 1L),
        reference_chunk_rows = 5L, chunk_rows = 4L,
        memory_limit = "2GB", checkpoint = TRUE)
    expect_identical(sharded$method, "sharded_exact")
    expect_identical(sharded$gpu_devices, c(0L, 1L))
    expect_identical(vapply(sharded$per_device, function(value) {
        value$resources$gpu_device
    }, 0L), c(0L, 1L))
    expect_equal(read_massive_knn(sharded), read_massive_knn(single),
        tolerance = 1e-5)
    expect_false(file.exists(paste0(
        file.path(directory, "multi"), ".multigpu.checkpoint.rds")))
    expect_false(file.exists(paste0(
        file.path(directory, "sharded_multi"),
        ".multigpu.checkpoint.rds")))
})

test_that("sharded CUDA landmark KNN keeps reference off device", {
    skip_if_not(fastEmbedR:::native_cuda_knn_available_cpp())
    set.seed(748)
    reference <- matrix(rnorm(23L * 5L), ncol = 5L)
    query <- matrix(rnorm(11L * 5L), ncol = 5L)
    query[1L, ] <- reference[1L, ]
    source_path <- make_massive_fixture(query)
    reference_path <- make_massive_fixture(reference)
    directory <- tempfile()
    dir.create(directory)
    on.exit(unlink(c(source_path, reference_path, directory),
        recursive = TRUE))
    source <- massive_matrix(source_path, 11L, 5L)
    landmarks <- massive_matrix(reference_path, 23L, 5L)
    graph <- massive_landmark_knn(source, landmarks, 3L,
        file.path(directory, "sharded"), backend = "cuda",
        method = "sharded_exact", chunk_rows = 4L,
        reference_chunk_rows = 7L, memory_limit = "256MB")
    observed <- read_massive_knn(graph)
    expected <- fastEmbedR:::native_exact_query_cpp(reference,
        query, 3L, n_threads = 1L, metric = "euclidean")
    expect_equal(observed$indices, expected$indices)
    expect_equal(observed$distances, expected$distances,
        tolerance = 1e-4)
    expect_identical(graph$backend, "cuda")
    expect_identical(graph$method, "sharded_exact")
    expect_identical(graph$query_storage, "native_float32")
    expect_true(graph$exact)
    expect_gt(graph$resources$n_shards, 1L)
    expect_identical(graph$reference_storage, "native_float32")
    expect_identical(observed$indices[1L, 1L], 1L)
    expect_equal(massive_audit_landmark_knn(graph, source,
        landmarks, sample_rows = 5L,
        memory_limit = "256MB")$observed_recall, 1)
})

test_that("sharded CUDA landmark KNN accepts 256 neighbors", {
    skip_if_not(fastEmbedR:::native_cuda_knn_available_cpp())
    set.seed(749)
    reference <- matrix(rnorm(260L * 3L), ncol = 3L)
    query <- matrix(rnorm(2L * 3L), ncol = 3L)
    query_path <- make_massive_fixture(query)
    reference_path <- make_massive_fixture(reference)
    directory <- tempfile()
    dir.create(directory)
    on.exit(unlink(c(query_path, reference_path, directory),
        recursive = TRUE))
    source <- massive_matrix(query_path, 2L, 3L)
    landmarks <- massive_matrix(reference_path, 260L, 3L)
    graph <- massive_landmark_knn(source, landmarks, 256L,
        file.path(directory, "sharded"), backend = "cuda",
        method = "sharded_exact", chunk_rows = 2L,
        reference_chunk_rows = 65L, memory_limit = "256MB")
    observed <- read_massive_knn(graph)
    expected <- fastEmbedR:::native_exact_query_cpp(reference,
        query, 256L, n_threads = 1L, metric = "euclidean")
    expect_equal(observed$indices, expected$indices)
    expect_equal(observed$distances, expected$distances,
        tolerance = 1e-4)
})

test_that("sharded CUDA landmark KNN resumes committed query rows", {
    skip_if_not(fastEmbedR:::native_cuda_knn_available_cpp())
    set.seed(750)
    reference <- matrix(rnorm(23L * 5L), ncol = 5L)
    query <- matrix(rnorm(11L * 5L), ncol = 5L)
    query_path <- make_massive_fixture(query)
    reference_path <- make_massive_fixture(reference)
    directory <- tempfile()
    dir.create(directory)
    on.exit(unlink(c(query_path, reference_path, directory),
        recursive = TRUE))
    source <- massive_matrix(query_path, 11L, 5L)
    landmarks <- massive_matrix(reference_path, 23L, 5L)
    expected <- massive_landmark_knn(source, landmarks, 3L,
        file.path(directory, "expected"), backend = "cuda",
        method = "sharded_exact", chunk_rows = 4L,
        reference_chunk_rows = 7L, memory_limit = "256MB")
    output <- file.path(directory, "resumed")
    request <- fastEmbedR:::massive_landmark_sharded_setup(
        source, landmarks, 3L, output, "cuda", 1L, 4L, 7L,
        "256MB", TRUE, FALSE, NULL)
    count <- ceiling(landmarks$nrow /
        request$setup$resources$n_shards)
    view <- fastEmbedR:::massive_matrix_rows(landmarks, 1L, count)
    state <- fastEmbedR:::massive_knn_reference(view, 3L,
        "exact", 1L, "cuda")
    first <- fastEmbedR:::massive_landmark_shard_batch(
        source, landmarks, request$setup, state, 1L, count,
        1L, TRUE, TRUE)
    request$saved$state$completed_rows <- first$rows
    request$saved$state$pilot_signature <- first$pilot
    fastEmbedR:::massive_checkpoint_write(request$saved$state,
        request$saved$path)
    fastEmbedR:::massive_landmark_shard_batch(source, landmarks,
        request$setup, state, 1L, count, 5L, TRUE)
    rm(state)
    invisible(gc(FALSE))
    expect_error(massive_landmark_knn(source, landmarks, 3L,
        output, backend = "cuda", method = "sharded_exact",
        chunk_rows = 5L, reference_chunk_rows = 7L,
        memory_limit = "256MB", checkpoint = TRUE,
        resume = TRUE), "checkpoint")
    observed <- massive_landmark_knn(source, landmarks, 3L,
        output, backend = "cuda", method = "sharded_exact",
        chunk_rows = 4L, reference_chunk_rows = 7L,
        memory_limit = "256MB", checkpoint = TRUE,
        resume = TRUE)
    expect_true(observed$resumed)
    expect_false(file.exists(request$saved$path))
    expect_equal(read_massive_knn(observed),
        read_massive_knn(expected))
})

test_that("sharded CUDA landmark requests reject unsupported modes", {
    source_path <- make_massive_fixture(matrix(1:30, ncol = 3L))
    source <- massive_matrix(source_path, 10L, 3L)
    on.exit(unlink(source_path))
    expect_error(massive_landmark_knn(source, source, 3L,
        tempfile(), backend = "cpu", method = "sharded_exact"),
        "needs CUDA")
    expect_error(massive_landmark_knn(source, source, 3L,
        tempfile(), backend = "cuda", method = "sharded_exact",
        resume = TRUE), "requires `checkpoint = TRUE`")
})

prepare_massive_knn_resume <- function(source, reference, graph,
                                        prefix, completed, chunk_rows,
                                        n.cores = 1L,
                                        reference_chunk_rows = NULL) {
    rows <- massive_read_knn_rows(graph, 1L, completed)
    paths <- paste0(prefix, c(".indices.u32.part",
        ".distances.f32.part"))
    writeBin(c(as.integer(t(rows$indices)), 0L), paths[[1L]],
        size = 4L, endian = "little")
    writeBin(c(as.vector(t(rows$distances)), 0), paths[[2L]],
        size = 4L, endian = "little")
    setup <- fastEmbedR:::massive_knn_setup(source, reference,
        graph$ncol, prefix, graph$backend, n.cores, graph$method,
        chunk_rows, "256MB", TRUE,
        reference_chunk_rows = reference_chunk_rows)
    saved <- fastEmbedR:::massive_knn_checkpoint(source, setup, FALSE)
    pilot <- massive_read_knn_rows(graph, 1L, chunk_rows)
    pilot$pilot_recall <- graph$pilot_recall
    pilot$pilot_id_recall <- graph$pilot_id_recall
    pilot$pilot_recall_metric <- graph$pilot_recall_metric
    pilot$pilot_rows <- graph$pilot_rows
    pilot$nlist <- graph$nlist
    pilot$nprobe <- graph$nprobe
    saved$state$completed_rows <- completed
    saved$state$pilot_signature <-
        fastEmbedR:::massive_knn_probe_signature(pilot)
    saved$state$pilot_metadata <-
        fastEmbedR:::massive_knn_metadata(pilot)
    fastEmbedR:::massive_checkpoint_write(saved$state, saved$path)
    paths
}

test_that("exact landmark KNN resumes without duplicating rows", {
    set.seed(817)
    query <- matrix(rnorm(23L * 4L), ncol = 4L)
    reference <- matrix(rnorm(17L * 4L), ncol = 4L)
    source <- massive_matrix(query)
    landmarks <- massive_matrix(reference)
    baseline_prefix <- tempfile()
    resume_prefix <- tempfile()
    files <- paste0(rep(c(baseline_prefix, resume_prefix), each = 2L),
        c(".indices.u32", ".distances.f32"))
    on.exit(unlink(c(files, paste0(resume_prefix,
        c(".indices.u32.part", ".distances.f32.part",
            ".checkpoint.rds")))))
    baseline <- massive_landmark_knn(source, landmarks, 4L,
        baseline_prefix, chunk_rows = 7L, memory_limit = "256MB")
    parts <- prepare_massive_knn_resume(source, landmarks, baseline,
        resume_prefix, 14L, 7L)
    expect_error(massive_landmark_knn(source, landmarks, 4L,
        resume_prefix, chunk_rows = 6L, memory_limit = "256MB",
        checkpoint = TRUE, resume = TRUE), "does not match")
    sidecar <- paste0(resume_prefix, ".checkpoint.rds")
    state <- readRDS(sidecar)
    saveRDS("broken", sidecar)
    expect_error(massive_landmark_knn(source, landmarks, 4L,
        resume_prefix, chunk_rows = 7L, memory_limit = "256MB",
        checkpoint = TRUE, resume = TRUE),
        "KNN checkpoint or partial output does not match")
    saveRDS(state, sidecar)
    state$pilot_signature <- "invalid pilot"
    fastEmbedR:::massive_checkpoint_write(state, sidecar)
    size_before <- file.info(parts)$size
    expect_error(massive_landmark_knn(source, landmarks, 4L,
        resume_prefix, chunk_rows = 7L, memory_limit = "256MB",
        checkpoint = TRUE, resume = TRUE), "pilot")
    expect_equal(file.info(parts)$size, size_before)
    pilot <- massive_read_knn_rows(baseline, 1L, 7L)
    state$pilot_signature <-
        fastEmbedR:::massive_knn_probe_signature(pilot)
    fastEmbedR:::massive_checkpoint_write(state, sidecar)
    expect_true(all(file.exists(parts)))
    resumed <- massive_landmark_knn(source, landmarks, 4L,
        resume_prefix, chunk_rows = 7L, memory_limit = "256MB",
        checkpoint = TRUE, resume = TRUE)
    expect_equal(read_massive_knn(resumed), read_massive_knn(baseline))
    expect_false(any(file.exists(c(parts,
        paste0(resume_prefix, ".checkpoint.rds")))))
    expect_equal(file.info(resumed$indices_path)$size, 23 * 4 * 4)
})

test_that("KNN resumes before the first output file was created", {
    set.seed(820)
    source <- massive_matrix(matrix(rnorm(13L * 3L), ncol = 3L))
    landmarks <- massive_matrix(matrix(rnorm(9L * 3L), ncol = 3L))
    prefix <- tempfile()
    on.exit(unlink(paste0(prefix, c(".indices.u32",
        ".distances.f32", ".checkpoint.rds"))))
    setup <- fastEmbedR:::massive_knn_setup(source, landmarks, 3L,
        prefix, "cpu", 1L, "exact", 5L, "256MB", TRUE)
    fastEmbedR:::massive_knn_checkpoint(source, setup, FALSE)
    graph <- massive_landmark_knn(source, landmarks, 3L, prefix,
        chunk_rows = 5L, memory_limit = "256MB",
        checkpoint = TRUE, resume = TRUE)
    expected <- fastEmbedR:::native_exact_query_cpp(landmarks$data,
        source$data, 3L, 1L, "euclidean", 0.99)
    expect_equal(read_massive_knn(graph)$indices, expected$indices)
    expect_false(file.exists(paste0(prefix, ".checkpoint.rds")))
})

test_that("HNSW landmark KNN resumes with a rebuilt index", {
    set.seed(818)
    source <- massive_matrix(matrix(rnorm(31L * 5L), ncol = 5L))
    landmarks <- massive_matrix(matrix(rnorm(140L * 5L), ncol = 5L))
    baseline_prefix <- tempfile()
    resume_prefix <- tempfile()
    on.exit(unlink(paste0(rep(c(baseline_prefix, resume_prefix),
        each = 2L), c(".indices.u32", ".distances.f32"))))
    baseline <- massive_landmark_knn(source, landmarks, 6L,
        baseline_prefix, method = "hnsw", n.cores = 2L,
        chunk_rows = 9L, memory_limit = "256MB")
    prepare_massive_knn_resume(source, landmarks, baseline,
        resume_prefix, 18L, 9L, n.cores = 2L)
    resumed <- massive_landmark_knn(source, landmarks, 6L,
        resume_prefix, method = "hnsw", n.cores = 2L,
        chunk_rows = 9L, memory_limit = "256MB",
        checkpoint = TRUE, resume = TRUE)
    expect_equal(read_massive_knn(resumed), read_massive_knn(baseline))
})

test_that("streamed exact landmark KNN retains every query row", {
    set.seed(109)
    reference <- matrix(rnorm(32L * 4L), ncol = 4L)
    query <- matrix(rnorm(67L * 4L), ncol = 4L)
    source_path <- make_massive_fixture(query)
    reference_path <- make_massive_fixture(reference)
    prefix <- tempfile()
    on.exit(unlink(c(source_path, reference_path,
        paste0(prefix, c(".indices.u32", ".distances.f32")))))
    source <- massive_matrix(source_path, nrow = 67, ncol = 4)
    landmarks <- massive_matrix(reference_path, nrow = 32, ncol = 4)
    graph <- massive_landmark_knn(source, landmarks, 5L, prefix,
        chunk_rows = 11L, memory_limit = "256MB")
    observed <- read_massive_knn(graph)
    expected <- fastEmbedR:::native_exact_query_cpp(
        reference, query, 5L, 1L, "euclidean", 0.99)
    expect_identical(graph$method, "exact")
    expect_true(graph$index_reused)
    expect_equal(observed$indices, expected$indices)
    expect_equal(observed$distances, expected$distances,
        tolerance = 1e-5)
    window <- massive_read_knn_rows(graph, first = 23, n = 17L)
    expect_equal(window$indices, observed$indices[23:39, ])
    expect_equal(window$distances, observed$distances[23:39, ])
    expect_equal(head(graph, 2L)$indices, observed$indices[1:2, ])
    expect_error(massive_read_knn_rows(graph, 66, 3L), "range")
    expect_equal(file.info(graph$indices_path)$size, 67 * 5 * 4)
    expect_equal(file.info(graph$distances_path)$size, 67 * 5 * 4)
    audited <- massive_audit_landmark_knn(graph, source, landmarks,
        sample_rows = 7L, memory_limit = "256MB")
    expect_identical(audited$audit_reference_backend,
        "cpu_exact_stream")
    expect_equal(audited$observed_recall, 1)
    expect_equal(audited$minimum_row_recall, 1)
    expect_equal(length(audited$audit_sample_rows), 7L)
    expect_identical(audited$audit_reference_passes, 1L)
    expect_true(audited$sample_target_met)
    expect_error(massive_audit_landmark_knn(graph, source,
        landmarks, sample_rows = 68L), "matching file-backed")
    if (!fastEmbedR:::native_cuda_knn_available_cpp()) {
        expect_error(massive_landmark_knn(source, landmarks, 5L,
            tempfile(), backend = "cuda"), "no CPU fallback")
    }
})

test_that("landmark HNSW reports sampled recall and source identity", {
    set.seed(110)
    query_path <- make_massive_fixture(matrix(rnorm(60L * 4L),
        ncol = 4L))
    reference_path <- make_massive_fixture(matrix(rnorm(90L * 4L),
        ncol = 4L))
    prefix <- tempfile()
    on.exit(unlink(c(query_path, reference_path,
        paste0(prefix, c(".indices.u32", ".distances.f32")))))
    query <- massive_matrix(query_path, 60L, 4L)
    reference <- massive_matrix(reference_path, 90L, 4L)
    knn <- massive_landmark_knn(query, reference, 5L,
        prefix, method = "hnsw", memory_limit = "256MB")
    expect_false(knn$recall_audited)
    audited <- massive_audit_landmark_knn(knn, query, reference,
        sample_rows = 8L, memory_limit = "256MB")
    expect_true(audited$recall_audited)
    expect_equal(length(audited$audit_sample_rows), 8L)
    expect_gte(audited$observed_recall, 0)
    expect_lte(audited$observed_recall, 1)
    expect_gte(audited$minimum_row_recall, 0)
    expect_lte(audited$minimum_row_recall, audited$observed_recall)
    Sys.setFileTime(reference_path, Sys.time() + 3600)
    expect_error(massive_audit_landmark_knn(knn, query, reference,
        sample_rows = 2L), "changed since fitting")
})

test_that("exact landmark search can stream a nonresident reference", {
    set.seed(1031)
    query <- matrix(rnorm(29L * 6L), ncol = 6L)
    reference <- matrix(rnorm(43L * 6L), ncol = 6L)
    query_path <- make_massive_fixture(query)
    reference_path <- make_massive_fixture(reference, "fbin")
    directory <- tempfile()
    dir.create(directory)
    on.exit(unlink(c(query_path, reference_path, directory),
        recursive = TRUE))
    source <- massive_matrix(query_path, 29L, 6L, access = "mmap")
    landmarks <- massive_matrix(reference_path, access = "stream")
    resident <- massive_landmark_knn(source, landmarks, 5L,
        file.path(directory, "resident"), method = "exact",
        chunk_rows = 7L, memory_limit = "256MB")
    streamed <- massive_landmark_knn(source, landmarks, 5L,
        file.path(directory, "streamed"), method = "stream_exact",
        chunk_rows = 7L, reference_chunk_rows = 9L,
        n.cores = 2L, memory_limit = "256MB")
    native <- fastEmbedR:::massive_knn_reference(
        landmarks, 5L, "exact", 1L, "cpu")
    first <- fastEmbedR:::massive_knn_batch(source, native,
        1L, 7L, 5L, "exact", 1L, "cpu")
    expect_identical(first$input_type, "float32")
    expect_identical(resident$query_storage, "native_float32")
    expect_identical(streamed$query_storage, "file_backed_float32")
    expect_identical(streamed$method, "stream_exact")
    expect_identical(streamed$reference_storage, "file_backed_float32")
    expect_true(streamed$recall_audited)
    expect_false(streamed$index_reused)
    expect_identical(streamed$resources$reference_chunk_rows, 9L)
    expect_lt(streamed$resources$peak_ram_bytes,
        resident$resources$peak_ram_bytes)
    expect_equal(read_massive_knn(streamed), read_massive_knn(resident),
        tolerance = 1e-5)
    mapped <- massive_matrix(reference_path, access = "mmap")
    mapped_search <- massive_landmark_knn(source, mapped, 5L,
        file.path(directory, "mapped"), method = "stream_exact",
        chunk_rows = 6L, reference_chunk_rows = 8L,
        memory_limit = "256MB")
    expect_equal(read_massive_knn(mapped_search),
        read_massive_knn(resident), tolerance = 1e-5)
    expect_error(massive_landmark_knn(source, landmarks, 5L,
        file.path(directory, "wrong"), method = "exact",
        reference_chunk_rows = 9L), "requires `stream_exact`")
    expect_error(massive_landmark_knn(source, massive_matrix(reference),
        5L, file.path(directory, "memory"), method = "stream_exact"),
        "file-backed reference")
    expect_error(massive_landmark_knn(source, landmarks, 5L,
        file.path(directory, "cuda"), backend = "cuda",
        method = "stream_exact", reference_chunk_rows = 9L),
        "no CUDA fallback")
})

test_that("streamed-reference KNN resumes with the same chunk policy", {
    set.seed(1032)
    query <- matrix(rnorm(27L * 5L), ncol = 5L)
    reference <- matrix(rnorm(33L * 5L), ncol = 5L)
    query_path <- make_massive_fixture(query)
    reference_path <- make_massive_fixture(reference)
    directory <- tempfile()
    dir.create(directory)
    on.exit(unlink(c(query_path, reference_path, directory),
        recursive = TRUE))
    source <- massive_matrix(query_path, 27L, 5L)
    landmarks <- massive_matrix(reference_path, 33L, 5L)
    baseline <- massive_landmark_knn(source, landmarks, 4L,
        file.path(directory, "baseline"), method = "stream_exact",
        chunk_rows = 7L, reference_chunk_rows = 11L,
        memory_limit = "256MB")
    prefix <- file.path(directory, "resumed")
    parts <- prepare_massive_knn_resume(source, landmarks, baseline,
        prefix, 14L, 7L, reference_chunk_rows = 11L)
    sizes <- file.info(parts)$size
    expect_error(massive_landmark_knn(source, landmarks, 4L,
        prefix, method = "stream_exact", chunk_rows = 7L,
        reference_chunk_rows = 10L, memory_limit = "256MB",
        checkpoint = TRUE, resume = TRUE), "does not match")
    expect_equal(file.info(parts)$size, sizes)
    resumed <- massive_landmark_knn(source, landmarks, 4L,
        prefix, method = "stream_exact", chunk_rows = 7L,
        reference_chunk_rows = 11L, memory_limit = "256MB",
        checkpoint = TRUE, resume = TRUE)
    expect_equal(read_massive_knn(resumed), read_massive_knn(baseline))
    expect_false(file.exists(paste0(prefix, ".checkpoint.rds")))
})

test_that("streamed-reference KNN accepts 64-bit query row offsets", {
    reference <- matrix(seq_len(33L) / 19, ncol = 3L)
    reference_path <- make_massive_fixture(reference)
    on.exit(unlink(reference_path))
    query <- fastEmbedR:::massive_synthetic_matrix(1e12, 3L)
    landmarks <- massive_matrix(reference_path, 11L, 3L)
    first <- 2^31 + 17
    observed <- fastEmbedR:::massive_exact_reference_batch_cpp(
        query, landmarks, first, 3L, 3L, 4L, 2L)
    query_rows <- massive_read_rows(query, first, 3L)
    stored_reference <- massive_read_rows(landmarks, 1, 11L)
    for (row in seq_len(3L)) {
        distances <- sqrt(rowSums((stored_reference - matrix(
            query_rows[row, ], 11L, 3L, byrow = TRUE))^2))
        nearest <- order(distances, seq_len(11L))[1:3]
        expect_identical(observed$indices[row, ], as.integer(nearest))
        expect_equal(observed$distances[row, ], distances[nearest],
            tolerance = 1e-6)
    }
    huge_reference <- fastEmbedR:::massive_synthetic_matrix(1e9, 3L)
    resources <- fastEmbedR:::massive_full_knn_resources(query,
        3L, 3L, 128L, "256MB", huge_reference)
    expect_lt(resources$peak_ram_bytes, 100e6)
    expect_identical(resources$reference_chunk_rows, 128L)
})

test_that("persistent CUDA landmark search preserves exact neighbors", {
    skip_if_not(fastEmbedR:::native_cuda_knn_available_cpp())
    set.seed(271)
    reference <- matrix(rnorm(47L * 5L), ncol = 5L)
    query <- matrix(rnorm(31L * 5L), ncol = 5L)
    reference_path <- make_massive_fixture(reference, "fbin")
    query_path <- make_massive_fixture(query, "fbin")
    prefix <- tempfile()
    on.exit(unlink(c(reference_path, query_path, paste0(prefix,
        c(".indices.u32", ".distances.f32")))))
    graph <- massive_landmark_knn(
        massive_matrix(query_path), massive_matrix(reference_path), 6L,
        prefix, backend = "cuda", method = "exact", chunk_rows = 7L,
        memory_limit = "256MB")
    observed <- read_massive_knn(graph)
    expected <- fastEmbedR:::native_exact_query_cpp(
        reference, query, 6L, 1L, "euclidean", 0.99)
    expect_identical(graph$backend, "cuda")
    expect_identical(graph$method, "exact")
    expect_identical(graph$reference_storage, "native_float32")
    expect_identical(graph$query_storage, "native_float32")
    expect_true(graph$index_reused)
    expect_equal(observed$indices, expected$indices)
    expect_equal(observed$distances, expected$distances,
        tolerance = 1e-4)
    expect_error(massive_landmark_knn(
        massive_matrix(query), massive_matrix(reference), 6L,
        tempfile(), backend = "cuda", method = "hnsw"),
        "no algorithm fallback")
})

test_that("persistent CUDA search reuses bounded query buffers", {
    skip_if_not(fastEmbedR:::native_cuda_knn_available_cpp())
    set.seed(219)
    reference <- matrix(rnorm(80L * 6L), ncol = 6L)
    query <- matrix(rnorm(24L * 6L), ncol = 6L)
    index <- fastEmbedR:::native_cuda_index_build_cpp(
        reference, 5L, "exact", 0.99)
    batches <- list(1:10, 11:14, 1:16, 1:4)
    reused <- c(FALSE, TRUE, FALSE, TRUE)
    for (i in seq_along(batches)) {
        rows <- batches[[i]]
        found <- fastEmbedR:::native_cuda_index_search_cpp(
            index, query[rows, , drop = FALSE], 5L)
        expected <- fastEmbedR:::native_exact_query_cpp(
            reference, query[rows, , drop = FALSE],
            5L, 1L, "euclidean", 0.99)
        expect_identical(found$backend, "native_cuda_cuvs_exact")
        expect_identical(found$query_buffers_reused, reused[[i]])
        expect_equal(found$indices, expected$indices)
        expect_equal(found$distances, expected$distances,
            tolerance = 1e-4)
    }
})

test_that("CUDA landmark KNN resumes on CUDA without fallback", {
    skip_if_not(fastEmbedR:::native_cuda_knn_available_cpp())
    set.seed(819)
    source <- massive_matrix(matrix(rnorm(19L * 5L), ncol = 5L))
    landmarks <- massive_matrix(matrix(rnorm(37L * 5L), ncol = 5L))
    baseline_prefix <- tempfile()
    resume_prefix <- tempfile()
    on.exit(unlink(paste0(rep(c(baseline_prefix, resume_prefix),
        each = 2L), c(".indices.u32", ".distances.f32"))))
    baseline <- massive_landmark_knn(source, landmarks, 5L,
        baseline_prefix, method = "exact", backend = "cuda",
        chunk_rows = 6L, memory_limit = "256MB")
    prepare_massive_knn_resume(source, landmarks, baseline,
        resume_prefix, 12L, 6L)
    resumed <- massive_landmark_knn(source, landmarks, 5L,
        resume_prefix, method = "exact", backend = "cuda",
        chunk_rows = 6L, memory_limit = "256MB",
        checkpoint = TRUE, resume = TRUE)
    expect_identical(resumed$backend, "cuda")
    expect_equal(read_massive_knn(resumed), read_massive_knn(baseline),
        tolerance = 1e-5)
})

test_that("massive CUDA search selects IVF without CPU fallback", {
    choose <- fastEmbedR:::massive_knn_method
    expect_identical(choose("auto", "cuda", 99999, 10L), "exact")
    expect_identical(choose("auto", "cuda", 100000, 10L), "ivf")
    expect_error(choose("ivf", "cpu", 100000, 10L),
        "no algorithm fallback")
    expect_error(choose("hnsw", "cuda", 100000, 10L),
        "no algorithm fallback")
})

test_that("CUDA KNN planning caps batches before allocation", {
    resources <- list(chunk_rows = 250000L)
    reference <- list(nrow = 1000, ncol = 1024)
    info <- list(device = 0L, free_bytes = 400 * 1024^2)
    plan <- fastEmbedR:::massive_knn_cuda_resources(
        resources, reference, 30L, "ivf", info)
    expect_lt(plan$chunk_rows, 32768L)
    expect_lte(plan$peak_vram_bytes, plan$vram_budget_bytes)
    expect_identical(plan$gpu_device, 0L)
    info$free_bytes <- 100 * 1024^2
    expect_error(fastEmbedR:::massive_knn_cuda_resources(
        resources, reference, 30L, "ivf", info), "free-VRAM")
})

test_that("persistent CUDA IVF reports calibrated pilot recall", {
    skip_if_not(fastEmbedR:::native_cuda_knn_available_cpp())
    set.seed(273)
    reference <- matrix(rnorm(15000L * 16L), ncol = 16L)
    query <- matrix(rnorm(128L * 16L), ncol = 16L)
    query_path <- make_massive_fixture(query)
    reference_path <- make_massive_fixture(reference)
    prefix <- tempfile()
    on.exit(unlink(c(query_path, reference_path, paste0(prefix,
        c(".indices.u32", ".distances.f32")))))
    query_source <- massive_matrix(query_path, 128L, 16L)
    reference_source <- massive_matrix(reference_path, 15000L, 16L)
    graph <- massive_landmark_knn(
        query_source, reference_source, 10L,
        prefix, backend = "cuda", method = "ivf",
        chunk_rows = 64L, memory_limit = "256MB")
    observed <- read_massive_knn(graph)
    expected <- fastEmbedR:::native_exact_query_cpp(
        reference, query, 10L, 1L, "euclidean", 0.99)
    recall <- mean(vapply(seq_len(nrow(query)), function(i) {
        length(intersect(observed$indices[i, ],
            expected$indices[i, ])) / 10
    }, numeric(1)))
    expect_identical(graph$method, "ivf")
    expect_true(graph$index_reused)
    expect_false(graph$recall_audited)
    expect_identical(graph$pilot_recall_metric, "distance_at_k")
    expect_equal(graph$pilot_rows, 64L)
    expect_gte(graph$pilot_recall, 0.99)
    expect_gte(graph$pilot_id_recall, 0.98)
    expect_lte(graph$nprobe, graph$nlist)
    expect_lte(graph$resources$peak_vram_bytes,
        graph$resources$vram_budget_bytes)
    expect_gte(recall, 0.98)
    expect_true(all(is.finite(observed$distances)))
    audited <- massive_audit_landmark_knn(graph,
        query_source, reference_source, sample_rows = 5L,
        memory_limit = "256MB")
    rows <- audited$audit_sample_rows
    exact_sample <- mean(vapply(rows, function(i) {
        length(intersect(observed$indices[i, ],
            expected$indices[i, ])) / 10
    }, numeric(1)))
    expect_equal(audited$observed_recall, exact_sample)
    expect_identical(audited$audit_reference_backend,
        "cpu_exact_stream")
})

test_that("persistent CUDA pilot accepts distance-equivalent ties", {
    skip_if_not(fastEmbedR:::native_cuda_knn_available_cpp())
    set.seed(274)
    base <- matrix(rnorm(100L * 16L), ncol = 16L)
    reference <- base[rep(seq_len(100L), each = 50L), ]
    query <- base[seq_len(32L), ]
    index <- fastEmbedR:::native_cuda_index_build_cpp(
        reference, 10L, "ivf", 0.99)
    result <- fastEmbedR:::native_cuda_index_search_cpp(
        index, query, 10L)
    expect_identical(result$backend, "native_cuda_cuvs_ivf_flat")
    expect_identical(result$pilot_recall_metric, "distance_at_k")
    expect_gte(result$pilot_recall, 0.99)
    expect_lte(result$pilot_id_recall, result$pilot_recall)
    expect_true(all(result$distances < 1e-4))
})

test_that("persistent CPU exact search reuses a float32 reference", {
    set.seed(211)
    reference <- matrix(rnorm(120L * 6L), ncol = 6L)
    query <- matrix(rnorm(29L * 6L), ncol = 6L)
    reference_path <- make_massive_fixture(reference)
    query_path <- make_massive_fixture(query)
    prefix <- tempfile()
    on.exit(unlink(c(reference_path, query_path,
        paste0(prefix, c(".indices.u32", ".distances.f32")))))
    source <- massive_matrix(query_path, nrow = 29, ncol = 6)
    landmarks <- massive_matrix(reference_path, nrow = 120, ncol = 6)
    graph <- massive_landmark_knn(source, landmarks, 5L, prefix,
        method = "exact", chunk_rows = 7L, n.cores = 2L,
        memory_limit = "256MB")
    observed <- read_massive_knn(graph)
    expected <- fastEmbedR:::native_exact_query_cpp(
        reference, query, 5L, 2L)
    expect_identical(observed$indices, expected$indices)
    expect_equal(observed$distances, expected$distances,
        tolerance = 1e-5)
    expect_true(graph$index_reused)
    expect_identical(graph$reference_storage, "native_float32")
    expect_identical(graph$query_storage, "native_float32")
    expect_error(fastEmbedR:::native_exact_index_search_cpp(
        new.env(), query, 5L), "persistent exact index")
})

test_that("persistent HNSW agrees across bounded query batches", {
    set.seed(203)
    reference <- matrix(rnorm(140L * 5L), ncol = 5L)
    query <- matrix(rnorm(53L * 5L), ncol = 5L)
    source_path <- make_massive_fixture(query)
    reference_path <- make_massive_fixture(reference)
    prefix <- tempfile()
    on.exit(unlink(c(source_path, reference_path,
        paste0(prefix, c(".indices.u32", ".distances.f32")))))
    source <- massive_matrix(source_path, nrow = 53, ncol = 5)
    landmarks <- massive_matrix(reference_path, nrow = 140, ncol = 5)
    graph <- massive_landmark_knn(source, landmarks, 7L, prefix,
        method = "hnsw", chunk_rows = 9L, n.cores = 2L,
        memory_limit = "256MB")
    observed <- read_massive_knn(graph)
    expected <- fastEmbedR:::native_hnsw_query_cpp(
        reference, query, 7L, 2L, "euclidean", 0.99)
    expect_true(graph$index_reused)
    expect_identical(graph$reference_storage, "native_float32")
    expect_identical(graph$query_storage, "native_float32")
    expect_false(graph$recall_audited)
    expect_equal(observed$indices, expected$indices)
    expect_equal(observed$distances, expected$distances,
        tolerance = 1e-5)
    expect_error(fastEmbedR:::native_hnsw_index_search_cpp(
        new.env(), query, 7L), "live persistent")
})

test_that("file-backed HNSW builds from one native float32 buffer", {
    set.seed(210)
    reference <- matrix(rnorm(180L * 6L), ncol = 6L)
    query <- matrix(rnorm(24L * 6L), ncol = 6L)
    path <- make_massive_fixture(reference, "fbin")
    on.exit(unlink(path))
    file <- massive_matrix(path, access = "mmap")
    resident <- massive_matrix(reference)
    native <- fastEmbedR:::massive_knn_reference(
        file, 5L, "hnsw", 1L, "cpu")
    memory <- fastEmbedR:::massive_knn_reference(
        resident, 5L, "hnsw", 1L, "cpu")
    search <- fastEmbedR:::native_hnsw_index_search_cpp
    observed <- search(native$index$pointer, query, 5L, 1L)
    expected <- search(memory$index$pointer, query, 5L, 1L)
    expect_identical(native$reference_storage, "native_float32")
    expect_identical(memory$reference_storage, "R_double")
    expect_true(native$spanning_links)
    expect_true(memory$spanning_links)
    expect_equal(observed$indices, expected$indices)
    expect_equal(observed$distances, expected$distances,
        tolerance = 1e-5)
    buffer <- fastEmbedR:::massive_reference_buffer_cpp(file)
    fastEmbedR:::native_hnsw_index_build_cpp(buffer, 5L)
    expect_error(fastEmbedR:::native_hnsw_index_build_cpp(
        buffer, 5L), "already consumed")
})

test_that("CUDA projection engine reflects reference and batch costs", {
    choose <- fastEmbedR:::massive_cuda_projection_engine
    expect_identical(choose(100000, 2, 30, 16384), "batch_cuda")
    expect_identical(choose(500000, 2, 30, 16384),
        "persistent_cuda")
    expect_identical(choose(100000, 2, 30, 1024),
        "persistent_cuda")
})

test_that("large-reference CUDA projection reuses the saved layout", {
    skip_if_not(isTRUE(fastEmbedR:::embedding_cuda_available_cpp()))
    set.seed(32)
    reference <- matrix(rnorm(500000L * 2L), ncol = 2L)
    fit <- umap(matrix(rnorm(24L * 4L), ncol = 4L),
        n_neighbors = 5L, backend = "cpu")
    fit$layout <- reference
    indices <- matrix(rep(1:5, 11L), nrow = 11L, byrow = TRUE)
    distances <- matrix(rep(seq(0.1, 0.5, length.out = 5L),
        11L), nrow = 11L, byrow = TRUE)
    index_path <- tempfile(fileext = ".u32")
    distance_path <- tempfile(fileext = ".f32")
    output <- tempfile(fileext = ".f32")
    on.exit(unlink(c(index_path, distance_path, output)))
    writeBin(as.integer(t(indices)), index_path, size = 4L,
        endian = "little")
    writeBin(as.vector(t(distances)), distance_path, size = 4L,
        endian = "little")
    graph <- structure(list(indices_path = index_path,
        distances_path = distance_path, nrow = 11L, ncol = 5L,
        n_reference = 500000L, metric = "euclidean",
        backend = "cpu"), class = "fastEmbedR_massive_knn")
    observed <- massive_project_landmarks(fit, graph, output,
        backend = "cuda", refinement_epochs = 0L,
        chunk_rows = 7L, memory_limit = "256MB")
    expected <- fastEmbedR:::project_embedding_knn_cuda_cpp(
        reference, indices, distances)
    expect_identical(observed$resources$projection_engine,
        "persistent_cuda")
    expect_equal(as.matrix(observed$layout), expected,
        tolerance = 1e-5)
})

test_that("file-backed UMAP projection reuses saved KNN rows", {
    set.seed(231)
    reference <- matrix(rnorm(24L * 4L), ncol = 4L)
    query <- matrix(rnorm(39L * 4L), ncol = 4L)
    source_path <- make_massive_fixture(query)
    reference_path <- make_massive_fixture(reference)
    prefix <- tempfile()
    output <- tempfile(fileext = ".f32")
    gpu_output <- tempfile(fileext = ".f32")
    gpu_output3 <- tempfile(fileext = ".f32")
    on.exit(unlink(c(source_path, reference_path, output, gpu_output,
        gpu_output3,
        paste0(prefix, c(".indices.u32", ".distances.f32")))))
    source <- massive_matrix(source_path, nrow = 39, ncol = 4)
    landmarks <- massive_matrix(reference_path, nrow = 24, ncol = 4)
    graph <- massive_landmark_knn(source, landmarks, 5L, prefix,
        chunk_rows = 9L, memory_limit = "256MB")
    fit <- umap(reference, n_neighbors = 5L, backend = "cpu")
    projected <- massive_project_landmarks(
        fit, graph, output, chunk_rows = 7L,
        refinement_epochs = 0L, memory_limit = "256MB")
    expected <- fastEmbedR:::project_embedding_knn_cpp(
        fit$layout, read_massive_knn(graph)$indices,
        read_massive_knn(graph)$distances)
    expect_equal(as.matrix(projected$layout), expected,
        tolerance = 1e-5)
    expect_equal(file.info(output)$size, 39 * 2 * 4)
    expect_identical(projected$backend, "cpu")
    expect_true(projected$experimental)
    first_path <- tempfile(fileext = ".f32")
    second_path <- tempfile(fileext = ".f32")
    joined_path <- tempfile(fileext = ".f32")
    on.exit(unlink(c(first_path, second_path, joined_path)), add = TRUE)
    first <- massive_project_landmarks(fit, graph, first_path,
        row_range = c(1, 18), chunk_rows = 7L,
        refinement_epochs = 0L, memory_limit = "256MB")
    second <- massive_project_landmarks(fit, graph, second_path,
        row_range = c(19, 39), chunk_rows = 7L,
        refinement_epochs = 0L, memory_limit = "256MB")
    expect_identical(dim(as.matrix(first$layout)), c(18L, 2L))
    expect_identical(dim(as.matrix(second$layout)), c(21L, 2L))
    fastEmbedR:::massive_concat_word_files_cpp(
        c(first_path, second_path), joined_path,
        c(18, 21) * 2 * 4)
    joined <- massive_matrix(joined_path, nrow = 39, ncol = 2)
    expect_equal(as.matrix(joined), as.matrix(projected$layout),
        tolerance = 1e-5)
    expect_error(massive_project_landmarks(fit, graph,
        tempfile(fileext = ".f32"), row_range = c(2, 40)),
        "valid inclusive row indices")
    expect_error(fastEmbedR:::massive_concat_word_files_cpp(
        c(first_path, second_path), tempfile(fileext = ".f32"),
        c(1, 21) * 2 * 4), "size mismatch")
    expect_error(massive_project_landmarks(
        fit, graph, gpu_output, backend = "cuda",
        refinement_epochs = 2L),
        "no CPU fallback")
    if (isTRUE(fastEmbedR:::embedding_cuda_available_cpp())) {
        gpu <- massive_project_landmarks(
            fit, graph, gpu_output, backend = "cuda",
            refinement_epochs = 0L, chunk_rows = 7L,
            memory_limit = "256MB")
        expect_equal(as.matrix(gpu$layout), expected,
            tolerance = 1e-4)
        expect_identical(gpu$backend, "cuda")
        expect_identical(gpu$resources$projection_engine,
            "batch_cuda")
        expect_gt(gpu$resources$peak_vram_bytes, 0)
        projector <- fastEmbedR:::massive_cuda_projector_create_cpp(
            fit$layout, graph$ncol, 7L)
        for (first in c(1L, 8L, 15L, 36L)) {
            count <- min(7L, graph$nrow - first + 1L)
            neighbors <- massive_read_knn_rows(graph, first, count)
            old <- fastEmbedR:::project_embedding_knn_cuda_cpp(
                fit$layout, neighbors$indices, neighbors$distances)
            reused <- fastEmbedR:::massive_cuda_projector_batch_cpp(
                projector, neighbors$indices, neighbors$distances)
            expect_equal(reused, old, tolerance = 1e-10)
        }
        fastEmbedR:::massive_cuda_projector_release_cpp(projector)
        expect_error(fastEmbedR:::massive_cuda_projector_batch_cpp(
            projector, neighbors$indices, neighbors$distances),
            "batch dimensions differ")
        fit3 <- umap(reference, n_neighbors = 5L,
            n_components = 3L, backend = "cpu")
        gpu3 <- massive_project_landmarks(
            fit3, graph, gpu_output3, backend = "cuda",
            refinement_epochs = 0L, chunk_rows = 7L,
            memory_limit = "256MB")
        expected3 <- fastEmbedR:::project_embedding_knn_cpp(
            fit3$layout, read_massive_knn(graph)$indices,
            read_massive_knn(graph)$distances)
        expect_equal(as.matrix(gpu3$layout), expected3,
            tolerance = 1e-4)
        multi_path <- tempfile(fileext = ".f32")
        on.exit(unlink(c(multi_path,
            sub("\\.f32$", ".gpu0.f32", multi_path),
            sub("\\.f32$", ".gpu0.f32.result.rds", multi_path),
            paste0(multi_path, ".multigpu.checkpoint.rds"))), add = TRUE)
        distributed <- massive_project_landmarks(fit, graph,
            multi_path, backend = "cuda", devices = 0L,
            refinement_epochs = 0L, chunk_rows = 7L,
            memory_limit = "256MB", checkpoint = TRUE)
        expect_identical(distributed$gpu_devices, 0L)
        expect_identical(distributed$resources$per_device[[1L]]$gpu_device,
            0L)
        expect_equal(as.matrix(distributed$layout), expected,
            tolerance = 1e-4)
        arguments <- list(fit = fit, graph = graph, output = multi_path,
            selection = NULL, backend = "cuda", n.cores = 1L,
            chunk_rows = 7L, memory_limit = "256MB",
            refinement_epochs = 0L, transform_iter = 250L,
            transform_perplexity = 5, seed = 4L)
        tasks <- fastEmbedR:::massive_row_shards(
            multi_path, graph$nrow, 2L, 0L)
        signature <- fastEmbedR:::massive_multigpu_signature(
            arguments, 0L, tasks, multi_path)
        sidecar <- paste0(multi_path, ".multigpu.checkpoint.rds")
        saveRDS(list(signature = signature), sidecar)
        expect_error(massive_project_landmarks(fit, graph,
            multi_path, backend = "cuda", devices = 0L,
            refinement_epochs = 0L, chunk_rows = 7L,
            memory_limit = "256MB", checkpoint = TRUE,
            resume = TRUE, seed = 17L), "checkpoint does not match")
        recovered <- massive_project_landmarks(fit, graph,
            multi_path, backend = "cuda", devices = 0L,
            refinement_epochs = 0L, chunk_rows = 7L,
            memory_limit = "256MB", checkpoint = TRUE, resume = TRUE)
        expect_equal(as.matrix(recovered$layout), expected,
            tolerance = 1e-4)
        saveRDS(list(signature = signature), sidecar)
        unlink(multi_path)
        resumed <- massive_project_landmarks(fit, graph,
            multi_path, backend = "cuda", devices = 0L,
            refinement_epochs = 0L, chunk_rows = 7L,
            memory_limit = "256MB", checkpoint = TRUE, resume = TRUE)
        expect_identical(resumed$resources$per_device[[1L]]$status,
            "reused")
        expect_equal(as.matrix(resumed$layout), expected,
            tolerance = 1e-4)
        expect_false(file.exists(sidecar))
        expect_error(massive_project_landmarks(fit, graph,
            tempfile(fileext = ".f32"), backend = "cuda",
            devices = fastEmbedR:::massive_cuda_device_count_cpp(),
            refinement_epochs = 0L), "distinct available CUDA")
    } else {
        expect_error(massive_project_landmarks(
            fit, graph, gpu_output, backend = "cuda",
            refinement_epochs = 0L), "unavailable")
        expect_error(fastEmbedR:::massive_cuda_projector_create_cpp(
            fit$layout, graph$ncol, 7L), "unavailable")
        expect_error(massive_project_landmarks(fit, graph,
            tempfile(fileext = ".f32"), backend = "cuda",
            devices = 0L, refinement_epochs = 0L), "unavailable")
    }
})

test_that("completed projections reopen only with unchanged files", {
    set.seed(234)
    reference <- matrix(rnorm(48L), ncol = 4L)
    query <- matrix(rnorm(64L), ncol = 4L)
    source_path <- make_massive_fixture(query)
    reference_path <- make_massive_fixture(reference)
    prefix <- tempfile()
    outputs <- paste0(tempfile(), c("a.f32", "b.f32"))
    graph_paths <- paste0(prefix, c(".indices.u32", ".distances.f32"))
    on.exit(unlink(c(source_path, reference_path, graph_paths,
        outputs, paste0(outputs, ".manifest.rds"))))
    source <- massive_matrix(source_path, nrow = 16L, ncol = 4L)
    landmarks <- massive_matrix(reference_path, nrow = 12L, ncol = 4L)
    graph <- massive_landmark_knn(source, landmarks, 5L, prefix,
        method = "exact", memory_limit = "256MB")
    fit <- umap(reference, n_neighbors = 5L, backend = "cpu")
    results <- lapply(outputs, function(output) {
        massive_project_landmarks(fit, graph, output,
            refinement_epochs = 0L, memory_limit = "256MB")
    })
    reopened <- massive_open_projection(outputs[[1L]])
    expect_s3_class(reopened, "fastEmbedR_massive_projection")
    expect_equal(as.matrix(reopened$layout),
        as.matrix(results[[1L]]$layout))
    expect_identical(reopened$backend, "cpu")
    changed_graph <- results[[1L]]
    changed_graph$graph$output_identity$bytes[[1L]] <- 0
    expect_error(fastEmbedR:::massive_projection_complete(changed_graph),
        "KNN files changed")
    expect_error(massive_open_projection(tempfile(fileext = ".f32")),
        "manifest is missing")
    manifest <- results[[2L]]$manifest_path
    saved <- readRDS(manifest)
    saveRDS(1L, manifest)
    expect_error(massive_open_projection(outputs[[2L]]),
        "manifest is invalid")
    saveRDS(saved, manifest)
    con <- file(outputs[[1L]], "ab")
    writeBin(as.raw(1L), con)
    close(con)
    expect_error(massive_open_projection(outputs[[1L]]), "changed")
    con <- file(graph$indices_path, "ab")
    writeBin(as.raw(1L), con)
    close(con)
    expect_error(massive_open_projection(outputs[[2L]]), "changed")
})

test_that("projection merge resumes across a shard boundary", {
    directory <- tempfile()
    dir.create(directory)
    on.exit(unlink(directory, recursive = TRUE))
    paths <- file.path(directory, c("first.f32", "second.f32"))
    output <- file.path(directory, "merged.f32")
    count <- 786432L
    writeBin(rep(1, count), paths[1L], size = 4L)
    writeBin(rep(2, count), paths[2L], size = 4L)
    writeBin(c(rep(1, count), rep(2, 262146L)),
        paste0(output, ".part"), size = 4L)
    fastEmbedR:::massive_concat_word_files_cpp(paths, output,
        rep(count * 4, 2L), resume = TRUE)
    expect_equal(file.info(output)$size, count * 8)
    values <- readBin(output, "numeric", n = count * 2L, size = 4L)
    expect_true(all(values[seq_len(count)] == 1))
    expect_true(all(values[count + seq_len(count)] == 2))
    expect_false(file.exists(paste0(output, ".part")))
    expect_silent(fastEmbedR:::massive_verify_merged_shards(
        paths, output, rep(count * 4, 2L)))
    connection <- file(output, "r+b")
    seek(connection, where = count * 4, origin = "start")
    writeBin(9, connection, size = 4L, endian = "little")
    close(connection)
    expect_error(fastEmbedR:::massive_verify_merged_shards(
        paths, output, rep(count * 4, 2L)),
        "differs from its shards")
})

test_that("landmark source rows retain their reference coordinates", {
    set.seed(235)
    data <- matrix(rnorm(41L * 4L), ncol = 4L)
    source_path <- make_massive_fixture(data)
    reference_path <- tempfile(fileext = ".f32")
    prefix <- tempfile()
    output <- tempfile(fileext = ".f32")
    gpu_output <- tempfile(fileext = ".f32")
    on.exit(unlink(c(source_path, reference_path, output, gpu_output,
        paste0(prefix, c(".indices.u32", ".distances.f32")))))
    source <- massive_matrix(source_path, nrow = 41, ncol = 4)
    selection <- massive_select_landmarks(
        source, 16L, reference_path, chunk_rows = 8L)
    graph <- massive_landmark_knn(source, selection, 5L, prefix,
        chunk_rows = 9L, memory_limit = "256MB")
    fit <- umap(as.matrix(selection$data), n_neighbors = 5L,
        backend = "cpu")
    projected <- massive_project_landmarks(
        fit, graph, output, selection = selection,
        chunk_rows = 7L, refinement_epochs = 2L,
        memory_limit = "256MB")
    layout <- as.matrix(projected$layout)
    expect_equal(layout[selection$indices, ],
        matrix(as.numeric(fit$layout), ncol = 2L), tolerance = 1e-5)
    expect_true(all(is.finite(layout)))
    expect_true(projected$reference_rows_preserved)
    invalid <- selection
    invalid$indices <- rev(invalid$indices)
    expect_error(massive_project_landmarks(
        fit, graph, tempfile(fileext = ".f32"),
        selection = invalid), "sorted query rows")
    invalid <- selection
    invalid$indices[1L] <- invalid$indices[1L] + 1
    expect_error(massive_project_landmarks(
        fit, graph, tempfile(fileext = ".f32"),
        selection = invalid), "row indices changed|sorted query rows")
    invalid <- selection
    invalid$indices_signature <- "tampered"
    expect_error(massive_project_landmarks(
        fit, graph, tempfile(fileext = ".f32"),
        selection = invalid), "row indices changed")
    invalid <- selection
    invalid$source_identity <- list(changed = TRUE)
    expect_error(massive_project_landmarks(
        fit, graph, tempfile(fileext = ".f32"),
        selection = invalid), "KNN query source")
})

test_that("local query graph refines bounded CPU UMAP windows", {
    set.seed(521)
    x <- rbind(matrix(rnorm(30L * 4L, -2, 0.4), ncol = 4L),
        matrix(rnorm(30L * 4L, 2, 0.4), ncol = 4L))
    source_path <- make_massive_fixture(x)
    other_path <- make_massive_fixture(x)
    directory <- tempfile()
    dir.create(directory)
    on.exit(unlink(c(source_path, other_path, directory),
        recursive = TRUE))
    source <- massive_matrix(source_path, nrow = 60, ncol = 4)
    other <- massive_matrix(other_path, nrow = 60, ncol = 4)
    selection <- massive_select_landmarks(source, 20L,
        file.path(directory, "landmarks.f32"), seed = 4L,
        chunk_rows = 9L)
    graph <- massive_landmark_knn(source, selection, 5L,
        file.path(directory, "graph"), chunk_rows = 9L,
        memory_limit = "256MB")
    fit <- umap(as.matrix(selection$data), n_neighbors = 5L,
        backend = "cpu", seed = 4L)
    expect_identical(graph$source_identity,
        fastEmbedR:::massive_checkpoint_source_identity(source))
    baseline <- massive_project_landmarks(fit, graph,
        file.path(directory, "baseline.f32"), selection = selection,
        chunk_rows = 8L, refinement_epochs = 3L,
        memory_limit = "256MB")
    local <- massive_project_landmarks(fit, graph,
        file.path(directory, "local.f32"), selection = selection,
        source = source, local_refine = TRUE, local_neighbors = 3L,
        overlap_rows = 3L, chunk_rows = 8L,
        refinement_epochs = 3L, memory_limit = "256MB")
    layout <- as.matrix(local$layout)
    expect_identical(dim(layout), c(60L, 2L))
    expect_true(all(is.finite(layout)))
    expect_equal(layout[selection$indices, ],
        matrix(as.numeric(fit$layout), ncol = 2L), tolerance = 1e-5)
    expect_gt(max(abs(layout - as.matrix(baseline$layout))), 1e-5)
    expect_true(local$local_refine)
    expect_identical(local$local_neighbors, 3L)
    expect_identical(local$overlap_rows, 3L)
    expect_error(massive_project_landmarks(fit, graph,
        file.path(directory, "wrong.f32"), source = other,
        local_refine = TRUE, local_neighbors = 3L,
        overlap_rows = 3L, refinement_epochs = 3L), "exactly")
    expect_error(massive_project_landmarks(fit, graph,
        file.path(directory, "too_little_overlap.f32"), source = source,
        local_refine = TRUE, local_neighbors = 3L,
        overlap_rows = 2L, refinement_epochs = 3L),
        "overlap must cover")
    expect_error(massive_project_landmarks(fit, graph,
        file.path(directory, "gpu.f32"), source = source,
        local_refine = TRUE, backend = "cuda",
        refinement_epochs = 3L), "no CPU fallback")
    expect_error(massive_project_landmarks(fit, graph,
        file.path(directory, "zero.f32"), source = source,
        local_refine = TRUE, refinement_epochs = 0L),
        "positive `refinement_epochs`")
    public <- umap(source, massive = "landmark", landmarks = 20L,
        output = file.path(directory, "public_local.f32"),
        backend = "cpu", n.cores = 2L, n_neighbors = 5L,
        chunk_rows = 8L, memory_limit = "256MB",
        local_refine = TRUE, local_neighbors = 3L,
        overlap_rows = 3L, refinement_epochs = 3L)
    expect_true(public$local_refine)
    expect_true(all(is.finite(as.matrix(public$layout))))
    expect_error(umap(source, massive = "landmark", landmarks = 20L,
        output = file.path(directory, "public_gpu.f32"),
        backend = "cuda", local_refine = TRUE,
        refinement_epochs = 3L), "no fallback")
})

test_that("file-backed t-SNE projection uses the native transform", {
    set.seed(232)
    reference <- matrix(rnorm(28L * 4L), ncol = 4L)
    query <- matrix(rnorm(13L * 4L), ncol = 4L)
    source_path <- make_massive_fixture(query)
    reference_path <- make_massive_fixture(reference)
    prefix <- tempfile()
    output <- tempfile(fileext = ".f32")
    gpu_output <- tempfile(fileext = ".f32")
    gpu3_output <- tempfile(fileext = ".f32")
    resumed_output <- tempfile(fileext = ".f32")
    empty_resume_output <- tempfile(fileext = ".f32")
    gpu_resumed_output <- tempfile(fileext = ".f32")
    gpu_empty_resume_output <- tempfile(fileext = ".f32")
    on.exit(unlink(c(source_path, reference_path, output, gpu_output,
        gpu3_output,
        resumed_output, paste0(resumed_output,
            c(".part", ".checkpoint.rds")),
        empty_resume_output, paste0(empty_resume_output,
            c(".part", ".checkpoint.rds")),
        gpu_resumed_output, paste0(gpu_resumed_output,
            c(".part", ".checkpoint.rds")),
        gpu_empty_resume_output, paste0(gpu_empty_resume_output,
            c(".part", ".checkpoint.rds")),
        paste0(prefix, c(".indices.u32", ".distances.f32")))))
    source <- massive_matrix(source_path, nrow = 13, ncol = 4)
    landmarks <- massive_matrix(reference_path, nrow = 28, ncol = 4)
    graph <- massive_landmark_knn(source, landmarks, 5L, prefix,
        chunk_rows = 6L, memory_limit = "256MB")
    fit <- tsne(reference, perplexity = 5, backend = "cpu",
        n_iter = 2L, early_exaggeration_iter = 0L)
    projected <- massive_project_landmarks(
        fit, graph, output, chunk_rows = 6L,
        transform_iter = 2L, transform_perplexity = 5,
        memory_limit = "256MB", checkpoint = TRUE)
    expected <- transform_tsne(fit$layout,
        knn = massive_read_knn_rows(graph, 1, 6),
        perplexity = 5, n_iter = 2L, backend = "cpu", seed = 4L)
    expect_equal(massive_read_rows(projected$layout, 1, 6),
        matrix(as.numeric(expected), nrow = 6L), tolerance = 1e-5)
    expect_true(all(is.finite(as.matrix(projected$layout))))
    expect_identical(projected$transform_iter, 2L)
    expect_equal(as.matrix(massive_open_projection(output)$layout),
        as.matrix(projected$layout))
    expect_false(file.exists(paste0(output, ".checkpoint.rds")))
    resources <- fastEmbedR:::massive_projection_resources(
        graph, 2L, 6L, "256MB", "cpu")
    settings <- list(backend = "cpu", workers = 1L,
        chunk_rows = resources$chunk_rows, epochs = 50L,
        iterations = 2L, perplexity = 5, seed = 4L,
        row_range = c(1, graph$nrow))
    signature <- fastEmbedR:::massive_projection_signature(
        fit, graph, resources, settings, NULL)
    saveRDS("broken", paste0(empty_resume_output, ".checkpoint.rds"))
    expect_error(massive_project_landmarks(fit, graph,
        empty_resume_output, chunk_rows = 6L, transform_iter = 2L,
        memory_limit = "256MB", checkpoint = TRUE, resume = TRUE),
        "Projection checkpoint or partial output does not match")
    saveRDS(list(signature = signature, completed_rows = 1),
        paste0(empty_resume_output, ".checkpoint.rds"))
    expect_error(massive_project_landmarks(fit, graph,
        empty_resume_output, chunk_rows = 6L, transform_iter = 2L,
        memory_limit = "256MB", checkpoint = TRUE, resume = TRUE),
        "does not match")
    saveRDS(list(signature = signature, completed_rows = 0),
        paste0(empty_resume_output, ".checkpoint.rds"))
    expect_false(file.exists(paste0(empty_resume_output, ".part")))
    empty_resumed <- massive_project_landmarks(fit, graph,
        empty_resume_output, chunk_rows = 6L, transform_iter = 2L,
        memory_limit = "256MB", checkpoint = TRUE, resume = TRUE)
    expect_equal(as.matrix(empty_resumed$layout),
        as.matrix(projected$layout), tolerance = 1e-5)
    part <- file(paste0(resumed_output, ".part"), "wb")
    writeBin(as.vector(t(massive_read_rows(projected$layout, 1, 6))),
        part, size = 4L, endian = "little")
    writeBin(rep(999, 2), part, size = 4L, endian = "little")
    close(part)
    saveRDS(list(signature = signature, completed_rows = 6),
        paste0(resumed_output, ".checkpoint.rds"))
    expect_error(massive_project_landmarks(fit, graph, resumed_output,
        chunk_rows = 6L, transform_iter = 2L,
        memory_limit = "256MB", checkpoint = TRUE,
        resume = TRUE, seed = 17L), "does not match")
    resumed <- massive_project_landmarks(fit, graph, resumed_output,
        chunk_rows = 6L, transform_iter = 2L,
        memory_limit = "256MB", checkpoint = TRUE, resume = TRUE)
    expect_equal(as.matrix(resumed$layout),
        as.matrix(projected$layout), tolerance = 1e-5)
    expect_false(file.exists(paste0(resumed_output, ".checkpoint.rds")))
    expect_error(massive_project_landmarks(fit, graph, tempfile(),
        resume = TRUE), "requires `checkpoint = TRUE`")
    expect_error(massive_project_landmarks(
        fit, graph, tempfile(fileext = ".f32"),
        transform_perplexity = 6), "graph k")
    if (isTRUE(fastEmbedR:::embedding_cuda_available_cpp())) {
        gpu <- massive_project_landmarks(fit, graph, gpu_output,
            backend = "cuda", chunk_rows = 6L,
            transform_iter = 2L, transform_perplexity = 5,
            memory_limit = "256MB", checkpoint = TRUE)
        direct <- transform_tsne(fit$layout,
            knn = massive_read_knn_rows(graph, 1, 6),
            perplexity = 5, n_iter = 2L, backend = "cuda",
            n.cores = 1L, seed = 4L)
        expect_equal(massive_read_rows(gpu$layout, 1, 6),
            matrix(as.numeric(direct), nrow = 6L),
            tolerance = 1e-4)
        expect_true(all(is.finite(as.matrix(gpu$layout))))
        expect_identical(gpu$backend, "cuda")
        expect_true(gpu$resources$peak_vram_bytes > 0)
        gpu_resources <- fastEmbedR:::massive_projection_resources(
            graph, 2L, 6L, "256MB", "cuda")
        gpu_settings <- settings
        gpu_settings$backend <- "cuda"
        gpu_settings$chunk_rows <- gpu_resources$chunk_rows
        gpu_signature <- fastEmbedR:::massive_projection_signature(
            fit, graph, gpu_resources, gpu_settings, NULL)
        saveRDS(list(signature = gpu_signature, completed_rows = 0),
            paste0(gpu_empty_resume_output, ".checkpoint.rds"))
        gpu_empty_resumed <- massive_project_landmarks(fit, graph,
            gpu_empty_resume_output, backend = "cuda", chunk_rows = 6L,
            transform_iter = 2L, memory_limit = "256MB",
            checkpoint = TRUE, resume = TRUE)
        expect_identical(gpu_empty_resumed$backend, "cuda")
        expect_equal(as.matrix(gpu_empty_resumed$layout),
            as.matrix(gpu$layout), tolerance = 1e-4)
        gpu_part <- file(paste0(gpu_resumed_output, ".part"), "wb")
        writeBin(as.vector(t(massive_read_rows(gpu$layout, 1, 6))),
            gpu_part, size = 4L, endian = "little")
        close(gpu_part)
        saveRDS(list(signature = gpu_signature, completed_rows = 6),
            paste0(gpu_resumed_output, ".checkpoint.rds"))
        gpu_resumed <- massive_project_landmarks(fit, graph,
            gpu_resumed_output, backend = "cuda", chunk_rows = 6L,
            transform_iter = 2L, memory_limit = "256MB",
            checkpoint = TRUE, resume = TRUE)
        expect_identical(gpu_resumed$backend, "cuda")
        expect_equal(as.matrix(gpu_resumed$layout),
            as.matrix(gpu$layout), tolerance = 1e-4)
        fit3 <- fit
        fit3$layout <- cbind(fit$layout,
            seq_len(nrow(fit$layout)) / 100)
        gpu3 <- massive_project_landmarks(fit3, graph,
            gpu3_output, backend = "cuda", chunk_rows = 6L,
            transform_iter = 2L, transform_perplexity = 5,
            memory_limit = "256MB")
        direct3 <- transform_tsne(fit3$layout,
            knn = massive_read_knn_rows(graph, 1, 6),
            perplexity = 5, n_iter = 2L, backend = "cuda",
            n.cores = 1L, seed = 4L)
        expect_equal(massive_read_rows(gpu3$layout, 1, 6),
            matrix(as.numeric(direct3), nrow = 6L),
            tolerance = 1e-4)
        expect_identical(gpu3$backend, "cuda")
        expect_equal(gpu3$layout$ncol, 3)
        fit4 <- fit3
        fit4$layout <- cbind(fit3$layout, 0)
        expect_error(massive_project_landmarks(fit4, graph,
            tempfile(fileext = ".f32"), backend = "cuda"),
            "requires a 2D or 3D reference")
    } else {
        expect_error(massive_project_landmarks(fit, graph,
            gpu_output, backend = "cuda"), "unavailable")
    }
})

test_that("all native graph methods vote into file-backed labels", {
    set.seed(246)
    reference <- rbind(
        matrix(rnorm(24, 0, 0.1), ncol = 2L),
        matrix(rnorm(24, 4, 0.1), ncol = 2L),
        matrix(rnorm(24, 8, 0.1), ncol = 2L))
    query <- rbind(
        matrix(rnorm(16, 0, 0.15), ncol = 2L),
        matrix(rnorm(16, 4, 0.15), ncol = 2L),
        matrix(rnorm(16, 8, 0.15), ncol = 2L))
    source_path <- make_massive_fixture(query)
    reference_path <- make_massive_fixture(reference)
    prefix <- tempfile()
    outputs <- vapply(c("leiden", "louvain", "walktrap"),
        function(x) tempfile(), character(1L))
    files <- c(source_path, reference_path,
        paste0(prefix, c(".indices.u32", ".distances.f32")),
        as.vector(outer(outputs,
            c(".clusters.u32", ".confidence.f32"), paste0)))
    on.exit(unlink(files))
    source <- massive_matrix(source_path, nrow = 24, ncol = 2L)
    landmarks <- massive_matrix(reference_path, nrow = 36, ncol = 2L)
    knn <- massive_landmark_knn(source, landmarks, 5L, prefix,
        chunk_rows = 7L, memory_limit = "256MB")
    graph <- knn_graph(reference, k = 5L, backend = "cpu")
    baseline <- NULL
    for (method in names(outputs)) {
        direct <- graph_cluster(graph, method = method, seed = 3L)
        streamed <- massive_cluster_landmarks(graph, knn,
            outputs[[method]], method = method, n.cores = 2L,
            chunk_rows = 6L, memory_limit = "256MB", seed = 3L)
        observed <- massive_read_cluster_rows(streamed, 1, 24L)
        neighbors <- massive_read_knn_rows(knn, 1, 24L)
        expected <- lapply(seq_len(24L), function(row) {
            labels <- direct$membership[neighbors$indices[row, ]]
            weights <- 1 / (neighbors$distances[row, ] + 1e-3)
            votes <- tapply(weights, labels, sum)
            best <- which.max(votes)
            c(label = as.integer(names(votes)[best]),
                confidence = unname(votes[best] / sum(votes)))
        })
        expected <- do.call(rbind, expected)
        expect_identical(streamed$reference_membership,
            direct$membership)
        expect_equal(observed$membership, as.integer(expected[, 1L]))
        expect_equal(observed$confidence, expected[, 2L],
            tolerance = 1e-5)
        if (method == "leiden") baseline <- observed
        expect_equal(head(streamed, 2L)$membership,
            observed$membership[1:2])
        expect_equal(file.info(streamed$membership_path)$size, 24 * 4)
        expect_error(massive_read_cluster_rows(streamed, 24, 2L),
            "row range")
    }
    cuda_knn <- knn
    cuda_knn$backend <- "cuda"
    reuse_output <- tempfile()
    on.exit(unlink(paste0(reuse_output,
        c(".clusters.u32", ".confidence.f32"))), add = TRUE)
    reused <- massive_cluster_landmarks(graph, cuda_knn,
        reuse_output, method = "leiden", n.cores = 2L,
        chunk_rows = 6L, memory_limit = "256MB", seed = 3L)
    expect_identical(reused$backend, "cpu")
    expect_identical(reused$graph_backend, "cuda")
    expect_equal(massive_read_cluster_rows(reused, 1, 24L), baseline)
    resources <- fastEmbedR:::massive_cluster_resources(knn, graph,
        2L, 6L, "256MB")
    expect_gt(resources$reference_graph_estimate_bytes, 0)
    expect_error(massive_cluster_landmarks(graph, knn,
        tempfile(), method = "walktrap", backend = "cuda"),
        "Walktrap is CPU-only")
    if (isTRUE(fastEmbedR:::graph_clustering_cuda_available_cpp())) {
        for (method in c("leiden", "louvain")) {
            gpu_output <- tempfile()
            on.exit(unlink(paste0(gpu_output,
                c(".clusters.u32", ".confidence.f32"))), add = TRUE)
            direct <- graph_cluster(graph, method = method,
                backend = "cuda", seed = 3L)
            gpu <- massive_cluster_landmarks(graph, knn,
                gpu_output, method = method, backend = "cuda",
                n.cores = 2L, chunk_rows = 6L,
                memory_limit = "256MB", seed = 3L)
            rows <- massive_read_knn_rows(knn, 1, 24L)
            voted <- fastEmbedR:::massive_vote_landmarks_cpp(
                rows$indices, rows$distances,
                direct$membership, 2L)
            expect_identical(gpu$reference_membership,
                direct$membership)
            expect_equal(massive_read_cluster_rows(gpu, 1, 24L),
                voted, tolerance = 1e-5)
            expect_identical(gpu$backend, "cuda")
            expect_identical(gpu$assignment_backend, "cpu")
            expect_gt(gpu$resources$peak_vram_bytes, 0)
        }
    } else {
        expect_error(massive_cluster_landmarks(graph, knn,
            tempfile(), backend = "cuda"), "unavailable")
    }
})

test_that("massive clustering composes selection, graph, and voting", {
    set.seed(421)
    x <- rbind(matrix(rnorm(48L, 0, 0.1), ncol = 2L),
        matrix(rnorm(48L, 4, 0.1), ncol = 2L),
        matrix(rnorm(48L, 8, 0.1), ncol = 2L))
    source_path <- make_massive_fixture(x)
    directory <- tempfile()
    dir.create(directory)
    on.exit(unlink(c(source_path, directory), recursive = TRUE))
    source <- massive_matrix(source_path, nrow = nrow(x),
        ncol = ncol(x))
    for (method in c("leiden", "louvain", "walktrap")) {
        prefix <- file.path(directory, method)
        result <- massive_cluster(source, method = method,
            landmarks = 24L, k = 4L, output = prefix,
            landmark_method = if (method == "walktrap") {
                "random"
            } else "reservoir",
            n.cores = 2L, chunk_rows = 7L,
            memory_limit = "512MB", seed = 3L)
        selected <- result$selection$indices
        reference <- as.matrix(result$selection$data)
        direct <- graph_cluster(knn_graph(reference, k = 4L,
            backend = "cpu", n.cores = 2L), method = method,
            backend = "cpu", seed = 3L)
        observed <- massive_read_cluster_rows(result, 1L, nrow(x))
        expect_identical(result$reference_membership,
            direct$membership)
        expect_equal(observed$membership[selected],
            direct$membership)
        expect_true(all(observed$confidence[selected] == 1))
        expect_true(all(is.finite(observed$confidence)))
        expect_true(all(observed$confidence >= 0 &
            observed$confidence <= 1))
        expect_equal(file.info(result$membership_path)$size,
            nrow(x) * 4)
        expect_equal(file.info(result$confidence_path)$size,
            nrow(x) * 4)
        expect_identical(result$graph_backend, "cpu")
        expect_identical(result$selection$method,
            if (method == "walktrap") "random" else "reservoir")
        expect_identical(result$reference_graph_parameters$weight,
            "snn")
    }
    automatic <- with_mocked_bindings(
        massive_available_ram_bytes = function() 4 * 1024^3,
        massive_cluster(source, massive = "auto",
            method = "leiden", k = 4L, memory_limit = "1GB"),
        .package = "fastEmbedR")
    expect_s3_class(automatic, "fastEmbedR_graph_cluster")
    expect_identical(automatic$massive_auto, "in_memory")
    auto_prefix <- file.path(directory, "automatic")
    approximate <- with_mocked_bindings(
        massive_available_ram_bytes = function() 4 * 1024^3,
        massive_cluster(source, massive = "auto",
            method = "leiden", k = 4L, output = auto_prefix,
            chunk_rows = 7L, memory_limit = "256MB"),
        .package = "fastEmbedR")
    expect_s3_class(approximate, "fastEmbedR_massive_clusters")
    expect_identical(approximate$massive_auto, "landmark")
    expect_equal(massive_read_cluster_rows(approximate, 1L,
        nrow(x))$membership[approximate$selection$indices],
        approximate$reference_membership)
})

test_that("landmark clustering reuses a verified embedding KNN", {
    skip_if_not_installed("float")
    expect_identical(names(formals(massive_cluster))[5L], "k")
    expect_identical(tail(names(formals(massive_cluster)), 1L), "nn")
    set.seed(422)
    x <- matrix(rnorm(60L * 4L), ncol = 4L)
    input <- make_massive_fixture(x)
    other <- make_massive_fixture(x)
    directory <- tempfile()
    dir.create(directory)
    on.exit(unlink(c(input, other, directory), recursive = TRUE))
    source <- massive_matrix(input, 60L, 4L)
    embedding <- umap(source, massive = "landmark",
        landmarks = 20L, n_neighbors = 5L, transform_k = 5L,
        output = file.path(directory, "umap.f32"),
        chunk_rows = 9L, memory_limit = "512MB", backend = "cpu")
    run <- function(data = source, k = 5L, mode = "landmark") {
        massive_cluster(data, method = "leiden", massive = mode,
            nn = embedding, k = k, output = file.path(directory, "cluster"),
            chunk_rows = 9L, memory_limit = "512MB",
            backend = "cpu", checkpoint = TRUE)
    }
    expect_error(run(k = 4L), "does not match")
    expect_error(run(massive_matrix(other, 60L, 4L)),
        "does not match")
    expect_error(run(mode = "auto"), "explicit landmark")
    with_mocked_bindings(
        massive_select_landmarks = function(...) stop("resampled"),
        massive_landmark_knn = function(...) stop("KNN repeated"),
        clustered <- run(),
        .package = "fastEmbedR"
    )
    expect_true(clustered$graph_reused)
    expect_identical(clustered$knn$indices_path,
        embedding$graph$indices_path)
    expect_identical(clustered$selection$indices,
        embedding$selection$indices)
    expect_false(file.exists(file.path(directory,
        "cluster.knn.indices.u32")))
    expect_equal(length(massive_read_cluster_rows(clustered,
        1L, 60L)$membership), 60L)
})

test_that("massive clustering rejects unsafe requests before writing", {
    x <- matrix(seq_len(120L), nrow = 30L)
    source_path <- make_massive_fixture(x)
    directory <- tempfile()
    dir.create(directory)
    on.exit(unlink(c(source_path, directory), recursive = TRUE))
    source <- massive_matrix(source_path, nrow = 30L, ncol = 4L)
    prefix <- file.path(directory, "rejected")
    expect_error(massive_cluster(source, landmarks = 15L,
        k = 4L, output = prefix, memory_limit = "64MB"),
        "budget")
    expect_error(massive_cluster(source, method = "walktrap",
        landmarks = 15L, k = 4L, output = prefix,
        backend = "cuda"), "Walktrap is CPU-only")
    expect_error(massive_cluster(source, landmarks = 15L,
        k = 4L, output = prefix, resolution = -1),
        "resolution")
    expect_error(massive_cluster(source, landmarks = 15L,
        k = 15L, output = prefix), "cannot support")
    expect_error(massive_cluster(source, landmarks = 15L,
        k = 4L, output = prefix, devices = 0L), "requires CUDA")
    expect_error(massive_cluster(source, massive = "auto",
        output = prefix, k = 4L, devices = 0L),
        "requires landmark clustering")
    expect_length(list.files(directory), 0L)
})

test_that("Walktrap landmark planning respects native RAM limits", {
    source <- fastEmbedR:::massive_synthetic_matrix(1e9, 96L)
    plan <- with_mocked_bindings(
        massive_available_ram_bytes = function() 16 * 1024^3,
        fastEmbedR:::massive_cluster_auto_plan(source,
            NULL, 30L, "cpu", 1L, "8GB", "auto", tempfile(),
            NULL, "walktrap"),
        .package = "fastEmbedR")
    expect_identical(plan$mode, "landmark")
    expect_identical(plan$count, 4000L)
    expect_lte(16 * plan$count^2 +
        plan$count * plan$budget$per_landmark,
        0.7 * plan$budget$limit - 168 * 1024^2)
    small <- fastEmbedR:::massive_cluster_auto_plan(source,
        NULL, 30L, "cpu", 1L, "256MB", "landmark",
        tempfile(), NULL, "walktrap")
    expect_lt(small$count, plan$count)
    expect_error(fastEmbedR:::massive_cluster_auto_plan(source,
        4001L, 30L, "cpu", 1L, "8GB", "landmark",
        tempfile(), NULL, "walktrap"), "Walktrap")
    expect_error(massive_cluster(source, method = "walktrap",
        landmarks = 4001L, k = 30L, output = tempfile(),
        memory_limit = "8GB"), "Walktrap")
    other <- fastEmbedR:::massive_cluster_reference_budget(
        1e9, 96L, 30L, "cpu", 1L, "8GB", NULL, "leiden")
    expect_gt(other$cap, plan$count)
})

test_that("Walktrap resource check precedes output creation", {
    knn <- list(nrow = 1e9, ncol = 30L, n_reference = 4000L)
    graph <- list(n_vertices = 4000L, weight = numeric())
    estimate <- fastEmbedR:::massive_cluster_resources(knn,
        graph, 1L, NULL, "512MB", "cpu", "walktrap")
    expect_gte(estimate$peak_ram_bytes, 16 * 4000^2)
    expect_error(fastEmbedR:::massive_cluster_resources(knn,
        graph, 1L, NULL, "256MB", "cpu", "walktrap"),
        "memory_limit")
    graph$n_vertices <- 4001L
    expect_error(fastEmbedR:::massive_cluster_resources(knn,
        graph, 1L, NULL, "8GB", "cpu", "walktrap"),
        "4,000 landmarks")
})

test_that("CUDA landmark clustering honors selected device", {
    skip_if_not(isTRUE(fastEmbedR:::embedding_cuda_available_cpp()))
    skip_if_not(isTRUE(fastEmbedR:::native_cuda_knn_available_cpp()))
    skip_if_not(isTRUE(fastEmbedR:::graph_clustering_cuda_available_cpp()))
    set.seed(427)
    x <- rbind(matrix(rnorm(36L, 0, 0.1), ncol = 2L),
        matrix(rnorm(36L, 4, 0.1), ncol = 2L))
    source_path <- make_massive_fixture(x)
    directory <- tempfile()
    dir.create(directory)
    on.exit(unlink(c(source_path, directory), recursive = TRUE))
    source <- massive_matrix(source_path, nrow = nrow(x), ncol = 2L)
    bad_device <- fastEmbedR:::massive_cuda_device_count_cpp()
    expect_error(massive_cluster(source, method = "louvain",
        landmarks = 18L, k = 4L,
        output = file.path(directory, "invalid"),
        backend = "cuda", devices = bad_device,
        memory_limit = "512MB"), "distinct available CUDA devices")
    expect_false(file.exists(file.path(directory, "invalid.clusters.u32")))
    result <- massive_cluster(source, method = "louvain",
        landmarks = 18L, k = 4L, output = file.path(directory, "gpu"),
        backend = "cuda", devices = 0L, chunk_rows = 8L,
        memory_limit = "512MB", seed = 4L)
    observed <- massive_read_cluster_rows(result, 1L, nrow(x))
    expect_identical(result$gpu_devices, 0L)
    expect_identical(result$knn$gpu_devices, 0L)
    expect_identical(result$backend, "cuda")
    expect_identical(result$graph_backend, "cuda")
    expect_identical(result$resources$gpu_device, 0L)
    expect_true(all(is.finite(observed$confidence)))
    expect_equal(observed$membership[result$selection$indices],
        result$reference_membership)
})

test_that("landmark clustering resumes saved communities and rows", {
    set.seed(248)
    reference <- matrix(rnorm(18L * 3L), ncol = 3L)
    query <- matrix(rnorm(27L * 3L), ncol = 3L)
    knn_prefix <- tempfile()
    baseline_prefix <- tempfile()
    resume_prefix <- tempfile()
    files <- c(paste0(knn_prefix,
        c(".indices.u32", ".distances.f32")),
        paste0(rep(c(baseline_prefix, resume_prefix), each = 2L),
            c(".clusters.u32", ".confidence.f32")),
        paste0(resume_prefix, c(".clusters.u32.part",
            ".confidence.f32.part", ".checkpoint.rds")))
    on.exit(unlink(files))
    knn <- massive_landmark_knn(massive_matrix(query),
        massive_matrix(reference), 4L, knn_prefix,
        chunk_rows = 7L, memory_limit = "256MB")
    graph <- knn_graph(reference, k = 4L)
    baseline <- massive_cluster_landmarks(graph, knn,
        baseline_prefix, method = "leiden", seed = 3L,
        chunk_rows = 5L, memory_limit = "256MB",
        checkpoint = TRUE, checkpoint_every = 2L)
    resources <- fastEmbedR:::massive_cluster_resources(knn, graph, 1L,
        5L, "256MB")
    paths <- fastEmbedR:::massive_cluster_paths(resume_prefix,
        resources$output_bytes, TRUE)
    controls <- list(method = "leiden", backend = "cpu", workers = 1L,
        chunk_rows = 5L, resolution = 1, n_iterations = 10L,
        n_runs = 1L, steps = 4L, seed = 3L)
    saved <- fastEmbedR:::massive_cluster_checkpoint(graph, knn,
        NULL, paths, controls, FALSE)
    clustered <- graph_cluster(graph, method = "leiden", seed = 3L)
    saved <- fastEmbedR:::massive_cluster_save_checkpoint(
        saved, clustered)
    rows <- massive_read_cluster_rows(baseline, 1L, 10L)
    parts <- stats::setNames(paste0(unname(paths), ".part"),
        names(paths))
    writeBin(c(rows$membership, 0L), parts[["membership"]],
        size = 4L, endian = "little")
    writeBin(c(rows$confidence, 0), parts[["confidence"]],
        size = 4L, endian = "little")
    saved$state$completed_rows <- 10
    fastEmbedR:::massive_checkpoint_write(saved$state, saved$path)
    expect_error(massive_cluster_landmarks(graph, knn,
        resume_prefix, method = "leiden", seed = 4L,
        chunk_rows = 5L, memory_limit = "256MB",
        checkpoint = TRUE, resume = TRUE), "does not match")
    changed <- graph
    changed$weight[[1L]] <- changed$weight[[1L]] + 0.1
    expect_error(massive_cluster_landmarks(changed, knn,
        resume_prefix, method = "leiden", seed = 3L,
        chunk_rows = 5L, memory_limit = "256MB",
        checkpoint = TRUE, resume = TRUE), "does not match")
    expect_equal(file.info(parts)$size, rep(44, 2L))
    resumed <- massive_cluster_landmarks(graph, knn,
        resume_prefix, method = "leiden", seed = 3L,
        chunk_rows = 5L, memory_limit = "256MB",
        checkpoint = TRUE, resume = TRUE)
    expect_equal(massive_read_cluster_rows(resumed, 1L, 27L),
        massive_read_cluster_rows(baseline, 1L, 27L))
    expect_identical(resumed$reference_membership,
        baseline$reference_membership)
    expect_false(any(file.exists(c(parts, saved$path))))
})

test_that("landmark clustering resumes before its first output", {
    set.seed(249)
    reference <- matrix(rnorm(15L * 3L), ncol = 3L)
    query <- matrix(rnorm(11L * 3L), ncol = 3L)
    knn_prefix <- tempfile()
    output <- tempfile()
    on.exit(unlink(c(paste0(knn_prefix,
        c(".indices.u32", ".distances.f32")),
        paste0(output, c(".clusters.u32", ".confidence.f32",
            ".checkpoint.rds")))))
    knn <- massive_landmark_knn(massive_matrix(query),
        massive_matrix(reference), 3L, knn_prefix,
        chunk_rows = 5L, memory_limit = "256MB")
    graph <- knn_graph(reference, k = 3L)
    resources <- fastEmbedR:::massive_cluster_resources(knn, graph, 1L,
        4L, "256MB")
    paths <- fastEmbedR:::massive_cluster_paths(output,
        resources$output_bytes, TRUE)
    controls <- list(method = "louvain", backend = "cpu", workers = 1L,
        chunk_rows = 4L, resolution = 1, n_iterations = 10L,
        n_runs = 1L, steps = 4L, seed = 3L)
    saved <- fastEmbedR:::massive_cluster_checkpoint(graph, knn,
        NULL, paths, controls, FALSE)
    clustered <- graph_cluster(graph, method = "louvain", seed = 3L)
    fastEmbedR:::massive_cluster_save_checkpoint(saved, clustered)
    resumed <- massive_cluster_landmarks(graph, knn, output,
        method = "louvain", seed = 3L, chunk_rows = 4L,
        memory_limit = "256MB", checkpoint = TRUE, resume = TRUE)
    expect_equal(resumed$reference_membership, clustered$membership)
    expect_equal(file.info(resumed$membership_path)$size, 11 * 4)
    expect_false(file.exists(saved$path))
})

test_that("public landmark clustering resumes completed stages", {
    set.seed(250)
    data <- matrix(rnorm(51L * 3L), ncol = 3L)
    input <- make_massive_fixture(data)
    directory <- tempfile()
    dir.create(directory)
    on.exit(unlink(c(input, directory), recursive = TRUE))
    source <- massive_matrix(input, 51L, 3L)
    run <- function(name, resume = FALSE, seed = 3L) {
        massive_cluster(source, method = "leiden", landmarks = 20L,
            k = 5L, output = file.path(directory, name),
            backend = "cpu", chunk_rows = 7L,
            memory_limit = "512MB", seed = seed,
            checkpoint = TRUE, checkpoint_every = 1L,
            resume = resume)
    }
    baseline <- run("baseline")
    original <- fastEmbedR:::massive_cluster_batch
    calls <- 0L
    with_mocked_bindings(
        massive_cluster_batch = function(...) {
            calls <<- calls + 1L
            if (calls == 2L) stop("interrupted labels")
            original(...)
        },
        expect_error(run("resumed"), "interrupted labels"),
        .package = "fastEmbedR"
    )
    prefix <- file.path(directory, "resumed")
    sidecar <- paste0(prefix, ".workflow.checkpoint.rds")
    expect_identical(readRDS(sidecar)$stage, "knn")
    expect_true(file.exists(paste0(prefix, ".checkpoint.rds")))
    expect_error(run("resumed", TRUE, 4L),
        "checkpoint does not match")
    with_mocked_bindings(
        massive_select_landmarks = function(...) stop("resampled"),
        knn_graph = function(...) stop("graph rebuilt"),
        massive_landmark_knn = function(...) stop("KNN repeated"),
        continued <- run("resumed", TRUE),
        .package = "fastEmbedR"
    )
    expect_equal(massive_read_cluster_rows(continued, 1L, 51L),
        massive_read_cluster_rows(baseline, 1L, 51L))
    expect_false(file.exists(sidecar))
    expect_false(file.exists(paste0(prefix, ".checkpoint.rds")))
})

test_that("public landmark clustering resumes saved selection", {
    set.seed(251)
    input <- make_massive_fixture(matrix(rnorm(45L * 3L), ncol = 3L))
    directory <- tempfile()
    dir.create(directory)
    on.exit(unlink(c(input, directory), recursive = TRUE))
    source <- massive_matrix(input, 45L, 3L)
    output <- file.path(directory, "clusters")
    run <- function(resume = FALSE) massive_cluster(source,
        method = "louvain", landmarks = 18L, k = 4L,
        output = output, chunk_rows = 8L, memory_limit = "512MB",
        seed = 4L, checkpoint = TRUE, resume = resume)
    with_mocked_bindings(
        massive_read_rows_cpp = function(...) stop("graph interrupted"),
        expect_error(run(), "graph interrupted"),
        .package = "fastEmbedR"
    )
    sidecar <- paste0(output, ".workflow.checkpoint.rds")
    saved <- readRDS(sidecar)
    expect_identical(saved$stage, "selected")
    with_mocked_bindings(
        massive_select_landmarks = function(...) stop("resampled"),
        resumed <- run(TRUE),
        .package = "fastEmbedR"
    )
    expect_identical(resumed$selection$indices,
        saved$selection$indices)
    expect_equal(length(massive_read_cluster_rows(resumed,
        1L, 45L)$membership), 45L)
    expect_false(file.exists(sidecar))
})

test_that("landmark cluster checkpoint reserves graph disk space", {
    input <- make_massive_fixture(matrix(0, nrow = 45L, ncol = 3L))
    output <- tempfile()
    on.exit(unlink(input))
    source <- massive_matrix(input, 45L, 3L)
    with_mocked_bindings(
        massive_disk_available_cpp = function(...) 5000,
        {
            expect_true(is.list(
                fastEmbedR:::massive_cluster_output_plan(
                    source, 18L, 4L, output)))
            expect_error(fastEmbedR:::massive_cluster_output_plan(
                source, 18L, 4L, output, checkpoint = TRUE),
                "free-disk budget")
        },
        .package = "fastEmbedR"
    )
})

test_that("file-backed cluster assignment preserves landmark rows", {
    set.seed(247)
    data <- matrix(rnorm(51L * 3L), ncol = 3L)
    source_path <- make_massive_fixture(data)
    reference_path <- tempfile(fileext = ".f32")
    prefix <- tempfile()
    output <- tempfile()
    on.exit(unlink(c(source_path, reference_path,
        paste0(prefix, c(".indices.u32", ".distances.f32")),
        paste0(output, c(".clusters.u32", ".confidence.f32")))))
    source <- massive_matrix(source_path, nrow = 51, ncol = 3L)
    selection <- massive_select_landmarks(
        source, 20L, reference_path, chunk_rows = 9L)
    knn <- massive_landmark_knn(source, selection, 5L, prefix,
        chunk_rows = 8L, memory_limit = "256MB")
    graph <- knn_graph(as.matrix(selection$data), k = 5L)
    clustered <- massive_cluster_landmarks(graph, knn, output,
        selection = selection, chunk_rows = 7L,
        memory_limit = "256MB")
    expect_identical(clustered$membership_identity,
        fastEmbedR:::massive_checkpoint_file_identity(
            clustered$membership_path))
    observed <- massive_read_cluster_rows(clustered, 1, 51L)
    expect_equal(observed$membership[selection$indices],
        clustered$reference_membership)
    expect_equal(observed$confidence[selection$indices],
        rep(1, 20L))
    expect_true(clustered$reference_rows_preserved)
})
