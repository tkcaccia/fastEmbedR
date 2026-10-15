test_that("Metal landmark clustering never falls back silently", {
    with_mocked_bindings(
        graph_clustering_metal_available_cpp = function() FALSE,
        expect_error(fastEmbedR:::massive_cluster_backend(
            "leiden", "metal"), "no CPU fallback"),
        .package = "fastEmbedR")
    expect_error(fastEmbedR:::massive_cluster_backend(
        "walktrap", "metal"), "CPU-only")
})

test_that("file-backed Metal exact KNN matches streamed CPU search", {
    skip_if_not(fastEmbedR:::native_metal_knn_available_cpp())
    skip_if_not(requireNamespace("float", quietly = TRUE))
    set.seed(1942)
    values <- matrix(stats::rnorm(37L * 5L), ncol = 5L)
    input <- tempfile(fileext = ".f32")
    cpu_prefix <- tempfile()
    metal_prefix <- tempfile()
    suffix <- c(".indices.u32", ".distances.f32")
    on.exit(unlink(c(input, paste0(cpu_prefix, suffix),
        paste0(metal_prefix, suffix),
        paste0(metal_prefix, ".checkpoint.rds"))))
    writeBin(as.numeric(t(values)), input, size = 4L,
        endian = "little")
    source <- massive_matrix(input, nrow(values), ncol(values))
    cpu <- massive_full_knn_graph(source, 4L, cpu_prefix,
        backend = "cpu", method = "exact", chunk_rows = 7L,
        reference_chunk_rows = 11L, memory_limit = "256MB")
    metal <- massive_full_knn_graph(source, 4L, metal_prefix,
        backend = "metal", method = "exact", chunk_rows = 7L,
        reference_chunk_rows = 11L, memory_limit = "256MB",
        checkpoint = TRUE, audit_rows = 8L)
    observed <- massive_read_graph_edges(metal, 1L, nrow(values))
    expected <- massive_read_graph_edges(cpu, 1L, nrow(values))
    expect_identical(metal$backend, "metal")
    expect_identical(metal$method, "native_metal_exact_sharded")
    expect_identical(metal$query_storage, "R_float32")
    expect_identical(metal$resources$metal_memory_model,
        "unified_shared")
    expect_gt(metal$resources$n_shards, 1L)
    expect_true(metal$exact)
    expect_true(metal$recall_audited)
    expect_equal(metal$observed_recall, 1)
    expect_identical(observed$to, expected$to)
    expect_equal(observed$distance, expected$distance,
        tolerance = 1e-3)
})

test_that("file-backed Metal KNN rejects unsupported routes", {
    source <- massive_matrix(matrix(as.double(seq_len(20L)),
        ncol = 2L))
    expect_error(massive_full_knn_graph(source, 2L, tempfile(),
        backend = "metal", method = "hnsw_sharded"),
        "Metal KNN supports exact shards only")
    expect_error(massive_full_knn_graph(source, 2L, tempfile(),
        backend = "metal", method = "ivf_sharded"),
        "Metal KNN supports exact shards only")
    expect_error(massive_full_knn_graph(source, 2L, tempfile(),
        backend = "metal", method = "exact", devices = 0L),
        "requires CUDA")
    expect_error(massive_landmark_knn(source, source, 2L,
        tempfile(), backend = "metal", method = "hnsw"),
        "Metal landmark KNN supports exact search only")
    expect_error(massive_landmark_knn(source, source, 2L,
        tempfile(), backend = "metal", n.cores = 4L),
        "n.cores = 1")
    expect_error(massive_full_knn_graph(source, 2L, tempfile(),
        backend = "metal", n.cores = 4L), "n.cores = 1")
})

test_that("file-backed Metal PCA matches CPU PCA and reopens scores", {
    skip_if_not(fastEmbedR:::embedding_metal_available_cpp())
    set.seed(1944)
    values <- sweep(matrix(stats::rnorm(57L * 5L), ncol = 5L),
        2L, seq(40, 200, length.out = 5L), `+`)
    input <- tempfile(fileext = ".f32")
    directory <- tempfile()
    dir.create(directory)
    on.exit(unlink(c(input, directory), recursive = TRUE))
    writeBin(as.numeric(t(values)), input, size = 4L,
        endian = "little")
    source <- massive_matrix(input, 57L, 5L)
    cpu <- pca(source, ncomp = 2L, backend = "cpu", n.cores = 1L,
        massive = "out_of_core", output = file.path(directory, "cpu.f32"),
        chunk_rows = 9L, memory_limit = "512MB", scale = TRUE)
    metal_path <- file.path(directory, "metal.f32")
    metal <- pca(source, ncomp = 2L, backend = "metal", n.cores = 1L,
        massive = "out_of_core", output = metal_path,
        chunk_rows = 9L, memory_limit = "512MB", checkpoint = TRUE,
        scale = TRUE)
    expect_identical(metal$backend, "metal")
    expect_identical(metal$fit_statistics_backend,
        "metal_streamed_crossprod")
    expect_identical(metal$decomposition_backend, "cpu_eigen")
    expect_identical(metal$resources$metal_memory_model,
        "unified_shared")
    expect_true(is.na(metal$resources$peak_vram_bytes))
    expect_equal(metal$singular_values, cpu$singular_values,
        tolerance = 1e-3)
    expect_equal(tcrossprod(as.matrix(metal$scores)),
        tcrossprod(as.matrix(cpu$scores)), tolerance = 1e-3)
    expect_equal(massive_read_rows(massive_open_pca(metal_path)$scores,
        1L, 57L), as.matrix(metal$scores))
    expect_error(pca(source, ncomp = 2L, backend = "metal",
        n.cores = 2L, massive = "out_of_core",
        output = file.path(directory, "invalid.f32")),
        "n.cores = 1")
})

test_that("Metal PCA resumes a saved fit without repeating covariance", {
    skip_if_not(fastEmbedR:::embedding_metal_available_cpp())
    set.seed(1945)
    values <- matrix(stats::rnorm(39L * 4L), ncol = 4L)
    input <- tempfile(fileext = ".f32")
    directory <- tempfile()
    dir.create(directory)
    on.exit(unlink(c(input, directory), recursive = TRUE))
    writeBin(as.numeric(t(values)), input, size = 4L,
        endian = "little")
    source <- massive_matrix(input, 39L, 4L)
    path <- file.path(directory, "resume.f32")
    run <- function(resume = FALSE) pca(source, ncomp = 2L,
        backend = "metal", n.cores = 1L,
        massive = "out_of_core", output = path,
        chunk_rows = 8L, memory_limit = "512MB",
        checkpoint = TRUE, resume = resume)
    with_mocked_bindings(
        massive_pca_project_metal = function(...) stop("interrupted"),
        expect_error(run(), "interrupted"),
        .package = "fastEmbedR")
    expect_identical(readRDS(paste0(path, ".checkpoint.rds"))$stage,
        "fit")
    with_mocked_bindings(
        massive_pca_covariance_metal = function(...) stop("fit repeated"),
        result <- run(resume = TRUE),
        .package = "fastEmbedR")
    expect_equal(dim(as.matrix(result$scores)), c(39L, 2L))
    expect_false(file.exists(paste0(path, ".checkpoint.rds")))
})

test_that("Metal landmark UMAP and t-SNE write file-backed layouts", {
    skip_if_not(fastEmbedR:::native_metal_knn_available_cpp())
    skip_if_not_installed("float")
    set.seed(1943)
    values <- matrix(stats::rnorm(48L * 4L), ncol = 4L)
    input <- tempfile(fileext = ".f32")
    directory <- tempfile()
    dir.create(directory)
    on.exit(unlink(c(input, directory), recursive = TRUE))
    writeBin(as.numeric(t(values)), input, size = 4L,
        endian = "little")
    source <- massive_matrix(input, 48L, 4L)
    umap_path <- file.path(directory, "umap.f32")
    tsne_path <- file.path(directory, "tsne.f32")
    embedding <- umap(source, massive = "landmark", landmarks = 20L,
        n_neighbors = 5L, transform_k = 5L, backend = "metal",
        output = umap_path, refinement_epochs = 0L,
        chunk_rows = 9L, memory_limit = "512MB")
    expect_identical(embedding$backend, "metal")
    expect_identical(embedding$graph$backend, "metal")
    expect_false(embedding$graph$index_reused)
    expect_true(is.na(embedding$resources$peak_vram_bytes))
    expect_identical(embedding$resources$metal_memory_model,
        "unified_shared")
    cpu <- massive_landmark_knn(source, embedding$selection$data,
        5L, file.path(directory, "cpu"), backend = "cpu",
        method = "exact", chunk_rows = 9L,
        memory_limit = "512MB")
    observed <- massive_read_knn_rows(embedding$graph, 1L, 48L)
    expected <- massive_read_knn_rows(cpu, 1L, 48L)
    expect_identical(observed$indices, expected$indices)
    expect_equal(observed$distances, expected$distances,
        tolerance = 1e-3)
    expect_equal(dim(as.matrix(embedding$layout)), c(48L, 2L))
    expect_true(all(is.finite(as.matrix(embedding$layout))))
    expect_identical(massive_open_projection(umap_path)$backend,
        "metal")
    transformed <- tsne(source, massive = "landmark", landmarks = 20L,
        perplexity = 3, transform_perplexity = 3,
        early_exaggeration_iter = 3L, n_iter = 3L,
        transform_iter = 3L, backend = "metal",
        output = tsne_path, chunk_rows = 9L,
        memory_limit = "512MB")
    expect_identical(transformed$backend, "metal")
    expect_identical(transformed$graph$backend, "metal")
    expect_equal(dim(as.matrix(transformed$layout)), c(48L, 2L))
    expect_true(all(is.finite(as.matrix(transformed$layout))))
    expect_identical(massive_open_projection(tsne_path)$backend,
        "metal")
})

test_that("Metal landmark communities retain every source row", {
    skip_if_not(fastEmbedR:::graph_clustering_metal_available_cpp())
    skip_if_not(fastEmbedR:::native_metal_knn_available_cpp())
    skip_if_not_installed("float")
    set.seed(1947)
    values <- rbind(
        matrix(stats::rnorm(48L, sd = 0.1), ncol = 3L),
        matrix(stats::rnorm(48L, mean = 5, sd = 0.1), ncol = 3L))
    input <- tempfile(fileext = ".f32")
    directory <- tempfile()
    dir.create(directory)
    on.exit(unlink(c(input, directory), recursive = TRUE))
    writeBin(as.numeric(t(values)), input, size = 4L,
        endian = "little")
    source <- massive_matrix(input, nrow(values), ncol(values))
    for (method in c("leiden", "louvain")) {
        result <- with_mocked_bindings(
            fastembedr_graph_cluster_cpp = function(...) {
                stop("CPU clustering fallback")
            },
            massive_cluster(source, method = method,
                landmarks = 12L, k = 4L,
                output = file.path(directory, method),
                backend = "metal", n.cores = 1L,
                chunk_rows = 7L, memory_limit = "512MB",
                checkpoint = TRUE, seed = 4L),
            .package = "fastEmbedR")
        labels <- massive_read_cluster_rows(result, 1L, nrow(values))
        expect_identical(result$backend, "metal")
        expect_identical(result$graph_backend, "metal")
        expect_identical(result$assignment_backend, "cpu")
        expect_identical(result$resources$metal_memory_model,
            "unified_shared")
        expect_true(is.na(result$resources$peak_vram_bytes))
        expect_gt(result$resources$metal_buffer_bytes_estimate, 0)
        expect_length(labels$membership, nrow(values))
        expect_true(all(labels$membership >= 1L))
        expect_true(all(is.finite(labels$confidence)))
        expect_identical(labels$membership[result$selection$indices],
            result$reference_membership)
    }
    expect_error(massive_cluster(source, method = "walktrap",
        landmarks = 12L, k = 4L, backend = "metal",
        output = file.path(directory, "walktrap")), "CPU-only")
    expect_error(massive_cluster(source, method = "leiden",
        massive = "out_of_core_graph", backend = "metal",
        output = file.path(directory, "full.clusters.u32")),
        "Full-graph clustering needs CPU or explicit CUDA")
})

test_that("full-graph Metal UMAP streams a prepared fuzzy graph", {
    skip_if_not(fastEmbedR:::embedding_metal_available_cpp())
    directory <- tempfile()
    dir.create(directory)
    on.exit(unlink(directory, recursive = TRUE))
    prefix <- file.path(directory, "knn")
    fuzzy_prefix <- file.path(directory, "fuzzy")
    rows <- 32L
    ids <- outer(seq_len(rows), seq_len(4L), function(i, j)
        (i + j - 1L) %% rows + 1L)
    writeBin(as.integer(t(ids)), paste0(prefix, ".indices.u32"),
        size = 4L, endian = "little")
    writeBin(rep(c(0.1, 0.3, 0.6, 1), rows),
        paste0(prefix, ".distances.f32"), size = 4L,
        endian = "little")
    knn <- massive_knn_graph(prefix, rows, 4L)
    directed <- massive_umap_memberships(knn)
    graph <- massive_umap_fuzzy_graph(directed, fuzzy_prefix,
        memory_limit = "64MB")
    set.seed(1946)
    initial <- matrix(stats::rnorm(rows * 2L, sd = 0.2), ncol = 2L)
    init_path <- file.path(directory, "init.f32")
    writeBin(as.numeric(t(initial)), init_path, size = 4L,
        endian = "little")
    source <- massive_matrix(init_path, rows, 2L)
    path <- file.path(directory, "metal.f32")
    fit <- with_mocked_bindings(
        massive_umap_optimize_cpp = function(...) stop("CPU fallback"),
        massive_umap_optimize(graph, source, path, backend = "metal",
            n_epochs = 16L, negative_sample_rate = 1L,
            chunk_rows = 5L, memory_limit = "256MB"),
        .package = "fastEmbedR")
    expect_identical(fit$backend, "metal")
    expect_identical(fit$optimizer, "streamed_metal_atomic")
    expect_identical(fit$resources$metal_memory_model,
        "unified_shared")
    expect_equal(fit$resources$edge_batch_capacity, 65536L)
    expect_gt(fit$updates$positive_updates, 0)
    expect_gt(fit$updates$negative_updates, 0)
    expect_equal(dim(massive_read_rows(fit, 1L, rows)), c(rows, 2L))
    expect_true(all(is.finite(massive_read_rows(fit, 1L, rows))))
    expect_gt(sum(abs(massive_read_rows(fit, 1L, rows) - initial)), 0)
    batched <- file.path(directory, "batched.f32")
    native <- fastEmbedR:::massive_umap_optimize_metal_cpp(
        graph$offsets_path, graph$indices_path, graph$weights_path,
        rows, init_path, "f32", batched, 16L, 1L, 1, 0.1, 1, 1L,
        5L, as.integer(graph$max_degree), 256 * 1024^2)
    expect_equal(native$n_vertices, rows)
    expect_equal(file.info(batched)$size, rows * 2L * 4L)
    mapped_path <- file.path(directory, "mapped.f32")
    mapped <- with_mocked_bindings(
        massive_umap_optimize_cpp = function(...) stop("CPU fallback"),
        massive_umap_optimize(graph, source, mapped_path,
            backend = "metal", layout_storage = "mmap",
            n_epochs = 16L, negative_sample_rate = 1L,
            chunk_rows = 5L, memory_limit = "256MB"),
        .package = "fastEmbedR")
    expect_identical(mapped$backend, "metal")
    expect_identical(mapped$layout_storage, "mmap")
    expect_identical(mapped$parameters$layout_storage, "mmap")
    expect_equal(mapped$resources$layout_mmap_bytes, rows * 8L)
    expect_lt(mapped$resources$layout_ram_bytes,
        fit$resources$layout_ram_bytes)
    expect_equal(file.info(mapped_path)$size, rows * 8L)
    expect_false(file.exists(paste0(mapped_path, ".part")))
    expect_true(all(is.finite(massive_read_rows(mapped, 1L, rows))))
    expect_equal(massive_read_rows(mapped, 1L, rows),
        massive_read_rows(fit, 1L, rows), tolerance = 1e-5)
    fbin_path <- file.path(directory, "init.fbin")
    stream <- file(fbin_path, "wb")
    writeBin(c(rows, 2L), stream, size = 4L, endian = "little")
    writeBin(as.numeric(t(initial)), stream, size = 4L,
        endian = "little")
    close(stream)
    fbin_fit <- massive_umap_optimize(graph,
        massive_matrix(fbin_path), file.path(directory, "fbin.f32"),
        backend = "metal", layout_storage = "mmap", n_epochs = 16L,
        negative_sample_rate = 1L, memory_limit = "256MB")
    expect_equal(massive_read_rows(fbin_fit, 1L, rows),
        massive_read_rows(mapped, 1L, rows), tolerance = 1e-5)
    public_path <- file.path(directory, "public-mapped.f32")
    public <- with_mocked_bindings(
        massive_umap_optimize_cpp = function(...) stop("CPU fallback"),
        umap(graph, massive = "out_of_core_graph", init = source,
            output = public_path, backend = "metal", n.cores = 1L,
            layout_storage = "mmap", memory_limit = "256MB"),
        .package = "fastEmbedR")
    expect_identical(public$backend, "metal")
    expect_identical(public$layout_storage, "mmap")
    expect_equal(file.info(public_path)$size, rows * 8L)
    expect_true(all(is.finite(massive_read_rows(public, 1L, rows))))
    expect_error(massive_umap_optimize(graph, source,
        file.path(directory, "multi.f32"), backend = "metal",
        devices = 0L), "requires CUDA")
    expect_error(massive_umap_optimize(graph, source,
        file.path(directory, "small.f32"), backend = "metal",
        memory_limit = "128MB"), "memory_limit")
    init_3d <- file.path(directory, "init3d.f32")
    writeBin(as.numeric(t(cbind(initial, 0))), init_3d,
        size = 4L, endian = "little")
    expect_error(massive_umap_optimize(graph,
        massive_matrix(init_3d, rows, 3L),
        file.path(directory, "three.f32"), backend = "metal"),
        "2D output")
    restart_path <- file.path(directory, "restart.f32")
    commit <- fastEmbedR:::massive_umap_checkpoint_commit
    with_mocked_bindings(
        massive_umap_checkpoint_commit = function(...) {
            commit(...)
            stop("interrupted after checkpoint")
        },
        expect_error(massive_umap_optimize(graph, source,
            restart_path, backend = "metal", n_epochs = 16L,
            negative_sample_rate = 1L, memory_limit = "256MB",
            checkpoint = TRUE, checkpoint_every = 4L),
            "interrupted after checkpoint"),
        .package = "fastEmbedR")
    expect_true(file.exists(paste0(restart_path, ".checkpoint.rds")))
    resumed <- massive_umap_optimize(graph, source, restart_path,
        backend = "metal", n_epochs = 16L,
        negative_sample_rate = 1L, memory_limit = "256MB",
        checkpoint = TRUE, resume = TRUE, checkpoint_every = 4L)
    expect_identical(resumed$parameters$resumed_from_epoch, 4L)
    expect_equal(dim(massive_read_rows(resumed, 1L, rows)),
        c(rows, 2L))
    expect_false(file.exists(paste0(restart_path, ".checkpoint.rds")))
    mapped_restart <- file.path(directory, "mapped-restart.f32")
    with_mocked_bindings(
        massive_umap_checkpoint_commit = function(...) {
            commit(...)
            stop("interrupted after checkpoint")
        },
        expect_error(massive_umap_optimize(graph, source,
            mapped_restart, backend = "metal", n_epochs = 16L,
            negative_sample_rate = 1L, memory_limit = "256MB",
            layout_storage = "mmap", checkpoint = TRUE,
            checkpoint_every = 4L), "interrupted after checkpoint"),
        .package = "fastEmbedR")
    expect_true(file.exists(paste0(mapped_restart, ".part")))
    mapped_resumed <- massive_umap_optimize(graph, source,
        mapped_restart, backend = "metal", n_epochs = 16L,
        negative_sample_rate = 1L, memory_limit = "256MB",
        layout_storage = "mmap", checkpoint = TRUE, resume = TRUE,
        checkpoint_every = 4L)
    expect_identical(mapped_resumed$parameters$resumed_from_epoch, 4L)
    expect_equal(file.info(mapped_restart)$size, rows * 8L)
    expect_false(file.exists(paste0(mapped_restart, ".part")))
})
