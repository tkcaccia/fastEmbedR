test_that("file-backed Louvain moves agree with resident communities", {
    skip_on_os("windows")
    source <- tempfile()
    fuzzy_prefix <- tempfile()
    output <- tempfile(fileext = ".clusters.u32")
    metal_output <- tempfile(fileext = ".clusters.u32")
    metal_dir <- tempfile()
    dir.create(metal_dir)
    on.exit(unlink(metal_dir, recursive = TRUE), add = TRUE)
    repeated <- tempfile(fileext = ".clusters.u32")
    complete <- tempfile(fileext = ".clusters.u32")
    resident_complete <- tempfile(fileext = ".clusters.u32")
    facade <- tempfile(fileext = ".clusters.u32")
    leiden <- tempfile(fileext = ".clusters.u32")
    files <- c(paste0(source,
        c(".indices.u32", ".distances.f32", ".weights.f32")),
        paste0(fuzzy_prefix,
            c(".offsets.u64", ".indices.u32", ".weights.f32")),
        output, metal_output, repeated, complete, resident_complete, facade,
        leiden,
        paste0(c(output, metal_output, repeated), ".part"),
        paste0(c(output, repeated), ".counts.part"),
        paste0(c(output, repeated), ".volumes.part"),
        paste0(complete, c(".part", ".level0.clusters.u32",
            ".coarse.edges.bin", ".coarse.edges.bin.work")),
        paste0(resident_complete, c(".part",
            ".level0.clusters.u32", ".level1.clusters.u32",
            ".coarse.edges.bin", ".coarse.edges.bin.work")),
        paste0(facade, c(".part", ".level0.clusters.u32",
            ".coarse.edges.bin", ".coarse.edges.bin.work")),
        paste0(fuzzy_prefix, ".work"))
    on.exit(unlink(files, recursive = TRUE))
    neighbors <- t(vapply(seq_len(8L), function(row) {
        setdiff(if (row <= 4L) 1:4 else 5:8, row)
    }, integer(3L)))
    distances <- matrix(rep(c(0.1, 0.3, 0.7), 8L),
        nrow = 8L, byrow = TRUE)
    writeBin(as.integer(t(neighbors)), paste0(source,
        ".indices.u32"), size = 4L, endian = "little")
    writeBin(as.numeric(t(distances)), paste0(source,
        ".distances.f32"), size = 4L, endian = "little")
    knn <- massive_knn_graph(source, 8L, 3L)
    directed <- massive_umap_memberships(knn)
    graph <- massive_umap_fuzzy_graph(directed, fuzzy_prefix,
        memory_limit = "64MB")
    if (!isTRUE(graph_clustering_cuda_available_cpp()))
        expect_error(massive_louvain_level(graph, output,
            backend = "cuda"), "unavailable")
    expect_false(file.exists(output))
    fit <- massive_louvain_level(graph, output, seed = 4L,
        chunk_rows = 2L, memory_limit = "128MB")
    again <- massive_louvain_level(graph, repeated, seed = 4L,
        chunk_rows = 4L, memory_limit = "128MB")
    labels <- massive_read_louvain_rows(fit, 1L, 8L)
    edges <- massive_read_graph_edges(graph, 1L, 8L)
    resident <- graph_cluster(list(from = edges$from,
        to = edges$to, weight = edges$weight, n_vertices = 8L),
        method = "louvain", seed = 4L)
    expect_identical(outer(labels, labels, "=="),
        outer(resident$membership, resident$membership, "=="))
    expect_identical(labels,
        massive_read_louvain_rows(again, 1L, 8L))
    expect_identical(head(fit, 4L), labels[1:4])
    expect_identical(fit$n_communities, 2L)
    expect_identical(fit$scan_order,
        "cyclic_blocks_shuffled_rows")
    if (isTRUE(graph_clustering_metal_available_cpp())) {
        metal_fit <- massive_louvain_level(
            graph, metal_output, seed = 4L,
            chunk_rows = 2L, memory_limit = "128MB",
            backend = "metal")
        metal_labels <- massive_read_louvain_rows(
            metal_fit, 1L, 8L)
        expect_identical(metal_fit$backend, "metal")
        expect_identical(metal_fit$scan_order,
            "metal_streamed_blocks_color_phases")
        expect_identical(outer(metal_labels, metal_labels, "=="),
            outer(labels, labels, "=="))
        expect_gte(metal_fit$modularity_final,
            metal_fit$modularity_initial - 1e-7)
        metal_full <- massive_louvain(graph,
            file.path(metal_dir, "full.u32"), seed = 4L,
            chunk_rows = 2L, memory_limit = "128MB",
            backend = "metal")
        expect_identical(metal_full$backend, "metal")
        expect_identical(metal_full$optimization_backend, "metal")
        expect_identical(metal_full$contraction_backend, "cpu")
        expect_identical(metal_full$objective_audit_backend, "cpu")
        expect_identical(outer(
            massive_read_louvain_rows(metal_full, 1L, 8L),
            massive_read_louvain_rows(metal_full, 1L, 8L), "=="),
            outer(labels, labels, "=="))
        metal_facade <- massive_cluster(graph,
            method = "louvain", massive = "out_of_core_graph",
            output = file.path(metal_dir, "facade.u32"),
            backend = "metal", seed = 4L,
            memory_limit = "128MB")
        expect_identical(metal_facade$backend, "metal")
        expect_identical(metal_facade$mode, "out_of_core_graph")
        expect_identical(metal_facade$resources$chunk_rows, 65536L)
    } else {
        expect_error(massive_louvain_level(
            graph, metal_output, backend = "metal"),
            "unavailable")
    }
    expect_gt(fit$modularity_final, fit$modularity_initial)
    expect_equal(fit$modularity_final,
        massive_graph_modularity(graph, output, 2L,
            memory_limit = "128MB")$modularity,
        tolerance = 1e-6)
    full <- massive_louvain(graph, complete, seed = 4L,
        chunk_rows = 2L, memory_limit = "128MB")
    expect_identical(full$method, "louvain_multilevel")
    expect_identical(full$n_communities, 2L)
    expect_gt(length(full$file_backed_levels), 0L)
    expect_identical(full$estimated_coarse_bytes, 0)
    expect_equal(full$contracted_edges,
        file.size(paste0(complete, ".coarse.edges.bin")) / 16)
    expect_identical(outer(massive_read_louvain_rows(full, 1L, 8L),
        massive_read_louvain_rows(full, 1L, 8L), "=="),
        outer(labels, labels, "=="))
    expect_gte(full$modularity_final, fit$modularity_final - 1e-7)
    resident_full <- massive_louvain(graph, resident_complete,
        seed = 4L, chunk_rows = 2L, memory_limit = "256MB")
    expect_length(resident_full$file_backed_levels, 0L)
    expect_equal(full$modularity_final,
        resident_full$modularity_final, tolerance = 1e-6)
    shared <- massive_cluster(graph, method = "louvain",
        massive = "out_of_core_graph", output = facade,
        chunk_rows = 2L, memory_limit = "128MB", seed = 4L)
    expect_identical(shared$mode, "out_of_core_graph")
    expect_identical(shared$backend, "cpu")
    expect_identical(massive_read_louvain_rows(shared, 1L, 8L),
        massive_read_louvain_rows(full, 1L, 8L))
    leiden_fit <- massive_cluster(graph, method = "leiden",
        massive = "out_of_core_graph", output = leiden,
        chunk_rows = 2L, memory_limit = "128MB", seed = 4L)
    resident_leiden <- graph_cluster(list(from = edges$from,
        to = edges$to, weight = edges$weight, n_vertices = 8L),
        method = "leiden", seed = 4L)
    expect_identical(outer(massive_read_leiden_rows(
        leiden_fit, 1L, 8L), massive_read_leiden_rows(
        leiden_fit, 1L, 8L), "=="),
        outer(resident_leiden$membership,
            resident_leiden$membership, "=="))
    expect_identical(leiden_fit$backend, "cpu")
    leiden_parallel <- massive_cluster(graph, method = "leiden",
        massive = "out_of_core_graph",
        output = file.path(metal_dir, "leiden-parallel.u32"),
        chunk_rows = 2L, memory_limit = "128MB", seed = 4L,
        n.cores = 2L, parallel.mode = "fit")
    expect_identical(massive_read_leiden_rows(leiden_parallel, 1L, 8L),
        massive_read_leiden_rows(leiden_fit, 1L, 8L))
    expect_identical(leiden_parallel$cpu_workers_effective, 2L)
    seeded_dir <- tempfile()
    dir.create(seeded_dir)
    on.exit(unlink(seeded_dir, recursive = TRUE), add = TRUE)
    initial_path <- file.path(seeded_dir, "landmark.clusters.u32")
    writeBin(c(1L, 1L, 2L, 2L, 1L, 1L, 2L, 2L),
        initial_path, size = 4L, endian = "little")
    initial <- structure(list(membership_path = initial_path,
        membership_identity = fastEmbedR:::massive_checkpoint_file_identity(
            initial_path),
        nrow = 8L, n_communities = 2L),
        class = "fastEmbedR_massive_clusters")
    initial_score <- massive_graph_modularity(graph, initial_path,
        2L, memory_limit = "128MB")$modularity
    seeded_louvain <- massive_louvain(graph,
        file.path(seeded_dir, "louvain.u32"), initial = initial,
        seed = 4L, chunk_rows = 2L, memory_limit = "128MB")
    expect_equal(seeded_louvain$modularity_initial, initial_score,
        tolerance = 1e-6)
    expect_gte(seeded_louvain$modularity_final, initial_score - 1e-7)
    expect_identical(seeded_louvain$initial_partition_path,
        initial_path)
    if (isTRUE(graph_clustering_metal_available_cpp())) {
        metal_seeded <- massive_louvain(graph,
            file.path(seeded_dir, "metal-seeded.u32"),
            initial = initial, backend = "metal", seed = 4L,
            chunk_rows = 2L, memory_limit = "128MB")
        expect_equal(metal_seeded$modularity_initial,
            initial_score, tolerance = 1e-6)
        expect_gte(metal_seeded$modularity_final,
            initial_score - 1e-7)
        expect_identical(metal_seeded$initial_partition_path,
            initial_path)
    }
    seeded_facade <- massive_cluster(graph, method = "louvain",
        massive = "out_of_core_graph",
        output = file.path(seeded_dir, "facade.u32"),
        initial = initial, seed = 4L, chunk_rows = 2L,
        memory_limit = "128MB")
    expect_identical(massive_read_louvain_rows(seeded_facade, 1L, 8L),
        massive_read_louvain_rows(seeded_louvain, 1L, 8L))
    seeded_leiden <- massive_leiden(graph,
        file.path(seeded_dir, "leiden.u32"), initial = initial,
        seed = 4L, chunk_rows = 2L, memory_limit = "128MB")
    expect_gte(seeded_leiden$modularity_final, initial_score - 1e-7)
    expect_identical(seeded_leiden$initial_partition_path,
        initial_path)
    expect_error(massive_cluster(graph, method = "leiden",
        massive = "landmark", initial = initial),
        "requires.*out_of_core_graph")
    invalid <- initial
    invalid$nrow <- 7L
    expect_error(massive_louvain(graph,
        file.path(seeded_dir, "wrong.u32"), initial = invalid),
        "do not match")
    invalid$nrow <- NA_real_
    expect_error(massive_louvain(graph,
        file.path(seeded_dir, "missing.u32"), initial = invalid),
        "do not match")
    con <- file(initial_path, "ab")
    writeBin(1L, con, size = 4L, endian = "little")
    close(con)
    expect_error(massive_leiden(graph,
        file.path(seeded_dir, "changed.u32"), initial = initial),
        "changed or has wrong size")
    if (!fastEmbedR:::graph_clustering_cuda_available_cpp())
        expect_error(massive_cluster(graph, method = "louvain",
            massive = "out_of_core_graph", output = tempfile(),
            backend = "cuda"), "unavailable")
    expect_error(massive_cluster(graph, method = "louvain",
        massive = "out_of_core_graph", output = tempfile(), k = 3L),
        "do not apply")
    parallel_path <- file.path(seeded_dir, "parallel.u32")
    parallel <- massive_cluster(graph, method = "louvain",
        massive = "out_of_core_graph", output = parallel_path,
        n.cores = 2L, seed = 4L, chunk_rows = 2L,
        memory_limit = "128MB", parallel.mode = "fit")
    expect_identical(parallel$cpu_workers, 2L)
    expect_identical(parallel$cpu_workers_effective, 2L)
    expect_identical(parallel$parallel_stage,
        "row_candidate_preparation")
    expect_identical(massive_read_louvain_rows(parallel, 1L, 8L),
        massive_read_louvain_rows(full, 1L, 8L))
    expect_error(massive_cluster(graph, method = "louvain",
        massive = "out_of_core_graph", output = tempfile(),
        n.cores = 2L), "parallel.mode")
    expect_false(file.exists(paste0(output, ".counts.part")))
    expect_false(file.exists(paste0(output, ".volumes.part")))
    expect_error(massive_louvain_level(graph, output),
        "already exists")
    con <- file(output, "ab")
    writeBin(1L, con, size = 4L, endian = "little")
    close(con)
    expect_error(massive_read_louvain_rows(fit),
        "changed since fitting")
})

test_that("shuffled file-backed moves do not collapse a ring", {
    skip_on_os("windows")
    n <- 1024L
    k <- 6L
    source <- tempfile()
    fuzzy_prefix <- tempfile()
    output <- tempfile(fileext = ".clusters.u32")
    complete <- tempfile(fileext = ".clusters.u32")
    files <- c(paste0(source,
        c(".indices.u32", ".distances.f32", ".weights.f32")),
        paste0(fuzzy_prefix,
            c(".offsets.u64", ".indices.u32", ".weights.f32")),
        paste0(fuzzy_prefix, ".work"), output, complete,
        paste0(complete, c(".part", ".level0.clusters.u32",
            ".coarse.edges.bin", ".coarse.edges.bin.work")))
    on.exit(unlink(files, recursive = TRUE))
    rows <- seq_len(n) - 1L
    neighbors <- outer(rows, seq_len(k), function(row, step) {
        (row + step) %% n + 1L
    })
    writeBin(as.integer(t(neighbors)), paste0(source,
        ".indices.u32"), size = 4L, endian = "little")
    writeBin(rep(as.numeric(seq_len(k)), n), paste0(source,
        ".distances.f32"), size = 4L, endian = "little")
    knn <- massive_knn_graph(source, n, k)
    directed <- massive_umap_memberships(knn)
    graph <- massive_umap_fuzzy_graph(directed, fuzzy_prefix,
        memory_limit = "64MB")
    fit <- massive_louvain_level(graph, output, seed = 4L,
        chunk_rows = 64L, memory_limit = "128MB")
    expect_gt(fit$n_communities, 10L)
    expect_gt(fit$modularity_final, 0.5)
    expect_equal(fit$modularity_final,
        massive_graph_modularity(graph, output,
            fit$n_communities, memory_limit = "128MB")$modularity,
        tolerance = 1e-6)
    full <- massive_louvain(graph, complete, seed = 4L,
        chunk_rows = 64L, memory_limit = "128MB")
    expect_gt(length(full$file_backed_levels), 0L)
    expect_gt(full$modularity_final, fit$modularity_final)
    expect_lt(full$n_communities, fit$n_communities)
    edges <- massive_read_graph_edges(graph, 1L, n)
    resident <- graph_cluster(list(from = edges$from,
        to = edges$to, weight = edges$weight, n_vertices = n),
        method = "louvain", seed = 4L)
    expect_gte(full$modularity_final,
        resident$modularity - 0.05)
    if (!fastEmbedR:::graph_clustering_cuda_available_cpp())
        expect_error(massive_louvain(graph, tempfile(),
            backend = "cuda"), "unavailable")
    expect_error(fastEmbedR:::massive_louvain_coarse_graph(
        list(n_vertices = 1e6, n_edges = 1e7), 128 * 1024^2),
        "No fallback")
})

test_that("file-backed Louvain resumes verified completed stages", {
    skip_on_os("windows")
    directory <- tempfile()
    dir.create(directory)
    on.exit(unlink(directory, recursive = TRUE))
    source <- file.path(directory, "knn")
    fuzzy <- file.path(directory, "fuzzy")
    neighbors <- t(vapply(seq_len(8L), function(row) {
        setdiff(if (row <= 4L) 1:4 else 5:8, row)
    }, integer(3L)))
    distances <- matrix(rep(c(0.1, 0.3, 0.7), 8L),
        nrow = 8L, byrow = TRUE)
    writeBin(as.integer(t(neighbors)), paste0(source,
        ".indices.u32"), size = 4L, endian = "little")
    writeBin(as.numeric(t(distances)), paste0(source,
        ".distances.f32"), size = 4L, endian = "little")
    knn <- massive_knn_graph(source, 8L, 3L)
    directed <- massive_umap_memberships(knn)
    graph <- massive_umap_fuzzy_graph(directed, fuzzy,
        memory_limit = "64MB")
    baseline <- massive_louvain(graph,
        file.path(directory, "baseline.u32"), seed = 4L,
        chunk_rows = 2L, memory_limit = "256MB")
    controls <- fastEmbedR:::massive_louvain_controls(10L, 1, 4L)

    after_level <- file.path(directory, "after_level.u32")
    level <- massive_louvain_level(graph,
        paste0(after_level, ".level0.clusters.u32"),
        seed = 4L, chunk_rows = 2L, memory_limit = "256MB")
    signature <- fastEmbedR:::massive_louvain_signature(graph,
        after_level, controls, 2L, "256MB")
    fastEmbedR:::massive_louvain_save_stage(after_level,
        signature, "level0", level)
    expect_error(massive_louvain(graph, after_level,
        seed = 4L, resolution = 2, chunk_rows = 2L,
        memory_limit = "256MB", checkpoint = TRUE,
        resume = TRUE), "checkpoint does not match")
    expect_false(file.exists(after_level))
    resumed_level <- massive_louvain(graph, after_level,
        seed = 4L, chunk_rows = 2L, memory_limit = "256MB",
        checkpoint = TRUE, resume = TRUE)
    expect_false(file.exists(paste0(after_level,
        ".checkpoint.rds")))
    expect_identical(massive_read_louvain_rows(resumed_level,
        1L, 8L), massive_read_louvain_rows(baseline, 1L, 8L))
    if (isTRUE(graph_clustering_metal_available_cpp())) {
        metal_path <- file.path(directory, "metal-resume.u32")
        metal_level <- massive_louvain_level(graph,
            paste0(metal_path, ".level0.clusters.u32"),
            seed = 4L, chunk_rows = 2L,
            memory_limit = "256MB", backend = "metal")
        metal_controls <- controls
        metal_controls$backend <- "metal"
        metal_signature <- fastEmbedR:::massive_louvain_signature(
            graph, metal_path, metal_controls, 2L, "256MB")
        fastEmbedR:::massive_louvain_save_stage(
            metal_path, metal_signature, "level0", metal_level)
        metal_resumed <- massive_louvain(graph, metal_path,
            seed = 4L, chunk_rows = 2L,
            memory_limit = "256MB", backend = "metal",
            checkpoint = TRUE, resume = TRUE)
        expect_identical(metal_resumed$backend, "metal")
        expect_identical(metal_resumed$n_communities, 2L)
    }

    after_contract <- file.path(directory, "after_contract.u32")
    prepared <- fastEmbedR:::massive_louvain_prepare(graph,
        after_contract, controls, 2L, "256MB", TRUE, FALSE)
    expect_identical(readRDS(paste0(after_contract,
        ".checkpoint.rds"))$stage, "contracted")
    expect_true(file.exists(prepared$contracted$path))
    with_mocked_bindings(
        massive_disk_available_cpp = function(...) 0,
        expect_silent(fastEmbedR:::massive_louvain_contract_resources(
            graph, after_contract, "256MB", "contracted")),
        .package = "fastEmbedR"
    )
    resumed_contract <- massive_cluster(graph, method = "louvain",
        massive = "out_of_core_graph", output = after_contract,
        seed = 4L, chunk_rows = 2L, memory_limit = "256MB",
        checkpoint = TRUE, resume = TRUE)
    expect_identical(resumed_contract$mode, "out_of_core_graph")
    expect_identical(massive_read_louvain_rows(resumed_contract,
        1L, 8L), massive_read_louvain_rows(baseline, 1L, 8L))

    tampered <- file.path(directory, "tampered.u32")
    prepared <- fastEmbedR:::massive_louvain_prepare(graph,
        tampered, controls, 2L, "256MB", TRUE, FALSE)
    con <- file(prepared$contracted$path, "ab")
    writeBin(as.raw(0), con)
    close(con)
    expect_error(massive_louvain(graph, tampered,
        seed = 4L, chunk_rows = 2L, memory_limit = "256MB",
        checkpoint = TRUE, resume = TRUE),
        "contraction changed")
    expect_false(file.exists(tampered))
    expect_error(massive_louvain(graph, tempfile(),
        resume = TRUE), "requires `checkpoint = TRUE`")
    expect_error(massive_cluster(graph, method = "louvain",
        massive = "landmark", checkpoint = TRUE),
        "needs a matrix source")
    expect_error(massive_cluster(graph, method = "louvain",
        massive = "landmark", checkpoint = NA),
        "requires `checkpoint = TRUE`")
})

test_that("contracted Louvain scans a row beyond 65,536 edges", {
    skip_on_os("windows")
    n_vertices <- 65538L
    n_edges <- n_vertices - 1L
    input <- tempfile(fileext = ".coarse.edges.bin")
    prefix <- tempfile()
    output <- tempfile(fileext = ".clusters.u32")
    paths <- c(input, paste0(prefix,
        c(".offsets.u64", ".indices.u32", ".weights.f64", ".work")),
        paste0(output, c("", ".part", ".counts.part",
            ".volumes.part")))
    on.exit(unlink(paths, recursive = TRUE))
    records <- matrix(as.raw(0), 16L, n_edges)
    records[1:4, ] <- matrix(writeBin(rep(1L, n_edges), raw(),
        size = 4L, endian = "little"), nrow = 4L)
    records[5:8, ] <- matrix(writeBin(seq.int(2L, n_vertices),
        raw(), size = 4L, endian = "little"), nrow = 4L)
    records[9:16, ] <- matrix(writeBin(rep(1, n_edges), raw(),
        size = 8L, endian = "little"), nrow = 8L)
    writeBin(as.raw(records), input)
    csr <- fastEmbedR:::massive_coarse_csr_cpp(input, n_vertices,
        n_edges, prefix, 128 * 1024^2)
    expect_equal(csr$max_degree, n_edges)
    expect_equal(csr$n_edges, 2 * n_edges)
    csr$file_identity <- fastEmbedR:::massive_checkpoint_file_identity(
        c(csr$offsets_path, csr$indices_path, csr$weights_path))
    csr$weight_dtype <- "float64"
    plan <- fastEmbedR:::massive_louvain_resources(csr, 128L,
        "128MB")
    expect_gte(plan$edge_budget, n_edges)
    fit <- fastEmbedR:::massive_louvain_run_level(csr, output,
        2L, 1, 4L, 128L, "128MB", TRUE, 1L)
    expect_true(is.finite(fit$modularity_final))
    expect_equal(fit$total_edge_weight, n_edges)
})
