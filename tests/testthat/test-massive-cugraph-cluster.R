test_that("full-graph CUDA clustering uses the explicit cuGraph route", {
    graph <- structure(list(), class = "fastEmbedR_massive_graph")
    with_mocked_bindings(
        {
            result <- massive_cluster(graph, method = "leiden",
                massive = "out_of_core_graph", backend = "cuda",
                output = tempfile(), devices = c(0L, 1L))
            expect_identical(result$mode, "out_of_core_graph")
            expect_identical(result$backend, "cuda")
            expect_identical(result$devices, c(0L, 1L))
            expect_error(massive_cluster(graph, method = "walktrap",
                massive = "out_of_core_graph", backend = "cuda",
                output = tempfile(), devices = c(0L, 1L)),
                "Walktrap|supports Leiden/Louvain")
            expect_error(massive_cluster(graph, method = "leiden",
                massive = "out_of_core_graph", backend = "cuda",
                output = tempfile(), devices = c(0L, 1L),
                checkpoint = TRUE), "without checkpoint")
        },
        massive_cugraph_cluster = function(graph, output, method,
                devices, ...) list(backend = "cuda", devices = devices),
        .package = "fastEmbedR"
    )
})

test_that("cuGraph plan rejects insufficient GPU capacity", {
    graph <- structure(list(n_vertices = 4, n_edges = 8,
        max_degree = 2, storage = "csr",
        weight_method = "umap_fuzzy_union", symmetrized = TRUE),
        class = "fastEmbedR_massive_graph")
    with_mocked_bindings(
        {
            plan <- fastEmbedR:::massive_cugraph_plan(graph,
                tempfile(), c(0L, 1L), 2L, "1GB")
            expect_identical(plan$uuids, c("GPU-A", "GPU-B"))
            expect_equal(length(plan$devices), 2L)
            selected <- fastEmbedR:::massive_cugraph_plan(graph,
                tempfile(), c(2L, 0L), 2L, "1GB")
            expect_identical(selected$devices, c(2L, 0L))
            expect_identical(selected$uuids, c("GPU-C", "GPU-A"))
            expect_error(fastEmbedR:::massive_cugraph_plan(graph,
                tempfile(), 0L, 2L, "1GB"),
                "two to four")
            stale <- tempfile()
            on.exit(unlink(paste0(stale, ".part")), add = TRUE)
            writeBin(1L, paste0(stale, ".part"), size = 4L)
            expect_error(fastEmbedR:::massive_cugraph_plan(graph,
                stale, c(0L, 1L), 2L, "1GB"),
                "output files already exist")
        },
        massive_validate_devices = function(devices, ...) devices,
        massive_auto_cuda_free = function(...) 4 * 1024^3,
        massive_louvain_output = function(output, ...) output,
        massive_cuda_device_uuids = function(devices)
            c("GPU-A", "GPU-B", "GPU-C")[devices + 1L],
        .package = "fastEmbedR"
    )
})

test_that("cuGraph RAM admission includes concurrent GPU workers", {
    graph <- structure(list(n_vertices = 1e6, n_edges = 3e7,
        max_degree = 30, storage = "csr",
        weight_method = "umap_fuzzy_union", symmetrized = TRUE),
        class = "fastEmbedR_massive_graph")
    with_mocked_bindings(
        {
            plan <- fastEmbedR:::massive_cugraph_plan(graph,
                tempfile(), 0:1, 1e5, "600MB")
            expect_equal(plan$ram_staging_estimate_bytes, 288e6)
            expect_error(fastEmbedR:::massive_cugraph_plan(graph,
                tempfile(), 0:3, 1e5, "600MB"), "RAM budget")
        },
        massive_validate_devices = function(devices, ...) devices,
        massive_auto_cuda_free = function(...) 100 * 1024^3,
        massive_louvain_output = function(output, ...) output,
        massive_cuda_device_uuids = function(devices)
            paste0("GPU-", devices),
        .package = "fastEmbedR"
    )
})

test_that("cuGraph reports the requested fit controls", {
    check <- fastEmbedR:::massive_cugraph_controls_check
    controls <- list(passes = 10L, resolution = 0.7, seed = 4L)
    metadata <- list(method = "leiden", backend = "cugraph_dask",
        iteration_semantics = "max_iter", seed_applied = TRUE,
        iterations_requested = 10L, resolution = 0.7,
        seed_requested = 4L)
    expect_silent(check(metadata, "leiden", controls))
    metadata$resolution <- 1
    expect_error(check(metadata, "leiden", controls),
        "requested clustering controls")
    metadata$resolution <- 0.7
    metadata$method <- "louvain"
    expect_error(check(metadata, "leiden", controls),
        "requested clustering controls")
    metadata$method <- "leiden"
    metadata$seed_requested <- 5L
    expect_error(check(metadata, "leiden", controls),
        "requested clustering controls")
})

test_that("cuGraph keeps the independently verified modularity", {
    path <- tempfile(fileext = ".clusters.u32")
    part <- paste0(path, ".part")
    on.exit(unlink(c(path, part)))
    writeBin(as.integer(c(1, 1, 2, 2)), part, size = 4L,
        endian = "little")
    graph <- list(n_vertices = 4, offsets_path = "a",
        indices_path = "b", weights_path = "c")
    before <- list(paths = c("a", "b", "c"))
    plan <- list(path = path, devices = c(0L, 1L),
        uuids = c("GPU-A", "GPU-B"), chunk_rows = 2L,
        free_vram = 1e9, vram_estimate = 1e8,
        ram_staging_estimate_bytes = 1000, limit = 3000)
    metadata <- list(rows = 4, communities = 2,
        iterations_requested = 10L,
        distributed_graph = TRUE, worker_count = 2L,
        modularity_reported = 0.8,
        device_spilling = FALSE,
        devices = list(list(uuid = "GPU-A",
                pci_bus_id = "20:00.0"),
            list(uuid = "GPU-B", pci_bus_id = "40:00.0")),
        gpu_memory = list(scope = "whole_physical_gpu",
            samples = 3L, host_process_tree = list(
                scope = "launcher_and_workers", samples = 3L,
                tracked_samples = 2L,
                baseline_rss_bytes = 1000,
                peak_rss_bytes = 2000), workers = list(
                list(address = "a", pid = 2L,
                    pci_bus_id = "20:00.0", uuid = "GPU-A"),
                list(address = "b", pid = 3L,
                    pci_bus_id = "40:00.0", uuid = "GPU-B")),
            devices = list(
                list(uuid = "GPU-A", baseline_used_bytes = 10,
                    peak_used_bytes = 100, peak_delta_bytes = 90),
                list(uuid = "GPU-B", baseline_used_bytes = 20,
                    peak_used_bytes = 200, peak_delta_bytes = 180))))
    with_mocked_bindings(
        {
            incomplete <- metadata
            incomplete$gpu_memory$samples <- 1L
            expect_error(fastEmbedR:::massive_cugraph_finish(
                graph, plan, "leiden", before, incomplete, 1,
                "1GB"), "telemetry is incomplete")
            incomplete <- metadata
            incomplete$gpu_memory$host_process_tree <- NULL
            expect_error(fastEmbedR:::massive_cugraph_finish(
                graph, plan, "leiden", before, incomplete, 1,
                "1GB"), "host RAM telemetry is incomplete")
            incomplete <- metadata
            incomplete$gpu_memory$host_process_tree$peak_rss_bytes <- 9
            expect_error(fastEmbedR:::massive_cugraph_finish(
                graph, plan, "leiden", before, incomplete, 1,
                "1GB"), "host RAM telemetry is incomplete")
            over_budget <- metadata
            over_budget$gpu_memory$host_process_tree$peak_rss_bytes <- 3001
            expect_error(fastEmbedR:::massive_cugraph_finish(
                graph, plan, "leiden", before, over_budget, 1,
                "1GB"), "requested RAM budget")
            single <- metadata
            single$distributed_graph <- FALSE
            expect_error(fastEmbedR:::massive_cugraph_finish(
                graph, plan, "leiden", before, single, 1,
                "1GB"), "telemetry is incomplete")
            single <- metadata
            single$worker_count <- 1L
            expect_error(fastEmbedR:::massive_cugraph_finish(
                graph, plan, "leiden", before, single, 1,
                "1GB"), "telemetry is incomplete")
            mapped <- metadata
            mapped$gpu_memory$workers[[2L]]$pci_bus_id <- "20:00.0"
            expect_error(fastEmbedR:::massive_cugraph_finish(
                graph, plan, "leiden", before, mapped, 1,
                "1GB"), "worker GPU map is incomplete")
            invalid <- metadata
            invalid$gpu_memory$devices[[1L]]$peak_used_bytes <- NA_real_
            expect_error(fastEmbedR:::massive_cugraph_finish(
                graph, plan, "leiden", before, invalid, 1,
                "1GB"), "GPU peak is invalid")
            invalid <- metadata
            invalid$gpu_memory$devices[[1L]]$peak_delta_bytes <- 91
            expect_error(fastEmbedR:::massive_cugraph_finish(
                graph, plan, "leiden", before, invalid, 1,
                "1GB"), "GPU peak is invalid")
            expect_error(fastEmbedR:::massive_cugraph_finish(
                graph, plan, "louvain", before, metadata, 1,
                "1GB"), "modularity disagrees")
            missing_score <- metadata
            missing_score$modularity_reported <- NA_real_
            expect_error(fastEmbedR:::massive_cugraph_finish(
                graph, plan, "leiden", before, missing_score, 1,
                "1GB"), "modularity disagrees")
            with_mocked_bindings(
                expect_error(fastEmbedR:::massive_cugraph_finish(
                    graph, plan, "leiden", before, metadata, 1,
                    "1GB"), "disconnected communities"),
                massive_graph_modularity = function(...)
                    list(modularity = 0.4,
                        connectivity_audited = TRUE,
                        disconnected_communities = 1L),
                .package = "fastEmbedR")
            result <- NULL
            expect_warning(result <-
                fastEmbedR:::massive_cugraph_finish(
                graph, plan, "leiden", before, metadata, 1,
                "1GB"), "verified graph score")
            expect_s3_class(result, "fastEmbedR_massive_leiden")
            expect_identical(result$membership_path, path)
            expect_equal(result$modularity_final, 0.4)
            expect_equal(result$modularity_reported, 0.8)
            expect_equal(result$modularity_gap, 0.4)
            expect_true(result$graph_audit$connectivity_audited)
            expect_identical(
                result$graph_audit$disconnected_communities, 0L)
            expect_equal(file.info(path)$size, 16)
            expect_equal(result$resources$measured_vram_peak_bytes,
                200)
            expect_equal(result$resources$measured_ram_peak_bytes, 2000)
            expect_equal(result$resources$ram_staging_estimate_bytes,
                1000)
            expect_identical(result$resources$measured_ram_scope,
                "launcher_and_workers")
            expect_match(paste(capture.output(print(result)),
                collapse = " "), "cuGraph on 2 GPUs")
        },
        massive_graph_modularity = function(...) list(
            modularity = 0.4, connectivity_audited = TRUE,
            disconnected_communities = 0L),
        massive_checkpoint_validated_graph_identity = function(graph,
                paths)
            list(paths = paths),
        .package = "fastEmbedR"
    )
})

test_that("cuGraph rejects changed graph data after fitting", {
    skip_on_os("windows")
    paths <- vapply(seq_len(3L), function(i) tempfile(), "")
    output <- tempfile()
    on.exit(unlink(c(paths, output)), add = TRUE)
    for (path in paths) writeBin(1L, path, size = 4L)
    graph <- list(offsets_path = paths[[1L]],
        indices_path = paths[[2L]], weights_path = paths[[3L]])
    graph$file_identity <-
        fastEmbedR:::massive_checkpoint_file_identity(paths)
    graph$validated_input_identity <-
        fastEmbedR:::massive_checkpoint_input_files_identity(paths)
    stamp <- file.info(paths[[3L]])$mtime
    Sys.sleep(1.1)
    with_mocked_bindings(
        expect_error(fastEmbedR:::massive_cugraph_cluster(
            graph, output, "leiden", c(0L, 1L), 2L,
            "1GB", 1, 10L, 1L),
            "Graph changed since validation"),
        massive_cugraph_plan = function(...) list(path = output),
        massive_cugraph_invoke = function(...) {
            writeBin(2L, paths[[3L]], size = 4L)
            Sys.setFileTime(paths[[3L]], stamp)
            list()
        },
        massive_cugraph_controls_check = function(...) NULL,
        .package = "fastEmbedR"
    )
    expect_identical(graph$file_identity,
        fastEmbedR:::massive_checkpoint_file_identity(paths))
})

test_that("cuGraph rejects graph changes during its modularity audit", {
    skip_on_os("windows")
    paths <- vapply(seq_len(3L), function(i) tempfile(), "")
    output <- tempfile()
    part <- paste0(output, ".part")
    on.exit(unlink(c(paths, output, part)), add = TRUE)
    for (path in paths) writeBin(1L, path, size = 4L)
    writeBin(as.integer(1:4), part, size = 4L)
    graph <- list(n_vertices = 4L, offsets_path = paths[[1L]],
        indices_path = paths[[2L]], weights_path = paths[[3L]])
    graph$validated_input_identity <-
        fastEmbedR:::massive_checkpoint_input_files_identity(paths)
    before <- fastEmbedR:::massive_cugraph_graph_identity(graph)
    plan <- list(path = output, devices = c(0L, 1L),
        uuids = c("GPU-A", "GPU-B"), chunk_rows = 2L)
    metadata <- list(rows = 4L, communities = 2L,
        modularity_reported = 0.8,
        devices = list(list(uuid = "GPU-A"), list(uuid = "GPU-B")))
    stamp <- file.info(paths[[3L]])$mtime
    Sys.sleep(1.1)
    with_mocked_bindings(
        expect_error(fastEmbedR:::massive_cugraph_finish(
            graph, plan, "louvain", before, metadata, 1, "1GB"),
            "Graph changed since validation"),
        massive_cugraph_memory = function(...) list(peak = 100),
        massive_graph_modularity = function(...) {
            writeBin(2L, paths[[3L]], size = 4L)
            Sys.setFileTime(paths[[3L]], stamp)
            list(modularity = 0.8)
        },
        .package = "fastEmbedR"
    )
    expect_false(file.exists(output))
})

test_that("cuGraph records the implementation and Python worker", {
    expected <- fastEmbedR:::massive_checkpoint_implementation_identity()
    worker <- system.file("python", "massive_cugraph_cluster.py",
        package = "fastEmbedR", mustWork = TRUE)
    with_mocked_bindings(
        {
            result <- fastEmbedR:::massive_cugraph_cluster(
                list(), tempfile(), "leiden", c(0L, 1L), 2L,
                "1GB", 1, 2L, 1L)
            expect_identical(result$implementation_identity,
                expected)
            expect_identical(result$python_worker_md5,
                unname(tools::md5sum(worker)))
        },
        massive_cugraph_plan = function(...) list(),
        massive_cugraph_graph_identity = function(...) list(),
        massive_cugraph_invoke = function(...) list(),
        massive_cugraph_controls_check = function(...) NULL,
        massive_cugraph_finish = function(...) list(),
        .package = "fastEmbedR"
    )
})

test_that("cuGraph rejects an implementation change during fitting", {
    calls <- 0L
    with_mocked_bindings(
        expect_error(fastEmbedR:::massive_cugraph_cluster(
            list(), tempfile(), "leiden", c(0L, 1L), 2L,
            "1GB", 1, 2L, 1L), "implementation changed"),
        massive_checkpoint_implementation_identity = function() {
            calls <<- calls + 1L
            calls
        },
        massive_cugraph_plan = function(...) list(),
        massive_cugraph_graph_identity = function(...) list(),
        massive_cugraph_invoke = function(...) list(),
        massive_cugraph_controls_check = function(...) NULL,
        massive_cugraph_finish = function(...) stop(
            "Changed implementation was accepted."),
        .package = "fastEmbedR"
    )
})

test_that("missing cuGraph runtime never falls back to CPU", {
    withr::local_envvar(FASTEMBEDR_CUGRAPH_PYTHON =
        "/nonexistent/fastembedr-python")
    expect_error(fastEmbedR:::massive_cugraph_invoke(
        list(), list(), "leiden", list()),
        "unavailable; no CPU fallback")
})

test_that("cuGraph passes large vertex counts as decimal integers", {
    withr::local_envvar(FASTEMBEDR_CUGRAPH_PYTHON = "/bin/echo")
    path <- tempfile()
    on.exit(unlink(paste0(path, c(".cugraph.out",
        ".cugraph.err"))))
    graph <- list(n_vertices = 1e8, offsets_path = "a",
        indices_path = "b", weights_path = "c")
    plan <- list(path = path, chunk_rows = 256L,
        uuids = c("GPU-A", "GPU-B"), vram_estimate = 1e9,
        limit = 4e9)
    expect_error(fastEmbedR:::massive_cugraph_invoke(graph,
        plan, "louvain", list(passes = 10L, resolution = 1,
            seed = 1L)), "did not write complete labels")
    command_args <- readLines(paste0(path, ".cugraph.out"))
    expect_match(paste(command_args, collapse = " "),
        "100000000")
    expect_false(any(grepl("1e\\+", command_args)))
})

test_that("cuGraph Louvain rejects a seed it cannot apply", {
    expect_error(fastEmbedR:::massive_cugraph_cluster(
        list(), tempfile(), "louvain", c(0L, 1L),
        2L, "1GB", 1, 10L, 17L),
        "no seed control")
})
