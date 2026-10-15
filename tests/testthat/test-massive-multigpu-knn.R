test_that("multi-GPU KNN worker detects input replacement", {
    directory <- tempfile()
    dir.create(directory)
    on.exit(unlink(directory, recursive = TRUE))
    input <- file.path(directory, "input.f32")
    reference <- file.path(directory, "reference.f32")
    writeBin(as.numeric(seq_len(12L)), input, size = 4L)
    writeBin(as.numeric(seq_len(12L)), reference, size = 4L)
    source <- massive_matrix(input, 4L, 3L)
    target <- massive_matrix(reference, 4L, 3L)
    task <- list(device = 0L, row_range = c(1L, 4L),
        output = file.path(directory, "shard"), status = "new")
    with_mocked_bindings(
        massive_cuda_worker_identity = function(...) NULL,
        massive_cuda_select_cpp = function(...) NULL,
        massive_landmark_knn = function(...) {
            con <- file(reference, "ab")
            writeBin(0, con, size = 4L)
            close(con)
            list(resources = list(gpu_device = 0L))
        },
        expect_error(fastEmbedR:::massive_knn_device_worker(
            task, list(x = source, reference = target, k = 2L),
            "mock", "mock"), "input changed during worker search"),
        .package = "fastEmbedR")
})

test_that("interrupted multi-GPU IVF shards restart from row one", {
    root <- tempfile("ivf_restart_")
    dir.create(root)
    on.exit(unlink(root, recursive = TRUE), add = TRUE)
    task <- list(output = file.path(root, "gpu0"), bytes = 16)
    writeBin(1L, paste0(task$output, ".indices.u32.part"),
        size = 4L)
    writeBin(1, paste0(task$output, ".distances.f32.part"),
        size = 4L)
    status <- fastEmbedR:::massive_knn_device_status
    restarted <- status(task, NULL, NULL, 2L, "sharded_ivf", TRUE)
    expect_identical(restarted$status, "restart")
    expect_equal(restarted$remaining, 32)
    expect_error(status(task, NULL, NULL, 2L,
        "sharded_exact", TRUE), "incomplete checkpoint")
})

test_that("coordinated IVF checkpoints complete GPU shards only", {
    root <- tempfile("ivf_checkpoint_")
    dir.create(root)
    on.exit(unlink(root, recursive = TRUE), add = TRUE)
    source_path <- file.path(root, "source.f32")
    writeBin(as.numeric(1:8), source_path, size = 4L)
    source <- massive_matrix(source_path, 4L, 2L)
    task <- list(device = 0L, row_range = c(1L, 4L),
        output = file.path(root, "gpu0"), status = "restart")
    observed <- NULL
    search <- function(...) {
        observed <<- list(...)
        ids <- paste0(observed$output, ".indices.u32")
        distances <- paste0(observed$output, ".distances.f32")
        writeBin(1L, ids, size = 4L)
        writeBin(1, distances, size = 4L)
        list(indices_path = ids, distances_path = distances,
            resources = list(gpu_device = 0L))
    }
    with_mocked_bindings(
        massive_cuda_worker_identity = function(...) NULL,
        massive_cuda_select_cpp = function(...) NULL,
        massive_landmark_knn = search,
        {
            result <- fastEmbedR:::massive_knn_device_worker(
                task, list(x = source, reference = source, k = 2L,
                    method = "sharded_ivf", checkpoint = TRUE),
                "mock", "mock")
            expect_identical(observed$method, "sharded_ivf")
            expect_identical(observed$checkpoint, FALSE)
            expect_identical(observed$resume, FALSE)
            expect_true(result$restarted)
            saved <- readRDS(paste0(task$output, ".result.rds"))
            expect_identical(saved$completed_output_identity,
                result$completed_output_identity)
        }, .package = "fastEmbedR")
})

test_that("multi-GPU IVF resume keeps only verified complete shards", {
    root <- tempfile("ivf_resume_")
    dir.create(root)
    on.exit(unlink(root, recursive = TRUE), add = TRUE)
    input <- file.path(root, "input.f32")
    writeBin(as.numeric(1:8), input, size = 4L)
    source <- massive_matrix(input, 4L, 2L)
    prefix <- file.path(root, "merged")
    paths <- fastEmbedR:::massive_knn_paths(prefix, 64)
    setup <- list(paths = paths, reference = source)
    devices <- c(0L, 1L)
    tasks <- fastEmbedR:::massive_knn_device_tasks(
        source, 2L, prefix, devices)
    controls <- list(x = source, reference = source,
        k = 2L, method = "sharded_ivf")
    checkpoint <- fastEmbedR:::massive_knn_devices_checkpoint
    status <- fastEmbedR:::massive_knn_device_status
    with_mocked_bindings(
        massive_cuda_device_uuids = function(...) c("GPU-A", "GPU-B"),
        {
            checkpoint(source, setup, tasks, devices, controls, FALSE)
            complete <- tasks[[1L]]
            finished <- paste0(complete$output,
                c(".indices.u32", ".distances.f32"))
            writeBin(rep(1L, 4L), finished[[1L]], size = 4L)
            writeBin(rep(0, 4L), finished[[2L]], size = 4L)
            view <- fastEmbedR:::massive_matrix_rows(source,
                complete$row_range[[1L]],
                diff(complete$row_range) + 1)
            result <- list(backend = "cuda", method = "sharded_ivf",
                nrow = view$nrow, ncol = 2L, n_reference = source$nrow,
                indices_path = finished[[1L]],
                distances_path = finished[[2L]],
                resources = list(gpu_device = 0L),
                source_identity = fastEmbedR:::
                    massive_checkpoint_source_identity(view),
                reference_identity = fastEmbedR:::
                    massive_checkpoint_source_identity(source),
                implementation_identity = fastEmbedR:::
                    massive_checkpoint_implementation_identity(),
                output_identity = fastEmbedR:::
                    massive_checkpoint_file_identity(finished),
                completed_output_identity = fastEmbedR:::
                    massive_checkpoint_input_files_identity(finished))
            saveRDS(result, paste0(complete$output, ".result.rds"))
            interrupted <- paste0(tasks[[2L]]$output,
                c(".indices.u32.part", ".distances.f32.part"))
            for (path in interrupted) writeBin(1L, path, size = 4L)
            found <- lapply(tasks, status, x = source,
                reference = source, k = 2L,
                method = "sharded_ivf", resume = TRUE)
            expect_identical(vapply(found, `[[`, "", "status"),
                c("complete", "restart"))
            checkpoint(source, setup, found, devices, controls, TRUE)
            expect_true(all(file.exists(finished)))
            expect_false(any(file.exists(interrupted)))
            stream <- file(finished[[1L]], "ab")
            writeBin(2L, stream, size = 4L)
            close(stream)
            expect_error(status(complete, source, source, 2L,
                "sharded_ivf", TRUE), "inconsistent")
        }, .package = "fastEmbedR")
})

test_that("multi-GPU full KNN counts its two live payload phases", {
    directory <- tempfile()
    dir.create(directory)
    on.exit(unlink(directory, recursive = TRUE))
    paths <- file.path(directory, paste0("part", seq_len(8L)))
    for (i in seq_along(paths))
        writeBin(as.integer(seq_len(i)), paths[[i]], size = 4L)
    graph <- list(indices_path = paths[[7L]],
        distances_path = paths[[8L]])
    candidates <- list(indices_path = paths[[1L]],
        distances_path = paths[[2L]],
        peak_payload_bytes = sum(file.info(paths[1:6])$size))
    expected <- max(candidates$peak_payload_bytes,
        sum(file.info(paths[c(1:2, 7:8)])$size))
    measure <- fastEmbedR:::massive_full_knn_payload_bytes
    unlink(paths[3:6])
    expect_equal(measure(graph, candidates, expected), expected)
    expect_error(measure(graph, candidates, expected - 4),
        "disk budget")
    unlink(paths[[7L]])
    expect_error(measure(graph, candidates, expected),
        "disk budget")
})

test_that("multi-GPU KNN records actual query ownership", {
    directory <- tempfile()
    dir.create(directory)
    on.exit(unlink(directory, recursive = TRUE))
    input <- file.path(directory, "input.f32")
    writeBin(as.numeric(seq_len(10L)), input, size = 4L)
    source <- massive_matrix(input, 5L, 2L)
    paths <- c(indices = file.path(directory, "ids.u32"),
        distances = file.path(directory, "dist.f32"))
    for (path in paths) writeBin(1L, path, size = 4L)
    setup <- list(paths = paths, reference = source,
        resources = list(output_bytes = 40))
    tasks <- list(list(row_range = c(1, 2)),
        list(row_range = c(3, 5)))
    result <- list(reference_identity = list(test = TRUE),
        indices_path = paths[[1L]], distances_path = paths[[2L]],
        resources = list(peak_ram_bytes = 10, chunk_rows = 1L),
        pilot_recall = 1, pilot_id_recall = 1,
        pilot_rows = 1, nlist = 1, nprobe = 1,
        build_seconds = 1, query_seconds = 1)
    with_mocked_bindings(
        {
            joined <- fastEmbedR:::massive_knn_devices_result(
                list(result, result), setup, source, c(0L, 1L),
                0.2, tasks)
            expect_equal(lapply(joined$per_device, `[[`,
                "query_row_range"), lapply(tasks, `[[`, "row_range"))
            expect_equal(joined$resources$minimum_reference_read_bytes,
                80)
            expect_identical(fastEmbedR:::massive_knn_holdout_rows(
                5L, joined$per_device, 2L), c(2L, 5L))
        },
        massive_cuda_device_uuids = function(...) c("GPU-A", "GPU-B"),
        .package = "fastEmbedR"
    )
})

test_that("multi-GPU KNN coordinator detects source changes", {
    directory <- tempfile()
    dir.create(directory)
    on.exit(unlink(directory, recursive = TRUE))
    input <- file.path(directory, "input.f32")
    writeBin(as.numeric(seq_len(18L)), input, size = 4L)
    source <- massive_matrix(input, 6L, 3L)
    paths <- fastEmbedR:::massive_knn_paths(
        file.path(directory, "result"), 6L * 2L * 8L)
    setup <- list(paths = paths, reference = source, k = 2L,
        method = "exact", resources = list(output_bytes = 96))
    with_mocked_bindings(
        massive_cuda_parallel_tasks = function(...) {
            con <- file(input, "ab")
            writeBin(0, con, size = 4L)
            close(con)
            list()
        },
        expect_error(fastEmbedR:::massive_knn_devices_execute(
            source, setup, c(0L, 1L),
            list(x = source, reference = source, k = 2L),
            FALSE, FALSE), "input changed during parallel search"),
        .package = "fastEmbedR")
})

test_that("multi-GPU full KNN removes self with duplicate-distance ties", {
    directory <- tempfile()
    dir.create(directory)
    on.exit(unlink(directory, recursive = TRUE))
    input <- file.path(directory, "input.f32")
    values <- matrix(c(0, 0, 0, 0, 1, 0, 1, 0), ncol = 2L,
        byrow = TRUE)
    writeBin(as.numeric(t(values)), input, size = 4L)
    source <- massive_matrix(input, 4L, 2L)
    prefix <- file.path(directory, "candidates")
    ids <- matrix(c(2L, 1L, 3L, 1L, 3L, 4L,
        3L, 4L, 1L, 3L, 4L, 2L), nrow = 4L, byrow = TRUE)
    distances <- matrix(c(0, 0, 1, 0, 1, 1,
        0, 0, 1, 0, 0, 1), nrow = 4L, byrow = TRUE)
    indices_path <- paste0(prefix, ".indices.u32")
    distances_path <- paste0(prefix, ".distances.f32")
    writeBin(as.integer(t(ids)), indices_path, size = 4L)
    writeBin(as.numeric(t(distances)), distances_path, size = 4L)
    candidates <- structure(list(indices_path = indices_path,
        distances_path = distances_path, nrow = 4L, ncol = 3L,
        n_reference = 4L, source_identity =
            fastEmbedR:::massive_checkpoint_source_identity(source),
        reference_identity =
            fastEmbedR:::massive_checkpoint_source_identity(source)),
        class = "fastEmbedR_massive_knn")
    graph <- fastEmbedR:::massive_full_knn_strip_self(candidates,
        source, 2L, file.path(directory, "merged"), 2L)
    observed <- massive_read_graph_edges(graph, 1L, 4L)
    expect_identical(observed$to,
        c(2L, 3L, 1L, 3L, 4L, 1L, 3L, 2L))
    expect_true(all(observed$to != rep(1:4, each = 2L)))
    expect_equal(observed$distance, rep(c(0, 1), 4L))
})

test_that("multi-GPU KNN rejects corrupted completed merge staging", {
    directory <- tempfile()
    dir.create(directory)
    on.exit(unlink(directory, recursive = TRUE))
    shards <- file.path(directory, c("gpu0.u32", "gpu1.u32"))
    output <- file.path(directory, "merged.u32")
    staged <- paste0(output, ".merge")
    writeBin(as.integer(c(1L, 2L)), shards[[1L]], size = 4L)
    writeBin(as.integer(c(3L, 4L)), shards[[2L]], size = 4L)
    writeBin(as.integer(c(1L, 2L, 3L, 9L)), staged, size = 4L)
    expect_error(fastEmbedR:::massive_knn_merge_one(
        shards, output, c(8, 8), TRUE), "differs from its shards")
    expect_false(file.exists(output))
    writeBin(as.integer(c(1L, 2L, 3L, 4L)), staged, size = 4L)
    expect_silent(fastEmbedR:::massive_knn_merge_one(
        shards, output, c(8, 8), TRUE))
    expect_false(file.exists(staged))
    expect_identical(readBin(output, integer(), 4L, size = 4L),
        as.integer(1:4))
})

test_that("multi-GPU KNN rejects a corrupted resumed merge prefix", {
    directory <- tempfile()
    dir.create(directory)
    on.exit(unlink(directory, recursive = TRUE))
    shards <- file.path(directory, c("gpu0.u32", "gpu1.u32"))
    output <- file.path(directory, "merged.u32")
    block <- 1048576L
    writeBin(seq_len(block), shards[[1L]], size = 4L)
    writeBin(as.integer(c(7L, 8L)), shards[[2L]], size = 4L)
    part <- paste0(output, ".merge.part")
    expect_true(file.copy(shards[[1L]], part))
    stream <- file(part, "r+b")
    writeBin(0L, stream, size = 4L)
    close(stream)
    expect_error(fastEmbedR:::massive_knn_merge_one(
        shards, output, c(4 * block, 8), TRUE),
        "differs from its shards")
    expect_false(file.exists(output))
})

test_that("multi-GPU KNN reports exact recall for every device", {
    graph <- list(audit_sample_rows = c(5L, 15L, 17L),
        audit_row_recall = c(1, 0.9, 0.8),
        per_device = list(
            list(query_row_range = c(1L, 10L),
                resources = list(gpu_device = 0L)),
            list(query_row_range = c(11L, 20L),
                resources = list(gpu_device = 1L))))
    audit <- fastEmbedR:::massive_full_knn_device_audit
    result <- audit(graph, c(0L, 1L))$per_device_audit
    expect_identical(lapply(result, `[[`, "sampled_rows"),
        list(5L, c(15L, 17L)))
    expect_equal(vapply(result, `[[`, 0, "recall_mean"), c(1, 0.85))
    expect_identical(vapply(result, `[[`, 0L, "rows_below_target"),
        c(0L, 2L))
    graph$audit_sample_rows <- c(5L, 15L, 21L)
    expect_error(audit(graph, c(0L, 1L)), "ownership is invalid")
    graph$audit_sample_rows <- 5L
    graph$audit_row_recall <- 1
    expect_error(audit(graph, c(0L, 1L)), "omitted a device")
    graph$audit_sample_rows <- c(5L, 15L, 15L)
    graph$audit_row_recall <- c(1, 0.9, 0.8)
    expect_error(audit(graph, c(0L, 1L)), "audit is incomplete")
    graph$audit_sample_rows <- c(5L, 15L, 17L)
    graph$audit_row_recall[[3L]] <- 1.1
    expect_error(audit(graph, c(0L, 1L)), "audit is incomplete")
    graph$audit_row_recall[[3L]] <- NaN
    expect_error(audit(graph, c(0L, 1L)), "audit is incomplete")
    graph$audit_row_recall[[3L]] <- 0.8
    graph$per_device[[2L]]$resources$gpu_device <- 2L
    expect_error(audit(graph, c(0L, 1L)), "mapping changed")
})

test_that("saved full KNN candidates reject changed device ownership", {
    directory <- tempfile()
    dir.create(directory)
    on.exit(unlink(directory, recursive = TRUE))
    input <- file.path(directory, "input.f32")
    writeBin(as.numeric(seq_len(12L)), input, size = 4L)
    source <- massive_matrix(input, 6L, 2L)
    prefix <- file.path(directory, "candidate")
    paths <- paste0(prefix, c(".indices.u32", ".distances.f32"))
    writeBin(rep(1L, 18L), paths[[1L]], size = 4L)
    writeBin(rep(1, 18L), paths[[2L]], size = 4L)
    devices <- c(0L, 1L)
    tasks <- fastEmbedR:::massive_knn_device_tasks(
        source, 3L, prefix, devices)
    shards <- unlist(lapply(tasks, function(task)
        paste0(task$output, c(".indices.u32", ".distances.f32"))),
        use.names = FALSE)
    identity <- fastEmbedR:::massive_checkpoint_source_identity(source)
    saved <- structure(list(indices_path = paths[[1L]],
        distances_path = paths[[2L]], nrow = 6L, ncol = 3L,
        n_reference = 6L, gpu_devices = devices,
        gpu_identity = c("GPU-A", "GPU-B"),
        source_identity = identity, reference_identity = identity,
        shard_paths = lapply(tasks, function(task)
            list(indices_path = paste0(task$output, ".indices.u32"),
                distances_path = paste0(task$output, ".distances.f32"))),
        peak_payload_bytes = 16 * source$nrow * 3L,
        output_identity =
            fastEmbedR:::massive_checkpoint_file_identity(paths),
        per_device = lapply(tasks, function(task)
            list(query_row_range = task$row_range,
                resources = list(gpu_device = task$device,
                    chunk_rows = 2L)))),
        class = "fastEmbedR_massive_knn")
    check <- fastEmbedR:::massive_full_knn_candidate_check
    with_mocked_bindings(
        massive_cuda_device_uuids = function(...) c("GPU-A", "GPU-B"),
        expect_silent(check(saved, source, 2L, prefix,
            devices, shards)),
        .package = "fastEmbedR")
    saved$per_device[[1L]]$query_row_range <- c(1, 4)
    with_mocked_bindings(
        massive_cuda_device_uuids = function(...) c("GPU-A", "GPU-B"),
        expect_error(check(saved, source, 2L, prefix,
            devices, shards), "candidates changed"),
        .package = "fastEmbedR")
    saved$per_device[[1L]]$query_row_range <- tasks[[1L]]$row_range
    saved$per_device[[2L]]$resources$chunk_rows <- 1.5
    with_mocked_bindings(
        massive_cuda_device_uuids = function(...) c("GPU-A", "GPU-B"),
        expect_error(check(saved, source, 2L, prefix,
            devices, shards), "candidates changed"),
        .package = "fastEmbedR")
})
