write_owner_offsets <- function(path, values) {
    bytes <- unlist(lapply(values, function(value)
        floor(value / 256^(0:7)) %% 256), use.names = FALSE)
    writeBin(as.raw(bytes), path)
}

test_that("GPU owner edge counts read 64-bit CSR offsets", {
    offsets <- tempfile(fileext = ".offsets.u64")
    on.exit(unlink(offsets))
    write_owner_offsets(offsets, c(0, 300, 700, 701, 1000))
    graph <- list(n_vertices = 4L, n_edges = 1000,
        offsets_path = offsets)
    expect_equal(fastEmbedR:::massive_graph_owner_edge_counts(
        graph, 0:1), c(700, 300))
    graph$n_edges <- 999
    expect_error(fastEmbedR:::massive_graph_owner_edge_counts(
        graph, 0:1), "cannot verify GPU edge ownership")
})

test_that("four GPU partitions cover every row and edge", {
    offsets <- tempfile(fileext = ".offsets.u64")
    on.exit(unlink(offsets))
    write_owner_offsets(offsets, 0:12)
    graph <- list(n_vertices = 12L, n_edges = 12,
        offsets_path = offsets)
    expect_equal(fastEmbedR:::massive_graph_owner_edge_counts(
        graph, 0:3), c(3, 3, 3, 3))
    tasks <- fastEmbedR:::massive_row_shards(
        tempfile(fileext = ".f32"), 12L, 2L, 0:3)
    expect_equal(lapply(tasks, `[[`, "row_range"), list(
        c(1, 3), c(4, 6), c(7, 9), c(10, 12)))
    expect_equal(vapply(tasks, `[[`, 0, "bytes"),
        c(24, 24, 24, 24))
})

test_that("t-SNE rejects an empty CUDA owner before native allocation", {
    select <- function(rows) with_mocked_bindings(
        fastEmbedR:::massive_tsne_devices(0:3, "cuda", rows),
        massive_validate_devices = function(devices, ...) devices,
        massive_nccl_available_cpp = function() TRUE,
        .package = "fastEmbedR")
    expect_error(select(9L), "needs rows on every device")
    expect_identical(select(12L), 0:3)
})

test_that("coordinated t-SNE accounts for every edge after resume", {
    offsets <- tempfile(fileext = ".offsets.u64")
    on.exit(unlink(offsets))
    write_owner_offsets(offsets, c(0, 2, 3, 3, 4, 6, 7))
    validate <- fastEmbedR:::massive_tsne_validate_cuda_fit
    plan <- list(early_iter = 2L, normal_iter = 3L,
        devices = 0:1)
    graph <- list(n_vertices = 6L, n_edges = 7L,
        offsets_path = offsets)
    saved <- list(iteration = 1L)
    fit <- list(
        backend_used = "native_cuda_fft_graph_coordinated_nccl",
        devices = 0:1, iterations = 5L,
        resumed_from_iteration = 1L,
        per_device_attraction_edges = c(12, 16),
        per_device_expected_edges = c(12, 16),
        stage_seconds = list(repulsion_enqueue = 1,
            graph_visit = 2, attraction_sync = 3,
            update_sync = 4, output_write = 5))
    expect_silent(validate(plan, graph, fit, saved))
    fit$stage_seconds$update_sync <- NA_real_
    expect_error(validate(plan, graph, fit, saved), "timing is incomplete")
    fit$stage_seconds$update_sync <- 4
    fit$per_device_attraction_edges <- c(16, 12)
    fit$per_device_expected_edges <- c(16, 12)
    expect_error(validate(plan, graph, fit, saved),
        "did not visit every graph edge")
    fit$per_device_expected_edges <- c(12, 16)
    fit$per_device_attraction_edges <- c(13, 15)
    expect_error(validate(plan, graph, fit, saved),
        "did not visit every graph edge")
    fit$per_device_attraction_edges <- c(12, 16)
    fit$per_device_attraction_edges[2L] <- 15
    expect_error(validate(plan, graph, fit, saved),
        "did not visit every graph edge")
    fit$per_device_attraction_edges[2L] <- 16
    fit$devices <- c(0L, 2L)
    expect_error(validate(plan, graph, fit, saved),
        "did not visit every graph edge")
})

test_that("coordinated UMAP accounts for each device's work", {
    offsets <- tempfile(fileext = ".offsets.u64")
    weights <- tempfile(fileext = ".weights.f32")
    on.exit(unlink(c(offsets, weights)))
    write_owner_offsets(offsets, c(0, 2, 4, 4, 5, 7, 8))
    writeBin(c(1, 0.8, 0.5, 0.1, 1, 0.6, 0.1, 0.1), weights,
        size = 4L, endian = "little")
    validate <- fastEmbedR:::massive_umap_validate_cuda_fit
    plan <- list(devices = 0:1, n_epochs = 3L)
    graph <- list(n_vertices = 6L, n_edges = 8L,
        offsets_path = offsets, weights_path = weights)
    saved <- list(edge_visits = 5, epoch = 1L)
    built <- list(backend_used = "native_cuda_umap_coordinated_nccl",
        devices = 0:1, n_vertices = 6, n_edges = 8,
        max_weight = 1, edge_visits = 15,
        per_device_edge_visits = c(6, 4),
        per_device_row_visits = c(6, 6),
        graph_visit_seconds = 1, device_step_seconds = 2,
        synchronization_seconds = 3,
        finalization_block_seconds = 4)
    expect_silent(validate(plan, graph, built, saved))
    built$synchronization_seconds <- -1
    expect_error(validate(plan, graph, built, saved),
        "timing is incomplete")
    built$synchronization_seconds <- 3
    built$backend_used <- "native_cuda_umap_stream"
    expect_error(validate(plan, graph, built, saved),
        "work accounting is inconsistent")
    built$backend_used <- "native_cuda_umap_coordinated_nccl"
    built$devices <- c(1L, 0L)
    expect_error(validate(plan, graph, built, saved),
        "work accounting is inconsistent")
    built$devices <- 0:1
    built$per_device_edge_visits <- c(7, 3)
    expect_error(validate(plan, graph, built, saved),
        "work accounting is inconsistent")
    built$per_device_edge_visits <- c(6, 4)
    built$per_device_row_visits[2L] <- 5
    expect_error(validate(plan, graph, built, saved),
        "work accounting is inconsistent")
    built$per_device_row_visits[2L] <- 6
    graph$n_vertices <- 7L
    write_owner_offsets(offsets, c(0, 2, 4, 4, 4, 5, 7, 8))
    built$n_vertices <- 7
    built$per_device_row_visits <- c(8, 6)
    expect_silent(validate(plan, graph, built, saved))
    built$per_device_row_visits <- c(7, 7)
    expect_error(validate(plan, graph, built, saved),
        "work accounting is inconsistent")
    graph$n_vertices <- 6L
    write_owner_offsets(offsets, c(0, 2, 4, 4, 5, 7, 8))
    built$n_vertices <- 6
    built$per_device_row_visits <- c(6, 6)
    built$per_device_edge_visits[2L] <- 3
    expect_error(validate(plan, graph, built, saved),
        "work accounting is inconsistent")
    built$per_device_edge_visits[2L] <- 4
    built$n_vertices <- 5
    expect_error(validate(plan, graph, built, saved),
        "work accounting is inconsistent")
    built$n_vertices <- 6
    built$max_weight <- 0.8
    expect_error(validate(plan, graph, built, saved),
        "Graph weights cannot verify")
})
