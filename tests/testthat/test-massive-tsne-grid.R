test_that("CUDA t-SNE checkpoint identity includes the FFT grid", {
    graph <- list(n_vertices = 1000L, n_edges = 2000L,
        offsets_path = "offsets", indices_path = "indices",
        weights_path = "weights")
    init <- list(nrow = 1000L, ncol = 2L, path = "init.f32")
    output <- tempfile(fileext = ".f32")
    make_plan <- function() fastEmbedR:::massive_tsne_optimize_plan(
        graph, init, output, 2L, 3L, 12, 1, 1, 0.8, 0.8,
        0.01, Inf, "1GB", "cuda", 1L, c(0L, 1L))
    with_mocked_bindings(
        {
            withr::local_envvar(FASTEMBEDR_TSNE_FFT_GRID = "128")
            first <- make_plan()
            expect_identical(first$fft_grid_size, 128L)
            expect_identical(first$parameters$fft_grid_size, 128L)
            Sys.setenv(FASTEMBEDR_TSNE_FFT_GRID = "256")
            second <- make_plan()
            expect_identical(second$parameters$fft_grid_size, 256L)
            expect_false(identical(first$parameters,
                second$parameters))
        },
        massive_tsne_validate_graph_init = function(...) NULL,
        massive_tsne_devices = function(...) c(0L, 1L),
        massive_cuda_device_uuids = function(...) c("GPU-A", "GPU-B"),
        massive_auto_cuda_free = function(...) 1e12,
        massive_disk_available_cpp = function(...) 1e12,
        massive_checkpoint_validated_graph_identity = function(...) NULL,
        massive_checkpoint_input_files_identity = function(...)
            list(source = "unchanged"),
        .package = "fastEmbedR"
    )
})

test_that("CUDA t-SNE uses and verifies its planned FFT grid", {
    graph <- list(n_vertices = 17L, offsets_path = "offsets",
        indices_path = "indices", weights_path = "weights",
        n_edges = 34L, access = "stream")
    init <- list(path = "init.f32", ncol = 2L)
    plan <- list(output = "output.f32", source_paths = "source",
        source_identity = list(source = "unchanged"),
        early_iter = 1L, normal_iter = 1L,
        learning_rate = 1, learning_rate_auto = FALSE,
        edge_capacity = 64L, devices = c(0L, 1L),
        fft_grid_size = 128L)
    saved <- list(iteration = 0L, state_path = "", every = 0L,
        progress = NULL, sidecar = NULL)
    withr::local_envvar(FASTEMBEDR_TSNE_FFT_GRID = "256")
    with_mocked_bindings(
        {
            result <- massive_tsne_optimize(graph, init,
                "output.f32", backend = "cuda")
            expect_identical(result$fft_grid_size, 128L)
            expect_identical(Sys.getenv("FASTEMBEDR_TSNE_FFT_GRID"),
                "256")
        },
        massive_tsne_optimize_plan = function(...) plan,
        massive_tsne_checkpoint_prepare = function(...) saved,
        massive_tsne_optimize_cuda_cpp = function(...) {
            expect_identical(Sys.getenv("FASTEMBEDR_TSNE_FFT_GRID"),
                "128")
            list(fft_grid_size = 128L,
                backend_used =
                    "native_cuda_fft_graph_coordinated_nccl",
                devices = c(0L, 1L), iterations = 2L,
                resumed_from_iteration = 0L,
                per_device_attraction_edges = c(36, 32),
                per_device_expected_edges = c(36, 32),
                stage_seconds = list(repulsion_enqueue = 0.01,
                    graph_visit = 0.02, attraction_sync = 0.03,
                    update_sync = 0.04, output_write = 0.01))
        },
        massive_graph_owner_edge_counts = function(...)
            c(18, 16),
        massive_checkpoint_input_files_identity = function(...)
            list(source = "unchanged"),
        massive_tsne_result = function(plan, graph, init, fit, saved)
            fit,
        .package = "fastEmbedR"
    )
    with_mocked_bindings(
        expect_error(massive_tsne_optimize(graph, init,
            "output.f32", backend = "cuda"), "planned FFT grid"),
        massive_tsne_optimize_plan = function(...) plan,
        massive_tsne_checkpoint_prepare = function(...) saved,
        massive_tsne_optimize_cuda_cpp = function(...)
            list(fft_grid_size = 256L),
        .package = "fastEmbedR"
    )
    expect_identical(Sys.getenv("FASTEMBEDR_TSNE_FFT_GRID"), "256")
})

test_that("CUDA t-SNE refuses a checkpoint made with another grid", {
    input <- tempfile()
    output <- tempfile(fileext = ".f32")
    writeBin(as.raw(1), input)
    snapshot <- paste0(output, ".iter_1.state.f32")
    sidecar <- paste0(output, ".checkpoint.rds")
    on.exit(unlink(c(input, snapshot, sidecar)))
    plan <- list(output = output, source_paths = input,
        source_identity = fastEmbedR:::
            massive_checkpoint_input_files_identity(input),
        n_vertices = 2L, dimensions = 2L, state_bytes = 48,
        early_iter = 1L, normal_iter = 1L,
        parameters = list(fft_grid_size = 128L))
    saved <- fastEmbedR:::massive_tsne_checkpoint_prepare(
        plan, TRUE, FALSE, 1L)
    writeBin(rep(0, 12L), snapshot, size = 4L)
    saved$progress(1L, snapshot, 0.1)
    changed <- plan
    changed$parameters$fft_grid_size <- 256L
    expect_error(fastEmbedR:::massive_tsne_checkpoint_prepare(
        changed, TRUE, TRUE, 1L), "does not match")
})
