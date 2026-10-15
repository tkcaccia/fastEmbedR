test_that("CUDA checkpoint IDs require physical UUIDs", {
    visible <- function(id, uuid, process_visible = TRUE) list(
        backend = "cuda", process_visible = process_visible,
        logical_id = id, uuid = uuid)
    manifest <- list(devices = list(
        visible(0L, "GPU-A"), visible(1L, "GPU-B"),
        visible(NA_integer_, "GPU-C", FALSE)))
    identify <- fastEmbedR:::massive_cuda_manifest_uuids
    expect_identical(identify(manifest, c(1L, 0L)),
        c("GPU-B", "GPU-A"))
    expect_error(identify(manifest, 2L), "identify")
    manifest$devices[[2L]]$uuid <- NA_character_
    expect_error(identify(manifest, c(0L, 1L)), "identify")
    manifest$devices[[2L]]$uuid <- "GPU-A"
    expect_error(identify(manifest, c(0L, 1L)), "identify")
})

test_that("single-GPU full KNN records physical device identity", {
    with_mocked_bindings(
        massive_validate_devices = function(...) 1L,
        massive_cuda_memory_cpp = function(...) list(device = 0L),
        massive_cuda_select_cpp = function(...) NULL,
        massive_sharded_graph = function(...) list(method = "ivf_sharded"),
        massive_cuda_device_uuids = function(...) "GPU-B",
        {
            graph <- fastEmbedR:::massive_full_knn_device_request(
                list(nrow = 8L), 2L, "unused", "cuda", 1L,
                2L, 4L, "256MB", FALSE, FALSE, 0L,
                100L, 1L, "ivf_sharded")
            expect_identical(graph$gpu_devices, 1L)
            expect_identical(graph$gpu_identity, "GPU-B")
        }, .package = "fastEmbedR")
})

test_that("optimizer checkpoints bind physical CUDA device identity", {
    directory <- tempfile()
    dir.create(directory)
    on.exit(unlink(directory, recursive = TRUE))
    graph_paths <- file.path(directory, c("offsets", "indices", "weights"))
    for (path in graph_paths) writeBin(1, path, size = 4L)
    init_path <- file.path(directory, "init.f32")
    writeBin(rep(0.1, 8L), init_path, size = 4L)
    init <- massive_matrix(init_path, 4L, 2L)
    graph <- structure(list(n_vertices = 4L, n_edges = 8L,
        max_degree = 2L, offsets_path = graph_paths[[1L]],
        indices_path = graph_paths[[2L]],
        weights_path = graph_paths[[3L]], symmetrized = TRUE),
        class = "fastEmbedR_massive_graph")
    selected <- c("GPU-A", "GPU-B")
    with_mocked_bindings({
        umap <- function() fastEmbedR:::massive_umap_optimize_plan(
            graph, init, file.path(directory, "umap.f32"),
            2L, 0L, 0.1, 0.1, 1, 1L, 2L, "256MB", "cuda", 1L,
            devices = c(0L, 1L))
        graph$weight_method <- "umap_fuzzy_union"
        first_umap <- umap()
        graph$weight_method <- "tsne_compact_affinity"
        tsne <- function() fastEmbedR:::massive_tsne_optimize_plan(
            graph, init, file.path(directory, "tsne.f32"),
            2L, 2L, 12, 1, 1, 0.8, 0.8, 0.01,
            Inf, "1GB", "cuda", 1L, c(0L, 1L))
        first_tsne <- tsne()
        selected <- rev(selected)
        graph$weight_method <- "umap_fuzzy_union"
        second_umap <- umap()
        graph$weight_method <- "tsne_compact_affinity"
        second_tsne <- tsne()
        for (plans in list(list(first_umap, second_umap),
                list(first_tsne, second_tsne))) {
            expect_identical(plans[[1L]]$devices, plans[[2L]]$devices)
            expect_identical(plans[[1L]]$gpu_identity,
                c("GPU-A", "GPU-B"))
            expect_identical(plans[[2L]]$gpu_identity,
                c("GPU-B", "GPU-A"))
            expect_false(identical(
                fastEmbedR:::massive_checkpoint_signature(
                    plans[[1L]]$parameters),
                fastEmbedR:::massive_checkpoint_signature(
                    plans[[2L]]$parameters)))
        }
        signatures <- character()
        with_mocked_bindings({
            umap_signatures <- c(
                fastEmbedR:::massive_umap_checkpoint_prepare(
                    first_umap, TRUE, FALSE, 1L),
                fastEmbedR:::massive_umap_checkpoint_prepare(
                    second_umap, TRUE, FALSE, 1L))
            expect_false(identical(umap_signatures[[1L]],
                umap_signatures[[2L]]))
            fastEmbedR:::massive_tsne_checkpoint_prepare(
                first_tsne, TRUE, TRUE, 1L)
            fastEmbedR:::massive_tsne_checkpoint_prepare(
                second_tsne, TRUE, TRUE, 1L)
            expect_false(identical(signatures[[1L]],
                signatures[[2L]]))
        },
        massive_checkpoint_implementation_identity = function() "fixed",
        massive_disk_available_cpp = function(...) 1e9,
        massive_umap_checkpoint_state = function(plan, sidecar,
                signature, ...) signature,
        massive_tsne_checkpoint_load = function(plan, sidecar,
                signature) {
            signatures <<- c(signatures, signature)
            list(iteration = 0L, state_path = "", elapsed_seconds = 0)
        }, .package = "fastEmbedR")
        sidecar <- file.path(directory, "remapped.checkpoint.rds")
        saveRDS(list(signature = umap_signatures[[1L]]), sidecar)
        expect_error(fastEmbedR:::massive_umap_checkpoint_load(
            second_umap, sidecar, umap_signatures[[2L]]),
            "does not match")
        saveRDS(list(signature = signatures[[1L]]), sidecar)
        expect_error(fastEmbedR:::massive_tsne_checkpoint_load(
            second_tsne, sidecar, signatures[[2L]]),
            "does not match")
    },
    massive_umap_validate_inputs = function(...) NULL,
    massive_umap_plan_storage = function(...) c(0L, 1L),
    massive_umap_cuda_capacity = function(...) 16L,
    massive_tsne_devices = function(...) c(0L, 1L),
    massive_tsne_resource_plan = function(...) list(
        state_bytes = 100, peak_ram = 0, peak_vram = 0,
        edge_capacity = 16L),
    embedding_cuda_available_cpp = function() TRUE,
    massive_checkpoint_validated_graph_identity = function(...) NULL,
    massive_cuda_device_uuids = function(devices) selected,
    .package = "fastEmbedR")
})

test_that("multi-GPU KNN and projection signatures reject a remap", {
    directory <- tempfile()
    dir.create(directory)
    on.exit(unlink(directory, recursive = TRUE))
    input <- file.path(directory, "input.f32")
    writeBin(rep(0.1, 16L), input, size = 4L)
    source <- massive_matrix(input, 8L, 2L)
    devices <- c(0L, 1L)
    selected <- c("GPU-A", "GPU-B")
    setup <- list(reference = source, paths = c(
        indices = file.path(directory, "knn.indices.u32"),
        distances = file.path(directory, "knn.distances.f32")))
    with_mocked_bindings({
        arguments <- list(x = source, reference = source,
            checkpoint = TRUE, resume = FALSE)
        fastEmbedR:::massive_knn_devices_checkpoint(
            source, setup, list(), devices, arguments, FALSE)
        fastEmbedR:::massive_full_knn_devices_checkpoint(
            source, 2L, file.path(directory, "full"), "exact",
            devices, 2L, 4L, "256MB", 0L, 1L, FALSE)
        projection <- list(graph = list(nrow = 8L),
            fit = list(layout = matrix(0, 2L, 2L)),
            selection = NULL, output = file.path(directory, "proj.f32"))
        tasks <- list(list(device = 0L, row_range = c(1L, 4L),
            bytes = 32), list(device = 1L,
            row_range = c(5L, 8L), bytes = 32))
        first <- fastEmbedR:::massive_multigpu_signature(
            projection, devices, tasks, projection$output)
        selected <- rev(selected)
        expect_error(fastEmbedR:::massive_knn_devices_checkpoint(
            source, setup, list(), devices, arguments, TRUE),
            "does not match")
        expect_error(fastEmbedR:::massive_full_knn_devices_checkpoint(
            source, 2L, file.path(directory, "full"), "exact",
            devices, 2L, 4L, "256MB", 0L, 1L, TRUE),
            "does not match")
        second <- fastEmbedR:::massive_multigpu_signature(
            projection, devices, tasks, projection$output)
        expect_false(identical(first, second))
    },
    massive_cuda_device_uuids = function(devices) selected,
    massive_checkpoint_implementation_identity = function() "fixed",
    massive_projection_signature = function(fit, graph, resources,
            settings, ...) settings$gpu_identity,
    massive_reference_indices = function(...) integer(),
    .package = "fastEmbedR")
})
