test_that("all-row landmark fits forward coordinated CUDA devices", {
    x <- matrix(seq_len(24), nrow = 8L)
    selection <- list(indices = seq_len(8L), query_indices = integer())
    state <- list(x = x, n = 8L, metric = "euclidean",
        selection = selection)
    tsne_request <- list(perplexity = 2, n_components = 2L,
        seed = 1L, backend = "cuda", keep_knn = FALSE,
        verbose = FALSE, n_threads = 1L, affinity_k = NULL,
        devices = c(0L, 1L), extra = list())
    with_mocked_bindings({
        expect_error(run_full_landmark_tsne(state, tsne_request),
            "t-SNE devices forwarded")
        expect_error(run_full_landmark_umap(state, 1L, FALSE,
            FALSE, c(0L, 1L)), "UMAP devices forwarded")
    },
    tsne = function(..., devices) {
        expect_identical(devices, c(0L, 1L))
        stop("t-SNE devices forwarded")
    },
    umap = function(..., devices) {
        expect_identical(devices, c(0L, 1L))
        stop("UMAP devices forwarded")
    }, .package = "fastEmbedR")
})

test_that("partial resident UMAP forwards devices to reference fit", {
    x <- matrix(seq_len(24), nrow = 8L)
    selection <- list(indices = 1:4, query_indices = 5:8)
    state <- list(x = x, n = 8L, metric = "euclidean",
        selection = selection, all_landmarks = FALSE,
        backend = "cuda",
        partition = list(query = x[5:8, , drop = FALSE]))
    with_mocked_bindings({
        expect_error(run_landmark_umap(x, 4L, 3L, 2L,
            FALSE, NULL, "euclidean", 1L, "cuda", NULL,
            1L, FALSE, "fuzzy", FALSE, devices = c(0L, 1L)),
            "reference devices forwarded")
    },
    prepare_landmark_umap = function(...) state,
    massive_validate_devices = function(devices, ...) devices,
    massive_nccl_available_cpp = function() TRUE,
    massive_cuda_device_uuids = function(...) c("GPU-A", "GPU-B"),
    massive_cuda_memory_cpp = function() list(device = 0L),
    massive_cuda_select_cpp = function(...) NULL,
    fit_landmark_umap_reference = function(...) {
        expect_identical(tail(list(...), 1L)[[1L]], c(0L, 1L))
        stop("reference devices forwarded")
    },
    .package = "fastEmbedR")
})

test_that("partial CUDA UMAP assembles query shards in original order", {
    x <- matrix(seq_len(24) / 10, nrow = 8L)
    state <- list(
        partition = list(landmarks = x[1:4, , drop = FALSE],
            query = x[5:8, , drop = FALSE]),
        selection = list(indices = 1:4, query_indices = 5:8),
        n = 8L, gpu_identity = c("GPU-A", "GPU-B")
    )
    fit <- list(layout = matrix(0.1, 4L, 2L))
    knn <- list(indices = matrix(1:16, 4L, 4L),
        distances = matrix(0.1, 4L, 4L))
    seen <- NULL
    with_mocked_bindings({
        result <- run_landmark_umap_cuda_devices_refinement(
            state, fit, list(knn = knn), 5L,
            list(min_dist = 0.1, negative_rate = 5L,
                learning_rate = 1, repulsion = 1),
            7L, c(0L, 1L))
        expect_equal(unname(result$layout[1:4, ]), fit$layout)
        expect_equal(unname(result$layout[5:8, ]),
            matrix(0.2, 4L, 2L))
        expect_identical(result$gpu_devices, c(0L, 1L))
        expect_identical(result$gpu_identity,
            c("GPU-A", "GPU-B"))
        expect_identical(seen, c(0L, 1L))
    },
    landmark_umap_project_refine_cuda_devices_cpp = function(...) {
        seen <<- tail(list(...), 1L)[[1L]]
        list(query_layout = matrix(0.2, 4L, 2L),
            config = list(negative_pool = "reference_only"))
    },
    .package = "fastEmbedR")
})

test_that("multi-GPU UMAP materializes KNN before query sharding", {
    x <- matrix(seq_len(24) / 10, nrow = 8L)
    state <- list(
        partition = list(landmarks = x[1:4, , drop = FALSE],
            query = x[5:8, , drop = FALSE]),
        n_neighbors = 3L, backend = "cuda", n_threads = 1L,
        metric = "euclidean"
    )
    gpu <- structure(list(), class = "test_gpu_knn")
    host <- list(indices = matrix(1L, 4L, 3L),
        distances = matrix(0.1, 4L, 3L))
    with_mocked_bindings({
        result <- project_landmark_umap_rows(
            state, list(), 3L, 1L, c(0L, 1L))
        expect_identical(result$knn$indices, host$indices)
        expect_true(result$resident)
        expect_null(result$projected)
    },
    landmark_projection_knn = function(...) gpu,
    fastembedr_is_gpu_knn = function(...) TRUE,
    fastembedr_gpu_knn_to_host = function(...) host,
    .package = "fastEmbedR")
})

test_that("held-out CUDA projections use sharded native routes", {
    model <- list(reference_data = matrix(0.1, 4L, 3L),
        fit = list(layout = matrix(0.2, 4L, 2L)), seed = 9L)
    query <- matrix(0.3, 4L, 3L)
    knn <- list(indices = matrix(1L, 4L, 3L),
        distances = matrix(0.1, 4L, 3L))
    seen <- NULL
    with_mocked_bindings({
        umap <- project_landmark_umap_cuda_devices(
            model, query, knn, 2L,
            list(min_dist = 0.1, negative_rate = 5L,
                learning_rate = 1, repulsion = 1), c(0L, 1L))
        expect_equal(dim(umap$layout), c(4L, 2L))
        expect_identical(seen, c(0L, 1L))
        tsne <- project_landmark_model_tsne_cuda_devices(
            model, query, knn, 2, 3L, 1L,
            list(learning_rate = 0.1, early_exaggeration = 4,
                exaggeration = 1.5, initial_momentum = 0.8,
                final_momentum = 0.8, max_grad_norm = 0.25,
                max_step_norm = Inf, n_negatives = 4L,
                exact_threshold = 100L), c(0L, 1L))
        expect_equal(dim(tsne$layout), c(4L, 2L))
        expect_identical(seen, c(0L, 1L))
    },
    landmark_umap_project_refine_cuda_devices_cpp = function(...) {
        seen <<- tail(list(...), 1L)[[1L]]
        list(query_layout = matrix(0.4, 4L, 2L),
            config = list(negative_pool = "reference_only"))
    },
    landmark_tsne_transform_cuda_devices_cpp = function(...) {
        seen <<- tail(list(...), 1L)[[1L]]
        list(Y = matrix(0.5, 4L, 2L), config = list())
    }, .package = "fastEmbedR")
})

test_that("held-out devices require explicit CUDA and NCCL", {
    model <- structure(list(
        backend = "cuda", n.cores = 1L, method = "umap",
        n_neighbors = 3L,
        reference_data = matrix(0, 4L, 3L)
    ), class = "fastEmbedR_landmark_model")
    query <- matrix(0, 4L, 3L)
    with_mocked_bindings({
        prepared <- prepare_landmark_projection(
            model, query, NULL, 3L, 2, "median", 1L,
            c(0L, 1L))
        expect_identical(prepared$devices, c(0L, 1L))
        model$backend <- "cpu"
        expect_error(prepare_landmark_projection(
            model, query, NULL, 3L, 2, "median", 1L,
            c(0L, 1L)), "requires CUDA")
    },
    landmark_query_data = function(...) list(data = query),
    massive_validate_devices = function(devices, backend, nrow) {
        if (backend != "cuda") stop("devices requires CUDA")
        expect_identical(nrow, 4L)
        as.integer(devices)
    },
    massive_nccl_available_cpp = function() TRUE,
    massive_cuda_device_uuids = function(...) c("GPU-A", "GPU-B"),
    .package = "fastEmbedR")
})

test_that("held-out multi-GPU KNN stays resident until host handoff", {
    gpu <- structure(list(), class = "test_gpu_knn")
    host <- list(indices = matrix(1L, 4L, 3L),
        distances = matrix(0.1, 4L, 3L))
    request <- list(model = list(reference_data = matrix(0, 4L, 3L),
        metric = "euclidean", backend = "cuda", method = "umap"),
        query = matrix(0, 4L, 3L), transform_k = 3L,
        n_threads = 1L, devices = c(0L, 1L))
    with_mocked_bindings({
        result <- compute_landmark_projection(
            request, 0L, 2, 3L, 0L, NULL, FALSE, list())
        expect_identical(result$knn$indices, host$indices)
        expect_equal(dim(result$projection$layout), c(4L, 2L))
    },
    massive_cuda_select_cpp = function(...) NULL,
    massive_cuda_memory_cpp = function() list(device = 0L),
    precompute_query_knn = function(...) gpu,
    fastembedr_is_gpu_knn = function(...) TRUE,
    fastembedr_gpu_knn_to_host = function(...) host,
    run_landmark_projection_method = function(request, knn, ...) {
        expect_identical(knn$indices, host$indices)
        list(result = list(layout = matrix(0.5, 4L, 2L)),
            refinement_epochs = 0L)
    }, .package = "fastEmbedR")
})

test_that("resident t-SNE shards projected queries over requested GPUs", {
    x <- matrix(seq_len(24) / 10, nrow = 8L)
    knn <- list(indices = matrix(1:16, nrow = 4L),
        distances = matrix(0.1, 4L, 4L))
    fake <- structure(list(), class = "test_gpu_knn")
    request <- list(backend = "cuda", devices = c(0L, 1L),
        transform_perplexity = 2, seed = 4L)
    controls <- list(n_iter = 2L, n_negatives = 4L,
        threshold = 100L, learning_rate = 0.1,
        early_exaggeration = 4, exaggeration = 1.5,
        initial_momentum = 0.8, final_momentum = 0.8,
        max_grad_norm = 0.25, max_step_norm = Inf)
    request$transform_early_exaggeration_iter <- 1L
    state <- list(x_landmarks = x[1:4, , drop = FALSE],
        x_query = x[5:8, , drop = FALSE],
        reference_fit = list(layout = matrix(0.1, 4L, 2L)))
    observed <- NULL
    with_mocked_bindings({
        reference <- landmark_tsne_reference_args(knn,
            state, request, list(n_neighbors = 3L, perplexity = 2))
        expect_identical(reference$devices, c(0L, 1L))
        projected <- project_landmark_tsne_devices(state, request,
            controls, list(value = fake, time = system.time(NULL)))
        expect_identical(projected$projection$knn$indices,
            knn$indices)
        expect_equal(dim(projected$projection$projected), c(4L, 2L))
        expect_identical(observed, c(0L, 1L))
    },
    fastembedr_is_gpu_knn = function(x) inherits(x, "test_gpu_knn"),
    fastembedr_gpu_knn_to_host = function(...) knn,
    massive_cuda_device_uuids = function(...) c("GPU-A", "GPU-B"),
    landmark_tsne_transform_cuda_devices_cpp = function(...) {
        observed <<- tail(list(...), 1L)[[1L]]
        list(Y = matrix(0.2, 4L, 2L),
            config = list(optimizer = "cuda_query_shards"))
    }, .package = "fastEmbedR")
})
