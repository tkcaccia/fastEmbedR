massive_embedding_mode <- function(mode) {
    if (identical(mode, "auto")) stop(
        "Automatic embedding cannot choose an approximation; use ",
        "`massive = \"landmark\"` explicitly.", call. = FALSE)
    match.arg(mode, c("off", "landmark", "out_of_core_graph"))
}

massive_embedding_paths <- function(data, count, k, dimensions,
                                    output, memory_limit, devices,
                                    reuse = FALSE, resume = FALSE) {
    part <- paste0(output, ".part")
    path <- massive_float_output(output, data$nrow * dimensions * 4,
        resume = resume && file.exists(part))
    stem <- sub("\\.f32$", "", path, ignore.case = TRUE)
    landmarks <- paste0(stem, ".landmarks.f32")
    model <- paste0(stem, ".reference.rds")
    graph <- paste0(stem, ".knn")
    if (file.exists(model) && !resume) {
        stop("Reference model output already exists.", call. = FALSE)
    }
    if (!reuse && !resume) {
        massive_float_output(landmarks, count * data$ncol * 4)
        massive_knn_paths(graph, data$nrow * k * 8)
    }
    limit <- massive_memory_bytes(memory_limit)
    reference_ram <- 128 * 1024^2 + count *
        (48 * data$ncol + 16 * k + 64 * dimensions)
    if (reference_ram > 0.7 * limit) {
        stop("Landmark fit exceeds `memory_limit`; reduce `landmarks`.",
            call. = FALSE)
    }
    disk <- count * data$ncol * if (reuse) 8 else 12
    disk <- disk + data$nrow *
        (if (reuse) 0 else 8 * k) + data$nrow * 4 * dimensions
    if (!is.null(devices)) disk <- disk + data$nrow * dimensions * 4
    if (length(devices) > 1L && !reuse) {
        disk <- disk + data$nrow * k * 8
    }
    if (resume) {
        finished <- c(landmarks, paste0(graph, ".indices.u32"),
            paste0(graph, ".distances.f32"), paste0(path, ".part"))
        finished <- finished[file.exists(finished)]
        sizes <- file.info(finished)$size
        if (anyNA(sizes)) stop("Cannot inspect resume files.",
            call. = FALSE)
        disk <- max(0, disk - sum(sizes))
    }
    if (disk > 0.8 * massive_disk_available_cpp(path)) {
        stop("Experimental embedding exceeds free disk space.",
            call. = FALSE)
    }
    list(path = path, landmarks = landmarks, model = model,
        graph = graph, reference_ram_bytes = reference_ram,
        output_bytes = data$nrow * dimensions * 4,
        graph_bytes = data$nrow * k * 8)
}

massive_embedding_request <- function(data, landmarks, backend,
        metric, standardize, pca_dims, nn, keep_knn, output,
        memory_limit, chunk_rows, devices, landmark_method,
        checkpoint, resume) {
    massive_checkpoint_validate_controls(checkpoint, resume)
    if (!inherits(data, "fastEmbedR_massive_matrix")) {
        stop("EXPERIMENTAL landmark mode requires `massive_matrix()`.",
            call. = FALSE)
    }
    count <- integer_scalar(landmarks)
    if (is.na(count) || count < 2L || count >= data$nrow) {
        stop("`landmarks` must be a count between 2 and nrow - 1.",
            call. = FALSE)
    }
    metric <- match.arg(metric, c("euclidean", "cosine", "correlation"))
    if (metric != "euclidean" || !identical(standardize, FALSE) ||
        !is.null(pca_dims) ||
        !identical(keep_knn, FALSE)) {
        stop("EXPERIMENTAL landmark mode needs Euclidean raw data, ",
            "without preprocessing or `keep_knn`.",
            call. = FALSE)
    }
    if (!backend %in% c("cpu", "cuda")) {
        stop("EXPERIMENTAL landmark mode requires CPU or CUDA; ",
            "no backend fallback was used.", call. = FALSE)
    }
    if (!requireNamespace("float", quietly = TRUE)) {
        stop("EXPERIMENTAL landmark embedding needs `float` for ",
            "float32 reference fitting.", call. = FALSE)
    }
    if (backend == "cuda" &&
        (!isTRUE(embedding_cuda_available_cpp()) ||
            !isTRUE(native_cuda_knn_available_cpp()))) {
        stop("EXPERIMENTAL CUDA landmark mode is unavailable; ",
            "no CPU fallback was used.", call. = FALSE)
    }
    if (!is.null(chunk_rows) &&
        (length(chunk_rows) != 1L || !is.numeric(chunk_rows) ||
            !is.finite(chunk_rows) || chunk_rows < 1 ||
            chunk_rows != floor(chunk_rows) ||
            chunk_rows > .Machine$integer.max)) {
        stop("`chunk_rows` must be one positive integer.", call. = FALSE)
    }
    list(count = count, output = output, reuse = nn,
        memory_limit = memory_limit, chunk_rows = chunk_rows,
        devices = massive_validate_devices(devices, backend, data$nrow),
        landmark_method = match.arg(landmark_method,
            c("reservoir", "random")), checkpoint = checkpoint,
        resume = resume)
}

massive_embedding_reused_graph <- function(data, previous, count,
                                            k, backend) {
    if (is.null(previous)) return(NULL)
    if (!inherits(previous, "fastEmbedR_massive_projection") ||
        !identical(previous$mode, "landmark") ||
        !inherits(previous$graph, "fastEmbedR_massive_knn")) {
        stop("`nn` must be a previous massive landmark embedding.",
            call. = FALSE)
    }
    graph <- previous$graph
    selection <- previous$selection
    massive_embedding_checked_graph(data, selection, graph,
        count, k, backend)
    list(selection = selection, graph = graph)
}

massive_embedding_checked_graph <- function(data, selection, graph,
                                            count, k, backend) {
    massive_reference_indices(selection, graph)
    if (graph$nrow != data$nrow || graph$n_reference != count ||
        graph$ncol != k || graph$backend != backend ||
        graph$metric != "euclidean" ||
        selection$data$ncol != data$ncol ||
        !identical(graph$source_identity,
            massive_checkpoint_source_identity(data)) ||
        !identical(graph$reference_identity,
            massive_checkpoint_source_identity(selection$data)) ||
        !identical(graph$output_identity,
            massive_checkpoint_file_identity(c(
                graph$indices_path, graph$distances_path)))) {
        stop("Saved landmark KNN does not match the source, ",
            "reference, k, or backend; it was not recomputed.",
            call. = FALSE)
    }
    graph
}

massive_embedding_workflow <- function(data, request, paths, config) {
    if (!request$checkpoint) return(NULL)
    path <- sub("\\.f32$", ".workflow.checkpoint.rds", paths$path,
        ignore.case = TRUE)
    reused <- request$reuse
    signature <- massive_checkpoint_signature(list(version = 1L,
        package_version = as.character(utils::packageVersion("fastEmbedR")),
        source = massive_checkpoint_source_identity(data),
        count = request$count, landmark_method = request$landmark_method,
        k = config$k, dimensions = config$dimensions,
        backend = config$backend, workers = config$workers,
        seed = config$seed, chunk_rows = request$chunk_rows,
        memory_limit = request$memory_limit, devices = request$devices,
        fit = config$fit, projection = config$projection,
        reused_graph = if (is.null(reused)) NULL else
            reused$graph$output_identity))
    if (request$resume) {
        state <- tryCatch(readRDS(path), error = function(e) NULL)
        if (!is.list(state) ||
            !identical(state$signature, signature) ||
            !is.character(state$stage) || length(state$stage) != 1L ||
            is.na(state$stage) ||
            !state$stage %in% c("initial", "selected", "fitted", "graph")) {
            stop("Landmark workflow checkpoint does not match the ",
                "current request.", call. = FALSE)
        }
    } else {
        if (file.exists(path)) stop(
            "Landmark workflow checkpoint exists; use `resume = TRUE`.",
            call. = FALSE)
        state <- list(signature = signature, stage = "initial")
        massive_checkpoint_write(state, path)
    }
    list(path = path, state = state)
}

massive_workflow_stage <- function(workflow, stage, values) {
    if (is.null(workflow)) return(NULL)
    workflow$state[names(values)] <- values
    workflow$state$stage <- stage
    massive_checkpoint_write(workflow$state, workflow$path)
    workflow
}

massive_workflow_selection <- function(data, request, paths, seed,
                                        workflow, reused) {
    stage <- workflow$state$stage %||% "initial"
    if (stage != "initial") {
        selection <- workflow$state$selection
        identity <- tryCatch(massive_checkpoint_file_identity(
            selection$data$path), error = function(e) NULL)
        if (is.null(identity) ||
            !identical(identity, workflow$state$selection_file) ||
            !identical(selection$source_identity,
                massive_checkpoint_source_identity(data)) ||
            !identical(selection$indices_signature,
                massive_checkpoint_signature(selection$indices))) {
            stop("Saved landmark selection changed; no resampling was ",
                "performed.", call. = FALSE)
        }
        return(list(selection = selection, workflow = workflow))
    }
    selection <- if (!is.null(reused)) reused$selection else
        massive_select_landmarks(data, request$count,
            paths$landmarks, seed = seed,
            chunk_rows = request$chunk_rows,
            memory_limit = request$memory_limit,
            landmark_method = request$landmark_method)
    workflow <- massive_workflow_stage(workflow, "selected", list(
        selection = selection,
        selection_file = if (is.null(workflow)) NULL else
            massive_checkpoint_file_identity(selection$data$path)))
    list(selection = selection, workflow = workflow)
}

massive_embedding_reference <- function(selection, data, request,
                                        dimensions, paths,
                                        fit_reference, workflow) {
    stage <- workflow$state$stage %||% "initial"
    if (stage %in% c("fitted", "graph")) {
        identity <- tryCatch(massive_checkpoint_file_identity(
            paths$model), error = function(e) NULL)
        if (is.null(identity) ||
            !identical(identity, workflow$state$model_file)) {
            stop("Saved landmark model changed; it was not refitted.",
                call. = FALSE)
        }
        fit <- tryCatch(readRDS(paths$model), error = function(e) NULL)
    } else {
        if (file.exists(paths$model)) stop(
            "Uncommitted landmark model exists; refusing to overwrite it.",
            call. = FALSE)
        reference <- massive_read_rows_cpp(selection$data, 1,
            request$count, 4 * request$count * data$ncol, TRUE)
        fit <- fit_reference(reference)
        rm(reference)
    }
    if (!is.list(fit) || is.null(dim(fit$layout)) ||
        nrow(fit$layout) != request$count ||
        ncol(fit$layout) != dimensions) {
        stop("Landmark fit dimensions differ from the request.",
            call. = FALSE)
    }
    if (!stage %in% c("fitted", "graph")) {
        massive_checkpoint_write(fit, paths$model)
        workflow <- massive_workflow_stage(workflow, "fitted", list(
            model_file = if (is.null(workflow)) NULL else
                massive_checkpoint_file_identity(paths$model)))
    }
    list(fit = fit, workflow = workflow)
}

massive_embedding_graph <- function(data, selection, request, paths,
                                    k, backend, workers, workflow, reused) {
    stage <- workflow$state$stage %||% "initial"
    if (stage == "graph") {
        graph <- workflow$state$graph
        massive_embedding_checked_graph(data, selection, graph,
            request$count, k, backend)
        return(list(graph = graph, workflow = workflow))
    }
    sidecar <- paste0(paths$graph, if (length(request$devices) > 1L)
        ".multigpu.checkpoint.rds" else ".checkpoint.rds")
    graph <- if (!is.null(reused)) reused$graph else
        massive_landmark_knn(data, selection, k, paths$graph,
            backend = backend, n.cores = workers,
            chunk_rows = request$chunk_rows,
            memory_limit = request$memory_limit,
            devices = request$devices, checkpoint = request$checkpoint,
            resume = request$resume && file.exists(sidecar))
    workflow <- massive_workflow_stage(workflow, "graph",
        list(graph = graph))
    list(graph = graph, workflow = workflow)
}

massive_embedding_result_resources <- function(result, paths,
                                                graph, reused) {
    result$resources$reference_ram_bytes <- paths$reference_ram_bytes
    result$resources$graph_bytes <- paths$graph_bytes
    result$resources$peak_ram_bytes <- max(
        paths$reference_ram_bytes,
        if (reused) 0 else graph$resources$peak_ram_bytes,
        result$resources$peak_ram_bytes)
    result$resources$peak_vram_bytes <- max(
        if (reused) 0 else graph$resources$peak_vram_bytes %||% 0,
        result$resources$peak_vram_bytes)
    result
}

massive_embedding_off <- function(output, chunk_rows, memory_limit,
                                    refinement_epochs = 0L,
                                    local_refine = FALSE,
                                    local_neighbors = 15L,
                                    overlap_rows = 500L,
                                    devices = NULL,
                                    landmark_method = "reservoir",
                                    init = NULL,
                                    layout_storage = "memory",
                                    checkpoint = FALSE, resume = FALSE) {
    massive_checkpoint_validate_controls(checkpoint, resume)
    if (checkpoint) stop("`checkpoint` requires massive mode.",
        call. = FALSE)
    if (!is.null(init)) stop("`init` requires full-graph UMAP.",
        call. = FALSE)
    if (!identical(layout_storage, "memory")) stop(
        "`layout_storage` requires full-graph UMAP.", call. = FALSE)
    if (!is.null(output) || !is.null(chunk_rows) ||
        !identical(memory_limit, "8GB") ||
        !identical(refinement_epochs, 0L) ||
        !identical(local_refine, FALSE) ||
        !identical(local_neighbors, 15L) ||
        !identical(overlap_rows, 500L) || !is.null(devices) ||
        !identical(landmark_method, "reservoir")) {
        stop("Experimental output controls require ",
            "`massive = \"landmark\"`.", call. = FALSE)
    }
}

dispatch_massive_umap <- function(data, landmarks, n_neighbors,
        n_components, standardize, pca_dims, metric, nn, seed,
        backend, n.cores, keep_knn, graph_mode, verbose,
        transform_k, massive, output, chunk_rows, memory_limit,
        refinement_epochs, local_refine, local_neighbors,
        overlap_rows, devices, landmark_method, init,
        layout_storage, checkpoint, resume) {
    if (identical(massive, "out_of_core_graph")) return(
        run_massive_umap_graph(data, init, output, n_components,
            seed, backend, n.cores, chunk_rows, memory_limit,
            n_neighbors, standardize, pca_dims, metric, nn,
            keep_knn, graph_mode, verbose, landmarks, transform_k,
            refinement_epochs, local_refine, local_neighbors,
            overlap_rows, devices, landmark_method,
            layout_storage, checkpoint, resume))
    if (!identical(layout_storage, "memory")) stop(
        "`layout_storage` requires full-graph UMAP.", call. = FALSE)
    if (!is.null(init)) stop("`init` requires full-graph UMAP.",
        call. = FALSE)
    run_massive_umap_landmark(data, landmarks, n_neighbors, n_components,
        standardize, pca_dims, metric, nn, seed, backend, n.cores,
        keep_knn, graph_mode, verbose, transform_k, massive, output,
        chunk_rows, memory_limit, refinement_epochs, local_refine,
        local_neighbors, overlap_rows, devices, landmark_method,
        checkpoint, resume)
}

run_massive_umap_graph <- function(data, init, output, n_components,
        seed, backend, n.cores, chunk_rows, memory_limit,
        n_neighbors, standardize, pca_dims, metric, nn,
        keep_knn, graph_mode, verbose, landmarks, transform_k,
        refinement_epochs, local_refine, local_neighbors,
        overlap_rows, devices, landmark_method,
        layout_storage, checkpoint, resume) {
    if (!is.null(n_neighbors) || !identical(standardize, FALSE) ||
        !is.null(pca_dims) || !is.null(nn) ||
        !identical(keep_knn, FALSE) || !identical(verbose, FALSE) ||
        !identical(landmarks, FALSE) || !is.null(transform_k) ||
        !identical(refinement_epochs, 0L) ||
        !identical(local_refine, FALSE) ||
        !identical(local_neighbors, 15L) ||
        !identical(overlap_rows, 500L) || !is.null(devices) ||
        !identical(landmark_method, "reservoir") ||
        !identical(metric, c("euclidean", "cosine", "correlation")) ||
        !identical(graph_mode, c("fuzzy", "binary"))) {
        stop("Full-graph UMAP accepts a prepared fuzzy graph, not ",
            "matrix preprocessing or landmark controls.", call. = FALSE)
    }
    if (is.null(init) || is.null(output)) stop(
        "Full-graph UMAP requires `init` and `output`.", call. = FALSE)
    dimensions <- validate_n_components(n_components)
    if (!inherits(init, "fastEmbedR_massive_matrix") ||
        init$ncol != dimensions) stop(
        "`n_components` must match file-backed `init`.", call. = FALSE)
    massive_umap_optimize(data, init, output, seed = seed,
        chunk_rows = chunk_rows %||% 8192L,
        memory_limit = memory_limit,
        backend = resolve_embedding_backend(backend),
        n.cores = resolve_n_cores(n.cores),
        layout_storage = layout_storage,
        checkpoint = checkpoint, resume = resume)
}

massive_auto_cuda_free <- function(devices, rows) {
    selected <- massive_validate_devices(devices, "cuda", rows)
    if (is.null(selected)) return(massive_cuda_memory_cpp()$free_bytes)
    original <- massive_cuda_memory_cpp()$device
    on.exit(massive_cuda_select_cpp(original))
    min(vapply(selected, function(device) {
        massive_cuda_select_cpp(device)
        massive_cuda_memory_cpp()$free_bytes
    }, 0))
}

massive_embedding_run <- function(data, request, k, dimensions, backend,
    n.cores, seed, fit_reference, refinement_epochs, transform_iter,
    transform_perplexity, local_refine = FALSE, local_neighbors = 15L,
    overlap_rows = 500L, fit_config = NULL) {
    reused <- massive_embedding_reused_graph(data, request$reuse,
        request$count, k, backend)
    paths <- massive_embedding_paths(data, request$count, k, dimensions,
        request$output, request$memory_limit, request$devices,
        !is.null(reused), request$resume)
    if (!is.null(request$devices)) massive_cuda_select_cpp(request$devices[1L])
    message("EXPERIMENTAL landmark embedding: ", data$nrow,
        " rows; ", request$count, " landmarks; backend=", backend)
    config <- list(k = k, dimensions = dimensions, backend = backend,
        workers = n.cores, seed = seed, fit = fit_config,
        projection = list(epochs = refinement_epochs,
            iterations = transform_iter, perplexity = transform_perplexity,
            local_refine = local_refine, local_neighbors = local_neighbors,
            overlap_rows = overlap_rows))
    workflow <- massive_embedding_workflow(data, request, paths, config)
    selected <- massive_workflow_selection(data, request, paths,
        seed, workflow, reused)
    selection <- selected$selection
    fitted <- massive_embedding_reference(selection, data, request,
        dimensions, paths, fit_reference, selected$workflow)
    built <- massive_embedding_graph(data, selection, request, paths,
        k, backend, n.cores, fitted$workflow, reused)
    graph <- built$graph
    projection_sidecar <- if (is.null(request$devices)) {
        paste0(paths$path, ".checkpoint.rds")
    } else paste0(paths$path, ".multigpu.checkpoint.rds")
    result <- massive_project_landmarks(fitted$fit, graph, paths$path,
        selection = selection, backend = backend, n.cores = n.cores,
        chunk_rows = request$chunk_rows,
        memory_limit = request$memory_limit,
        refinement_epochs = refinement_epochs, transform_iter = transform_iter,
        transform_perplexity = transform_perplexity, seed = seed,
        source = if (local_refine) data else NULL,
        local_refine = local_refine,
        local_neighbors = local_neighbors,
        overlap_rows = overlap_rows, devices = request$devices,
        checkpoint = request$checkpoint,
        resume = request$resume && file.exists(projection_sidecar))
    if (!is.null(built$workflow)) unlink(built$workflow$path)
    result$reference_fit_path <- paths$model
    result$selection <- selection
    result$graph_reused <- !is.null(reused)
    massive_embedding_result_resources(result, paths, graph,
        !is.null(reused))
}

run_massive_umap_landmark <- function(data, landmarks, n_neighbors,
                            n_components, standardize, pca_dims,
                            metric, nn, seed, backend, n.cores,
                            keep_knn, graph_mode, verbose, transform_k,
                            massive, output, chunk_rows, memory_limit,
                            refinement_epochs, local_refine,
                            local_neighbors, overlap_rows, devices,
                            landmark_method, checkpoint, resume) {
    match.arg(massive, c("landmark"))
    state <- validate_umap_request(backend, graph_mode,
        n_components, n.cores, keep_knn)
    epochs <- integer_scalar(refinement_epochs)
    if (is.na(epochs) || epochs < 0L)
        stop("Invalid experimental refinement epochs.", call. = FALSE)
    if (!identical(local_refine, FALSE) && !identical(local_refine, TRUE)) {
        stop("`local_refine` must be TRUE or FALSE.", call. = FALSE)
    }
    if (local_refine && (state$backend != "cpu" || epochs < 1L)) {
        stop("EXPERIMENTAL local graph refinement requires CPU UMAP ",
            "and positive epochs; no fallback was used.",
            call. = FALSE)
    }
    request <- massive_embedding_request(data, landmarks,
        state$backend, metric, standardize, pca_dims, nn,
        keep_knn, output, memory_limit, chunk_rows, devices,
        landmark_method, checkpoint, resume)
    n_neighbors <- validate_umap_n_neighbors(n_neighbors, request$count)
    k <- integer_scalar(transform_k %||% n_neighbors)
    if (is.na(k) || k < 1L || k > request$count)
        stop("Invalid experimental projection neighbors.", call. = FALSE)
    if (state$backend == "cuda" && epochs != 0L) {
        stop("EXPERIMENTAL CUDA UMAP requires ",
            "`refinement_epochs = 0`; no CPU fallback was used.",
            call. = FALSE)
    }
    workers <- normalize_nn_threads(resolve_n_cores(n.cores))
    if (state$backend == "cuda" && workers != 1L) {
        stop("CUDA projection requires `n.cores = 1`.", call. = FALSE)
    }
    fit_reference <- function(reference) umap(reference,
        n_neighbors = n_neighbors, n_components = state$n_components,
        backend = state$backend, n.cores = workers,
        graph_mode = state$graph_mode, seed = seed, verbose = verbose)
    massive_embedding_run(data, request, k, state$n_components,
        state$backend, workers, seed, fit_reference, epochs, 250L, 5,
        local_refine, local_neighbors, overlap_rows,
        fit_config = list(method = "umap", neighbors = n_neighbors,
            graph_mode = state$graph_mode, verbose = verbose))
}

run_massive_tsne <- function(data, nn, settings, extra, controls,
                            massive, output, chunk_rows, memory_limit,
                            devices, landmark_method,
                            checkpoint, resume) {
    massive <- massive_embedding_mode(massive)
    if (identical(massive, "out_of_core_graph")) return(
        run_massive_tsne_graph(data, nn, settings, extra,
            controls, output, chunk_rows, memory_limit,
            devices, landmark_method, checkpoint, resume))
    run_massive_tsne_landmark(data, nn, settings, extra,
        controls, massive, output, chunk_rows, memory_limit,
        devices, landmark_method, checkpoint, resume)
}

run_massive_tsne_graph <- function(data, nn, settings, extra,
        controls, output, chunk_rows, memory_limit,
        devices, landmark_method, checkpoint, resume) {
    massive_tsne_graph_controls(data, nn, settings, extra,
        controls, output, chunk_rows, devices,
        landmark_method)
    optimizer <- settings$optimizer
    early <- optimizer$early_exaggeration_iter %||% 250L
    normal <- optimizer$n_iter %||% 500L
    early_exag <- if (identical(optimizer$early_exaggeration,
        "auto")) 12 else optimizer$early_exaggeration
    exag <- optimizer$exaggeration %||% 1
    step <- if (identical(optimizer$max_step_norm, "auto")) {
        5
    } else if (is.null(optimizer$max_step_norm)) {
        Inf
    } else optimizer$max_step_norm
    workers <- normalize_nn_threads(resolve_n_cores(settings$n_threads))
    massive_tsne_optimize(data, settings$Y_init, output,
        early_exaggeration_iter = early, n_iter = normal,
        early_exaggeration = early_exag, exaggeration = exag,
        learning_rate = optimizer$learning_rate,
        initial_momentum = optimizer$initial_momentum,
        final_momentum = optimizer$final_momentum,
        max_step_norm = step, memory_limit = memory_limit,
        backend = settings$backend, n.cores = workers,
        checkpoint = checkpoint, resume = resume)
}

massive_tsne_graph_controls <- function(data, nn, settings, extra,
        controls, output, chunk_rows, devices,
        landmark_method) {
    if (!inherits(data, "fastEmbedR_massive_graph") ||
        !identical(data$weight_method, "tsne_compact_affinity") ||
        !inherits(settings$Y_init, "fastEmbedR_massive_matrix") ||
        is.null(output)) stop(
        "Full-graph t-SNE requires affinities, file-backed `Y_init`, ",
        "and `output`.", call. = FALSE)
    if (!is.null(nn) || !is.null(settings$init_data) ||
        !identical(settings$standardize, FALSE) ||
        !is.null(settings$pca_dims) ||
        !identical(settings$keep_knn, FALSE) ||
        !identical(settings$metric,
            c("euclidean", "cosine", "correlation")) ||
        length(extra) || !identical(controls$landmarks, FALSE) ||
        !is.null(controls$transform_k) ||
        !identical(controls$transform_perplexity, 5) ||
        !identical(controls$transform_iter, 250L) ||
        !identical(controls$transform_early_exaggeration_iter, 0L) ||
        !is.null(controls$transform_n_negatives) ||
        !identical(controls$initialization,
            c("median", "weighted", "random")) ||
        !is.null(chunk_rows) ||
        !is.null(devices) ||
        !identical(landmark_method, "reservoir") ||
        isTRUE(settings$verbose) ||
        isTRUE(settings$optimizer$record_costs) ||
        !settings$optimizer$negative_gradient_method %in%
            c("auto", "fft")) stop(
        "Full-graph t-SNE does not accept matrix preprocessing, ",
        "landmark, or diagnostic controls.",
        call. = FALSE)
    if (settings$n_components != settings$Y_init$ncol ||
        (!is.null(settings$perplexity) &&
            settings$perplexity != data$perplexity)) stop(
        "Full-graph t-SNE dimensions or perplexity differ ",
        "from the prepared inputs.", call. = FALSE)
    invisible(NULL)
}

run_massive_tsne_landmark <- function(data, nn, settings, extra, controls,
                            massive, output, chunk_rows, memory_limit,
                            devices, landmark_method,
                            checkpoint, resume) {
    match.arg(massive, c("landmark"))
    request <- massive_embedding_request(data, controls$landmarks,
        settings$backend, settings$metric, settings$standardize,
        settings$pca_dims, nn, settings$keep_knn, output,
        memory_limit, chunk_rows, devices, landmark_method,
        checkpoint, resume)
    if (!is.null(settings$init_data) || !is.null(settings$Y_init) ||
        !identical(controls$transform_early_exaggeration_iter, 0L) ||
        !is.null(controls$transform_n_negatives) ||
        match.arg(controls$initialization,
            c("median", "weighted", "random")) != "median") {
        stop("EXPERIMENTAL t-SNE requires default transform controls ",
            "and no supplied initialization.", call. = FALSE)
    }
    policy <- opentsne_neighbor_policy(request$count, settings$perplexity)
    settings$perplexity <- policy$perplexity
    value <- controls$transform_perplexity
    iterations <- integer_scalar(controls$transform_iter)
    if (!is.numeric(value) || length(value) != 1L ||
        !is.finite(value) || value <= 0 ||
        is.na(iterations) || iterations < 1L) {
        stop("Invalid experimental t-SNE transform controls.", call. = FALSE)
    }
    k <- integer_scalar(controls$transform_k %||% ceiling(value))
    if (is.na(k) || k < value || k > request$count) {
        stop("`transform_k` must cover transform perplexity and fit ",
            "the landmark count.", call. = FALSE)
    }
    workers <- normalize_nn_threads(resolve_n_cores(settings$n_threads))
    settings$n_threads <- workers
    if (settings$backend == "cuda" && workers != 1L) {
        stop("CUDA projection requires `n.cores = 1`.", call. = FALSE)
    }
    if (settings$backend == "cuda" &&
        (!(settings$n_components %in% c(2L, 3L)) || k > 128L)) {
        stop("EXPERIMENTAL CUDA t-SNE projection needs 2D or 3D ",
            "and k <= 128; no CPU fallback was used.", call. = FALSE)
    }
    fit_reference <- function(reference) run_matrix_input_tsne(
        reference, NULL, settings, extra, TRUE)
    massive_embedding_run(data, request, k, settings$n_components,
        settings$backend, workers, settings$seed, fit_reference,
        0L, controls$transform_iter, controls$transform_perplexity,
        fit_config = list(method = "tsne", settings = settings,
            extra = extra))
}
