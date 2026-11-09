full_backed_pipeline_paths <- function(data, output, dimensions,
        memory_limit, init = NULL, resume = FALSE) {
    if (!fastembedr_file_backed_input(data) ||
        inherits(data, "fastEmbedR_massive_graph")) stop(
        "Full file-backed analysis needs a disk-backed matrix.",
        call. = FALSE)
    dimensions <- validate_n_components(dimensions)
    if (!dimensions %in% c(2L, 3L)) stop(
        "Full file-backed UMAP and t-SNE need 2D or 3D output.",
        call. = FALSE)
    massive_memory_bytes(memory_limit)
    path <- if (resume && file.exists(output)) {
        if (tolower(tools::file_ext(output)) != "f32") stop(
            "Output must have a .f32 extension.", call. = FALSE)
        file.path(normalizePath(dirname(output), mustWork = TRUE),
            basename(output))
    } else massive_float_output(output,
        as.double(data$nrow) * dimensions * 4,
        resume = resume && file.exists(paste0(output, ".part")))
    stem <- sub("\\.f32$", "", path, ignore.case = TRUE)
    paths <- list(output = path, knn = paste0(stem, ".knn"),
        graph = paste0(stem, ".graph"), pca = paste0(stem, ".pca.f32"),
        spectral = paste0(stem, ".spectral.f32"),
        checkpoint = paste0(stem, ".pipeline.rds"))
    knn_files <- paste0(paths$knn, c(".indices.u32", ".distances.f32"))
    graph_files <- paste0(paths$graph,
        c(".offsets.u64", ".indices.u32", ".weights.f32",
            ".fuzzy.rds", ".affinity.rds", ".work",
            ".sort-checkpoint.rds"))
    pca_files <- if (is.null(init)) c(paths$pca,
        paste0(paths$pca, ".manifest.rds"),
        paste0(paths$pca, ".checkpoint.rds"),
        sub("\\.f32$", ".init.f32", paths$pca))
        else character()
    created <- c(knn_files, graph_files, pca_files,
        if (is.null(init)) paths$spectral)
    occupied <- c(created, paths$checkpoint, paste0(created, ".part"),
        paste0(paths$knn, c(".checkpoint.rds",
            ".checkpoint.rds.part", ".weights.f32",
            ".weights.f32.part", ".weights.f32.checkpoint.rds",
            ".weights.f32.checkpoint.rds.part")))
    if (!resume && any(file.exists(occupied))) stop(
        "One-call intermediate output already exists; choose a new output.",
        call. = FALSE)
    inputs <- c(data$path, if (!is.null(init)) init$path)
    if (any(created %in% inputs)) stop(
        "One-call intermediate output overlaps an input file.",
        call. = FALSE)
    paths
}

full_backed_umap_layout_controls <- function(backend, layout_storage,
        devices, cpu_update) {
    if (!is.character(layout_storage) ||
        length(layout_storage) != 1L || is.na(layout_storage) ||
        !layout_storage %in% c("memory", "mmap", "managed")) stop(
        "`layout_storage` must be 'memory', 'mmap', or 'managed'.",
        call. = FALSE)
    if (backend == "cuda" && layout_storage == "mmap") stop(
        "CUDA full-graph UMAP does not support mmap.", call. = FALSE)
    if (backend != "cuda" && layout_storage == "managed") stop(
        "Managed layout storage requires CUDA.", call. = FALSE)
    if (!is.null(devices) && layout_storage != "memory") stop(
        "Coordinated CUDA UMAP requires device layout storage.",
        call. = FALSE)
    if (backend != "cpu" && cpu_update != "sgd") stop(
        "Synchronized full-graph UMAP requires CPU.", call. = FALSE)
    invisible(NULL)
}

full_backed_pipeline_init <- function(data, init, paths,
        dimensions, backend, n.cores, seed, memory_limit,
        devices, graph = NULL, tsne = FALSE,
        checkpoint = FALSE, resume = FALSE) {
    if (!is.null(init)) {
        return(list(layout = init, fit = NULL,
            backend = "external", method = "supplied"))
    }
    if (!tsne) {
        if (is.null(graph)) stop("UMAP initialization needs its graph.",
            call. = FALSE)
        source <- c(graph$offsets_path, graph$indices_path,
            graph$weights_path)
        identity <- massive_checkpoint_input_files_identity(source)
        built <- massive_spectral_init_stream_cpp(
            source[[1L]], source[[2L]], source[[3L]], graph$n_vertices,
            paths$spectral, dimensions, 50L, seed,
            8192L, massive_memory_bytes(memory_limit))
        if (!identical(identity,
                massive_checkpoint_input_files_identity(source))) stop(
            "UMAP graph changed during spectral initialization.",
            call. = FALSE)
        return(list(layout = massive_matrix(paths$spectral,
            data$nrow, dimensions), fit = NULL, backend = "cpu",
            method = "streamed_graph_spectral", resources = built))
    }
    workers <- if (identical(backend, "metal")) 1L else n.cores
    fit <- pca(data, ncomp = dimensions, backend = backend,
        n.cores = workers, seed = seed, output = paths$pca,
        storage = "file_backed", memory_limit = memory_limit,
        devices = devices, tsne_init = tsne,
        checkpoint = checkpoint, resume = resume)
    list(layout = fit$tsne_init, fit = fit,
        backend = backend, method = "pca",
        resources = fit$resources)
}

full_backed_validate_init <- function(data, init, dimensions,
        tsne = FALSE) {
    if (is.null(init)) {
        if (tsne && data$ncol < dimensions) stop(
            "Automatic initialization needs at least as many input ",
            "columns as output dimensions; supply `init`.",
            call. = FALSE)
        return(invisible(NULL))
    }
    if (!fastembedr_file_backed_input(init) ||
        inherits(init, "fastEmbedR_massive_graph") ||
        init$nrow != data$nrow || init$ncol != dimensions) stop(
        "`init` needs a matching file-backed coordinate matrix.",
        call. = FALSE)
    invisible(NULL)
}

full_backed_pipeline_record <- function(fit, data, neighbors, graph,
        prepared, knn_seconds, graph_seconds, init_seconds,
        fit_seconds,
        graph_workers, fit_workers, backend, keep_knn,
        memory_limit) {
    optimize_backend <- if (is.null(fit$optimizer_backend_stages))
        backend else paste(fit$optimizer_backend_stages, collapse = "+")
    fit$pipeline <- list(input = data,
        neighbors = if (keep_knn) neighbors else NULL,
        knn_released = !keep_knn,
        graph = graph, init = prepared$layout, pca = prepared$fit,
        initialization = prepared[c("method", "backend", "resources")],
        fit_scope = "all_rows_optimized",
        stage_backend = c(knn = neighbors$backend %||% backend,
            graph = "cpu",
            init = prepared$backend, optimize = optimize_backend),
        knn_method = neighbors$method %||% NA_character_,
        stage_resources = list(knn = neighbors$resources,
            graph = list(working_memory_budget_bytes =
                    massive_memory_bytes(memory_limit),
                peak_disk_bytes = graph$peak_graph_stage_bytes,
                sort_disk_budget_bytes =
                    graph$sort_resources$budget_bytes),
            init = prepared$resources, optimize = fit$resources),
        graph_stage_cpu_workers = graph_workers,
        optimizer_cpu_workers = if (optimize_backend == "metal+cpu")
            1L else fit_workers,
        optimizer_stage_seconds = fit$optimizer_stage_seconds,
        stage_seconds = c(knn = knn_seconds, graph = graph_seconds,
            init = init_seconds, optimize = fit_seconds))
    fit
}

full_backed_umap_controls <- function(landmarks, standardize,
        pca_dims, metric, nn, keep_knn, graph_mode, verbose,
        transform_k, refinement_epochs, local_refine, local_neighbors,
        overlap_rows, landmark_method, checkpoint, resume, ef_search) {
    if (!identical(landmarks, FALSE) ||
        !identical(standardize, FALSE) || !is.null(pca_dims) ||
        !is.null(nn) ||
        !identical(verbose, FALSE) || !is.null(transform_k) ||
        !identical(refinement_epochs, 0L) ||
        !identical(local_refine, FALSE) ||
        !identical(local_neighbors, 15L) ||
        !identical(overlap_rows, 500L) ||
        !identical(landmark_method, "reservoir") ||
        !identical(ef_search, 0L) ||
        !identical(match.arg(graph_mode, c("fuzzy", "binary")),
            "fuzzy")) stop(
        "One-call full file-backed UMAP needs raw Euclidean data and ",
        "full-graph controls only.", call. = FALSE)
    if (!is.logical(keep_knn) || length(keep_knn) != 1L ||
        is.na(keep_knn)) stop("`keep_knn` must be TRUE or FALSE.",
            call. = FALSE)
    massive_checkpoint_validate_controls(checkpoint, resume)
    invisible(NULL)
}

full_backed_umap_request <- function(data, output, dimensions,
        memory_limit, init, k, backend, workers, fit_workers,
        seed, devices, chunk_rows, layout_storage, cpu_update,
        keep_knn, checkpoint, resume) {
    controls <- list(method = "umap", k = k,
        dimensions = dimensions, backend = backend,
        workers = workers, fit_workers = fit_workers, seed = seed,
        memory_bytes = massive_memory_bytes(memory_limit),
        devices = devices, chunk_rows = chunk_rows,
        layout_storage = layout_storage,
        cpu_update = cpu_update, keep_knn = keep_knn)
    full_backed_pipeline_start(data, output, dimensions,
        memory_limit, init, controls, checkpoint, resume, keep_knn)
}

full_backed_umap_fit <- function(graph, init, paths, dimensions,
        seed, backend, workers, chunk_rows, memory_limit,
        devices, layout_storage, cpu_update, checkpoint) {
    started <- proc.time()[[3L]]
    fit <- umap(graph, init = init, output = paths$output,
        n_components = dimensions, seed = seed, backend = backend,
        n.cores = workers, chunk_rows = chunk_rows,
        memory_limit = memory_limit, devices = devices,
        layout_storage = layout_storage, cpu_update = cpu_update,
        storage = "file_backed", checkpoint = checkpoint,
        resume = checkpoint && file.exists(paste0(paths$output,
            ".checkpoint.rds")))
    list(fit = fit, seconds = proc.time()[[3L]] - started)
}

run_full_backed_umap <- function(data, landmarks, n_neighbors,
        n_components, standardize, pca_dims, metric, nn, seed,
        backend, n.cores, keep_knn, graph_mode, verbose,
        transform_k, output, chunk_rows, memory_limit,
        refinement_epochs, local_refine, local_neighbors,
        overlap_rows, devices, landmark_method, init,
        layout_storage, checkpoint, resume, cpu_update, ef_search) {
    full_backed_umap_controls(landmarks, standardize, pca_dims,
        metric, nn, keep_knn, graph_mode, verbose, transform_k,
        refinement_epochs, local_refine, local_neighbors,
        overlap_rows, landmark_method, checkpoint, resume, ef_search)
    if (!identical(resolve_embedding_metric(metric), "euclidean")) stop(
        "One-call file-backed KNN supports Euclidean distance only.",
        call. = FALSE)
    backend <- resolve_embedding_backend(backend)
    workers <- resolve_n_cores(n.cores)
    full_backed_umap_layout_controls(backend, layout_storage,
        devices, cpu_update)
    fit_workers <- if (backend == "cpu" &&
        cpu_update == "synchronized") workers else 1L
    k <- n_neighbors %||% auto_embedding_k(data$nrow, "umap")
    k <- validate_precompute_k(k, data$nrow - 1L, "nrow(data) - 1")
    full_backed_validate_init(data, init, n_components)
    request <- full_backed_umap_request(data, output, n_components,
        memory_limit, init, k, backend, workers, fit_workers,
        seed, devices, chunk_rows, layout_storage, cpu_update,
        keep_knn, checkpoint, resume)
    if (!is.null(request$completed)) return(request$completed)
    paths <- request$paths
    knn <- full_backed_knn_stage(data, k, backend, workers,
        paths, memory_limit, devices, request$state)
    graph_stage <- full_backed_umap_graph_stage(knn$object, paths,
        memory_limit, keep_knn, workers, knn$state)
    prepared_stage <- full_backed_init_stage(data, init, paths,
        n_components, backend, workers, seed, memory_limit,
        devices, FALSE, graph_stage$state, graph = graph_stage$object)
    graph <- graph_stage$object
    prepared <- prepared_stage$object
    optimized <- full_backed_umap_fit(graph, prepared$layout,
        paths, n_components, seed, backend, fit_workers,
        chunk_rows, memory_limit, devices, layout_storage,
        cpu_update, checkpoint)
    fit <- full_backed_pipeline_record(optimized$fit, data,
        knn$object, graph, prepared, knn$seconds, graph_stage$seconds,
        prepared_stage$seconds, optimized$seconds,
        workers, fit_workers, backend, keep_knn, memory_limit)
    full_backed_checkpoint_finalize(prepared_stage$state, fit,
        paths, keep_knn)
}

full_backed_tsne_controls <- function(data, nn, settings, extra,
        controls, chunk_rows, landmark_method, checkpoint, resume) {
    if (!is.null(nn) || length(extra) ||
        !identical(controls$landmarks, FALSE) ||
        !is.null(controls$transform_k) ||
        !is.null(settings$init_data) ||
        !identical(settings$standardize, FALSE) ||
        !is.null(settings$pca_dims) ||
        !is.null(chunk_rows) ||
        !identical(landmark_method, "reservoir") ||
        isTRUE(settings$verbose) ||
        isTRUE(settings$optimizer$record_costs) ||
        !settings$optimizer$negative_gradient_method %in%
            c("auto", "fft") ||
        !identical(controls$transform_perplexity, 5) ||
        !identical(controls$transform_iter, 250L) ||
        !identical(controls$transform_early_exaggeration_iter, 0L) ||
        !is.null(controls$transform_n_negatives) ||
        !identical(controls$initialization,
            c("median", "weighted", "random")) ||
        !identical(resolve_embedding_metric(settings$metric),
            "euclidean")) stop(
        "One-call full file-backed t-SNE needs raw Euclidean data and ",
        "full-graph controls only.", call. = FALSE)
    massive_checkpoint_validate_controls(checkpoint, resume)
    if (data$nrow < 4) stop(
        "Full file-backed t-SNE needs at least four rows.",
        call. = FALSE)
    invisible(NULL)
}

full_backed_tsne_request <- function(data, output, policy,
        settings, workers, memory_limit, devices,
        checkpoint, resume) {
    controls <- list(method = "tsne", policy = policy,
        backend = settings$backend, workers = workers,
        seed = settings$seed, optimizer = settings$optimizer,
        dimensions = settings$n_components,
        memory_bytes = massive_memory_bytes(memory_limit),
        devices = devices, keep_knn = settings$keep_knn)
    full_backed_pipeline_start(data, output,
        settings$n_components, memory_limit, settings$Y_init,
        controls, checkpoint, resume, settings$keep_knn)
}

full_backed_tsne_fit <- function(graph, prepared, paths,
        settings, extra, controls, memory_limit, devices,
        landmark_method, checkpoint) {
    settings$Y_init <- prepared$layout
    settings$metric <- c("euclidean", "cosine", "correlation")
    settings$keep_knn <- FALSE
    started <- proc.time()[[3L]]
    fit <- run_massive_tsne_graph(graph, NULL, settings,
        extra, controls, paths$output, NULL, memory_limit,
        devices, landmark_method, checkpoint,
        checkpoint && file.exists(paste0(paths$output,
            ".checkpoint.rds")))
    list(fit = fit, seconds = proc.time()[[3L]] - started)
}

run_full_backed_tsne <- function(data, nn, settings, extra, controls,
        output, chunk_rows, memory_limit, devices,
        landmark_method, checkpoint, resume) {
    full_backed_tsne_controls(data, nn, settings, extra, controls,
        chunk_rows, landmark_method, checkpoint, resume)
    policy <- opentsne_neighbor_policy(data$nrow,
        settings$perplexity, affinity_k = settings$affinity_k)
    full_backed_validate_init(data, settings$Y_init,
        settings$n_components, tsne = TRUE)
    backend <- settings$backend
    workers <- resolve_n_cores(settings$n_threads)
    request <- full_backed_tsne_request(data, output, policy,
        settings, workers, memory_limit, devices,
        checkpoint, resume)
    if (!is.null(request$completed)) return(request$completed)
    paths <- request$paths
    knn <- full_backed_knn_stage(data, policy$n_neighbors,
        backend, workers, paths, memory_limit, devices,
        request$state)
    graph_stage <- full_backed_tsne_graph_stage(knn$object,
        policy$perplexity, paths, memory_limit, workers,
        settings$keep_knn, knn$state)
    prepared_stage <- full_backed_init_stage(data, settings$Y_init,
        paths, settings$n_components, backend, workers,
        settings$seed, memory_limit, devices, TRUE,
        graph_stage$state)
    keep_knn <- settings$keep_knn
    settings$perplexity <- policy$perplexity
    settings$n_threads <- if (backend == "metal") 1L else workers
    optimized <- full_backed_tsne_fit(graph_stage$object,
        prepared_stage$object, paths, settings, extra, controls,
        memory_limit, devices, landmark_method, checkpoint)
    fit <- full_backed_pipeline_record(optimized$fit, data,
        knn$object, graph_stage$object, prepared_stage$object,
        knn$seconds, graph_stage$seconds, prepared_stage$seconds,
        optimized$seconds, workers, settings$n_threads,
        backend, keep_knn, memory_limit)
    full_backed_checkpoint_finalize(prepared_stage$state, fit,
        paths, keep_knn)
}
