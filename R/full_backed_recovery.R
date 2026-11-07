full_backed_checkpoint_open <- function(data, paths, controls,
        checkpoint, resume) {
    massive_checkpoint_validate_controls(checkpoint, resume)
    if (!checkpoint) return(NULL)
    signature <- massive_checkpoint_signature(list(
        version = 1L,
        implementation = massive_checkpoint_implementation_identity(),
        input = massive_checkpoint_source_identity(data),
        paths = paths, controls = controls))
    if (!resume) {
        if (file.exists(paths$checkpoint)) stop(
            "One-call checkpoint already exists; use `resume = TRUE`.",
            call. = FALSE)
        state <- list(version = 1L, signature = signature,
            stages = list(), path = paths$checkpoint)
        massive_checkpoint_write(state, paths$checkpoint)
        return(state)
    }
    state <- tryCatch(readRDS(paths$checkpoint),
        error = function(e) NULL)
    if (!is.list(state) || !identical(state$version, 1L) ||
        !identical(state$signature, signature) ||
        !identical(state$path, paths$checkpoint) ||
        !is.list(state$stages)) stop(
        "One-call checkpoint does not match source or controls.",
        call. = FALSE)
    state
}

full_backed_checkpoint_stage <- function(state, name, object,
        files, seconds) {
    if (is.null(state)) return(NULL)
    state$stages[[name]] <- list(object = object,
        identity = massive_checkpoint_input_files_identity(files),
        seconds = seconds)
    massive_checkpoint_write(state, state$path)
    state
}

full_backed_checkpoint_get <- function(state, name, files) {
    if (is.null(state)) return(NULL)
    stage <- state$stages[[name]]
    if (is.null(stage)) return(NULL)
    identity <- tryCatch(massive_checkpoint_input_files_identity(files),
        error = function(e) NULL)
    if (!identical(identity, stage$identity)) stop(
        "One-call checkpoint artifact changed or is missing.",
        call. = FALSE)
    stage
}

full_backed_checkpoint_finish <- function(state) {
    if (is.null(state)) return(invisible(NULL))
    if (unlink(state$path) != 0L || file.exists(state$path)) stop(
        "Could not retire completed one-call checkpoint.",
        call. = FALSE)
    invisible(NULL)
}

full_backed_knn_stage <- function(data, k, backend, workers,
        paths, memory_limit, devices, state) {
    files <- paste0(paths$knn,
        c(".indices.u32", ".distances.f32"))
    saved <- full_backed_checkpoint_get(state, "knn", files)
    if (!is.null(saved)) return(list(object = saved$object,
        seconds = saved$seconds, state = state))
    if (any(file.exists(files))) stop(
        "Completed one-call KNN lacks a stage checkpoint.",
        call. = FALSE)
    sidecar <- paste0(paths$knn, ".checkpoint.rds")
    started <- proc.time()[[3L]]
    neighbors <- precompute_knn(data, k = k, backend = backend,
        n.cores = workers, storage = "file_backed", output = paths$knn,
        memory_limit = memory_limit, devices = devices,
        checkpoint = !is.null(state),
        resume = !is.null(state) && file.exists(sidecar))
    seconds <- proc.time()[[3L]] - started
    state <- full_backed_checkpoint_stage(state, "knn",
        neighbors, files, seconds)
    list(object = neighbors, seconds = seconds, state = state)
}

full_backed_umap_graph_stage <- function(neighbors, paths,
        memory_limit, keep_knn, workers, state) {
    started <- proc.time()[[3L]]
    if (is.null(state)) return(list(
        object = massive_umap_fuzzy_graph(neighbors,
            output = paths$graph, memory_limit = memory_limit,
            sort_method = "bucketed", consume_knn = !keep_knn),
        seconds = proc.time()[[3L]] - started, state = NULL))
    membership_files <- c(neighbors$indices_path,
        paste0(paths$knn, ".weights.f32"))
    saved <- full_backed_checkpoint_get(state, "membership",
        membership_files)
    if (is.null(saved)) {
        weight <- membership_files[[2L]]
        if (file.exists(weight)) stop(
            "Completed UMAP memberships lack a stage checkpoint.",
            call. = FALSE)
        membership <- massive_umap_memberships(neighbors,
            checkpoint = TRUE,
            resume = file.exists(paste0(weight,
                ".checkpoint.rds")), n.cores = workers)
        state <- full_backed_checkpoint_stage(state, "membership",
            membership, membership_files,
            proc.time()[[3L]] - started)
    } else membership <- saved$object
    membership_seconds <- state$stages$membership$seconds
    result <- full_backed_checkpoint_graph(state, "graph",
        paths$graph, ".fuzzy.rds", function(resume) {
            massive_umap_fuzzy_graph(membership,
                output = paths$graph, memory_limit = memory_limit,
                sort_method = "global", checkpoint = TRUE,
                resume = resume)
        })
    list(object = result$object,
        seconds = membership_seconds + result$seconds,
        state = result$state)
}

full_backed_checkpoint_graph <- function(state, stage, prefix,
        manifest_suffix, build) {
    files <- paste0(prefix, c(".offsets.u64", ".indices.u32",
        ".weights.f32", manifest_suffix))
    saved <- full_backed_checkpoint_get(state, stage, files)
    if (!is.null(saved)) return(list(object = saved$object,
        seconds = saved$seconds, state = state))
    sidecar <- paste0(prefix, ".sort-checkpoint.rds")
    if (any(file.exists(files))) stop(
        "Completed one-call graph lacks a stage checkpoint.",
        call. = FALSE)
    started <- proc.time()[[3L]]
    graph <- build(file.exists(sidecar))
    seconds <- proc.time()[[3L]] - started
    state <- full_backed_checkpoint_stage(state, stage,
        graph, files, seconds)
    list(object = graph, seconds = seconds, state = state)
}

full_backed_tsne_graph_stage <- function(neighbors, perplexity,
        paths, memory_limit, workers, keep_knn, state) {
    started <- proc.time()[[3L]]
    if (is.null(state)) return(list(
        object = massive_tsne_affinities(neighbors,
            perplexity = perplexity, output = paths$graph,
            memory_limit = memory_limit, n.cores = workers,
            sort_method = "bucketed", consume_knn = !keep_knn),
        seconds = proc.time()[[3L]] - started, state = NULL))
    result <- full_backed_checkpoint_graph(state, "graph",
        paths$graph, ".affinity.rds", function(resume) {
            massive_tsne_affinities(neighbors,
                perplexity = perplexity, output = paths$graph,
                memory_limit = memory_limit, n.cores = workers,
                sort_method = "global", checkpoint = TRUE,
                resume = resume)
        })
    list(object = result$object,
        seconds = result$seconds,
        state = result$state)
}

full_backed_init_stage <- function(data, init, paths, dimensions,
        backend, workers, seed, memory_limit, devices, tsne, state,
        graph = NULL) {
    files <- if (!is.null(init)) init$path else if (!tsne) {
        paths$spectral
    } else c(paths$pca, paste0(paths$pca, ".manifest.rds"),
        sub("\\.f32$", ".init.f32", paths$pca))
    saved <- full_backed_checkpoint_get(state, "init", files)
    if (!is.null(saved)) return(list(object = saved$object,
        seconds = saved$seconds, state = state))
    generated <- if (tsne) paths$pca else paths$spectral
    if (is.null(init) && file.exists(generated)) stop(
        "Completed one-call initialization lacks a stage checkpoint.",
        call. = FALSE)
    if (is.null(init) && !tsne &&
        file.exists(paste0(generated, ".part"))) {
        if (is.null(state) || unlink(paste0(generated, ".part")) != 0L)
            stop("Could not retire incomplete spectral output.",
                call. = FALSE)
    }
    sidecar <- paste0(paths$pca, ".checkpoint.rds")
    started <- proc.time()[[3L]]
    prepared <- full_backed_pipeline_init(data, init, paths,
        dimensions, backend, workers, seed, memory_limit,
        devices, graph = graph, tsne = tsne,
        checkpoint = !is.null(state),
        resume = !is.null(state) && file.exists(sidecar))
    seconds <- proc.time()[[3L]] - started
    state <- full_backed_checkpoint_stage(state, "init",
        prepared, files, seconds)
    list(object = prepared, seconds = seconds, state = state)
}

full_backed_checkpoint_completed <- function(state, paths,
        keep_knn) {
    saved <- full_backed_checkpoint_get(state, "fit",
        paths$output)
    if (is.null(saved)) return(NULL)
    full_backed_checkpoint_finalize(state, saved$object,
        paths, keep_knn)
}

full_backed_checkpoint_finalize <- function(state, fit, paths,
        keep_knn) {
    if (is.null(state)) return(fit)
    if (is.null(state$stages$fit)) state <-
        full_backed_checkpoint_stage(state, "fit", fit,
            paths$output, NA_real_)
    if (!keep_knn) {
        generated <- paste0(paths$knn,
            c(".indices.u32", ".distances.f32",
                ".weights.f32"))
        existing <- generated[file.exists(generated)]
        if (length(existing) && unlink(existing) != 0L) stop(
            "Could not retire one-call KNN files.", call. = FALSE)
        if (any(file.exists(generated))) stop(
            "One-call KNN files remain after cleanup.", call. = FALSE)
    }
    full_backed_checkpoint_finish(state)
    fit
}

full_backed_pipeline_start <- function(data, output, dimensions,
        memory_limit, init, controls, checkpoint, resume,
        keep_knn) {
    paths <- full_backed_pipeline_paths(data, output,
        dimensions, memory_limit, init, resume)
    state <- full_backed_checkpoint_open(data, paths,
        c(controls, list(init = if (is.null(init)) NULL else
            massive_checkpoint_source_identity(init))),
        checkpoint, resume)
    completed <- if (resume) full_backed_checkpoint_completed(
        state, paths, keep_knn) else NULL
    list(paths = paths, state = state, completed = completed)
}
