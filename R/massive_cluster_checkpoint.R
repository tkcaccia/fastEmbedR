massive_cluster_workflow_checkpoint <- function(x, plan, paths,
        controls, checkpoint, resume, reused) {
    massive_checkpoint_validate_controls(checkpoint, resume)
    if (!checkpoint) return(NULL)
    path <- paste0(paths$prefix, ".workflow.checkpoint.rds")
    signature <- massive_checkpoint_signature(list(version = 1L,
        package_version = as.character(utils::packageVersion("fastEmbedR")),
        source = massive_checkpoint_source_identity(x),
        output = paths$prefix, landmarks = plan$count,
        controls = controls,
        reused = if (is.null(reused)) NULL else list(
            selection = reused$selection$indices_signature,
            graph = reused$graph$output_identity)))
    if (resume) {
        state <- tryCatch(readRDS(path), error = function(e) NULL)
        if (!is.list(state) ||
            !identical(state$signature, signature) ||
            !is.character(state$stage) || length(state$stage) != 1L ||
            is.na(state$stage) ||
            !state$stage %in% c("initial", "selected", "graph", "knn")) {
            stop("Cluster workflow checkpoint does not match the ",
                "current request.", call. = FALSE)
        }
    } else {
        if (file.exists(path)) stop(
            "Cluster workflow checkpoint exists; use `resume = TRUE`.",
            call. = FALSE)
        state <- list(signature = signature, stage = "initial")
        massive_checkpoint_write(state, path)
    }
    list(path = path, state = state)
}

massive_cluster_workflow_controls <- function(k, method, backend,
        workers, devices, landmark_method, knn_method, chunk_rows,
        memory_limit, resolution, n_iterations, n_runs, steps,
        seed, checkpoint_every) {
    list(k = k, method = method, backend = backend,
        workers = workers, devices = devices,
        landmark_method = landmark_method, knn_method = knn_method,
        chunk_rows = chunk_rows, memory_limit = memory_limit,
        resolution = resolution, n_iterations = n_iterations,
        n_runs = n_runs, steps = steps, seed = seed,
        checkpoint_every = checkpoint_every)
}

massive_cluster_checkpoint_controls <- function(checkpoint, resume,
        checkpoint_every) {
    massive_checkpoint_validate_controls(checkpoint, resume)
    every <- integer_scalar(checkpoint_every)
    if (is.na(every) || every < 1L) stop(
        "`checkpoint_every` must be a positive integer.", call. = FALSE)
    every
}

massive_cluster_workflow_report <- function(x, plan, paths, backend,
        method) {
    walktrap_bytes <- 16 * plan$count^2 * (method == "walktrap")
    estimate <- 168 * 1024^2 +
        plan$count * plan$budget$per_landmark + walktrap_bytes
    message("EXPERIMENTAL landmark clustering: ", x$nrow,
        " rows; ", plan$count, " landmarks; backend=", backend,
        "; estimated RAM=", round(estimate / 1024^3, 2),
        " GiB; disk=", round(paths$disk_bytes / 1024^3, 2), " GiB")
    list(walktrap_bytes = walktrap_bytes,
        reference_ram_estimate_bytes = estimate)
}

massive_cluster_reference_graph <- function(x, selection, plan, k,
        backend, workers, workflow) {
    stage <- workflow$state$stage %||% "initial"
    if (stage %in% c("graph", "knn")) {
        graph <- tryCatch(validate_fastembedr_graph(
            workflow$state$graph), error = function(e) NULL)
        if (is.null(graph) || graph$n_vertices != plan$count) stop(
            "Saved landmark graph is invalid; it was not rebuilt.",
            call. = FALSE)
        return(list(graph = graph, workflow = workflow))
    }
    reference <- massive_read_rows_cpp(selection$data, 1,
        plan$count, 8 * plan$count * x$ncol)
    graph <- knn_graph(reference, k = k, backend = backend,
        n.cores = workers)
    workflow <- massive_workflow_stage(workflow, "graph", list(
        graph = graph))
    list(graph = graph, workflow = workflow)
}

massive_cluster_query_knn <- function(x, selection, plan, paths,
        k, backend, workers, method, chunk_rows, memory_limit,
        workflow, resume, checkpoint, reused) {
    stage <- workflow$state$stage %||% "initial"
    if (stage == "knn") {
        knn <- workflow$state$knn
        massive_embedding_checked_graph(x, selection, knn,
            plan$count, k, backend)
        return(list(knn = knn, workflow = workflow))
    }
    sidecar <- paste0(paths$knn, if (length(plan$devices) > 1L)
        ".multigpu.checkpoint.rds" else ".checkpoint.rds")
    knn_workers <- if (length(plan$devices) > 1L) 1L else workers
    knn <- if (!is.null(reused)) reused$graph else
        massive_landmark_knn(x, selection, k, paths$knn,
            backend = backend, n.cores = knn_workers,
            devices = plan$devices, method = method,
            chunk_rows = chunk_rows, memory_limit = memory_limit,
            checkpoint = checkpoint,
            resume = resume && file.exists(sidecar))
    massive_embedding_checked_graph(x, selection, knn,
        plan$count, k, backend)
    workflow <- massive_workflow_stage(workflow, "knn", list(knn = knn))
    list(knn = knn, workflow = workflow)
}

massive_cluster_precompute <- function(x, plan, paths, k, backend,
        workers, landmark_method, knn_method, chunk_rows,
        memory_limit, seed, workflow, checkpoint, resume, reused) {
    request <- list(count = plan$count, chunk_rows = chunk_rows,
        memory_limit = memory_limit, landmark_method = landmark_method)
    selected <- massive_workflow_selection(x, request, paths,
        seed, workflow, reused)
    built <- massive_cluster_reference_graph(x, selected$selection,
        plan, k, backend, workers, selected$workflow)
    searched <- massive_cluster_query_knn(x, selected$selection,
        plan, paths, k, backend, workers, knn_method,
        chunk_rows, memory_limit, built$workflow, resume,
        checkpoint, reused)
    list(selection = selected$selection, graph = built$graph,
        knn = searched$knn, workflow = searched$workflow)
}
