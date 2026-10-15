massive_validate_devices <- function(devices, backend, nrow) {
    if (is.null(devices)) return(NULL)
    if (backend != "cuda") {
        stop("`devices` requires CUDA; no backend fallback was used.",
            call. = FALSE)
    }
    count <- massive_cuda_device_count_cpp()
    if (!is.numeric(devices) || !length(devices) ||
        any(!is.finite(devices)) || any(devices != floor(devices)) ||
        any(devices < 0) || any(devices >= count) ||
        anyDuplicated(devices) || length(devices) > nrow) {
        stop("`devices` must name distinct available CUDA devices.",
            call. = FALSE)
    }
    as.integer(devices)
}

massive_cuda_device_uuids <- function(devices) {
    if (!requireNamespace("gpuinfo", quietly = TRUE) ||
        utils::packageVersion("gpuinfo") < "0.2.0" ||
        !requireNamespace("jsonlite", quietly = TRUE)) stop(
        "CUDA device identity needs gpuinfo >= 0.2.0 and jsonlite.",
        call. = FALSE)
    manifest <- jsonlite::fromJSON(gpuinfo::hardware_manifest(
        memory = "none", as = "json"), simplifyVector = FALSE)
    massive_cuda_manifest_uuids(manifest, devices)
}

massive_cuda_manifest_uuids <- function(manifest, devices) {
    visible <- Filter(function(item) identical(item$backend, "cuda") &&
        isTRUE(item$process_visible), manifest$devices)
    ids <- vapply(visible, function(item) as.integer(item$logical_id), 0L)
    uuids <- vapply(visible, function(item) item$uuid %||% "", "")
    if (anyNA(ids) || anyNA(uuids) ||
        anyDuplicated(ids) || anyDuplicated(uuids) ||
        !all(devices %in% ids) || any(!nzchar(uuids))) stop(
        "gpuinfo did not identify every requested CUDA device.",
        call. = FALSE)
    unname(uuids[match(devices, ids)])
}

massive_row_shards <- function(path, nrow, dimensions, devices) {
    bounds <- floor(seq(0, nrow, length.out = length(devices) + 1L))
    lapply(seq_along(devices), function(i) {
        range <- c(bounds[i] + 1, bounds[i + 1L])
        shard <- sub("\\.f32$", sprintf(".gpu%d.f32", devices[i]),
            path, ignore.case = TRUE)
        bytes <- (diff(range) + 1) * dimensions * 4
        list(device = devices[i], row_range = range,
            output = shard, bytes = bytes)
    })
}

massive_multigpu_signature <- function(arguments, devices, tasks, path) {
    graph <- arguments$graph
    settings <- arguments[setdiff(names(arguments),
        c("fit", "graph", "selection", "output"))]
    settings$devices <- devices
    settings$gpu_identity <- massive_cuda_device_uuids(devices)
    settings$shards <- lapply(tasks, function(task) {
        task[c("device", "row_range", "bytes")]
    })
    settings$package_version <- as.character(
        utils::packageVersion("fastEmbedR"))
    settings$implementation <-
        massive_checkpoint_implementation_identity()
    settings$output <- path
    massive_projection_signature(arguments$fit, graph,
        list(output_bytes = graph$nrow * ncol(arguments$fit$layout) * 4),
        settings, massive_reference_indices(arguments$selection, graph))
}

massive_multigpu_checkpoint <- function(path, signature,
                                        checkpoint, resume) {
    sidecar <- paste0(path, ".multigpu.checkpoint.rds")
    if (resume) {
        state <- tryCatch(readRDS(sidecar), error = function(e) NULL)
        if (!is.list(state) ||
            !identical(state$signature, signature)) {
            stop("Multi-GPU checkpoint does not match current inputs.",
                call. = FALSE)
        }
    } else if (file.exists(sidecar)) {
        stop("Multi-GPU checkpoint exists; use `resume = TRUE`.",
            call. = FALSE)
    }
    sidecar
}

massive_multigpu_record_shard <- function(task) {
    record <- list(task = task[c("device", "row_range", "bytes")],
        identity = massive_checkpoint_file_identity(task$output))
    if (record$identity$bytes != task$bytes) stop(
        "Completed CUDA shard has an unexpected byte count.",
        call. = FALSE)
    massive_checkpoint_write(record,
        paste0(task$output, ".result.rds"))
}

massive_multigpu_verify_shard <- function(task, partial, sidecar) {
    path <- task$output
    manifest <- paste0(path, ".result.rds")
    saved <- if (file.exists(manifest)) {
        tryCatch(readRDS(manifest), error = function(e) NULL)
    } else NULL
    size <- file.info(path)$size
    if (is.na(size) || size != task$bytes ||
        file.exists(partial) || file.exists(sidecar) ||
        !is.list(saved) || !identical(saved$task,
            task[c("device", "row_range", "bytes")]) ||
        !identical(saved$identity,
            massive_checkpoint_file_identity(path))) stop(
        "Completed CUDA shard is inconsistent.", call. = FALSE)
}

massive_multigpu_task_status <- function(task, resume) {
    path <- task$output
    partial <- paste0(path, ".part")
    sidecar <- paste0(path, ".checkpoint.rds")
    manifest <- paste0(path, ".result.rds")
    if (resume && file.exists(path)) {
        massive_multigpu_verify_shard(task, partial, sidecar)
        task$status <- "complete"
        task$remaining <- 0
        return(task)
    }
    if (file.exists(manifest)) stop(
        "CUDA shard manifest exists without a completed output.",
        call. = FALSE)
    partial_resume <- resume && file.exists(partial) &&
        file.exists(sidecar)
    if (resume && xor(file.exists(partial), file.exists(sidecar))) {
        stop("CUDA projection shard has incomplete checkpoint files.",
            call. = FALSE)
    }
    massive_float_output(path, task$bytes, resume = partial_resume)
    task$status <- if (partial_resume) "partial" else "new"
    committed <- 0
    if (partial_resume) {
        state <- tryCatch(readRDS(sidecar), error = function(e) NULL)
        rows <- if (is.list(state)) state$completed_rows else NULL
        if (!is.list(state) || !is.numeric(rows) ||
            length(rows) != 1L || !is.finite(rows) ||
            rows != floor(rows)) {
            stop("CUDA projection shard checkpoint is invalid.",
                call. = FALSE)
        }
        committed <- rows * task$bytes / (diff(task$row_range) + 1)
        size <- file.info(partial)$size
        if (committed < 0 || committed > task$bytes ||
            is.na(size) || size < committed || size > task$bytes) {
            stop("CUDA projection shard checkpoint is invalid.",
                call. = FALSE)
        }
    }
    task$remaining <- task$bytes - committed
    task
}

massive_multigpu_disk_check <- function(path, bytes, tasks) {
    partial <- paste0(path, ".part")
    size <- if (file.exists(partial)) file.info(partial)$size else 0
    merge_remaining <- bytes - floor(size / 4194304) * 4194304
    needed <- merge_remaining + sum(vapply(tasks, `[[`, 0, "remaining"))
    if (!is.finite(needed) || needed < 0 ||
        needed > 0.8 * massive_disk_available_cpp(path)) {
        stop("CUDA projection resume exceeds free disk space.",
            call. = FALSE)
    }
    invisible(NULL)
}

massive_verify_merged_shards <- function(paths, output, bytes) {
    sizes <- file.info(paths)$size
    merged_size <- file.info(output)$size
    if (anyNA(sizes) || anyNA(merged_size) ||
        !identical(as.numeric(sizes), as.numeric(bytes)) ||
        merged_size != sum(bytes)) stop(
        "Completed CUDA merge has inconsistent file sizes.",
        call. = FALSE)
    merged <- file(output, "rb")
    on.exit(close(merged))
    for (i in seq_along(paths)) {
        shard <- file(paths[[i]], "rb")
        tryCatch({
            remaining <- bytes[[i]]
            while (remaining > 0) {
                count <- as.integer(min(remaining, 4194304))
                source <- readBin(shard, raw(), n = count)
                target <- readBin(merged, raw(), n = count)
                if (length(source) != count ||
                    !identical(source, target)) stop(
                    "Completed CUDA merge differs from its shards.",
                    call. = FALSE)
                remaining <- remaining - count
            }
        }, finally = close(shard))
    }
    invisible(NULL)
}

massive_multigpu_output <- function(output, bytes, resume) {
    if (resume && is.character(output) && length(output) == 1L &&
        !is.na(output) && file.exists(output)) {
        path <- file.path(normalizePath(dirname(output), mustWork = TRUE),
            basename(output))
        size <- file.info(path)$size
        if (tolower(tools::file_ext(path)) != "f32" || is.na(size) ||
            file.exists(paste0(path, ".part")) ||
            size != bytes) {
            stop("Completed CUDA projection output is inconsistent.",
                call. = FALSE)
        }
        return(path)
    }
    massive_float_output(output, bytes, resume = resume &&
        file.exists(paste0(output, ".part")))
}

massive_projection_device_worker <- function(task, arguments,
        package_path, implementation) {
    massive_cuda_worker_identity(package_path, implementation)
    if (!requireNamespace("float", quietly = TRUE)) {
        stop("CUDA landmark projection requires the float package.")
    }
    massive_cuda_select_cpp(task$device)
    info <- massive_cuda_memory_cpp()
    if (info$device != task$device) {
        stop("CUDA worker did not select the requested device.")
    }
    arguments$output <- task$output
    arguments$row_range <- task$row_range
    arguments$resume <- identical(task$status, "partial")
    result <- do.call(massive_project_landmarks, arguments)
    if (result$resources$gpu_device != task$device) {
        stop("Projection metadata reports a different CUDA device.")
    }
    if (isTRUE(arguments$checkpoint))
        massive_multigpu_record_shard(task)
    result
}

massive_cuda_parallel_tasks <- function(tasks, worker, arguments) {
    if (!length(tasks)) return(list())
    package_path <- normalizePath(find.package("fastEmbedR"))
    implementation <- massive_checkpoint_implementation_identity()
    cluster <- parallel::makePSOCKcluster(length(tasks))
    on.exit(parallel::stopCluster(cluster))
    bootstrap <- evalq(function(path) {
        .libPaths(c(path, .libPaths()))
    }, baseenv())
    parallel::clusterCall(cluster, bootstrap, dirname(package_path))
    parallel::clusterApply(cluster, tasks, worker, arguments = arguments,
        package_path = package_path, implementation = implementation)
}

massive_cuda_worker_identity <- function(package_path, implementation) {
    if (!identical(normalizePath(find.package("fastEmbedR")),
            package_path) ||
        !identical(massive_checkpoint_implementation_identity(),
            implementation)) stop(
        "CUDA worker loaded a different fastEmbedR implementation.",
        call. = FALSE)
}

massive_pca_covariance_device_worker <- function(task, arguments,
        package_path, implementation) {
    massive_cuda_worker_identity(package_path, implementation)
    massive_cuda_select_cpp(task$device)
    view <- massive_matrix_rows(arguments$source,
        task$row_range[[1L]], diff(task$row_range) + 1)
    resources <- massive_pca_resources(view, arguments$rank, 1L,
        arguments$chunk_rows, arguments$memory_limit, "cuda")
    if (resources$gpu_device != task$device) stop(
        "PCA statistics used a different CUDA device.")
    identity <- massive_checkpoint_source_identity(view)
    started <- proc.time()[[3L]]
    stats <- massive_pca_covariance_cuda_cpp(view,
        resources$chunk_rows, 1L, arguments$center)
    if (!identical(identity, massive_checkpoint_source_identity(view)))
        stop("PCA source changed during CUDA statistics.")
    c(stats, list(rows = as.double(view$nrow), device = task$device,
        row_range = c(view$first, view$first + view$nrow - 1),
        seconds = unname(proc.time()[[3L]] - started),
        estimated_ram_bytes = resources$peak_ram_bytes,
        estimated_vram_bytes = resources$peak_vram_bytes))
}

massive_pca_combine_covariance <- function(parts, center) {
    rows <- vapply(parts, `[[`, 0, "rows")
    n <- sum(rows)
    p <- length(parts[[1L]]$center)
    means <- if (center) Reduce(`+`, Map(function(part, count)
        part$center * count, parts, rows)) / n else numeric(p)
    cross <- matrix(0, p, p)
    for (i in seq_along(parts)) {
        part <- parts[[i]]
        if (length(part$center) != p ||
            !identical(dim(part$covariance), c(p, p)) ||
            any(!is.finite(c(part$center, part$covariance)))) stop(
            "CUDA PCA shard statistics are invalid.", call. = FALSE)
        delta <- part$center - means
        cross <- cross + (rows[i] - 1) * part$covariance +
            rows[i] * tcrossprod(delta)
    }
    list(center = means, covariance = cross / (n - 1),
        fit_devices = vapply(parts, `[[`, 0L, "device"),
        fit_per_device = lapply(parts, function(part)
            part[c("device", "row_range", "rows", "seconds",
                "estimated_ram_bytes", "estimated_vram_bytes")]))
}

massive_pca_fit_tasks <- function(source, devices) {
    if (source$nrow < 2 * length(devices)) stop(
        "Multi-GPU PCA needs at least two rows per device.",
        call. = FALSE)
    bounds <- floor(seq(0, source$nrow,
        length.out = length(devices) + 1L))
    lapply(seq_along(devices), function(i) list(
        device = devices[i], row_range = c(bounds[i] + 1,
            bounds[i + 1L])))
}

massive_pca_fit_budget <- function(resources, devices) {
    available <- resources$memory_limit_bytes -
        resources$peak_ram_bytes
    if (!is.finite(available) || available <= 0) stop(
        "Multi-GPU PCA workers exceed the RAM budget.", call. = FALSE)
    available / length(devices)
}

massive_pca_covariance_devices <- function(source, rank, center,
        resources) {
    devices <- resources$gpu_devices
    tasks <- massive_pca_fit_tasks(source, devices)
    identity <- massive_checkpoint_source_identity(source)
    args <- list(source = source, rank = rank, center = center,
        chunk_rows = resources$chunk_rows,
        memory_limit = massive_pca_fit_budget(resources, devices))
    started <- proc.time()[[3L]]
    parts <- massive_cuda_parallel_tasks(tasks,
        massive_pca_covariance_device_worker, args)
    if (!identical(identity, massive_checkpoint_source_identity(source)))
        stop("PCA source changed during multi-GPU fit.", call. = FALSE)
    if (!identical(vapply(parts, `[[`, 0L, "device"), devices))
        stop("Multi-GPU PCA returned unexpected devices.", call. = FALSE)
    expected_rows <- vapply(tasks, function(task)
        as.double(diff(task$row_range) + 1), 0)
    ranges <- Map(function(part, task) identical(part$row_range,
        task$row_range), parts, tasks)
    if (!all(unlist(ranges)) ||
        !identical(vapply(parts, `[[`, 0, "rows"), expected_rows))
        stop("Multi-GPU PCA fit did not cover every row.",
            call. = FALSE)
    combined <- massive_pca_combine_covariance(parts, center)
    combined$fit_statistics_seconds <- unname(
        proc.time()[[3L]] - started)
    combined
}

massive_pca_wide_action_device_worker <- function(task, arguments,
        package_path, implementation) {
    massive_cuda_worker_identity(package_path, implementation)
    massive_cuda_select_cpp(task$device)
    view <- massive_matrix_rows(arguments$source,
        task$row_range[[1L]], diff(task$row_range) + 1)
    resources <- massive_pca_resources(view, arguments$rank, 1L,
        arguments$chunk_rows, arguments$memory_limit, "cuda")
    if (resources$gpu_device != task$device) stop(
        "PCA sketch used a different CUDA device.")
    identity <- massive_checkpoint_source_identity(view)
    started <- proc.time()[[3L]]
    action <- massive_pca_wide_action_cuda_cpp(view,
        arguments$center, arguments$scale, arguments$basis,
        resources$chunk_rows)
    if (!identical(identity, massive_checkpoint_source_identity(view)))
        stop("PCA source changed during CUDA sketch.")
    list(action = action, device = task$device,
        row_range = c(view$first, view$first + view$nrow - 1),
        rows = as.double(view$nrow),
        seconds = unname(proc.time()[[3L]] - started),
        estimated_ram_bytes = resources$peak_ram_bytes,
        estimated_vram_bytes = resources$peak_vram_bytes)
}

massive_pca_wide_action_devices <- function(source, rank, center,
        scale, basis, resources) {
    devices <- resources$gpu_devices
    tasks <- massive_pca_fit_tasks(source, devices)
    identity <- massive_checkpoint_source_identity(source)
    args <- list(source = source, rank = rank, center = center,
        scale = scale, basis = basis,
        chunk_rows = resources$chunk_rows,
        memory_limit = massive_pca_fit_budget(resources, devices))
    started <- proc.time()[[3L]]
    parts <- massive_cuda_parallel_tasks(tasks,
        massive_pca_wide_action_device_worker, args)
    if (!identical(identity, massive_checkpoint_source_identity(source)))
        stop("PCA source changed during multi-GPU sketch.",
            call. = FALSE)
    if (!identical(vapply(parts, `[[`, 0L, "device"), devices) ||
        any(vapply(parts, function(part)
            !identical(dim(part$action), dim(basis)) ||
            any(!is.finite(part$action)), logical(1L)))) stop(
        "Multi-GPU PCA returned invalid sketch shards.",
        call. = FALSE)
    expected_rows <- vapply(tasks, function(task)
        as.double(diff(task$row_range) + 1), 0)
    ranges <- Map(function(part, task) identical(part$row_range,
        task$row_range), parts, tasks)
    if (!all(unlist(ranges)) ||
        !identical(vapply(parts, `[[`, 0, "rows"), expected_rows))
        stop("Multi-GPU PCA sketch did not cover every row.",
            call. = FALSE)
    list(value = Reduce(`+`, lapply(parts, `[[`, "action")),
        per_device = lapply(parts, function(part) part[
            c("device", "row_range", "rows", "seconds",
                "estimated_ram_bytes",
                "estimated_vram_bytes")]),
        seconds = unname(proc.time()[[3L]] - started))
}

massive_pca_device_worker <- function(task, arguments, package_path,
        implementation) {
    massive_cuda_worker_identity(package_path, implementation)
    massive_cuda_select_cpp(task$device)
    source <- massive_matrix_rows(arguments$source,
        task$row_range[[1L]], diff(task$row_range) + 1)
    resources <- massive_pca_resources(source, arguments$rank, 1L,
        arguments$chunk_rows, arguments$memory_limit, "cuda")
    if (resources$gpu_device != task$device) stop(
        "PCA resource metadata reports a different CUDA device.")
    fit <- arguments$fit
    saved <- massive_pca_shard_checkpoint(task, source, fit,
        resources, arguments$checkpoint)
    massive_pca_project_cuda_cpp(source, task$output, fit$loadings,
        fit$center, fit$scale, resources$chunk_rows,
        saved$rows, saved$progress, saved$resume)
    if (file.info(task$output)$size != task$bytes) stop(
        "CUDA PCA shard has an unexpected byte count.")
    if (isTRUE(arguments$checkpoint))
        massive_multigpu_record_shard(task)
    if (!is.null(saved$sidecar)) unlink(saved$sidecar)
    resources$status <- task$status
    resources
}

massive_pca_shard_checkpoint <- function(task, source, fit,
        resources, checkpoint) {
    if (!checkpoint) return(list(rows = 0, progress = NULL,
        resume = FALSE, sidecar = NULL))
    path <- task$output
    sidecar <- paste0(path, ".checkpoint.rds")
    fit_hash <- massive_checkpoint_signature(fit)
    signature <- massive_checkpoint_signature(list(
        source = massive_checkpoint_source_identity(source),
        output = path, model = fit_hash,
        chunk_rows = resources$chunk_rows, device = task$device))
    resume <- task$status == "partial"
    if (resume) {
        state <- massive_pca_checkpoint_load(sidecar, signature,
            source, ncol(fit$loadings), resources,
            paste0(path, ".part"), 1L, "cuda")
        if (!identical(state$fit_hash, fit_hash)) stop(
            "CUDA PCA shard model does not match the checkpoint.",
            call. = FALSE)
    } else {
        state <- list(signature = signature, stage = "fit", fit = fit,
            fit_hash = fit_hash,
            completed_rows = 0)
        massive_checkpoint_write(state, sidecar)
    }
    rows <- state$completed_rows
    progress <- function(completed) {
        state$completed_rows <- completed
        massive_checkpoint_write(state, sidecar)
    }
    list(rows = rows, progress = progress, resume = resume,
        sidecar = sidecar)
}

massive_pca_shards_check <- function(source, path, rank,
        output_bytes, devices, resume = FALSE) {
    origin <- source
    while (origin$format == "view") origin <- origin$source
    if (!origin$format %in% c("f32", "fbin")) stop(
        "Multi-GPU PCA needs a file-backed source.", call. = FALSE)
    tasks <- massive_row_shards(path, source$nrow, rank, devices)
    paths <- vapply(tasks, `[[`, "", "output")
    sidecars <- paste0(paths, ".checkpoint.rds")
    if (!resume && any(file.exists(c(paths,
        paste0(paths, ".part"), sidecars)))) stop(
        "CUDA PCA shard output already exists.", call. = FALSE)
    tasks <- lapply(tasks, massive_multigpu_task_status,
        resume = resume)
    massive_multigpu_disk_check(path, output_bytes, tasks)
    tasks
}

massive_pca_project_devices <- function(source, path, fit,
        resources, memory_limit, devices, checkpoint, resume) {
    rank <- ncol(fit$loadings)
    tasks <- massive_pca_shards_check(source, path, rank,
        resources$output_bytes, devices, resume)
    paths <- vapply(tasks, `[[`, "", "output")
    arguments <- list(source = source, fit = fit, rank = rank,
        chunk_rows = resources$chunk_rows,
        memory_limit = massive_memory_bytes(memory_limit) /
            length(devices), checkpoint = checkpoint)
    pending <- Filter(function(task) task$status != "complete", tasks)
    results <- massive_cuda_parallel_tasks(pending,
        massive_pca_device_worker, arguments)
    names(results) <- vapply(pending, `[[`, "", "output")
    per_device <- lapply(tasks, function(task) {
        results[[task$output]] %||% list(gpu_device = task$device,
            peak_ram_bytes = NA_real_, status = "reused")
    })
    massive_concat_word_files_cpp(paths, path,
        vapply(tasks, `[[`, 0, "bytes"),
        resume = resume && file.exists(paste0(path, ".part")))
    resources$gpu_devices <- devices
    resources$per_device <- per_device
    resources$shard_paths <- paths
    resources$parent_peak_ram_bytes <- resources$peak_ram_bytes
    resources$projection_peak_ram_bytes <- if (
            length(pending) < length(tasks)) {
        NA_real_
    } else resources$parent_peak_ram_bytes +
        sum(vapply(per_device, `[[`, 0, "peak_ram_bytes"))
    resources$peak_ram_bytes <- resources$projection_peak_ram_bytes
    resources
}

massive_project_pending_devices <- function(tasks, arguments) {
    pending <- Filter(function(task) task$status != "complete", tasks)
    results <- massive_cuda_parallel_tasks(pending,
        massive_projection_device_worker, arguments)
    names(results) <- vapply(pending, `[[`, "", "output")
    results
}

massive_knn_device_tasks <- function(x, k, output, devices) {
    bounds <- floor(seq(0, x$nrow, length.out = length(devices) + 1L))
    lapply(seq_along(devices), function(i) {
        rows <- c(bounds[i] + 1, bounds[i + 1L])
        list(device = devices[i], row_range = rows,
            output = paste0(output, ".gpu", devices[i]),
            bytes = (diff(rows) + 1) * k * 4)
    })
}

massive_knn_inputs_identity <- function(x, reference) {
    list(source = massive_checkpoint_source_identity(x),
        reference = massive_checkpoint_source_identity(reference))
}

massive_knn_inputs_unchanged <- function(identity, x, reference,
        phase) {
    if (!identical(identity, massive_knn_inputs_identity(x, reference)))
        stop("Multi-GPU KNN input changed during ", phase, ".",
            call. = FALSE)
}

massive_knn_device_worker <- function(task, arguments, package_path,
        implementation) {
    massive_cuda_worker_identity(package_path, implementation)
    massive_cuda_select_cpp(task$device)
    arguments$x <- massive_matrix_rows(arguments$x,
        task$row_range[1L], diff(task$row_range) + 1)
    identity <- massive_knn_inputs_identity(arguments$x,
        arguments$reference)
    arguments$output <- task$output
    coordinated_checkpoint <- isTRUE(arguments$checkpoint)
    if (identical(arguments$method, "sharded_ivf")) {
        arguments$checkpoint <- FALSE
        arguments$resume <- FALSE
    } else {
        arguments$resume <- identical(task$status, "partial")
    }
    result <- do.call(massive_landmark_knn, arguments)
    massive_knn_inputs_unchanged(identity, arguments$x,
        arguments$reference, "worker search")
    if (result$resources$gpu_device != task$device) {
        stop("KNN metadata reports a different CUDA device.")
    }
    paths <- c(result$indices_path, result$distances_path)
    result$completed_output_identity <-
        massive_checkpoint_input_files_identity(paths)
    result$restarted <- identical(task$status, "restart")
    if (coordinated_checkpoint) {
        massive_checkpoint_write(result,
            paste0(task$output, ".result.rds"))
    }
    result
}

massive_knn_device_result_check <- function(result, task, view,
        reference, k, method, paths) {
    if (!is.list(result) ||
        !identical(result$backend, "cuda") ||
        !identical(result$method, method) ||
        !identical(result$nrow, view$nrow) ||
        !identical(result$ncol, k) ||
        !identical(result$n_reference, reference$nrow) ||
        !identical(result$resources$gpu_device, task$device) ||
        !identical(normalizePath(c(result$indices_path,
            result$distances_path), mustWork = TRUE),
            normalizePath(unname(paths), mustWork = TRUE)) ||
        !identical(result$source_identity,
            massive_checkpoint_source_identity(view)) ||
        !identical(result$reference_identity,
            massive_checkpoint_source_identity(reference)) ||
        !identical(result$implementation_identity,
            massive_checkpoint_implementation_identity()) ||
        !identical(result$completed_output_identity,
            massive_checkpoint_input_files_identity(paths)) ||
        !identical(result$output_identity,
            massive_checkpoint_file_identity(paths)) ||
        !identical(as.numeric(file.info(paths)$size),
            rep(as.numeric(task$bytes), length(paths)))) stop(
            "Completed multi-GPU KNN shard is inconsistent.",
            call. = FALSE)
}

massive_knn_device_status <- function(task, x, reference, k,
        method, resume) {
    paths <- paste0(task$output,
        c(".indices.u32", ".distances.f32"))
    parts <- paste0(paths, ".part")
    sidecar <- paste0(task$output, ".checkpoint.rds")
    manifest <- paste0(task$output, ".result.rds")
    complete <- file.exists(paths)
    partial <- file.exists(parts)
    if (!resume && any(file.exists(c(paths, parts, sidecar,
        manifest)))) stop("Multi-GPU KNN shard output already exists.",
            call. = FALSE)
    if (resume && all(complete) && file.exists(manifest) &&
        !any(partial) && !file.exists(sidecar)) {
        result <- tryCatch(readRDS(manifest), error = function(e) NULL)
        view <- massive_matrix_rows(x, task$row_range[1L],
            diff(task$row_range) + 1)
        massive_knn_device_result_check(result, task, view,
            reference, k, method, paths)
        task$status <- "complete"
        task$result <- result
        task$remaining <- 0
        return(task)
    }
    if (resume && identical(method, "sharded_ivf") &&
        any(file.exists(c(paths, parts, sidecar, manifest)))) {
        task$status <- "restart"
        task$remaining <- 2 * task$bytes
        return(task)
    }
    if (resume && any(complete) && !file.exists(manifest)) {
        task$status <- "restart"
        task$remaining <- 2 * task$bytes
        return(task)
    }
    if (resume && (any(complete) || file.exists(manifest) ||
        any(partial) != file.exists(sidecar) ||
        (any(partial) && !all(partial)))) stop(
            "Multi-GPU KNN shard has incomplete checkpoint files.",
            call. = FALSE)
    task$status <- if (all(partial)) "partial" else "new"
    task$remaining <- 2 * task$bytes -
        if (all(partial)) sum(file.info(parts)$size) else 0
    if (!is.finite(task$remaining) || task$remaining < 0) stop(
        "Multi-GPU KNN shard has an invalid byte count.",
        call. = FALSE)
    task
}

massive_knn_merge_one <- function(inputs, path, bytes, resume) {
    staged <- paste0(path, ".merge")
    part <- paste0(staged, ".part")
    if (file.exists(path)) {
        if (!resume || any(file.exists(c(staged, part))) ||
            file.info(path)$size != sum(bytes)) stop(
                "Completed KNN merge is inconsistent.", call. = FALSE)
        massive_verify_merged_shards(inputs, path, bytes)
        return(invisible(NULL))
    }
    if (file.exists(staged)) {
        if (!resume || file.exists(part) ||
            file.info(staged)$size != sum(bytes)) stop(
                "Staged KNN merge is inconsistent.", call. = FALSE)
    } else {
        if (file.exists(part) && !resume) stop(
            "KNN merge staging files already exist.", call. = FALSE)
        massive_concat_word_files_cpp(inputs, staged, bytes,
            resume = file.exists(part))
    }
    if (resume) massive_verify_merged_shards(inputs, staged, bytes)
    if (!file.rename(staged, path)) stop(
        "Could not finalize merged KNN output.", call. = FALSE)
    invisible(NULL)
}

massive_knn_merge_shards <- function(results, paths, tasks,
                                    resume = FALSE) {
    merged <- stats::setNames(paste0(unname(paths), ".merge"),
        names(paths))
    if (!resume && (any(file.exists(merged)) ||
        any(file.exists(paste0(merged, ".part"))))) {
        stop("KNN merge staging files already exist.", call. = FALSE)
    }
    for (field in names(paths)) {
        inputs <- vapply(results, `[[`, "", paste0(field, "_path"))
        massive_knn_merge_one(inputs, paths[[field]],
            vapply(tasks, `[[`, 0, "bytes"), resume)
    }
}

massive_knn_devices_result <- function(results, setup, x,
                                        devices, merge_seconds, tasks) {
    out <- results[[1L]]
    out$indices_path <- setup$paths[["indices"]]
    out$distances_path <- setup$paths[["distances"]]
    out$nrow <- x$nrow
    out$source_identity <- massive_checkpoint_source_identity(x)
    references <- lapply(results, `[[`, "reference_identity")
    if (!all(vapply(references, identical, TRUE, references[[1L]]))) stop(
        "KNN shards used different references.", call. = FALSE)
    out$reference_identity <- references[[1L]]
    out$output_identity <- massive_checkpoint_file_identity(unname(setup$paths))
    out$implementation_identity <- massive_checkpoint_implementation_identity()
    out$gpu_devices <- devices
    out$gpu_identity <- massive_cuda_device_uuids(devices)
    out$shard_paths <- lapply(results, function(value)
        value[c("indices_path", "distances_path")])
    out$per_device <- Map(function(value, task) {
        c(list(query_row_range = task$row_range),
            value[c("resources", "build_seconds", "query_seconds",
                "pilot_recall", "pilot_id_recall",
                "pilot_recall_metric", "pilot_rows", "nlist", "nprobe",
                "ivf_settings", "restarted")])
    }, results, tasks)
    reference_bytes <- if (setup$reference$format == "memory") 0 else
        as.double(setup$reference$nrow) * setup$reference$ncol * 4
    out$resources <- list(output_bytes = setup$resources$output_bytes,
        reference_replication_factor = length(devices),
        minimum_reference_read_bytes = reference_bytes * length(devices),
        peak_ram_bytes = sum(vapply(results, function(value)
            value$resources$peak_ram_bytes, 0)),
        per_device = lapply(results, `[[`, "resources"))
    out$pilot_rows <- sum(vapply(results,
        function(value) as.numeric(value$pilot_rows), 0))
    recall <- vapply(results, `[[`, 0, "pilot_recall")
    out$pilot_recall <- if (all(is.finite(recall))) min(recall) else NA_real_
    out$pilot_id_recall <- min(vapply(results, function(value)
        value$pilot_id_recall %||% NA_real_, 0))
    for (field in c("nlist", "nprobe")) {
        values <- vapply(results,
            function(value) as.numeric(value[[field]]), 0)
        out[[field]] <- if (length(unique(values)) == 1L) {
            values[[1L]]
        } else NA_real_
    }
    out$build_seconds <- max(vapply(results, `[[`, 0, "build_seconds"))
    out$query_seconds <- max(vapply(results, `[[`, 0, "query_seconds"))
    out$merge_seconds <- merge_seconds
    out
}

massive_knn_devices_checkpoint <- function(x, setup, tasks,
        devices, arguments, resume) {
    source <- setup$reference
    while (identical(source$format, "view")) source <- source$source
    if (is.null(source$path)) stop(
        "Multi-GPU KNN checkpoint needs a file-backed reference.",
        call. = FALSE)
    controls <- arguments[setdiff(names(arguments),
        c("x", "reference", "checkpoint", "resume"))]
    signature <- massive_checkpoint_signature(list(version = 1L,
        implementation = massive_checkpoint_implementation_identity(),
        source = massive_checkpoint_source_identity(x),
        reference = massive_checkpoint_source_identity(setup$reference),
        output = setup$paths, controls = controls,
        devices = devices, gpu_identity = massive_cuda_device_uuids(devices),
        shards = lapply(tasks, function(task) {
            task[c("device", "row_range", "bytes")]
        })))
    prefix <- sub("\\.indices\\.u32$", "", setup$paths[["indices"]])
    finished <- paste0(prefix, ".multigpu.result.rds")
    sidecar <- paste0(prefix, ".multigpu.checkpoint.rds")
    if (!resume && file.exists(finished)) stop(
        "Completed multi-GPU KNN result already exists.", call. = FALSE)
    if (resume && !file.exists(sidecar)) {
        saved <- if (file.exists(finished))
            suppressWarnings(tryCatch(readRDS(finished),
                error = function(e) NULL)) else NULL
        if (!is.list(saved) ||
            !identical(saved$signature, signature) ||
            !identical(saved$output_identity,
                massive_checkpoint_file_identity(unname(setup$paths))))
            stop("Completed multi-GPU KNN result does not match.",
                call. = FALSE)
        massive_checkpoint_write(list(signature = signature),
            sidecar)
    }
    massive_multigpu_checkpoint(prefix, signature, TRUE, resume)
    if (!resume) massive_checkpoint_write(
        list(signature = signature), sidecar)
    for (task in tasks) if (identical(task$status, "restart")) {
        paths <- paste0(task$output, c(".indices.u32",
            ".distances.f32", ".indices.u32.part",
            ".distances.f32.part", ".checkpoint.rds",
            ".checkpoint.rds.tmp", ".result.rds", ".result.rds.tmp"))
        if (any(unlink(paths) != 0L)) stop(
            "Could not clear an orphaned multi-GPU KNN shard.",
            call. = FALSE)
    }
    sidecar
}

massive_knn_devices_disk_check <- function(setup, tasks) {
    paths <- setup$paths
    staged <- unlist(lapply(paths, function(path) {
        c(paste0(path, ".merge"), paste0(path, ".merge.part"))
    }))
    staged_bytes <- sum(file.info(staged)$size[
        file.exists(staged)])
    remaining <- sum(vapply(tasks, `[[`, 0, "remaining")) +
        sum(!file.exists(paths)) * setup$resources$output_bytes / 2 -
        staged_bytes
    if (!is.finite(remaining) || remaining < 0 ||
        remaining > 0.75 * massive_disk_available_cpp(paths[[1L]])) {
        stop("Multi-GPU KNN shards exceed free disk space.",
            call. = FALSE)
    }
}

massive_knn_devices_finish_checkpoint <- function(sidecar, prefix,
        output_identity) {
    if (is.null(sidecar)) return(invisible(NULL))
    massive_checkpoint_write(list(
        signature = readRDS(sidecar)$signature,
        output_identity = output_identity),
        paste0(prefix, ".multigpu.result.rds"))
    unlink(sidecar)
}

massive_knn_devices_execute <- function(x, setup, devices, arguments,
        checkpoint, resume) {
    identity <- massive_knn_inputs_identity(x, setup$reference)
    prefix <- sub("\\.indices\\.u32$", "", setup$paths[["indices"]])
    tasks <- massive_knn_device_tasks(x, setup$k, prefix, devices)
    tasks <- lapply(tasks, massive_knn_device_status,
        x = x, reference = setup$reference, k = setup$k,
        method = setup$method, resume = resume)
    shard_paths <- unlist(lapply(tasks, function(task)
        paste0(task$output, c(".indices.u32", ".distances.f32"))))
    merged_paths <- unname(setup$paths)
    cleanup_paths <- c(shard_paths, paste0(shard_paths, ".part"),
        merged_paths, paste0(merged_paths, ".part"),
        unlist(lapply(merged_paths, function(path)
            paste0(path, c(".merge", ".merge.part")))))
    complete <- FALSE
    on.exit(if (!checkpoint && !complete) unlink(cleanup_paths), add = TRUE)
    sidecar <- if (checkpoint) massive_knn_devices_checkpoint(
        x, setup, tasks, devices, arguments, resume) else NULL
    massive_knn_devices_disk_check(setup, tasks)
    pending <- Filter(function(task) task$status != "complete", tasks)
    arguments$checkpoint <- checkpoint
    started <- proc.time()[[3L]]
    completed <- massive_cuda_parallel_tasks(pending,
        massive_knn_device_worker, arguments)
    massive_knn_inputs_unchanged(identity, x, setup$reference,
        "parallel search")
    names(completed) <- vapply(pending, `[[`, "", "output")
    results <- lapply(tasks, function(t) t$result %||% completed[[t$output]])
    for (i in seq_along(tasks)) {
        task <- tasks[[i]]
        view <- massive_matrix_rows(x, task$row_range[[1L]],
            diff(task$row_range) + 1)
        paths <- paste0(task$output, c(".indices.u32", ".distances.f32"))
        massive_knn_device_result_check(results[[i]], task, view,
            setup$reference, setup$k, setup$method, paths)
    }
    merge_started <- proc.time()[[3L]]
    massive_knn_merge_shards(results, setup$paths, tasks, resume)
    massive_knn_inputs_unchanged(identity, x, setup$reference,
        "result merge")
    result <- massive_knn_devices_result(results, setup, x, devices,
        unname(proc.time()[[3L]] - merge_started), tasks)
    result$total_seconds <- unname(proc.time()[[3L]] - started)
    result$timing_scope <- "parallel_workers_merge"
    massive_knn_devices_finish_checkpoint(sidecar, prefix,
        result$output_identity)
    complete <- TRUE
    result
}

massive_knn_devices_check <- function(x, n.cores) {
    if (!inherits(x, "fastEmbedR_massive_matrix") ||
        x$format == "memory") stop("Multi-GPU KNN needs file-backed input.",
            call. = FALSE)
    if (normalize_nn_threads(n.cores) != 1L) {
        stop("Multi-GPU KNN requires `n.cores = 1`.", call. = FALSE)
    }
}

massive_knn_devices <- function(x, reference, k, output, n.cores,
                                method, chunk_rows, memory_limit,
                                checkpoint, resume, devices) {
    massive_knn_devices_check(x, n.cores)
    worker_limit <- massive_memory_bytes(memory_limit) / length(devices)
    setup <- massive_knn_setup(x, reference, k, output, "cuda", 1L,
        method, chunk_rows, worker_limit, resume,
        cuda_memory = FALSE, allow_final_resume = resume)
    arguments <- list(x = x, reference = setup$reference, k = setup$k,
        backend = "cuda", n.cores = 1L, method = setup$method,
        chunk_rows = chunk_rows, memory_limit = worker_limit)
    massive_knn_devices_execute(x, setup, devices, arguments,
        checkpoint, resume)
}

massive_knn_sharded_devices <- function(x, reference, k, output,
        n.cores, chunk_rows, reference_chunk_rows, memory_limit,
        checkpoint, resume, devices, method = "sharded_exact",
        checkpoint_every = 100L) {
    massive_knn_devices_check(x, n.cores)
    worker_limit <- massive_memory_bytes(memory_limit) / length(devices)
    request <- massive_landmark_sharded_setup(x, reference, k,
        output, "cuda", 1L, chunk_rows, reference_chunk_rows,
        worker_limit, FALSE, resume, devices[[1L]],
        allow_final_resume = resume, method = method,
        checkpoint_every = checkpoint_every)
    setup <- request$setup
    arguments <- list(x = x, reference = request$reference,
        k = setup$k, backend = "cuda", n.cores = 1L,
        method = method, chunk_rows = chunk_rows,
        reference_chunk_rows = reference_chunk_rows,
        memory_limit = worker_limit,
        checkpoint_every = checkpoint_every)
    massive_knn_devices_execute(x, setup, devices, arguments,
        checkpoint, resume)
}

massive_full_knn_strip_self <- function(knn, x, k, output,
        chunk_rows) {
    identity <- massive_checkpoint_source_identity(x)
    if (!inherits(knn, "fastEmbedR_massive_knn") ||
        knn$nrow != x$nrow || knn$n_reference != x$nrow ||
        knn$ncol != k + 1L ||
        !identical(knn$source_identity, identity) ||
        !identical(knn$reference_identity, identity)) stop(
            "Multi-GPU KNN candidates do not match the full source.",
            call. = FALSE)
    paths <- massive_knn_paths(output, 8 * x$nrow * k)
    parts <- stats::setNames(paste0(paths, ".part"), names(paths))
    complete <- FALSE
    on.exit(if (!complete) unlink(c(parts, paths)), add = TRUE)
    rows <- as.integer(min(chunk_rows %||% 8192L,
        floor(64e6 / (8 * (k + 1L)))))
    if (is.na(rows) || rows < 1L) stop("Invalid merge batch size.")
    for (first in seq(1, x$nrow, by = rows)) {
        batch <- massive_read_knn_rows(knn, first,
            as.integer(min(rows, x$nrow - first + 1)))
        massive_merge_knn_shard_cpp(parts[["indices"]],
            parts[["distances"]], batch$indices, batch$distances,
            first, 1, x$nrow, x$nrow, k, TRUE)
    }
    massive_knn_finish(NULL, NULL, parts, paths, x$nrow, k)
    graph <- massive_knn_graph(output, x$nrow, k)
    complete <- TRUE
    graph
}

massive_knn_holdout_rows <- function(nrow, per_device, count) {
    ranges <- vapply(per_device, function(item)
        as.double(item$query_row_range), numeric(2L))
    chunks <- vapply(per_device, function(item)
        as.double(item$resources$chunk_rows), 0)
    if (!length(per_device) || any(!is.finite(ranges)) ||
        any(ranges != floor(ranges)) || any(!is.finite(chunks)) ||
        any(chunks < 1 | chunks != floor(chunks)) ||
        ranges[1L, 1L] != 1 ||
        utils::tail(ranges[2L, ], 1L) != nrow ||
        any(ranges[2L, ] < ranges[1L, ]) ||
        (length(per_device) > 1L && any(ranges[1L, -1L] !=
            ranges[2L, -length(per_device)] + 1))) stop(
        "Multi-GPU KNN query ranges are invalid.", call. = FALSE)
    starts <- ranges[1L, ] + chunks
    lengths <- pmax(0, ranges[2L, ] - starts + 1)
    keep <- lengths > 0
    if (any(!keep)) stop(
        "IVF recall audit needs untouched query rows on every device.",
        call. = FALSE)
    starts <- starts[keep]
    lengths <- lengths[keep]
    available <- sum(lengths)
    if (count < length(lengths) || count > available) stop(
        "IVF recall audit needs untouched query rows.",
        call. = FALSE)
    assigned <- rep(1L, length(lengths))
    for (i in seq_len(count - length(lengths))) {
        priority <- lengths / (assigned + 1L)
        priority[assigned >= lengths] <- -Inf
        chosen <- which.max(priority)
        assigned[[chosen]] <- assigned[[chosen]] + 1L
    }
    positions <- lapply(seq_along(lengths), function(i)
        starts[[i]] + floor((seq_len(assigned[[i]]) - 0.5) *
            lengths[[i]] / assigned[[i]]))
    as.integer(unlist(positions, use.names = FALSE))
}

massive_full_knn_payload_bytes <- function(graph, candidates,
        expected) {
    paths <- c(candidates$indices_path, candidates$distances_path,
        graph$indices_path, graph$distances_path)
    sizes <- file.info(paths)$size
    peak <- max(candidates$peak_payload_bytes, sum(sizes))
    if (anyNA(sizes) || !is.finite(peak) || peak > expected) stop(
        "Multi-GPU KNN payload does not match its disk budget.",
        call. = FALSE)
    peak
}

massive_full_knn_devices_result <- function(graph, candidates,
        method, devices, output_bytes, disk_bytes, started) {
    graph$backend <- "cuda"
    graph$exact <- method == "exact"
    graph$method <- paste0("native_cuda_cuvs_",
        if (graph$exact) "exact" else "ivf", "_multigpu_sharded")
    graph$spanning_links <- graph$recall_audited <- FALSE
    graph$target_recall <- if (graph$exact) NA_real_ else 0.99
    graph$query_storage <- "native_float32"
    graph$gpu_devices <- devices
    graph$gpu_identity <- candidates$gpu_identity
    graph$per_device <- candidates$per_device
    graph$source_identity <- candidates$source_identity
    graph$implementation_identity <-
        massive_checkpoint_implementation_identity()
    graph$resources <- candidates$resources
    graph$resources$candidate_output_bytes <- graph$resources$output_bytes
    graph$resources$output_bytes <- output_bytes
    graph$resources$disk_budget_bytes <- disk_bytes
    graph$resources$payload_disk_bytes_before_cleanup <-
        massive_full_knn_payload_bytes(graph, candidates,
            disk_bytes)
    graph$build_seconds <- candidates$build_seconds
    graph$search_seconds <- candidates$query_seconds
    graph$total_seconds <- unname(proc.time()[[3L]] - started)
    graph$checkpoint <- graph$resumed <- FALSE
    graph$timing_scope <- "current_invocation"
    graph
}

massive_full_knn_device_audit <- function(graph, devices) {
    rows <- graph$audit_sample_rows
    recall <- graph$audit_row_recall
    parts <- graph$per_device
    if (length(parts) != length(devices) ||
        !is.numeric(rows) || !is.numeric(recall) ||
        length(rows) != length(recall) ||
        any(!is.finite(rows)) || any(rows != floor(rows)) ||
        any(rows < 1) || anyDuplicated(rows) ||
        any(!is.finite(recall)) || any(recall < 0 | recall > 1))
        stop("Multi-GPU KNN audit is incomplete.",
            call. = FALSE)
    graph$per_device_audit <- Map(function(part, device) {
        range <- part$query_row_range
        if (length(range) != 2L || anyNA(range) ||
            !identical(part$resources$gpu_device, device)) stop(
            "Multi-GPU KNN audit device mapping changed.",
            call. = FALSE)
        selected <- which(rows >= range[[1L]] & rows <= range[[2L]])
        if (!length(selected)) stop(
            "Multi-GPU KNN audit omitted a device.", call. = FALSE)
        list(device = device, query_row_range = range,
            sampled_rows = rows[selected],
            recall_mean = mean(recall[selected]),
            recall_min = min(recall[selected]),
            rows_below_target = sum(recall[selected] < 0.99))
    }, parts, devices)
    assigned <- unlist(lapply(graph$per_device_audit, `[[`,
        "sampled_rows"), use.names = FALSE)
    if (!identical(sort(assigned), sort(rows))) stop(
        "Multi-GPU KNN audit row ownership is invalid.",
        call. = FALSE)
    graph
}

massive_full_knn_device_request <- function(x, k, output, backend,
        n.cores, chunk_rows, reference_chunk_rows, memory_limit,
        checkpoint, resume, audit_rows, checkpoint_every, devices,
        method) {
    devices <- massive_validate_devices(devices, backend, x$nrow)
    if (length(devices) > 1L) return(
        massive_full_knn_devices(x, k, output, n.cores,
            chunk_rows, reference_chunk_rows, memory_limit,
            checkpoint, resume, audit_rows, checkpoint_every,
            devices, method))
    previous <- massive_cuda_memory_cpp()$device
    on.exit(massive_cuda_select_cpp(previous), add = TRUE)
    massive_cuda_select_cpp(devices[[1L]])
    graph <- massive_sharded_graph(x, k, output, n.cores,
        chunk_rows, reference_chunk_rows, memory_limit, audit_rows,
        "cuda", method, checkpoint, resume, checkpoint_every,
        0L, NULL)
    graph$gpu_devices <- devices
    graph$gpu_identity <- massive_cuda_device_uuids(devices)
    graph
}

massive_full_knn_devices_checkpoint <- function(x, k, output,
        method, devices, chunk_rows, reference_chunk_rows,
        memory_limit, audit_rows, checkpoint_every, resume) {
    signature <- massive_checkpoint_signature(list(version = 1L,
        implementation = massive_checkpoint_implementation_identity(),
        source = massive_checkpoint_source_identity(x),
        output = output, k = k, method = method, devices = devices,
        gpu_identity = massive_cuda_device_uuids(devices),
        chunk_rows = chunk_rows,
        reference_chunk_rows = reference_chunk_rows,
        memory_limit = massive_memory_bytes(memory_limit),
        audit_rows = audit_rows,
        checkpoint_every = checkpoint_every))
    sidecar <- massive_multigpu_checkpoint(output, signature,
        TRUE, resume)
    if (!resume) massive_checkpoint_write(
        list(signature = signature), sidecar)
    sidecar
}

massive_full_knn_devices_restart_output <- function(output, bytes) {
    paths <- c(paste0(output, ".indices.u32"),
        paste0(output, ".distances.f32"))
    files <- c(paths, paste0(paths, ".part"))
    sizes <- file.info(files)$size[file.exists(files)]
    if (anyNA(sizes) || any(sizes > bytes / 2)) stop(
        "Full KNN output files have invalid sizes.", call. = FALSE)
    unlink(files)
}

massive_full_knn_devices_finish <- function(candidates, x, k,
        output, chunk_rows, reference_chunk_rows, method, devices,
        audit_rows, bytes, needed, started, checkpoint, resume) {
    rows <- if (method == "ivf_sharded" && audit_rows > 0L) {
        massive_knn_holdout_rows(x$nrow, candidates$per_device,
            audit_rows)
    } else NULL
    graph <- massive_full_knn_strip_self(candidates, x, k, output,
        chunk_rows)
    graph <- massive_full_knn_devices_result(graph, candidates,
        method, devices, bytes, needed, started)
    setup <- list(resources = list(
        chunk_rows = chunk_rows %||% 8192L,
        reference_chunk_rows = reference_chunk_rows %||% 200000L),
        workers = 1L)
    graph <- massive_full_knn_audit(graph, x, audit_rows, setup,
        rows = rows)
    if (audit_rows > 0L) graph <- massive_full_knn_device_audit(
        graph, devices)
    if (!is.null(rows)) graph$audit_scope <- "untouched_query_rows"
    graph$checkpoint <- checkpoint
    graph$resumed <- resume
    graph
}

massive_full_knn_candidate_check <- function(saved, x, k,
        candidate_path, devices, shard_files) {
    paths <- c(saved$indices_path, saved$distances_path)
    identity <- massive_checkpoint_source_identity(x)
    tasks <- massive_knn_device_tasks(x, k + 1L,
        candidate_path, devices)
    owned <- length(saved$per_device) == length(tasks) &&
        all(vapply(seq_along(tasks), function(i) {
            part <- saved$per_device[[i]]
            is.list(part) && is.list(part$resources) &&
                identical(part$query_row_range,
                    tasks[[i]]$row_range) &&
                identical(part$resources$gpu_device,
                    tasks[[i]]$device) &&
                is.numeric(part$resources$chunk_rows) &&
                length(part$resources$chunk_rows) == 1L &&
                is.finite(part$resources$chunk_rows) &&
                part$resources$chunk_rows >= 1L &&
                part$resources$chunk_rows ==
                    floor(part$resources$chunk_rows)
        }, TRUE))
    if (!inherits(saved, "fastEmbedR_massive_knn") ||
        saved$nrow != x$nrow || saved$ncol != k + 1L ||
        saved$n_reference != x$nrow ||
        !massive_checkpoint_same_paths(paths,
            paste0(candidate_path,
                c(".indices.u32", ".distances.f32"))) ||
        !identical(unname(unlist(saved$shard_paths)), shard_files) ||
        !identical(saved$gpu_devices, devices) ||
        !identical(saved$gpu_identity,
            massive_cuda_device_uuids(devices)) ||
        !owned ||
        !identical(saved$source_identity, identity) ||
        !identical(saved$reference_identity, identity) ||
        !identical(saved$peak_payload_bytes,
            16 * x$nrow * (k + 1L)) ||
        !identical(saved$output_identity,
            massive_checkpoint_file_identity(paths))) stop(
        "Checkpointed KNN candidates changed.", call. = FALSE)
}

massive_full_knn_candidates <- function(x, k, candidate_path,
        devices, checkpoint, resume, controls, sidecar) {
    state <- if (!is.null(sidecar)) readRDS(sidecar) else NULL
    saved <- state$candidates
    shard_files <- unlist(lapply(devices, function(device)
        paste0(candidate_path, ".gpu", device,
            c(".indices.u32", ".distances.f32"))), use.names = FALSE)
    obsolete <- c(shard_files, paste0(candidate_path, ".gpu",
        devices, ".result.rds"))
    if (!is.null(saved)) {
        massive_full_knn_candidate_check(saved, x, k,
            candidate_path, devices, shard_files)
        if (any(unlink(obsolete) != 0L)) stop(
            "Could not remove completed GPU KNN shards.",
            call. = FALSE)
        return(saved)
    }
    candidates <- do.call(massive_landmark_knn,
        c(list(x = x, reference = x, k = k + 1L,
            output = candidate_path, backend = "cuda", n.cores = 1L,
            devices = devices, checkpoint = checkpoint,
            resume = resume), controls))
    paths <- c(candidates$indices_path, candidates$distances_path,
        unlist(candidates$shard_paths, use.names = FALSE))
    sizes <- file.info(paths)$size
    expected <- c(rep(4 * x$nrow * (k + 1L), 2L),
        unlist(lapply(massive_knn_device_tasks(x, k + 1L,
            candidate_path, devices), function(task)
            rep(task$bytes, 2L)), use.names = FALSE))
    if (!identical(unname(unlist(candidates$shard_paths)),
            shard_files) ||
        anyNA(sizes) || !identical(as.numeric(sizes), expected)) stop(
        "Multi-GPU KNN candidates are incomplete.", call. = FALSE)
    candidates$peak_payload_bytes <- sum(sizes)
    massive_full_knn_candidate_check(candidates, x, k,
        candidate_path, devices, shard_files)
    if (!is.null(sidecar)) massive_checkpoint_write(
        c(state, list(candidates = candidates)), sidecar)
    if (any(unlink(obsolete) != 0L)) stop(
        "Could not remove completed GPU KNN shards.", call. = FALSE)
    candidates
}

massive_full_knn_devices <- function(x, k, output, n.cores,
        chunk_rows, reference_chunk_rows, memory_limit,
        checkpoint, resume, audit_rows, checkpoint_every,
        devices, method) {
    if (k > 255L) stop("Multi-GPU CUDA KNN requires k <= 255.")
    if (audit_rows > 0L && audit_rows < length(devices))
        stop("Multi-GPU KNN audit needs one row per device.")
    massive_knn_devices_check(x, n.cores)
    bytes <- 8 * x$nrow * (k + 1L)
    output_bytes <- 8 * x$nrow * k
    needed <- max(2 * bytes, bytes + output_bytes)
    if (!resume && needed > 0.8 * massive_disk_available_cpp(output))
        stop("Multi-GPU full KNN exceeds the free-disk budget.")
    output <- file.path(normalizePath(dirname(output)), basename(output))
    candidate_path <- paste0(output, ".candidates")
    if (!resume) {
        massive_knn_paths(output, output_bytes)
        if (length(Sys.glob(paste0(candidate_path, "*")))) stop(
            "Multi-GPU KNN candidate files already exist.",
            call. = FALSE)
    }
    sidecar <- if (checkpoint) massive_full_knn_devices_checkpoint(
        x, k, output, method, devices, chunk_rows,
        reference_chunk_rows, memory_limit, audit_rows,
        checkpoint_every, resume) else NULL
    started <- proc.time()[[3L]]
    search <- if (method == "ivf_sharded") "sharded_ivf" else "sharded_exact"
    controls <- list(method = search, chunk_rows = chunk_rows,
        reference_chunk_rows = reference_chunk_rows,
        memory_limit = memory_limit, checkpoint_every = checkpoint_every)
    candidates <- massive_full_knn_candidates(x, k,
        candidate_path, devices, checkpoint, resume, controls, sidecar)
    temporary <- c(candidates$indices_path, candidates$distances_path,
        unlist(candidates$shard_paths),
        paste0(candidate_path, ".gpu", devices, ".result.rds"),
        paste0(candidate_path, ".multigpu.result.rds"))
    complete <- FALSE
    on.exit(if (complete || !checkpoint) unlink(temporary), add = TRUE)
    if (resume) massive_full_knn_devices_restart_output(output, output_bytes)
    graph <- massive_full_knn_devices_finish(candidates, x, k,
        output, chunk_rows, reference_chunk_rows, method,
        devices, audit_rows, output_bytes, needed, started,
        checkpoint, resume)
    complete <- TRUE
    if (!is.null(sidecar)) unlink(sidecar)
    graph
}

massive_multigpu_result <- function(path, graph, fit, arguments,
        tasks, devices, results, checkpoint, elapsed) {
    controls <- massive_projection_controls(fit, graph, "cuda",
        arguments$refinement_epochs, arguments$transform_iter,
        arguments$transform_perplexity, arguments$seed)
    per_device <- lapply(tasks, function(task) {
        result <- results[[task$output]]
        if (!is.null(result)) return(result$resources)
        list(gpu_device = task$device, output_bytes = task$bytes,
            peak_vram_bytes = NA_real_, status = "reused")
    })
    resources <- list(output_bytes = graph$nrow * ncol(fit$layout) * 4,
        projection_engine = if (fit$method == "umap") {
            "persistent_cuda"
        } else "batch", per_device = per_device)
    result <- massive_projection_result(path, graph, fit,
        ncol(fit$layout), "cuda", 1L, resources, controls,
        checkpoint, NULL,
        massive_reference_indices(arguments$selection, graph), elapsed,
        c(1, graph$nrow))
    result$gpu_devices <- devices
    result$gpu_identity <- massive_cuda_device_uuids(devices)
    result$shard_paths <- vapply(tasks, `[[`, "", "output")
    massive_projection_complete(result)
}

massive_project_devices <- function(arguments, devices, row_range,
        checkpoint, resume, local_refine) {
    massive_checkpoint_validate_controls(checkpoint, resume)
    graph <- arguments$graph
    fit <- arguments$fit
    devices <- massive_validate_devices(devices, arguments$backend,
        graph$nrow)
    if (!is.null(row_range) || local_refine) {
        stop("Multi-GPU projection requires a full range without ",
            "local refinement.", call. = FALSE)
    }
    massive_projection_controls(fit, graph, "cuda",
        arguments$refinement_epochs, arguments$transform_iter,
        arguments$transform_perplexity, arguments$seed)
    if (normalize_nn_threads(arguments$n.cores) != 1L) {
        stop("CUDA projection requires `n.cores = 1`.", call. = FALSE)
    }
    bytes <- graph$nrow * ncol(fit$layout) * 4
    path <- massive_multigpu_output(arguments$output, bytes, resume)
    merged <- file.exists(path)
    merge_resume <- resume && file.exists(paste0(path, ".part"))
    tasks <- massive_row_shards(path, graph$nrow,
        ncol(fit$layout), devices)
    signature <- if (checkpoint) massive_multigpu_signature(
        arguments, devices, tasks, path) else NULL
    sidecar <- massive_multigpu_checkpoint(path, signature, checkpoint, resume)
    tasks <- lapply(tasks, massive_multigpu_task_status, resume = resume)
    paths <- vapply(tasks, `[[`, "", "output")
    sizes <- vapply(tasks, `[[`, 0, "bytes")
    if ((merge_resume || merged) && any(vapply(tasks, `[[`, "",
        "status") != "complete")) stop(
        "CUDA merge requires completed projection shards.", call. = FALSE)
    if (merged) massive_verify_merged_shards(paths, path, sizes)
    if (!merged) massive_multigpu_disk_check(path, bytes, tasks)
    if (checkpoint && !resume) {
        massive_checkpoint_write(list(signature = signature), sidecar)
    }
    arguments$checkpoint <- checkpoint
    started <- proc.time()[[3L]]
    results <- if (merged) list() else {
        massive_project_pending_devices(tasks, arguments)
    }
    if (!merged) massive_concat_word_files_cpp(paths, path, sizes,
        resume = merge_resume)
    result <- massive_multigpu_result(path, graph, fit, arguments,
        tasks, devices, results, checkpoint,
        unname(proc.time()[[3L]] - started))
    if (checkpoint) unlink(sidecar)
    result
}
