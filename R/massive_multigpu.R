massive_validate_devices <- function(devices, backend, nrow) {
    if (is.null(devices)) return(NULL)
    if (backend != "cuda") {
        stop("`devices` requires CUDA; no CPU fallback was used.",
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
    settings$shards <- lapply(tasks, function(task) {
        task[c("device", "row_range", "bytes")]
    })
    settings$package_version <- as.character(
        utils::packageVersion("fastEmbedR"))
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
                                                package_path) {
    if (normalizePath(find.package("fastEmbedR")) != package_path) {
        stop("CUDA worker loaded a different fastEmbedR installation.")
    }
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
    cluster <- parallel::makePSOCKcluster(length(tasks))
    on.exit(parallel::stopCluster(cluster))
    bootstrap <- evalq(function(path) {
        .libPaths(c(path, .libPaths()))
    }, baseenv())
    parallel::clusterCall(cluster, bootstrap, dirname(package_path))
    parallel::clusterApply(cluster, tasks, worker, arguments = arguments,
        package_path = package_path)
}

massive_pca_device_worker <- function(task, arguments, package_path) {
    if (normalizePath(find.package("fastEmbedR")) != package_path) {
        stop("CUDA worker loaded a different fastEmbedR installation.")
    }
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
    resources$peak_ram_bytes <- if (length(pending) < length(tasks)) {
        NA_real_
    } else max(resources$peak_ram_bytes,
        sum(vapply(per_device, `[[`, 0, "peak_ram_bytes")))
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

massive_knn_device_worker <- function(task, arguments, package_path) {
    if (normalizePath(find.package("fastEmbedR")) != package_path) {
        stop("CUDA worker loaded a different fastEmbedR installation.")
    }
    massive_cuda_select_cpp(task$device)
    arguments$x <- massive_matrix_rows(arguments$x,
        task$row_range[1L], diff(task$row_range) + 1)
    arguments$output <- task$output
    arguments$resume <- identical(task$status, "partial")
    result <- do.call(massive_landmark_knn, arguments)
    if (result$resources$gpu_device != task$device) {
        stop("KNN metadata reports a different CUDA device.")
    }
    if (isTRUE(arguments$checkpoint)) {
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
                                        devices, merge_seconds) {
    out <- results[[1L]]
    out$indices_path <- setup$paths[["indices"]]
    out$distances_path <- setup$paths[["distances"]]
    out$nrow <- x$nrow
    out$source_identity <- massive_checkpoint_source_identity(x)
    references <- lapply(results, `[[`, "reference_identity")
    if (!all(vapply(references, identical, TRUE, references[[1L]]))) {
        stop("KNN shards used different references.", call. = FALSE)
    }
    out$reference_identity <- references[[1L]]
    out$output_identity <- massive_checkpoint_file_identity(
        unname(setup$paths))
    out$gpu_devices <- devices
    out$shard_paths <- lapply(results, function(value) {
        value[c("indices_path", "distances_path")]
    })
    out$per_device <- lapply(results, function(value) {
        value[c("resources", "build_seconds", "query_seconds",
            "pilot_recall", "pilot_id_recall",
            "pilot_recall_metric", "pilot_rows", "nlist", "nprobe")]
    })
    out$resources <- list(output_bytes = setup$resources$output_bytes,
        peak_ram_bytes = sum(vapply(results, function(value) {
            value$resources$peak_ram_bytes
        }, 0)), per_device = lapply(results, `[[`, "resources"))
    out$pilot_rows <- sum(vapply(results, function(value) {
        as.numeric(value$pilot_rows)
    }, 0))
    recall <- vapply(results, `[[`, 0, "pilot_recall")
    out$pilot_recall <- if (all(is.finite(recall))) min(recall) else NA_real_
    out$pilot_id_recall <- min(vapply(results, function(value)
        value$pilot_id_recall %||% NA_real_, 0))
    for (field in c("nlist", "nprobe")) {
        values <- vapply(results, function(value) {
            as.numeric(value[[field]])
        }, 0)
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
    reference <- setup$reference
    source <- reference
    while (identical(source$format, "view")) source <- source$source
    if (is.null(source$path)) stop(
        "Multi-GPU KNN checkpoint needs a file-backed reference.",
        call. = FALSE)
    controls <- arguments[setdiff(names(arguments),
        c("x", "reference", "checkpoint", "resume"))]
    signature <- massive_checkpoint_signature(list(version = 1L,
        package_version = as.character(
            utils::packageVersion("fastEmbedR")),
        source = massive_checkpoint_source_identity(x),
        reference = massive_checkpoint_source_identity(reference),
        output = setup$paths, controls = controls,
        devices = devices, shards = lapply(tasks, function(task) {
            task[c("device", "row_range", "bytes")]
        })))
    prefix <- sub("\\.indices\\.u32$", "",
        setup$paths[["indices"]])
    sidecar <- massive_multigpu_checkpoint(prefix, signature,
        TRUE, resume)
    if (!resume) massive_checkpoint_write(
        list(signature = signature), sidecar)
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

massive_knn_devices_execute <- function(x, setup, devices, arguments,
        checkpoint, resume) {
    prefix <- sub("\\.indices\\.u32$", "", setup$paths[["indices"]])
    tasks <- massive_knn_device_tasks(x, setup$k, prefix, devices)
    tasks <- lapply(tasks, massive_knn_device_status,
        x = x, reference = setup$reference, k = setup$k,
        method = setup$method, resume = resume)
    massive_knn_devices_disk_check(setup, tasks)
    sidecar <- if (checkpoint) massive_knn_devices_checkpoint(
        x, setup, tasks, devices, arguments, resume) else NULL
    pending <- Filter(function(task) task$status != "complete", tasks)
    arguments$checkpoint <- checkpoint
    completed <- massive_cuda_parallel_tasks(pending,
        massive_knn_device_worker, arguments)
    names(completed) <- vapply(pending, `[[`, "", "output")
    results <- lapply(tasks, function(task) {
        task$result %||% completed[[task$output]]
    })
    for (i in seq_along(tasks)) {
        task <- tasks[[i]]
        view <- massive_matrix_rows(x, task$row_range[[1L]],
            diff(task$row_range) + 1)
        paths <- paste0(task$output,
            c(".indices.u32", ".distances.f32"))
        massive_knn_device_result_check(results[[i]], task, view,
            setup$reference, setup$k, setup$method, paths)
    }
    started <- proc.time()[[3L]]
    massive_knn_merge_shards(results, setup$paths, tasks, resume)
    result <- massive_knn_devices_result(results, setup, x, devices,
        unname(proc.time()[[3L]] - started))
    if (!is.null(sidecar)) unlink(sidecar)
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
        checkpoint, resume, devices) {
    massive_knn_devices_check(x, n.cores)
    worker_limit <- massive_memory_bytes(memory_limit) / length(devices)
    request <- massive_landmark_sharded_setup(x, reference, k,
        output, "cuda", 1L, chunk_rows, reference_chunk_rows,
        worker_limit, FALSE, resume, devices[[1L]],
        allow_final_resume = resume)
    setup <- request$setup
    arguments <- list(x = x, reference = request$reference,
        k = setup$k, backend = "cuda", n.cores = 1L,
        method = "sharded_exact", chunk_rows = chunk_rows,
        reference_chunk_rows = reference_chunk_rows,
        memory_limit = worker_limit)
    massive_knn_devices_execute(x, setup, devices, arguments,
        checkpoint, resume)
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
