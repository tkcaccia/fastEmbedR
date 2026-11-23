precompute_knn_file_cagra <- function(data, k, output, memory_bytes,
        selection, audit_rows, devices, checkpoint) {
    if (checkpoint || !is.null(devices)) stop(
        "File-backed CAGRA requires checkpointing off and one GPU.",
        call. = FALSE)
    k <- validate_precompute_k(k, data$nrow - 1L, "nrow(data) - 1")
    audit_rows <- integer_scalar(audit_rows)
    if (is.na(audit_rows) || audit_rows < 0L ||
        audit_rows > min(data$nrow, 10000L)) stop(
        "`audit_rows` must be between zero and 10,000 rows.", call. = FALSE)
    entry <- if (selection$backend == "metal") "metal_cagra_file" else
        "cuvs_cagra_wide_file"
    builder <- get0(entry, envir = asNamespace("faissR"), inherits = FALSE)
    if (!is.function(builder)) stop(
        "The installed faissR lacks file-backed CAGRA support.", call. = FALSE)
    resources <- list(memory_limit_bytes = memory_bytes,
        output_bytes = 8 * as.double(data$nrow) * k, chunk_rows = 1024L,
        reference_chunk_rows = min(data$nrow, floor(192e6 / (4 * data$ncol))))
    paths <- massive_knn_paths(output, resources$output_bytes)
    identity <- massive_checkpoint_source_identity(data)
    arguments <- list(data, k, resources$chunk_rows, memory_bytes,
        selection$target_recall, selection$method == "cagra_width4")
    if (!is.null(selection$cuda_cagra_profile))
        arguments$profile <- selection$cuda_cagra_profile
    index <- do.call(builder, arguments)
    live <- TRUE
    on.exit(if (live) index$release(), add = TRUE)
    resources$chunk_rows <- index$query_batch_rows %||% resources$chunk_rows
    resources$memory_limit_bytes <- index$params$memory_limit_bytes %||%
        memory_bytes
    massive_full_knn_assert_source(data, identity)
    setup <- list(paths = paths, k = k, workers = 1L, resources = resources,
        target_recall = selection$target_recall, exact = FALSE,
        backend_used = index$backend_used,
        search_batch = index$search_batch, checkpoint_every = 1L)
    measurement <- massive_full_knn_stream(data, setup, NULL)
    index$release()
    live <- FALSE
    graph <- massive_file_cagra_graph(data, output, setup,
        index, selection, measurement, identity)
    massive_full_knn_audit(graph, data, audit_rows, setup)
}

massive_file_cagra_graph <- function(data, output, setup,
        index, selection, measurement, identity) {
    graph <- massive_knn_graph(output, data$nrow, setup$k)
    graph$backend <- selection$backend
    graph$method <- index$backend_used
    graph$knn_provider <- "faissR"
    graph$knn_selection <- selection
    graph$residency <- "file-backed"
    graph$index_residency <- selection$backend
    graph$query_storage <- "file_backed_float32"
    graph$index_reused <- TRUE
    graph$device_ids <- index$device_ids
    graph$exact <- FALSE
    graph$target_recall <- setup$target_recall
    graph$recall_audited <- FALSE
    graph$source_identity <- identity
    graph$implementation_identity <-
        massive_checkpoint_implementation_identity()
    graph$faissr_identity <- massive_faissr_identity()
    graph$resources <- setup$resources
    graph$resources$reference_bytes <- index$reference_bytes
    graph$resources$device_bytes_estimate <- index$device_bytes_estimate
    graph$resources$host_batch_bytes_estimate <- index$host_batch_bytes_estimate
    graph$resources$owned_metal_bytes <- index$owned_metal_bytes
    graph$resources$observed_metal_owned_buffer_peak_bytes <-
        index$observed_metal_owned_buffer_peak_bytes
    graph$resources$metal_memory_measurement_scope <-
        index$metal_memory_measurement_scope
    graph$resources$device_free_bytes_at_build <-
        index$device_free_bytes_at_build
    graph$cagra_settings <- index$params
    graph$source_read_and_upload_seconds <- index$source_read_and_upload_seconds
    graph$build_seconds <- index$build_seconds
    graph$build_timing <- index$build_timing
    graph$knn_search_seconds <- measurement$search_seconds
    graph$graph_write_seconds <- measurement$graph_write_seconds
    graph$search_and_write_seconds <- measurement$seconds
    graph$total_seconds <- graph$source_read_and_upload_seconds +
        graph$build_seconds + graph$search_and_write_seconds
    graph
}

file_cagra_shard_fit <- function(data, first, rows, k, selection, parts,
        initial, identity, builder) {
    resources <- selection$resources
    profile <- list(graph_degree = resources$graph_degree,
        intermediate_degree = resources$intermediate_degree,
        batch_rows = resources$chunk_rows,
        memory_limit_bytes = resources$memory_limit_bytes)
    if (!is.null(resources$builder)) profile$builder <- resources$builder
    reference <- massive_matrix_rows(data, first, rows)
    index <- builder(reference, k, resources$chunk_rows,
        resources$memory_limit_bytes, selection$target_recall, TRUE, profile)
    on.exit(index$release(), add = TRUE)
    if (!is.function(index$query_batch)) stop(
        "The installed faissR lacks external file-backed CAGRA queries.",
        call. = FALSE)
    times <- c(search = 0, write = 0)
    query_first <- 1L
    while (query_first <= data$nrow) {
        massive_full_knn_assert_source(data, identity)
        count <- as.integer(min(resources$chunk_rows,
            data$nrow - query_first + 1L))
        started <- proc.time()[[3L]]
        result <- index$query_batch(data, query_first, count)
        times[["search"]] <- times[["search"]] + proc.time()[[3L]] - started
        if (!identical(result$backend_used, "cuda_cuvs_cagra_file_query") ||
            !isTRUE(result$index_reused)) stop(
            "Sharded CAGRA changed backend or rebuilt a query index.",
            call. = FALSE)
        started <- proc.time()[[3L]]
        massive_merge_knn_shard_cpp(parts[["indices"]], parts[["distances"]],
            result$indices, result$distances, query_first, first, rows,
            data$nrow, k, initial)
        times[["write"]] <- times[["write"]] + proc.time()[[3L]] - started
        query_first <- query_first + count
    }
    list(timing = c(upload = index$source_read_and_upload_seconds,
        build = index$build_seconds, times), params = index$params,
        device = index$device_ids)
}

precompute_knn_file_cagra_sharded <- function(data, k, output, selection,
        audit_rows, devices, checkpoint) {
    if (checkpoint || !is.null(devices)) stop(
        "File-backed sharded CAGRA requires checkpointing off and one GPU.",
        call. = FALSE)
    k <- validate_precompute_k(k, data$nrow - 1L, "nrow(data) - 1")
    audit_rows <- integer_scalar(audit_rows)
    if (is.na(audit_rows) || audit_rows < 0L ||
        audit_rows > min(data$nrow, 10000L)) stop(
        "`audit_rows` must be between zero and 10,000 rows.", call. = FALSE)
    builder <- get0("cuvs_cagra_wide_file", envir = asNamespace("faissR"),
        inherits = FALSE)
    if (!is.function(builder)) stop(
        "The installed faissR lacks file-backed CAGRA support.", call. = FALSE)
    resources <- selection$resources
    paths <- massive_knn_paths(output, resources$output_bytes)
    parts <- stats::setNames(paste0(unname(paths), ".part"), names(paths))
    on.exit(unlink(parts), add = TRUE)
    identity <- massive_checkpoint_source_identity(data)
    first <- 1L
    base <- floor(data$nrow / resources$n_shards)
    remainder <- data$nrow %% resources$n_shards
    timing <- c(upload = 0, build = 0, search = 0, write = 0)
    for (shard in seq_len(resources$n_shards)) {
        count <- base + as.integer(shard <= remainder)
        fit <- file_cagra_shard_fit(data, first, count, k, selection, parts,
            shard == 1L, identity, builder)
        timing <- timing + fit$timing
        first <- first + count
    }
    massive_full_knn_assert_source(data, identity)
    finish <- proc.time()[[3L]]
    massive_knn_finish(NULL, NULL, parts, paths, data$nrow, k)
    timing[["write"]] <- timing[["write"]] + proc.time()[[3L]] - finish
    index <- list(backend_used = "cuda_cuvs_cagra_file_sharded",
        device_ids = fit$device, params = c(fit$params, resources[
            c("n_shards", "reference_chunk_rows")]),
        device_bytes_estimate = resources$device_bytes_estimate,
        host_batch_bytes_estimate = resources$host_batch_bytes_estimate,
        source_read_and_upload_seconds = timing[["upload"]],
        build_seconds = timing[["build"]])
    measurement <- list(search_seconds = timing[["search"]],
        graph_write_seconds = timing[["write"]],
        seconds = sum(timing[c("search", "write")]))
    setup <- list(k = k, workers = 1L, resources = resources,
        target_recall = selection$target_recall)
    graph <- massive_file_cagra_graph(data, output, setup,
        index, selection, measurement, identity)
    massive_full_knn_audit(graph, data, audit_rows, setup)
}
