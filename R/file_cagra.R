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
