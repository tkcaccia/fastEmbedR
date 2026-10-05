massive_local_controls <- function(fit, graph, backend, source,
        local_refine, local_neighbors, overlap_rows, epochs) {
    if (!identical(local_refine, TRUE)) {
        if (!identical(local_refine, FALSE) || !is.null(source) ||
            !identical(local_neighbors, 15L) ||
            !identical(overlap_rows, 500L)) {
            stop("Local graph controls require `local_refine = TRUE`.",
                call. = FALSE)
        }
        return(NULL)
    }
    if (fit$method != "umap" || backend != "cpu" || epochs < 1L) {
        stop("EXPERIMENTAL local graph refinement requires CPU UMAP ",
            "with positive `refinement_epochs`; no fallback was used.",
            call. = FALSE)
    }
    if (!inherits(source, "fastEmbedR_massive_matrix") ||
        is.null(source$path) || source$nrow != graph$nrow ||
        is.null(graph$source_identity) ||
        !identical(massive_checkpoint_source_identity(source),
            graph$source_identity)) {
        stop("Local graph source must match the file-backed KNN query ",
            "source exactly.", call. = FALSE)
    }
    neighbors <- integer_scalar(local_neighbors)
    overlap <- integer_scalar(overlap_rows)
    if (is.na(neighbors) || neighbors < 1L ||
        is.na(overlap) || overlap < neighbors ||
        neighbors >= source$nrow) {
        stop("Local overlap must cover the neighbor count, and ",
            "neighbors must be fewer than query rows.",
            call. = FALSE)
    }
    list(source = source, neighbors = neighbors, overlap = overlap,
        identity = graph$source_identity)
}

massive_local_resources <- function(resources, graph, dimensions, local) {
    limit <- resources$memory_limit_bytes
    fixed <- 64 * 1024^2 + 32 * graph$n_reference * dimensions
    per_row <- 48 * local$source$ncol + 2048 +
        96 * (graph$ncol + local$neighbors) + 64 * dimensions
    maximum <- floor((0.7 * limit - fixed) / per_row)
    maximum <- min(maximum,
        floor(128e6 / (8 * local$source$ncol)),
        floor(128e6 / (8 * graph$ncol)),
        .Machine$integer.max - graph$n_reference)
    core <- floor(maximum - 2 * local$overlap)
    if (!is.finite(core) || core <= local$neighbors) {
        stop("Local graph buffers exceed `memory_limit`; reduce ",
            "`overlap_rows` or `local_neighbors`.", call. = FALSE)
    }
    resources$chunk_rows <- as.integer(min(resources$chunk_rows,
        core, 32768))
    if (resources$chunk_rows <= local$neighbors) {
        stop("Local graph needs more chunk rows than neighbors.",
            call. = FALSE)
    }
    resources$peak_ram_bytes <- max(resources$peak_ram_bytes,
        fixed + per_row * (resources$chunk_rows + 2 * local$overlap))
    resources
}

massive_local_window <- function(local, graph, layout, first,
        count, reference_rows, workers) {
    left <- max(1, first - local$overlap)
    right <- min(graph$nrow, first + count - 1 + local$overlap)
    rows <- as.integer(right - left + 1)
    data <- massive_read_rows(local$source, left, rows)
    anchors <- massive_read_knn_rows(graph, left, rows)
    initial <- project_embedding_knn_cpp(
        layout, anchors$indices, anchors$distances)
    mapped <- if (is.null(reference_rows)) {
        integer(rows)
    } else {
        match(seq.int(left, right), reference_rows, nomatch = 0L)
    }
    fixed <- which(mapped > 0L)
    if (length(fixed)) {
        initial[fixed, ] <- layout[mapped[fixed], , drop = FALSE]
    }
    neighbors <- precompute_knn(data, k = local$neighbors,
        backend = "cpu", n.cores = workers)
    core <- seq.int(first - left + 1L, first - left + count)
    list(anchors = anchors, initial = initial, neighbors = neighbors,
        core = core, movable = core[mapped[core] == 0L])
}

massive_project_local_batch <- function(local, graph, layout,
                                        params, first, count, epochs,
                                        workers, seed, reference_rows) {
    window <- massive_local_window(local, graph, layout,
        first, count, reference_rows, workers)
    projected <- window$initial[window$core, , drop = FALSE]
    active <- window$movable
    if (!length(active)) return(projected)
    reference_count <- nrow(layout)
    indices <- cbind(
        window$anchors$indices[active, , drop = FALSE],
        window$neighbors$indices[active, , drop = FALSE] +
            reference_count)
    distances <- cbind(
        window$anchors$distances[active, , drop = FALSE],
        embedding_dense_double_matrix(
            window$neighbors$distances[active, , drop = FALSE]))
    refined <- knn_umap_refine_rows_cpp(
        indices, distances, as.integer(reference_count + active),
        rbind(layout, window$initial), epochs,
        params$min_dist, params$negative_rate,
        params$learning_rate, params$repulsion, workers, seed, FALSE)
    projected[match(active, window$core), ] <-
        refined[reference_count + active, , drop = FALSE]
    projected
}
