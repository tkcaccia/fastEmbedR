landmark_memory_estimate <- function(
    n, n_landmarks, n_neighbors, transform_k, n_components = 2L
) {
    values <- as.numeric(c(
        n, n_landmarks, n_neighbors, transform_k, n_components
    ))
    if (any(!is.finite(values)) || any(values < 0)) {
        stop("Landmark memory inputs must be finite and non-negative.",
            call. = FALSE)
    }
    n <- values[[1L]]
    m <- min(n, values[[2L]])
    k <- min(n, values[[3L]])
    transform_k <- min(m, values[[4L]])
    dimensions <- values[[5L]]
    query <- max(0, n - m)
    edge_bytes <- 8
    layout_bytes <- 4
    full_graph <- n * k * edge_bytes
    reference_graph <- m * k * edge_bytes
    full_layout <- n * dimensions * layout_bytes
    reference_layout <- m * dimensions * layout_bytes
    full_optimizer <- full_graph + full_layout
    reference_optimizer <- reference_graph + reference_layout
    list(
        landmark_reference_graph_bytes = reference_graph,
        full_graph_bytes = full_graph,
        landmark_reference_layout_bytes = reference_layout,
        full_layout_bytes = full_layout,
        landmark_reference_optimizer_bytes = reference_optimizer,
        full_optimizer_bytes = full_optimizer,
        landmark_projection_knn_bytes = query * transform_k * edge_bytes,
        landmark_reference_memory_ratio = reference_optimizer / full_optimizer,
        landmark_reference_memory_saving = 1 - reference_optimizer /
            full_optimizer
    )
}
