graph_layout_merge_controls <- function(merge, n) {
    allowed <- c("layout", "min_connectivity", "max_stretch",
        "min_cross_edges")
    if (!is.list(merge) || is.null(merge$layout) ||
        anyDuplicated(names(merge)) ||
        any(!names(merge) %in% allowed)) {
        stop("`layout_merge` needs a layout and known controls.",
            call. = FALSE)
    }
    layout <- merge$layout
    if (inherits(layout, "fastEmbedR_embedding")) layout <- layout$layout
    if (is_float32_matrix(layout)) {
        layout <- embedding_dense_double_matrix(layout)
    }
    if (!is.matrix(layout) || !is.numeric(layout) ||
        nrow(layout) != n || ncol(layout) != 2L ||
        any(!is.finite(layout))) {
        stop("`layout_merge$layout` must be a finite n-by-2 matrix.",
            call. = FALSE)
    }
    connectivity <- merge$min_connectivity %||% 0.2
    stretch <- merge$max_stretch %||% 2
    edges <- merge$min_cross_edges %||% 2L
    if (!is.numeric(connectivity) || length(connectivity) != 1L ||
        !is.finite(connectivity) || connectivity <= 0 ||
        !is.numeric(stretch) || length(stretch) != 1L ||
        !is.finite(stretch) || stretch <= 0 ||
        !is.numeric(edges) || length(edges) != 1L ||
        !is.finite(edges) || edges < 1 || edges != floor(edges)) {
        stop("Invalid layout merge thresholds.", call. = FALSE)
    }
    list(layout = layout, min_connectivity = connectivity,
        max_stretch = stretch, min_cross_edges = as.integer(edges))
}

graph_layout_pair_scores <- function(graph, membership, layout) {
    keep <- graph$weight > 0
    from <- graph$from[keep]
    to <- graph$to[keep]
    left <- membership[from]
    right <- membership[to]
    distance <- sqrt(rowSums((layout[from, , drop = FALSE] -
        layout[to, , drop = FALSE])^2))
    within <- left == right
    if (!any(within) || !any(!within)) return(data.frame())
    baseline <- stats::median(distance[within])
    if (baseline <= 0) stop(
        "Within-community layout edges have zero median length.",
        call. = FALSE)
    first <- pmin(left[!within], right[!within])
    second <- pmax(left[!within], right[!within])
    key <- paste(first, second, sep = ":")
    grouped <- split(distance[!within], key)
    ids <- strsplit(names(grouped), ":", fixed = TRUE)
    pairs <- matrix(as.integer(unlist(ids)), ncol = 2L, byrow = TRUE)
    degree <- tabulate(c(left, right), nbins = max(membership))
    counts <- lengths(grouped)
    expected <- degree[pairs[, 1L]] * degree[pairs[, 2L]] /
        (2 * length(from))
    data.frame(from = pairs[, 1L], to = pairs[, 2L],
        cross_edges = counts, connectivity = counts / expected,
        stretch = vapply(grouped, stats::median, 0.0) / baseline,
        row.names = NULL)
}

graph_layout_merge_labels <- function(membership, pairs) {
    parent <- seq_len(max(membership))
    root <- function(x) {
        while (parent[x] != x) x <- parent[x]
        x
    }
    for (i in seq_len(nrow(pairs))) {
        left <- root(pairs$from[i])
        right <- root(pairs$to[i])
        parent[right] <- left
    }
    roots <- vapply(parent, root, 1L)
    match(roots[membership], unique(roots[membership]))
}

graph_layout_modularity <- function(graph, membership, resolution) {
    keep <- graph$weight > 0
    from <- graph$from[keep]
    to <- graph$to[keep]
    weight <- graph$weight[keep]
    total <- sum(weight)
    if (total <= 0) return(NA_real_)
    volume <- numeric(max(membership))
    sums <- rowsum(c(weight, weight),
        c(membership[from], membership[to]), reorder = FALSE)
    volume[as.integer(rownames(sums))] <- sums[, 1L]
    sum(weight[membership[from] == membership[to]]) / total -
        resolution * sum((volume / (2 * total))^2)
}

graph_cluster_layout_merge <- function(result, graph, merge) {
    controls <- graph_layout_merge_controls(merge, graph$n_vertices)
    original <- result$membership
    pairs <- graph_layout_pair_scores(graph, original, controls$layout)
    if (nrow(pairs)) {
        pairs$selected <- pairs$cross_edges >= controls$min_cross_edges &
            pairs$connectivity >= controls$min_connectivity &
            pairs$stretch <= controls$max_stretch
    }
    chosen <- pairs[pairs$selected, , drop = FALSE]
    merged <- graph_layout_merge_labels(original, chosen)
    result$membership_original <- original
    result$membership <- as.integer(merged)
    result$n_communities <- length(unique(merged))
    result$modularity_original <- result$modularity
    result$modularity <- graph_layout_modularity(graph, merged,
        result$parameters$resolution)
    result$parameters$layout_merge <- controls[names(controls) != "layout"]
    result$layout_merge_pairs <- pairs
    result
}
