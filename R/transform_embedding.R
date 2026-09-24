transform_embedding_matrix <- function(x, name, min_rows = 1L) {
    x <- embedding_dense_double_matrix(x)
    if (nrow(x) < min_rows || ncol(x) < 1L) {
        stop(
            "`", name, "` must have at least ", min_rows,
            " row(s) and one column.",
            call. = FALSE
        )
    }
    if (any(!is.finite(x))) {
        stop("`", name, "` must contain only finite values.", call. = FALSE)
    }
    x
}

embedding_layout_dims_match <- function(x, n, p) {
    (is.matrix(x) || is_float32_matrix(x)) &&
        identical(as.integer(dim(x)), as.integer(c(n, p)))
}

assemble_landmark_layout <- function(reference_layout,
                                        projected_layout,
                                        landmark_indices,
                                        non_landmark_indices,
                                        n,
                                        prefix,
                                        return_float32 = FALSE) {
    n_components <- ncol(reference_layout)
    if (!embedding_layout_dims_match(
        reference_layout, length(landmark_indices), n_components
    )) {
        stop("Invalid landmark reference layout dimensions.", call. = FALSE)
    }
    if (!embedding_layout_dims_match(
        projected_layout, length(non_landmark_indices), n_components
    )) {
        stop("Invalid projected landmark layout dimensions.", call. = FALSE)
    }

    # Only the two-dimensional layouts cross through double here. The large
    # feature, KNN-distance, graph, and optimizer buffers remain float32.
    layout <- matrix(NA_real_, nrow = n, ncol = n_components)
    layout[landmark_indices, ] <- embedding_dense_double_matrix(
        reference_layout
    )
    layout[non_landmark_indices, ] <- embedding_dense_double_matrix(
        projected_layout
    )
    finalize_embedding_layout(layout, prefix, return_float32 = return_float32)
}

split_landmark_data <- function(x,
                                landmark_indices,
                                non_landmark_indices,
                                n_threads = NULL) {
    if (is_float32_matrix(x)) {
        return(split_float32_rows_cpp(
            x,
            as.integer(landmark_indices),
            as.integer(non_landmark_indices),
            as.integer(normalize_nn_threads(n_threads))
        ))
    }
    list(
        landmarks = x[landmark_indices, , drop = FALSE],
        query = x[non_landmark_indices, , drop = FALSE]
    )
}

transform_embedding_k <- function(k, max_k) {
    k <- integer_scalar(k)
    if (is.na(k) || k < 1L) {
        stop("`k` must be NULL or a positive integer.", call. = FALSE)
    }
    if (k > max_k) {
        stop("`k` cannot be larger than the available reference count.",
            call. = FALSE
        )
    }
    k
}

transform_projection_knn <- function(knn, n_reference, k = NULL) {
    if (!is.list(knn) || !all(c("indices", "distances") %in% names(knn))) {
        stop("`knn` must contain `indices` and `distances`.", call. = FALSE)
    }

    indices <- knn$indices
    distances <- knn$distances
    if (!is.matrix(indices)) indices <- as.matrix(indices)
    if (is_float32_matrix(distances)) {
        distances <- embedding_dense_double_matrix(distances)
    } else if (!is.matrix(distances)) {
        distances <- as.matrix(distances)
    }
    if (!is.integer(indices)) storage.mode(indices) <- "integer"
    if (!identical(typeof(distances), "double")) {
        storage.mode(distances) <-
            "double"
    }

    if (!identical(dim(indices), dim(distances))) {
        stop("KNN `indices` and `distances` must have the same dimensions.",
            call. = FALSE
        )
    }
    if (nrow(indices) < 1L || ncol(indices) < 1L) {
        stop("`knn` must have at least one row and one neighbor column.",
            call. = FALSE
        )
    }

    k <- if (is.null(k)) {
        ncol(indices)
    } else {
        transform_embedding_k(k,
            max_k = ncol(indices)
        )
    }
    out <- validate_projection_knn_cpp(
        indices,
        distances,
        as.integer(n_reference),
        as.integer(k)
    )

    list(indices = out$indices, distances = out$distances)
}
