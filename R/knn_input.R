knn_input_source <- function(indices, distances, arg_name) {
    if (!is.null(distances)) {
        return(list(
            indices = indices,
            distances = distances,
            backend = NA_character_,
            gpu_resident = FALSE
        ))
    }
    gpu_resident <- fastembedr_is_gpu_knn(indices)
    if (gpu_resident) {
        indices <- fastembedr_gpu_knn_to_host(indices)
    }
    required <- c("indices", "distances")
    if (!is.list(indices) || !all(required %in% names(indices))) {
        stop(
            "`distances` is required unless `", arg_name,
            "` is a list containing `indices` and `distances`.",
            call. = FALSE
        )
    }
    list(
        indices = indices$indices,
        distances = indices$distances,
        backend = attr(indices, "backend"),
        gpu_resident = gpu_resident ||
            isTRUE(attr(indices, "gpu_resident_source"))
    )
}

normalize_knn_input_matrices <- function(indices, distances) {
    distance_is_float32 <- is_float32_matrix(distances)
    if (!is.matrix(indices)) indices <- as.matrix(indices)
    if (!distance_is_float32 && !is.matrix(distances)) {
        distances <- as.matrix(distances)
    }
    if (!is.integer(indices)) storage.mode(indices) <- "integer"
    if (!distance_is_float32 && !identical(typeof(distances), "double")) {
        storage.mode(distances) <- "double"
    }
    if (!identical(dim(indices), dim(distances))) {
        stop("KNN `indices` and `distances` must have the same dimensions.",
            call. = FALSE
        )
    }
    if (nrow(indices) < 2L || ncol(indices) < 1L) {
        stop("KNN input must have at least two rows and one neighbor column.",
            call. = FALSE
        )
    }
    list(
        indices = indices,
        distances = distances,
        distance_is_float32 = distance_is_float32
    )
}

coerce_knn_input <- function(
    indices, distances = NULL, arg_name = "indices"
) {
    source <- knn_input_source(indices, distances, arg_name)
    matrices <- normalize_knn_input_matrices(
        source$indices,
        source$distances
    )
    stripped <- if (matrices$distance_is_float32) {
        strip_self_neighbors_float_cpp(
            matrices$indices,
            matrices$distances
        )
    } else {
        strip_self_neighbors_cpp(matrices$indices, matrices$distances)
    }
    if (stripped$n_neighbors < 1L) {
        stop("KNN input must contain at least one non-self neighbor.",
            call. = FALSE
        )
    }
    list(
        indices = stripped$indices,
        distances = stripped$distances,
        has_self = isTRUE(stripped$has_self),
        col_start = as.integer(stripped$col_start),
        n_neighbors = as.integer(stripped$n_neighbors),
        materialized = isTRUE(stripped$materialized),
        input_backend = source$backend %||% NA_character_,
        input_gpu_resident_source = source$gpu_resident,
        distance_type = stripped$distance_type %||%
            if (matrices$distance_is_float32) "float32" else "double"
    )
}

is_knn_input <- function(x) {
    fastembedr_is_gpu_knn(x) ||
        (is.list(x) && all(c("indices", "distances") %in% names(x)))
}

is_float32_matrix <- function(x) {
    inherits(x, "float32")
}

fastembedr_matrix_dimensions <- function(x) {
    if (is_float32_matrix(x)) {
        return(dim(methods::slot(x, "Data")))
    }
    dim(x)
}

public_core_config <- function(config) {
    if (!is.list(config)) {
        return(config)
    }
    if (!is.null(config$n_threads)) {
        config$n.cores <- config$n_threads
        config$n_threads <- NULL
    }
    config
}

fastembedr_knn_output_type <- function(data, backend) {
    if (!requireNamespace("float", quietly = TRUE)) {
        return("double")
    }
    if (is_float32_matrix(data)) {
        return("float")
    }
    if (backend %in% c("cpu", "cuda")) {
        return("float")
    }
    "double"
}

knn_index_base <- function(indices, n = nrow(indices)) {
    observed <- indices[!is.na(indices)]
    if (!length(observed)) {
        return("zero")
    }
    min_idx <- min(observed)
    max_idx <- max(observed)
    if (is.finite(min_idx) && is.finite(max_idx) && min_idx >= 1L && max_idx <=
        n) {
        return("one")
    }
    "zero"
}

empty_self_neighbor_result <- function(indices, distances) {
    list(
        indices = indices,
        distances = distances,
        has_self = FALSE,
        col_start = 0L,
        n_neighbors = 0L,
        materialized = FALSE
    )
}

remove_arbitrary_self_neighbors <- function(
    indices, distances, self_mask
) {
    n <- nrow(indices)
    k <- ncol(indices)
    if (k == 1L) {
        return(list(
            indices = indices[, 0L, drop = FALSE],
            distances = distances[, 0L, drop = FALSE],
            has_self = TRUE,
            col_start = 0L,
            n_neighbors = 0L,
            materialized = TRUE
        ))
    }
    out_indices <- matrix(0L, nrow = n, ncol = k - 1L)
    out_distances <- matrix(0, nrow = n, ncol = k - 1L)
    cols <- seq_len(k)
    for (i in seq_len(n)) {
        keep <- cols[-which(self_mask[i, ])[1L]]
        out_indices[i, ] <- indices[i, keep]
        out_distances[i, ] <- distances[i, keep]
    }
    list(
        indices = out_indices,
        distances = out_distances,
        has_self = TRUE,
        col_start = 0L,
        n_neighbors = as.integer(k - 1L),
        materialized = TRUE
    )
}

strip_self_neighbors <- function(indices, distances) {
    if (ncol(indices) < 1L) {
        return(empty_self_neighbor_result(indices, distances))
    }
    n <- nrow(indices)
    k <- ncol(indices)
    expected <- if (identical(knn_index_base(indices, n), "one")) {
        seq_len(n)
    } else {
        seq_len(n) - 1L
    }
    tolerance <- max(sqrt(.Machine$double.eps), 1e-12)
    first_self <- indices[, 1L] == expected & distances[, 1L] <= tolerance
    if (all(first_self)) {
        return(list(
            indices = indices,
            distances = distances,
            has_self = TRUE,
            col_start = 1L,
            n_neighbors = as.integer(k - 1L),
            materialized = FALSE
        ))
    }
    self_mask <- indices == expected & distances <= tolerance
    if (!all(rowSums(self_mask) > 0L)) {
        return(list(
            indices = indices, distances = distances, has_self = FALSE,
            col_start = 0L, n_neighbors = as.integer(k), materialized = FALSE
        ))
    }
    remove_arbitrary_self_neighbors(indices, distances, self_mask)
}

materialize_knn_range <- function(indices,
                                    distances,
                                    col_start = 0L,
                                    n_neighbors = ncol(indices) - col_start) {
    col_start <- as.integer(col_start)
    n_neighbors <- as.integer(n_neighbors)
    if (col_start == 0L && n_neighbors == ncol(indices)) {
        return(list(indices = indices, distances = distances))
    }
    cols <- seq.int(col_start + 1L, length.out = n_neighbors)
    list(
        indices = indices[, cols, drop = FALSE],
        distances = distances[, cols, drop = FALSE]
    )
}

knn_has_self_column <- function(indices, distances) {
    strip_self_neighbors(indices, distances)$has_self
}

set_embedding_colnames <- function(layout, prefix) {
    if (!is.matrix(layout) && !is_float32_matrix(layout)) {
        layout <- as.matrix(
            layout
        )
    }
    dimnames(layout)[[2L]] <- paste0(prefix, seq_len(ncol(layout)))
    layout
}

embedding_dense_double_matrix <- function(x) {
    if (is_float32_matrix(x)) {
        if (!requireNamespace("float", quietly = TRUE)) {
            stop("The float package is required to decode float32 matrices.",
                call. = FALSE
            )
        }
        x <- float::dbl(x)
    } else {
        x <- as.matrix(x)
    }
    storage.mode(x) <- "double"
    x
}

finalize_embedding_layout <- function(layout, prefix, return_float32 = FALSE) {
    attrs <- attributes(layout)
    layout <- if (isTRUE(return_float32) && requireNamespace("float",
        quietly = TRUE
    )) {
        if (is_float32_matrix(layout)) layout else float::fl(as.matrix(layout))
    } else {
        embedding_dense_double_matrix(layout)
    }
    layout <- set_embedding_colnames(layout, prefix)
    keep <- setdiff(names(attrs), c(
        "dim", "dimnames", "names", "class",
        "Data"
    ))
    for (name in keep) {
        attr(layout, name) <- attrs[[name]]
    }
    attr(layout, "precision") <- if (is_float32_matrix(
        layout
    )) {
        "float32"
    } else {
        "double"
    }
    layout
}

validate_n_components <- function(n_components) {
    n_components <- integer_scalar(n_components)
    invalid <- is.na(n_components) || n_components < 1L
    if (invalid) {
        stop("`n_components` must be a positive integer.", call. = FALSE)
    }
    n_components
}

finish_nn_result <- function(out,
                                backend,
                                k,
                                self_query,
                                exact = TRUE,
                                metric = "euclidean") {
    attr(out, "backend") <- backend
    attr(out, "k") <- as.integer(k)
    attr(out, "self_query") <- isTRUE(self_query)
    attr(out, "exact") <- isTRUE(exact)
    attr(out, "metric") <- metric
    class(out) <- c("fastEmbedR_nn", "list")
    out
}
