massive_landmark_resources <- function(x, count, chunk_rows,
                                        memory_limit, landmark_method) {
    limit <- massive_memory_bytes(memory_limit)
    selection_bytes <- if (landmark_method == "random") 64 else 16
    fixed <- selection_bytes * count + 16 * 1024^2
    maximum <- floor((0.7 * limit - fixed) / (4 * x$ncol))
    if (!is.finite(maximum) || maximum < 1) {
        stop("Landmark buffers exceed `memory_limit`.", call. = FALSE)
    }
    requested <- chunk_rows %||% 250000L
    if (!is.numeric(requested) || length(requested) != 1L ||
        !is.finite(requested) || requested < 1 ||
        requested != floor(requested) ||
        requested > .Machine$integer.max) {
        stop("`chunk_rows` must be one positive integer.",
            call. = FALSE)
    }
    list(
        chunk_rows = as.integer(min(requested, maximum)),
        peak_ram_bytes = fixed + 4 * x$ncol * min(requested, maximum),
        output_bytes = count * x$ncol * 4
    )
}

#' Experimental streaming reservoir landmarks
#'
#' Selects rows without loading a file-backed matrix into RAM. Reservoir
#' sampling scans the row count and then reads the source sequentially.
#' Random sampling takes work proportional to the landmark count and reads
#' only sampled rows; use it when random access is efficient. Both write
#' selected rows in source order to a row-major float32 `.f32` file.
#' Returned row identifiers are one-based doubles, including above 2^31.
#'
#' @param x A `massive_matrix()` descriptor.
#' @param landmarks Number of rows to select.
#' @param output New `.f32` output path.
#' @param seed Integer random seed.
#' @param chunk_rows Maximum source rows read per block.
#' @param memory_limit Conservative RAM budget for sampling buffers.
#' @param landmark_method `"reservoir"` or `"random"`.
#' @return A list with `data`, a file-backed landmark matrix descriptor;
#'   `indices`, sorted one-based source rows; and resource metadata.
#' @export
massive_select_landmarks <- function(x, landmarks, output, seed = 42L,
                                        chunk_rows = NULL,
                                        memory_limit = "512MB",
                                        landmark_method = c("reservoir",
                                            "random")) {
    if (!inherits(x, "fastEmbedR_massive_matrix")) {
        stop("`x` must be a massive_matrix() descriptor.", call. = FALSE)
    }
    count <- integer_scalar(landmarks)
    random_seed <- integer_scalar(seed)
    if (is.na(count) || count < 1L || count > x$nrow ||
        is.na(random_seed)) {
        stop("`landmarks` and `seed` must be valid integers.",
            call. = FALSE)
    }
    landmark_method <- match.arg(landmark_method)
    resources <- massive_landmark_resources(
        x, count, chunk_rows, memory_limit, landmark_method
    )
    path <- massive_float_output(output, resources$output_bytes)
    message("EXPERIMENTAL ", landmark_method, " landmarks: ", count,
        " of ",
        x$nrow, "; chunk rows=", resources$chunk_rows)
    indices <- massive_landmark_sample_cpp(
        x, count, random_seed, resources$chunk_rows, path,
        landmark_method
    )
    list(
        data = massive_matrix(path, nrow = count, ncol = x$ncol),
        indices = indices, method = landmark_method, seed = random_seed,
        indices_signature = massive_checkpoint_signature(indices),
        source_identity = if (x$format == "memory") NULL else
            massive_checkpoint_source_identity(x),
        backend = "cpu", resources = resources, experimental = TRUE
    )
}
