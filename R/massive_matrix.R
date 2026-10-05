#' Experimental disk-backed float32 matrix
#'
#' `massive_matrix()` describes a row-major float32 file without loading it.
#' Raw `.f32` files require dimensions; ANN `.fbin` files contain two
#' little-endian unsigned 32-bit dimensions followed by row-major float32
#' values. `access = "mmap"` maps the file into virtual address space on
#' POSIX systems; it does not read the whole file into RAM. A descriptor
#' records the file size and modification time and rejects later reads if
#' either changes; recreate it after intentionally replacing the input.
#'
#' @param x File path or an ordinary double-precision R matrix. A matrix
#'   source is useful for checking experimental algorithms, but is already
#'   resident in RAM.
#' @param nrow,ncol Dimensions of a headerless `.f32` file.
#' @param format One of `"f32"` or `"fbin"`; inferred from the suffix.
#' @param access `"stream"` or `"mmap"` for file sources.
#' @return A lightweight `fastEmbedR_massive_matrix` descriptor.
#' @export
massive_matrix <- function(x, nrow = NULL, ncol = NULL,
                            format = NULL,
                            access = c("stream", "mmap")) {
    if (is.matrix(x)) {
        if (typeof(x) != "double" || any(!is.finite(x))) {
            stop("A memory source must be a finite double matrix.",
                call. = FALSE)
        }
        source <- list(
            data = x, format = "memory", access = "memory",
            nrow = base::nrow(x), ncol = base::ncol(x),
            dtype = "float32", experimental = TRUE
        )
    } else {
        if (!is.character(x) || length(x) != 1L || is.na(x)) {
            stop("`x` must be one file path or a double matrix.",
                call. = FALSE)
        }
        path <- normalizePath(x, winslash = "/", mustWork = TRUE)
        format <- format %||% sub("^.*\\.", "", path)
        format <- match.arg(format, c("f32", "fbin"))
        access <- match.arg(access)
        if (format == "f32" && (is.null(nrow) || is.null(ncol))) {
            stop("Raw `.f32` files require `nrow` and `ncol`.",
                call. = FALSE)
        }
        if (format == "fbin" && (!is.null(nrow) || !is.null(ncol))) {
            stop("`.fbin` dimensions come from its header.",
                call. = FALSE)
        }
        info <- massive_file_info_cpp(
            path, format, nrow %||% NA_real_, ncol %||% NA_real_
        )
        source <- c(list(
            path = path, format = format, access = access,
            dtype = "float32", experimental = TRUE
        ), info)
    }
    class(source) <- "fastEmbedR_massive_matrix"
    source
}

massive_auto_materialize <- function(source) {
    massive_read_rows_cpp(source, 1, as.integer(source$nrow),
        8 * source$nrow * source$ncol)
}

massive_synthetic_matrix <- function(nrow, ncol) {
    dimensions <- c(nrow, ncol)
    if (!is.numeric(dimensions) || length(dimensions) != 2L ||
        any(!is.finite(dimensions)) || any(dimensions < 1) ||
        any(dimensions != floor(dimensions)) ||
        any(dimensions > 2^53 - 1)) {
        stop("Synthetic dimensions must be exact positive integers.",
            call. = FALSE)
    }
    source <- list(
        nrow = as.numeric(nrow), ncol = as.numeric(ncol),
        format = "synthetic", access = "synthetic",
        dtype = "float32", experimental = TRUE
    )
    class(source) <- "fastEmbedR_massive_matrix"
    source
}

massive_matrix_rows <- function(x, first, n) {
    if (!inherits(x, "fastEmbedR_massive_matrix") ||
        x$format == "memory" ||
        !is.numeric(first) || length(first) != 1L ||
        !is.numeric(n) || length(n) != 1L ||
        !is.finite(first) || !is.finite(n) ||
        first < 1 || n < 1 || first != floor(first) ||
        n != floor(n) || n > x$nrow - first + 1) {
        stop("Massive row view needs a valid file-backed range.",
            call. = FALSE)
    }
    result <- list(format = "view", source = x, first = first,
        nrow = n, ncol = x$ncol, dtype = "float32",
        access = x$access, experimental = TRUE)
    class(result) <- "fastEmbedR_massive_matrix"
    result
}

#' Read a bounded range from an experimental massive matrix
#'
#' @param x A `fastEmbedR_massive_matrix`.
#' @param first First one-based row to read.
#' @param n Number of consecutive rows; the returned R matrix is limited
#'   to 128 MB.
#' @return A double-precision R matrix for inspection.
#' @export
massive_read_rows <- function(x, first = 1, n = 6L) {
    if (!inherits(x, "fastEmbedR_massive_matrix")) {
        stop("`x` must be a massive_matrix() descriptor.", call. = FALSE)
    }
    first <- as.numeric(first)
    if (length(first) != 1L || !is.numeric(n) || length(n) != 1L ||
        !is.finite(first) || !is.finite(n) ||
        first < 1 || first != floor(first) ||
        n < 1 || n != floor(n) || n > .Machine$integer.max) {
        stop("`first` and `n` must be finite scalar counts.",
            call. = FALSE)
    }
    massive_read_rows_cpp(x, first, as.integer(n), 128e6)
}

#' @export
print.fastEmbedR_massive_matrix <- function(x, ...) {
    cat("EXPERIMENTAL fastEmbedR massive matrix\n")
    cat("  dimensions: ", format(x$nrow, scientific = FALSE), " x ",
        format(x$ncol, scientific = FALSE), "\n", sep = "")
    origin <- x$path %||% if (x$format == "view") {
        paste0("rows ", x$first, "-", x$first + x$nrow - 1,
            " of ", x$source$path %||% x$source$format)
    } else if (x$format == "synthetic") {
        "synthetic"
    } else {
        "R matrix"
    }
    cat("  source: ", origin, "\n", sep = "")
    if (!is.null(x$method)) cat("  method: ", x$method,
        "; mode: ", x$mode, "; backend: ", x$backend,
        "\n", sep = "")
    invisible(x)
}

#' @export
head.fastEmbedR_massive_matrix <- function(x, n = 6L, ...) {
    massive_read_rows(x, n = min(n, x$nrow))
}

#' @export
as.matrix.fastEmbedR_massive_matrix <- function(x, ...) {
    bytes <- x$nrow * x$ncol * 8
    if (!is.finite(bytes) || bytes > 128e6 ||
        x$nrow > .Machine$integer.max ||
        x$ncol > .Machine$integer.max) {
        stop("Materialization exceeds the 128 MB safety limit; ",
            "use massive_read_rows() instead.", call. = FALSE)
    }
    massive_read_rows(x, n = x$nrow)
}
