massive_checkpoint_file_identity <- function(paths) {
    paths <- normalizePath(paths, winslash = "/", mustWork = TRUE)
    details <- file.info(paths)
    if (anyNA(details$size) || anyNA(details$mtime)) {
        stop("Cannot inspect checkpoint inputs.", call. = FALSE)
    }
    list(paths = paths, bytes = as.numeric(details$size),
        modified = as.numeric(details$mtime))
}

massive_checkpoint_same_paths <- function(left, right) {
    if (length(left) != length(right) ||
        !all(file.exists(left)) || !all(file.exists(right))) return(FALSE)
    identical(normalizePath(left, winslash = "/", mustWork = TRUE),
        normalizePath(right, winslash = "/", mustWork = TRUE))
}

massive_checkpoint_same_names <- function(left, right) {
    if (.Platform$OS.type == "windows") {
        left <- gsub("\\\\", "/", left)
        right <- gsub("\\\\", "/", right)
    }
    identical(left, right)
}

massive_checkpoint_source_identity <- function(source) {
    if (identical(source$format, "view")) return(list(
        format = "view", first = source$first, nrow = source$nrow,
        source = massive_checkpoint_source_identity(source$source)))
    if (is.null(source$path)) return(source)
    list(format = source$format, nrow = source$nrow,
        ncol = source$ncol,
        file = massive_checkpoint_file_identity(source$path))
}

massive_checkpoint_validate_controls <- function(checkpoint, resume) {
    if (!is.logical(checkpoint) || length(checkpoint) != 1L ||
        is.na(checkpoint) || !is.logical(resume) ||
        length(resume) != 1L || is.na(resume) ||
        (resume && !checkpoint)) {
        stop("`resume` requires `checkpoint = TRUE`; both must be ",
            "TRUE or FALSE.", call. = FALSE)
    }
}

massive_checkpoint_signature <- function(value) {
    path <- tempfile("fastembedr_checkpoint_")
    on.exit(unlink(path))
    saveRDS(value, path, version = 3L)
    result <- unname(tools::md5sum(path))
    if (is.na(result)) {
        stop("Cannot hash checkpoint inputs.", call. = FALSE)
    }
    result
}

massive_checkpoint_write <- function(state, path) {
    temporary <- paste0(path, ".tmp")
    on.exit(unlink(temporary))
    saveRDS(state, temporary, version = 3L)
    if (file.rename(temporary, path)) return(invisible(NULL))
    if (.Platform$OS.type != "windows" ||
        !file.copy(temporary, path, overwrite = TRUE)) {
        stop("Could not save experimental checkpoint.", call. = FALSE)
    }
    invisible(NULL)
}
