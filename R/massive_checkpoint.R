massive_checkpoint_file_identity <- function(paths) {
    paths <- normalizePath(paths, winslash = "/", mustWork = TRUE)
    details <- file.info(paths)
    if (anyNA(details$size) || anyNA(details$mtime)) {
        stop("Cannot inspect checkpoint inputs.", call. = FALSE)
    }
    list(paths = paths, bytes = as.numeric(details$size),
        modified = as.numeric(details$mtime))
}

massive_checkpoint_input_files_identity <- function(paths) {
    files <- massive_checkpoint_file_identity(paths)
    changed <- file.info(files$paths)$ctime
    if (anyNA(changed)) stop(
        "Cannot inspect checkpoint input status time.",
        call. = FALSE)
    files$status_changed <- as.numeric(changed)
    files
}

massive_checkpoint_validated_graph_identity <- function(graph, paths) {
    current <- massive_checkpoint_input_files_identity(paths)
    if (is.null(graph$validated_input_identity) ||
        !identical(graph$validated_input_identity, current)) stop(
        "Graph changed since validation; re-import it.",
        call. = FALSE)
    current
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
    file <- massive_checkpoint_input_files_identity(source$path)
    list(format = source$format, nrow = source$nrow,
        ncol = source$ncol, file = file)
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

massive_checkpoint_implementation_identity <- function() {
    dll <- getLoadedDLLs()[["fastEmbedR"]]
    if (is.null(dll)) stop(
        "Cannot identify the loaded fastEmbedR implementation.",
        call. = FALSE)
    native <- unname(tools::md5sum(dll[["path"]]))
    if (is.na(native)) stop(
        "Cannot hash the loaded fastEmbedR implementation.",
        call. = FALSE)
    code_dir <- file.path(find.package("fastEmbedR"), "R")
    code <- list.files(code_dir, pattern = "\\.[Rr]$",
        full.names = TRUE)
    if (!length(code)) code <- file.path(code_dir,
        c("fastEmbedR.rdb", "fastEmbedR.rdx"))
    if (!length(code) || !all(file.exists(code))) stop(
        "Cannot identify the loaded fastEmbedR R code.",
        call. = FALSE)
    code_hash <- unname(tools::md5sum(code))
    if (anyNA(code_hash)) stop(
        "Cannot hash the loaded fastEmbedR R code.",
        call. = FALSE)
    names(code_hash) <- basename(code)
    list(package_version = as.character(
        utils::packageVersion("fastEmbedR")), native = native,
        r_code = code_hash)
}

massive_faissr_identity <- function() {
    if (!requireNamespace("faissR", quietly = TRUE)) stop(
        "Cannot identify the required faissR implementation.",
        call. = FALSE)
    dll <- getLoadedDLLs()[["faissR"]]
    if (is.null(dll)) stop(
        "Cannot identify the loaded faissR native library.",
        call. = FALSE)
    native <- unname(tools::md5sum(dll[["path"]]))
    if (is.na(native)) stop(
        "Cannot hash the loaded faissR native library.",
        call. = FALSE)
    list(package_version = as.character(
        utils::packageVersion("faissR")), native = native)
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
