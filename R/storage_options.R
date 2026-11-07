fastembedr_file_backed_input <- function(x) {
    if (inherits(x, "fastEmbedR_massive_graph")) return(TRUE)
    inherits(x, "fastEmbedR_massive_matrix") &&
        !identical(x$format, "memory")
}

fastembedr_storage_option <- function(x, storage = NULL) {
    choice <- if (is.null(storage))
        getOption("storage", "auto") else storage
    if (!is.character(choice) || length(choice) != 1L ||
        is.na(choice) || !choice %in%
            c("auto", "resident", "file_backed")) {
        stop("Storage must be 'auto', 'resident', or 'file_backed'.",
            call. = FALSE)
    }
    backed <- fastembedr_file_backed_input(x)
    if (choice == "file_backed" && !backed) stop(
        "File-backed storage needs a disk-backed massive_matrix() or graph; ",
        "a resident matrix is not silently written to disk.",
        call. = FALSE)
    if (choice == "resident" && backed) stop(
        "Resident storage cannot load a file-backed descriptor implicitly.",
        call. = FALSE)
    choice
}

fastembedr_memory_option <- function(default) {
    value <- getOption("memory_limit", default)
    massive_memory_bytes(value)
    value
}

fastembedr_embedding_storage <- function(x, landmarks, massive,
        automatic, memory_limit, memory_missing, method, storage = NULL) {
    storage <- fastembedr_storage_option(x, storage)
    if (automatic) massive <- embedding_default_massive(x, landmarks)
    if (storage == "file_backed" && identical(massive, "off")) {
        graph <- if (method == "t-SNE") "affinity" else "fuzzy"
        stop("Full file-backed ", method, " needs a prepared ", graph,
            " graph; a file-backed matrix needs `landmarks` for ",
            "projection.", call. = FALSE)
    }
    if (memory_missing && !identical(massive, "off"))
        memory_limit <- fastembedr_memory_option("8GB")
    list(massive = massive, memory_limit = memory_limit)
}
