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

fastembedr_pca_storage_mode <- function(x, massive, storage) {
    if (storage == "resident" && massive != "off") stop(
        "Resident storage cannot use automatic or out-of-core PCA.",
        call. = FALSE)
    if (fastembedr_file_backed_input(x) && massive == "off") stop(
        "A file-backed input cannot use resident PCA mode.",
        call. = FALSE)
    if (storage == "file_backed" && massive == "auto")
        return("out_of_core")
    massive
}

fastembedr_embedding_storage <- function(x, landmarks, massive,
        automatic, memory_limit, memory_missing, method, storage = NULL) {
    storage <- fastembedr_storage_option(x, storage)
    if (automatic) massive <- embedding_default_massive(x, landmarks)
    if (storage == "resident" && !identical(massive, "off")) stop(
        "Resident storage cannot use a file-backed embedding mode.",
        call. = FALSE)
    if (storage == "file_backed" && identical(massive, "off")) {
        graph <- if (method == "t-SNE") "affinity" else "fuzzy"
        stop("Full file-backed ", method, " needs a prepared ", graph,
            " graph; a file-backed matrix needs `landmarks` for ",
            "projection.", call. = FALSE)
    }
    if (fastembedr_file_backed_input(x) &&
        identical(massive, "off")) stop(
        "A file-backed input cannot use resident embedding mode.",
        call. = FALSE)
    if (memory_missing && !identical(massive, "off"))
        memory_limit <- fastembedr_memory_option("8GB")
    list(massive = massive, memory_limit = memory_limit)
}

fastembedr_cluster_storage_mode <- function(x, massive, storage,
        landmarks) {
    if (storage != "file_backed" ||
        (!is.null(massive) && massive != "auto")) return(massive)
    if (inherits(x, "fastEmbedR_massive_graph"))
        return("out_of_core_graph")
    if (!is.null(landmarks)) return("landmark")
    stop("Full file-backed clustering needs a prepared graph; ",
        "set `landmarks` for landmark clustering.", call. = FALSE)
}
