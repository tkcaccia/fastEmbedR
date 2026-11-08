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

fastembedr_resident_file_requested <- function(x, storage) {
    choice <- if (is.null(storage))
        getOption("storage", "auto") else storage
    identical(choice, "resident") && fastembedr_file_backed_input(x)
}

fastembedr_resident_controls_ok <- function(output, chunk_rows,
        checkpoint, resume) {
    is.null(output) && is.null(chunk_rows) &&
        identical(checkpoint, FALSE) && identical(resume, FALSE)
}

fastembedr_explicit_resident_data <- function(x, storage,
        memory_limit, massive = NULL, controls_ok = TRUE) {
    if (!fastembedr_resident_file_requested(x, storage)) return(x)
    if (!is.null(massive) && !identical(massive, "off")) stop(
        "Resident file loading conflicts with the requested ",
        "`massive` mode.", call. = FALSE)
    if (!isTRUE(controls_ok)) stop(
        "Resident file loading conflicts with file-backed controls.",
        call. = FALSE)
    if (!inherits(x, "fastEmbedR_massive_matrix")) stop(
        "A prepared file-backed graph cannot be made resident by this ",
        "route.", call. = FALSE)
    if (!requireNamespace("float", quietly = TRUE)) stop(
        "Explicit resident file loading requires `float`.",
        call. = FALSE)
    bytes <- 4 * as.double(x$nrow) * x$ncol
    limit <- massive_memory_bytes(memory_limit)
    available <- massive_available_ram_bytes()
    if (!is.finite(bytes) || !is.finite(available) ||
        1.25 * bytes > min(limit, 0.8 * available)) stop(
        "Resident float32 loading exceeds the memory limit or ",
        "available RAM; use a larger explicit `memory_limit` only ",
        "when the machine has enough free RAM.", call. = FALSE)
    massive_auto_materialize(x)
}

fastembedr_pca_resident_data <- function(x, storage, memory_limit,
        memory_missing, massive, output, chunk_rows, checkpoint, resume) {
    resident_limit <- if (memory_missing &&
        fastembedr_resident_file_requested(x, storage))
        fastembedr_memory_option("512MB") else memory_limit
    fastembedr_explicit_resident_data(x, storage, resident_limit,
        massive, fastembedr_resident_controls_ok(output,
            chunk_rows, checkpoint, resume))
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
        automatic, memory_limit, memory_missing, method, storage = NULL,
        controls_ok = TRUE) {
    resident_source <- fastembedr_resident_file_requested(x, storage)
    resident_limit <- if (memory_missing && resident_source)
        fastembedr_memory_option("8GB") else memory_limit
    x <- fastembedr_explicit_resident_data(x, storage,
        resident_limit, massive, controls_ok)
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
    if (resident_source) memory_limit <- resident_limit
    list(data = x, massive = massive, memory_limit = memory_limit,
        control_memory_limit = if (resident_source) "8GB" else memory_limit)
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
