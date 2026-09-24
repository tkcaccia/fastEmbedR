validate_environment_backend <- function(backend, label = "backend") {
    backend <- tolower(as.character(backend))
    if (length(backend) != 1L || is.na(backend) || !nzchar(backend) ||
        !backend %in% embedding_backend_choices()) {
        stop("`", label, "` must be one of \"cpu\", \"cuda\", or \"metal\".",
            call. = FALSE
        )
    }
    backend
}

backend_flag <- function(fn) {
    tryCatch(isTRUE(fn()), error = function(e) FALSE)
}

resolve_native_gpu_backend <- function(need_knn = FALSE,
                                        need_embedding = FALSE) {
    backend <- available_native_gpu_backend(
        need_knn = need_knn,
        need_embedding = need_embedding
    )
    if (!is.na(backend)) return(backend)
    need <- c(
        if (isTRUE(need_knn)) "KNN" else NULL,
        if (isTRUE(need_embedding)) "embedding" else NULL
    )
    if (length(need) == 0L) need <- "requested"
    stop(
        "No native GPU backend is available for ",
        paste(need, collapse = " and "),
        ". Rebuild fastEmbedR with the requested native backend enabled.",
        call. = FALSE
    )
}

available_native_gpu_backend <- function(need_knn = FALSE,
                                            need_embedding = FALSE) {
    cuda_ok <- (!isTRUE(need_knn) ||
        backend_flag(native_cuda_knn_available_cpp)) &&
        (!isTRUE(need_embedding) ||
            backend_flag(embedding_cuda_available_cpp))
    if (cuda_ok) return("cuda")
    metal_ok <- (!isTRUE(need_knn) ||
        backend_flag(native_metal_knn_available_cpp)) &&
        (!isTRUE(need_embedding) ||
            backend_flag(embedding_metal_available_cpp))
    if (metal_ok) return("metal")
    NA_character_
}

resolve_backend_request <- function(backend,
                                    need_knn = FALSE,
                                    need_embedding = FALSE) {
    if (identical(backend, "gpu")) {
        return(resolve_native_gpu_backend(
            need_knn = need_knn,
            need_embedding = need_embedding
        ))
    }
    backend
}

embedding_backend_choices <- function() {
    c("cpu", "cuda", "metal")
}

resolve_embedding_backend <- function(backend) {
    if (!is.null(backend) && length(backend) == 1L) {
        return(validate_environment_backend(backend))
    }
    option <- getOption("backend", NULL)
    if (!is.null(option)) {
        return(validate_environment_backend(option, "option backend"))
    }
    environment <- Sys.getenv("FASTEMBEDR_BACKEND", unset = "")
    if (nzchar(environment)) {
        return(validate_environment_backend(
            environment,
            "FASTEMBEDR_BACKEND"
        ))
    }
    "cpu"
}

embedding_knn_backend <- function(backend) {
    resolve_embedding_backend(backend)
}

fixed_embedding_knn_backend <- function(backend) {
    embedding_knn_backend(backend)
}
