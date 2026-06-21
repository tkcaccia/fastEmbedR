# Internal backend summary used by tests and diagnostics.
backend_info <- function() {
  nn_info <- tryCatch(
    faissR::backend_info(),
    error = function(e) data.frame(
      backend = c("cpu", "faiss", "cuvs", "cuda", "metal"),
      available = c(TRUE, FALSE, FALSE, FALSE, FALSE),
      knn_available = c(TRUE, FALSE, FALSE, FALSE, FALSE),
      explicit_backend = c("cpu", "faiss", "cuda_cuvs", "cuda", "metal"),
      device = c(cpu_summary(), NA_character_, NA_character_, NA_character_, NA_character_),
      runtime = c(R.version$platform, conditionMessage(e), conditionMessage(e), conditionMessage(e), conditionMessage(e)),
      note = c(
        "Native CPU KNN path is available through faissR.",
        "faissR backend_info() failed.",
        "faissR backend_info() failed.",
        "faissR backend_info() failed.",
        "faissR backend_info() failed."
      ),
      stringsAsFactors = FALSE
    )
  )

  backends <- c("cpu", "faiss", "cuvs", "cuda", "metal")
  nn_info <- nn_info[match(backends, nn_info$backend), , drop = FALSE]
  embedding_available <- c(
    TRUE,
    FALSE,
    FALSE,
    backend_flag(embedding_cuda_available_cpp),
    backend_flag(embedding_metal_available_cpp)
  )

  nn_info$available <- isTRUE_VECTOR(nn_info$knn_available) | embedding_available
  nn_info$embedding_available <- embedding_available
  nn_info$note <- paste(
    nn_info$note,
    c(
      "CPU embedding is always available.",
      "FAISS is used only by faissR for KNN, not by fastEmbedR embeddings.",
      "cuVS is used only by faissR for KNN, not by fastEmbedR embeddings.",
      if (embedding_available[4L]) {
        "fastEmbedR CUDA embedding kernels are available."
      } else {
        "fastEmbedR CUDA embedding kernels are unavailable."
      },
      if (embedding_available[5L]) {
        "fastEmbedR Metal embedding kernels are available."
      } else {
        "fastEmbedR Metal embedding kernels are unavailable."
      }
    )
  )
  nn_info
}

backend_flag <- function(fn) {
  tryCatch(isTRUE(fn()), error = function(e) FALSE)
}

isTRUE_VECTOR <- function(x) {
  x <- as.logical(x)
  x[is.na(x)] <- FALSE
  x
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
    ". Use `faissR::backend_info()` to inspect available KNN backends.",
    call. = FALSE
  )
}

available_native_gpu_backend <- function(need_knn = FALSE,
                                         need_embedding = FALSE) {
  cuda_ok <- (!isTRUE(need_knn) || backend_flag(faissR::cuda_available)) &&
    (!isTRUE(need_embedding) || backend_flag(embedding_cuda_available_cpp))
  if (cuda_ok) return("cuda")

  metal_ok <- (!isTRUE(need_knn) || backend_flag(faissR::metal_available)) &&
    (!isTRUE(need_embedding) || backend_flag(embedding_metal_available_cpp))
  if (metal_ok) return("metal")

  NA_character_
}

resolve_backend_request <- function(backend,
                                    need_knn = FALSE,
                                    need_embedding = FALSE) {
  if (identical(backend, "gpu")) {
    resolve_native_gpu_backend(
      need_knn = need_knn,
      need_embedding = need_embedding
    )
  } else {
    backend
  }
}

embedding_backend_choices <- function() {
  c("cpu", "cuda", "metal")
}

resolve_embedding_backend <- function(backend) {
  backend <- match.arg(backend, embedding_backend_choices())
  backend
}

embedding_knn_backend <- function(backend) {
  backend <- resolve_embedding_backend(backend)
  if (identical(backend, "cuda")) {
    "cuda"
  } else {
    "cpu"
  }
}

fixed_embedding_knn_backend <- function(backend) {
  embedding_knn_backend(backend)
}

cpu_summary <- function() {
  cores <- suppressWarnings(parallel::detectCores(logical = TRUE))
  if (length(cores) != 1L || is.na(cores) || !is.finite(cores)) {
    "CPU"
  } else {
    paste0("CPU (", cores, " logical cores)")
  }
}
