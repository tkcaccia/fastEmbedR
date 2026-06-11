#' Summarize native backend availability
#'
#' `backend_info()` reports which native package backends can currently run.
#' It is a lightweight availability summary for users who want to check GPU
#' support before requesting `backend = "cuda"`, `"metal"`, or `"gpu"`.
#'
#' The function does not load or depend on external GPU packages such as
#' `cuda.ml`; it only reports the native backends compiled into `fastEmbedR`
#' and visible on the current machine. CUDA builds report CUDA runtime device
#' details when available; non-CUDA builds report the unavailable state and may
#' include `nvidia-smi` device hints if that command is present.
#'
#' @return A data frame with one row per native backend and columns describing
#'   KNN availability, embedding availability, device/runtime hints, and a short
#'   note. The function never intentionally falls back from a requested GPU
#'   backend to CPU; this table is informational only.
#' @export
backend_info <- function() {
  cuda_knn <- backend_flag(cuda_available)
  cuda_embedding <- backend_flag(embedding_cuda_available_cpp)
  metal_knn <- backend_flag(metal_available)
  metal_embedding <- backend_flag(embedding_metal_available_cpp)
  faiss_knn <- backend_flag(faiss_available)
  cuvs_knn <- backend_flag(cuvs_available)
  cuda <- cuda_summary()
  metal <- metal_summary()
  faiss <- faiss_summary()
  cuvs <- cuvs_summary()

  data.frame(
    backend = c("cpu", "faiss", "cuvs", "cuda", "metal"),
    available = c(TRUE, faiss_knn, cuvs_knn, cuda_knn || cuda_embedding, metal_knn || metal_embedding),
    knn_available = c(TRUE, faiss_knn, cuvs_knn, cuda_knn, metal_knn),
    embedding_available = c(TRUE, FALSE, FALSE, cuda_embedding, metal_embedding),
    explicit_backend = c("cpu", "faiss", "cuda_cuvs", "cuda", "metal"),
    device = c(cpu_summary(), faiss$device, cuvs$device, cuda$device, metal$device),
    runtime = c(R.version$platform, faiss$runtime, cuvs$runtime, cuda$runtime, metal$runtime),
    note = c(
      "Native CPU path is always available.",
      if (faiss_knn) {
        "Real FAISS C++ KNN is available for explicit FAISS requests."
      } else {
        "Real FAISS C++ KNN is unavailable; explicit FAISS requests will fail."
      },
      if (cuvs_knn) {
        "RAPIDS cuVS CUDA KNN is available for explicit cuVS requests."
      } else {
        "RAPIDS cuVS CUDA KNN is unavailable; explicit cuVS requests will fail."
      },
      if (cuda_knn || cuda_embedding) {
        "Native CUDA path is available for explicit CUDA requests."
      } else {
        "Native CUDA path is unavailable; explicit CUDA requests will fail."
      },
      if (metal_knn || metal_embedding) {
        "Native Metal path is available for explicit Metal requests."
      } else {
        "Native Metal path is unavailable; explicit Metal requests will fail."
      }
    ),
    stringsAsFactors = FALSE
  )
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
    ". Use `backend_info()` to inspect available backends.",
    call. = FALSE
  )
}

available_native_gpu_backend <- function(need_knn = FALSE,
                                         need_embedding = FALSE) {
  cuda_ok <- (!isTRUE(need_knn) || backend_flag(cuda_available)) &&
    (!isTRUE(need_embedding) || backend_flag(embedding_cuda_available_cpp))
  if (cuda_ok) return("cuda")

  metal_ok <- (!isTRUE(need_knn) || backend_flag(metal_available)) &&
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

cpu_summary <- function() {
  cores <- suppressWarnings(parallel::detectCores(logical = TRUE))
  if (length(cores) != 1L || is.na(cores) || !is.finite(cores)) {
    "CPU"
  } else {
    paste0("CPU (", cores, " logical cores)")
  }
}

nvidia_smi_summary <- function() {
  smi <- Sys.which("nvidia-smi")
  if (!nzchar(smi)) {
    return(list(device = NA_character_, runtime = NA_character_))
  }
  out <- tryCatch(
    system2(
      smi,
      c(
        "--query-gpu=name,driver_version,memory.total",
        "--format=csv,noheader,nounits"
      ),
      stdout = TRUE,
      stderr = FALSE
    ),
    error = function(e) character()
  )
  if (length(out) < 1L || !nzchar(out[1L])) {
    return(list(device = NA_character_, runtime = NA_character_))
  }
  parts <- trimws(strsplit(out[1L], ",", fixed = TRUE)[[1L]])
  device <- if (length(parts) >= 1L) parts[1L] else NA_character_
  driver <- if (length(parts) >= 2L) parts[2L] else NA_character_
  memory <- if (length(parts) >= 3L) parts[3L] else NA_character_
  runtime <- paste(
    c(
      if (!is.na(driver) && nzchar(driver)) paste0("driver ", driver) else NULL,
      if (!is.na(memory) && nzchar(memory)) paste0(memory, " MiB") else NULL
    ),
    collapse = ", "
  )
  if (!nzchar(runtime)) runtime <- NA_character_
  list(device = device, runtime = runtime)
}

cuda_summary <- function() {
  native <- cuda_native_summary()
  smi <- nvidia_smi_summary()

  device <- first_nonempty(native$device, smi$device)
  runtime <- combine_nonempty(native$runtime, smi$runtime)
  list(device = device, runtime = runtime)
}

cuda_native_summary <- function() {
  text <- tryCatch(
    cuda_device_info_json_cpp(),
    error = function(e) NA_character_
  )
  if (length(text) != 1L || is.na(text) || !nzchar(text)) {
    return(list(device = NA_character_, runtime = NA_character_))
  }

  available <- json_get_bool(text, "available")
  if (isTRUE(available)) {
    device <- json_get_string(text, "name")
    compute <- json_get_string(text, "compute_capability")
    total_memory <- json_get_number(text, "total_memory")
    free_memory <- json_get_number(text, "free_memory")
    memory <- cuda_memory_summary(free_memory, total_memory)
    runtime <- combine_nonempty(
      if (!is.na(compute)) paste0("compute capability ", compute) else NA_character_,
      memory
    )
    return(list(device = device, runtime = runtime))
  }

  reason <- json_get_string(text, "reason")
  list(device = NA_character_, runtime = reason)
}

faiss_summary <- function() {
  text <- tryCatch(
    faiss_info_json_cpp(),
    error = function(e) NA_character_
  )
  available <- json_get_bool(text, "available")
  reason <- json_get_string(text, "reason")
  runtime <- if (isTRUE(available)) {
    "FAISS C++ library"
  } else if (!is.na(reason)) {
    reason
  } else {
    NA_character_
  }
  list(device = "CPU", runtime = runtime)
}

cuvs_summary <- function() {
  text <- tryCatch(
    cuvs_info_json_cpp(),
    error = function(e) NA_character_
  )
  available <- json_get_bool(text, "available")
  reason <- json_get_string(text, "reason")
  device <- json_get_string(text, "device")
  compute <- json_get_string(text, "compute_capability")
  total_memory <- json_get_number(text, "total_memory")
  runtime <- if (isTRUE(available)) {
    combine_nonempty(
      "RAPIDS cuVS C API",
      if (!is.na(compute)) paste0("compute capability ", compute) else NA_character_,
      cuda_memory_summary(NA_real_, total_memory)
    )
  } else if (!is.na(reason)) {
    reason
  } else {
    NA_character_
  }
  list(device = if (!is.na(device)) device else "CUDA GPU", runtime = runtime)
}

json_get_bool <- function(text, key) {
  value <- json_capture(text, key, "(true|false)")
  if (is.na(value)) return(NA)
  identical(tolower(value), "true")
}

json_get_number <- function(text, key) {
  value <- json_capture(text, key, "([0-9]+(?:\\.[0-9]+)?)")
  if (is.na(value)) return(NA_real_)
  as.numeric(value)
}

json_get_string <- function(text, key) {
  value <- json_capture(text, key, "\"((?:\\\\.|[^\"\\\\])*)\"")
  if (is.na(value)) return(NA_character_)
  json_unescape(value)
}

json_capture <- function(text, key, value_pattern) {
  key_pattern <- paste0("\"", gsub("([\\W])", "\\\\\\1", key), "\"")
  pattern <- paste0(key_pattern, "\\s*:\\s*", value_pattern)
  match <- regexec(pattern, text, perl = TRUE)
  parts <- regmatches(text, match)[[1L]]
  if (length(parts) < 2L) NA_character_ else parts[2L]
}

json_unescape <- function(value) {
  value <- gsub("\\\\n", "\n", value)
  value <- gsub("\\\\r", "\r", value)
  value <- gsub("\\\\t", "\t", value)
  value <- gsub("\\\\\"", "\"", value)
  gsub("\\\\\\\\", "\\\\", value)
}

cuda_memory_summary <- function(free_memory, total_memory) {
  if (is.na(total_memory) || total_memory <= 0) {
    return(NA_character_)
  }
  total <- bytes_to_gib(total_memory)
  if (!is.na(free_memory) && free_memory >= 0) {
    return(paste0(bytes_to_gib(free_memory), " GiB free / ", total, " GiB total"))
  }
  paste0(total, " GiB total")
}

bytes_to_gib <- function(bytes) {
  format(round(bytes / 1024^3, 2), nsmall = 2L, trim = TRUE)
}

first_nonempty <- function(...) {
  values <- unlist(list(...), use.names = FALSE)
  values <- values[!is.na(values) & nzchar(values)]
  if (length(values) == 0L) NA_character_ else values[1L]
}

combine_nonempty <- function(...) {
  values <- unlist(list(...), use.names = FALSE)
  values <- values[!is.na(values) & nzchar(values)]
  values <- unique(values)
  if (length(values) == 0L) NA_character_ else paste(values, collapse = ", ")
}

metal_summary <- function() {
  if (!identical(Sys.info()[["sysname"]], "Darwin")) {
    return(list(device = NA_character_, runtime = NA_character_))
  }
  arch <- R.version$platform
  device <- if (grepl("arm64|aarch64", arch)) {
    "Apple Silicon Metal"
  } else {
    "macOS Metal"
  }
  version <- tryCatch(
    system2("sw_vers", "-productVersion", stdout = TRUE, stderr = FALSE),
    error = function(e) character()
  )
  runtime <- if (length(version) >= 1L && nzchar(version[1L])) {
    paste0("macOS ", version[1L])
  } else {
    "macOS"
  }
  list(device = device, runtime = runtime)
}
