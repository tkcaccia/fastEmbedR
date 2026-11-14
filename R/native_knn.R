normalize_nn_threads <- function(n_threads) {
    resolve_n_cores(n_threads)
}

fastembedr_optional_namespace_available <- function(package) {
    requireNamespace(package, quietly = TRUE)
}

fastembedr_optional_export <- function(package, name) {
    getExportedValue(package, name)
}

fastembedr_has_gpu_knn_shape <- function(x) {
    is.list(x) &&
        all(c("indices_ptr", "distances_ptr", "n_query", "k") %in% names(x)) &&
        identical(
            as.character(x$result_residency %||%
                attr(x, "result_residency") %||% ""),
            "cuda"
        )
}

fastembedr_as_gpu_knn <- function(x) {
    if (!fastembedr_has_gpu_knn_shape(x) || inherits(x, "fastEmbedR_gpu_knn")) {
        return(x)
    }
    class(x) <- unique(c("fastEmbedR_external_gpu_knn", class(x), "list"))
    x
}

fastembedr_is_gpu_knn <- function(x) {
    inherits(x, "fastEmbedR_gpu_knn") || fastembedr_has_gpu_knn_shape(x)
}

fastembedr_gpu_knn_to_host <- function(knn) {
    if (!fastembedr_is_gpu_knn(knn)) {
        return(knn)
    }
    if (inherits(knn, "faissR_gpu_knn")) {
        out <- faissR::gpu_knn_to_host(knn)
        attr(out, "gpu_resident_source") <- TRUE
        attr(out, "gpu_backend_used") <- knn$backend_used
        attr(out, "backend") <- knn$backend_used
        attr(out, "exclude_self") <- knn$exclude_self
        return(out)
    }
    native_provider <- inherits(knn, "fastEmbedR_gpu_knn") ||
        startsWith(as.character(knn$gpu_provider %||% ""), "fastEmbedR_native_")
    if (!native_provider) {
        stop(
            "fastEmbedR does not call another package to materialize ",
            "an external GPU KNN object. Convert it explicitly before ",
            "passing it to fastEmbedR, ",
            "or use fastEmbedR's native KNN route.",
            call. = FALSE
        )
    }
    out <- native_cuda_knn_to_host_cpp(knn)
    attr(out, "gpu_resident_source") <- TRUE
    attr(out, "gpu_backend_used") <-
        knn$backend_used %||% attr(knn, "backend") %||% "cuda"
    attr(out, "backend") <- attr(out, "backend") %||% "native_cuda_host"
    metric <- knn$metric %||% attr(knn, "metric")
    if (!is.null(metric)) attr(out, "metric") <- metric
    out
}

fastembedr_gpu_knn_info <- function(knn) {
    if (!fastembedr_is_gpu_knn(knn)) {
        stop("Expected a CUDA GPU-resident KNN object.", call. = FALSE)
    }
    n <- as.integer(knn$n_query %||% knn$n %||% NA_integer_)
    k <- as.integer(knn$k %||% NA_integer_)
    if (length(n) != 1L || is.na(n) || n < 2L ||
        length(k) != 1L || is.na(k) || k < 1L) {
        stop("Invalid CUDA GPU-resident KNN dimensions.", call. = FALSE)
    }
    list(
        n = n,
        k = k,
        has_self = !isTRUE(
            knn$exclude_self %||% attr(knn, "exclude_self") %||% FALSE
        ),
        input_backend = knn$backend_used %||% attr(knn, "backend_used") %||%
            attr(knn, "backend") %||% "cuda",
        metric = knn$metric %||% attr(knn, "metric") %||% NA_character_,
        distance_type = knn$distance_type %||%
            attr(knn, "distance_type") %||% "float32",
        result_residency = knn$result_residency %||%
            attr(knn, "result_residency") %||% "cuda"
    )
}

fastembedr_convert_knn_distances <- function(knn, output) {
    if (!identical(output, "float") || !is.list(knn) ||
        !("distances" %in% names(knn)) || is_float32_matrix(knn$distances)) {
        return(knn)
    }
    if (requireNamespace("float", quietly = TRUE)) {
        knn$distances <- float::fl(knn$distances)
        attr(knn, "distance_type") <- "float32"
    }
    knn
}

fastembedr_cpu_knn_method <- function(n) {
    n <- integer_scalar(n %||% NA_integer_)
    if (length(n) == 1L && !is.na(n) && n < 5000L) {
        "exact"
    } else {
        "hnsw"
    }
}

fastembedr_embedding_nn_policy <- function(embedding_backend, n = NULL,
                                            p = NULL) {
    embedding_backend <- resolve_embedding_backend(embedding_backend)
    n <- integer_scalar(n %||% NA_integer_)
    p <- integer_scalar(p %||% NA_integer_)
    small_enough_for_exact <- length(n) == 1L && !is.na(n) && n < 100000L
    method <- if (isTRUE(small_enough_for_exact)) "exact" else "ivf"
    if (identical(embedding_backend, "cuda")) {
        return(list(
            backend = "cuda", method = method, tuning = "auto",
            target_recall = 0.99
        ))
    }
    if (identical(embedding_backend, "metal")) {
        bounded_exact <- !is.na(n) && !is.na(p) &&
            n <= 50000L && p <= 16384L &&
            as.double(n) * n * p <= 1.25e11
        return(list(
            backend = "metal",
            method = if ((!is.na(n) && n < 4096L) || bounded_exact) {
                "exact"
            } else {
                "ivf"
            },
            tuning = "auto",
            target_recall = 0.99
        ))
    }
    method <- fastembedr_cpu_knn_method(n)
    list(
        backend = "cpu", method = method,
        target_recall = 0.99,
        recall_status = if (identical(method, "exact")) {
            "exact_by_construction"
        } else {
            "calibration_informed_not_runtime_audited"
        }
    )
}

fastembedr_metal_query_method <- function(n_reference, n_query, p) {
    complete <- all(vapply(
        list(n_reference, n_query, p),
        function(x) length(x) == 1L && !is.na(x),
        logical(1L)
    ))
    work <- if (complete) {
        as.double(n_reference) * as.double(n_query) * as.double(p)
    } else {
        NA_real_
    }
    use_ivf <- if (is.finite(work)) {
        n_reference >= 4096L && work >= 5e9
    } else {
        length(n_reference) == 1L &&
            !is.na(n_reference) &&
            n_reference >= 20000L
    }
    if (isTRUE(use_ivf)) "ivf" else "exact"
}

fastembedr_cuda_query_method <- function(n_reference, n_query, p) {
    complete <- all(vapply(
        list(n_reference, n_query, p),
        function(x) length(x) == 1L && !is.na(x),
        logical(1L)
    ))
    if (!complete) {
        return(if (
            length(n_reference) == 1L &&
                !is.na(n_reference) &&
                n_reference >= 100000L
        ) "ivf" else "exact")
    }
    work <- as.double(n_reference) * as.double(n_query) * as.double(p)
    use_ivf <- n_reference >= 100000L && n_query > 64L && work >= 5e9
    if (isTRUE(use_ivf)) "ivf" else "exact"
}

fastembedr_query_nn_policy <- function(embedding_backend,
                                        n_reference = NULL,
                                        n_query = NULL,
                                        p = NULL) {
    embedding_backend <- resolve_embedding_backend(embedding_backend)
    n_reference <- integer_scalar(n_reference %||% NA_integer_)
    n_query <- integer_scalar(n_query %||% NA_integer_)
    p <- integer_scalar(p %||% NA_integer_)
    if (identical(embedding_backend, "cuda")) {
        return(list(
            backend = "cuda",
            method = fastembedr_cuda_query_method(
                n_reference,
                n_query,
                p
            ),
            tuning = "auto",
            target_recall = 0.99
        ))
    }
    if (identical(embedding_backend, "metal")) {
        return(list(
            backend = "metal",
            method = fastembedr_metal_query_method(
                n_reference,
                n_query,
                p
            ),
            tuning = "auto",
            target_recall = 0.99
        ))
    }
    method <- fastembedr_cpu_knn_method(n_reference)
    list(
        backend = "cpu", method = method,
        target_recall = 0.99,
        recall_status = if (identical(method, "exact")) {
            "exact_by_construction"
        } else {
            "calibration_informed_not_runtime_audited"
        }
    )
}

fastembedr_nn_policy_engine <- function(policy, keep_gpu = FALSE,
                                        metric = "euclidean") {
    if (is.list(policy) && identical(policy$backend, "cuda")) {
        prefix <- if (isTRUE(keep_gpu)) {
            "native_cuvs_gpu_"
        } else {
            "native_cuvs_cuda_host_"
        }
        return(paste0(prefix, policy$method %||% "auto"))
    }
    if (is.list(policy) && identical(policy$backend, "cpu")) {
        if (identical(policy$method, "hnsw")) {
            return("faissr_cpu_hnsw")
        }
        if (identical(policy$method, "exact")) {
            return("faissr_cpu_flat")
        }
        return(paste0("native_cpu_", policy$method %||% "exact"))
    }
    if (is.list(policy) && identical(policy$backend, "metal")) {
        return(paste0("native_metal_", policy$method %||% "ivf"))
    }
    "native_unavailable"
}

validate_precompute_k <- function(k, upper, label) {
    k <- integer_scalar(k)
    if (length(k) != 1L ||
        is.na(k) ||
        !is.finite(k) ||
        k < 1L ||
        k > upper) {
        stop(
            "`k` must be one integer between 1 and ",
            label,
            ".",
            call. = FALSE
        )
    }
    k
}

finish_precomputed_knn <- function(out, metadata, policy, elapsed) {
    keep_gpu <- isTRUE(metadata$keep_gpu)
    out$n <- as.integer(metadata$n)
    out$k <- as.integer(metadata$k)
    out$metric <- metadata$metric
    out$exclude_self <- metadata$exclude_self
    out$backend_requested <- metadata$backend
    out$execution_backend <- metadata$backend
    out$engine <- fastembedr_nn_policy_engine(
        policy, keep_gpu, metadata$metric
    )
    out$elapsed_sec <- unname(elapsed[["elapsed"]])
    if (identical(policy$backend, "cpu") &&
        !identical(policy$method, "exact")) {
        out$target_recall <- policy$target_recall
        out$target_met <- NA
        out$recall_audited <- FALSE
        out$recall_status <- out$recall_status %||% policy$recall_status
    } else {
        out$target_recall <- policy$target_recall
    }
    out$result_residency <- if (keep_gpu) "cuda" else "host"
    classes <- if (keep_gpu) {
        c("fastEmbedR_gpu_knn", "fastEmbedR_knn", class(out), "list")
    } else {
        c("fastEmbedR_knn", class(out), "list")
    }
    class(out) <- unique(classes)
    attr(out, "backend") <- metadata$backend
    attr(out, "backend_requested") <- metadata$backend
    attr(out, "metric") <- metadata$metric
    attr(out, "exclude_self") <- metadata$exclude_self
    attr(out, "result_residency") <- out$result_residency
    out
}

run_precompute_knn <- function(x, k, metric, policy, n_threads,
                                keep_gpu) {
    fastembedr_nn_without_self(
        x,
        k = k,
        backend = policy$backend,
        method = policy$method,
        metric = metric,
        output = fastembedr_knn_output_type(x, policy$backend),
        n_threads = n_threads,
        tuning = policy$tuning,
        target_recall = policy$target_recall,
        keep_gpu = keep_gpu
    )
}

#' Precompute nearest neighbors
#'
#' `precompute_knn()` exposes the same nearest-neighbor search
#' used internally by [umap()] and [tsne()]. The search algorithm is chosen
#' by fastEmbedR for the requested backend and is deliberately not a user
#' parameter.
#'
#' @param data Numeric matrix, numeric data frame, `float::float32` matrix,
#'   or [massive_matrix()] file descriptor with observations in rows.
#' @param k Number of non-self nearest neighbors to return.
#' @param metric Distance metric: `"euclidean"`, `"cosine"`, or
#'   `"correlation"`.
#' @param backend Search backend: `"cpu"`, `"cuda"`, or `"metal"`.
#' @param n.cores Number of CPU cores. Native GPU backends ignore
#'   this argument.
#' @param storage `"auto"`, `"resident"`, or `"file_backed"`. When omitted,
#'   `options(storage = ...)` is used. A file-backed input must be
#'   a disk-backed [massive_matrix()] descriptor; a resident matrix is never
#'   silently copied to disk.
#' @param output New file prefix required for a file-backed KNN result.
#' @param memory_limit Host working-memory budget for file-backed KNN. When
#'   omitted, `options(memory_limit = ...)` is used, or `"8GB"`.
#' @param devices Optional CUDA device indices for file-backed KNN. Resident
#'   `precompute_knn()` does not accept this argument.
#' @param checkpoint Save completed file-backed KNN query blocks.
#' @param resume Continue a matching file-backed KNN checkpoint.
#' @param checkpoint_every Save progress after this many query blocks.
#'
#' @details
#' CPU uses faissR Flat exact search below 5,000 observations and
#' faissR's compiled FAISS HNSW interface for Euclidean distance otherwise.
#' Cosine and correlation use faissR HNSW. HNSW targets recall 0.99; its
#' recall is not audited during each call. Metal uses native exact search below
#' 4,096 rows and for inputs with at most 50,000 rows, 16,384 columns, and
#' 125 billion row-pair feature operations; otherwise it uses recall-tuned
#' IVF-Flat. CUDA uses RAPIDS
#' cuVS brute-force exact search below 100,000 observations and cuVS IVF-Flat
#' above that threshold. Approximate Metal and CUDA routes use an internal
#' recall target of 0.99 and report whether their pilot audit met it.
#'
#' A file-backed descriptor uses bounded KNN graph construction and writes
#' indices and distances under `output`. CPU selects exact search up to 5,000
#' rows when estimated pair-feature work is at most 125 billion, and sharded
#' HNSW otherwise. CUDA selects exact search below 100,000 rows under the same
#' work limit and sharded IVF-Flat otherwise. Metal selects bounded exact
#' search. File-backed KNN currently supports Euclidean distance only. The
#' result is a `fastEmbedR_massive_graph` descriptor; construct UMAP fuzzy
#' weights or t-SNE affinities from it before full-graph fitting.
#'
#' The resident CUDA result remains on the GPU and can be passed directly to
#' [umap_knn()] or [tsne_knn()] with `backend = "cuda"`. CPU and Metal
#' results contain one-based `indices` and `distances` matrices. Every result
#' excludes the observation itself. An unavailable requested backend raises an
#' error; no CPU fallback is reported as GPU work.
#'
#' @return A resident `fastEmbedR_knn` object, or a file-backed
#'   `fastEmbedR_massive_graph` descriptor. Resident host results contain
#'   `indices` and `distances`; resident CUDA results additionally inherit
#'   from `fastEmbedR_gpu_knn` and own device-resident buffers.
#' @examples
#' x <- scale(as.matrix(iris[, 1:4]))
#' knn <- precompute_knn(x, k = 15, backend = "cpu", n.cores = 2)
#' layout <- umap_knn(knn, backend = "cpu", seed = 1)
#' @export
precompute_knn <- function(data, k = 30L,
        metric = c("euclidean", "cosine", "correlation"),
        backend = NULL, n.cores = NULL, storage = NULL,
        output = NULL, memory_limit = NULL, devices = NULL,
        checkpoint = FALSE, resume = FALSE,
        checkpoint_every = 100L) {
    massive_checkpoint_validate_controls(checkpoint, resume)
    if (inherits(data, "fastEmbedR_massive_graph")) stop(
        "precompute_knn() needs a data matrix, not a prepared graph.",
        call. = FALSE)
    fastembedr_storage_option(data, storage)
    backend <- resolve_embedding_backend(backend)
    metric <- resolve_embedding_metric(metric, data)
    if (fastembedr_file_backed_input(data)) return(
        precompute_knn_backed(data, k, metric, backend, n.cores,
            output, memory_limit, devices, checkpoint, resume,
            checkpoint_every))
    if (inherits(data, "fastEmbedR_massive_matrix")) stop(
        "A memory-backed massive_matrix() is not a disk-backed source; ",
        "pass its resident matrix to precompute_knn().", call. = FALSE)
    if (!is.null(output) || !is.null(memory_limit) || !is.null(devices) ||
        checkpoint || resume || !identical(checkpoint_every, 100L)) stop(
        "File-backed KNN controls require a disk-backed massive_matrix().",
        call. = FALSE)
    prepared <- prepare_embedding_data(
        data,
        standardize = FALSE,
        pca_dims = NULL,
        seed = 4L,
        backend = backend
    )
    x <- prepared$data
    n <- nrow(x)
    k <- validate_precompute_k(k, n - 1L, "nrow(data) - 1")
    policy <- fastembedr_embedding_nn_policy(
        backend, n = n, p = ncol(x))
    keep_gpu <- identical(backend, "cuda")
    result <- timed_do_call(run_precompute_knn, list(
        x, k, metric, policy, n.cores, keep_gpu
    ))
    metadata <- list(
        n = n, k = k, metric = metric, exclude_self = TRUE,
        backend = backend, keep_gpu = keep_gpu
    )
    finish_precomputed_knn(result$value, metadata, policy,
        result$time)
}

precompute_knn_backed <- function(data, k, metric, backend, n.cores,
        output, memory_limit, devices, checkpoint, resume,
        checkpoint_every) {
    if (!identical(metric, "euclidean")) stop(
        "File-backed KNN currently supports Euclidean distance only.",
        call. = FALSE)
    if (is.null(output)) stop(
        "File-backed KNN requires a new `output` file prefix.",
        call. = FALSE)
    limit <- if (is.null(memory_limit))
        fastembedr_memory_option("8GB") else memory_limit
    massive_memory_bytes(limit)
    pair_work <- as.double(data$nrow) * data$nrow * data$ncol
    bounded_exact <- is.finite(pair_work) && pair_work <= 125e9
    method <- if (backend == "cpu") {
        if (data$nrow <= 5000L && bounded_exact)
            "exact" else "hnsw_sharded"
    } else if (backend == "cuda") {
        if (data$nrow < 100000L && bounded_exact)
            "exact" else "ivf_sharded"
    } else "exact"
    massive_full_knn_graph(data, k = k, output = output,
        backend = backend, n.cores = resolve_n_cores(n.cores),
        memory_limit = limit, method = method, devices = devices,
        checkpoint = checkpoint, resume = resume,
        checkpoint_every = checkpoint_every)
}

prepare_query_knn_inputs <- function(reference, query, backend) {
    args <- list(
        standardize = FALSE,
        pca_dims = NULL,
        seed = 4L,
        backend = backend
    )
    reference <- do.call(
        prepare_embedding_data,
        c(list(reference), args)
    )$data
    query <- do.call(
        prepare_embedding_data,
        c(list(query), args)
    )$data
    list(reference = reference, query = query)
}

run_precompute_query_knn <- function(reference, query, k, metric,
                                        policy, n_threads, keep_gpu) {
    fastembedr_native_query_knn(
        reference,
        query,
        k = k,
        metric = metric,
        n_threads = n_threads,
        target_recall = policy$target_recall,
        output = fastembedr_knn_output_type(
            reference,
            policy$backend
        ),
        backend = policy$backend,
        method = policy$method,
        keep_gpu = keep_gpu
    )
}

#' Precompute query-to-reference nearest neighbors
#'
#' `precompute_query_knn()` searches a fixed reference matrix for every row of
#' a query matrix. It uses the same backend family and routing
#' policy as [precompute_knn()], but does not compute unnecessary
#' reference-to-reference or query-to-query neighbors.
#'
#' @param reference Reference observations in rows.
#' @param query Query observations in rows and the same feature space as
#'   `reference`.
#' @inheritParams precompute_knn
#'
#' @details
#' CPU uses faissR Flat search when the reference has fewer than
#' 5,000 rows and faissR's compiled FAISS HNSW interface for Euclidean
#' distance otherwise. Cosine and correlation use faissR HNSW. HNSW
#' targets recall 0.99 but does not claim a per-call recall audit. Metal
#' routes between a native
#' query-only exact kernel and recall-tuned IVF-Flat from the estimated
#' reference-query distance workload. CUDA routes between cuVS brute-force
#' exact search and cuVS IVF-Flat using the reference size, query batch size,
#' and feature count. CUDA results remain device-resident for direct consumption
#' by landmark UMAP and t-SNE transformations.
#'
#' @return A `fastEmbedR_knn` object with one row per query observation and
#'   one-based indices into `reference`.
#' @examples
#' x <- scale(as.matrix(iris[, 1:4]))
#' ref <- x[1:100, , drop = FALSE]
#' qry <- x[101:150, , drop = FALSE]
#' knn <- precompute_query_knn(ref, qry, k = 10, backend = "cpu")
#' @export
#' @keywords internal
precompute_query_knn <- function(reference,
                                    query,
                                    k = 30L,
                                    metric = c(
                                        "euclidean", "cosine", "correlation"
                                    ),
                                    backend = NULL,
                                    n.cores = NULL) {
    backend <- resolve_embedding_backend(backend)
    metric <- resolve_embedding_metric(metric, reference)
    inputs <- prepare_query_knn_inputs(reference, query, backend)
    reference <- inputs$reference
    query <- inputs$query
    if (ncol(reference) != ncol(query)) {
        stop("`reference` and `query` must have the same number of columns.",
            call. = FALSE
        )
    }
    k <- validate_precompute_k(k, nrow(reference), "nrow(reference)")
    policy <- fastembedr_query_nn_policy(
        backend,
        n_reference = nrow(reference),
        n_query = nrow(query),
        p = ncol(reference)
    )
    keep_gpu <- identical(backend, "cuda")
    result <- timed_do_call(run_precompute_query_knn, list(
        reference, query, k, metric, policy, n.cores, keep_gpu
    ))
    metadata <- list(
        n = nrow(query), k = k, metric = metric, exclude_self = FALSE,
        backend = backend, keep_gpu = keep_gpu
    )
    out <- finish_precomputed_knn(
        result$value, metadata, policy, result$time
    )
    out$n_query <- as.integer(nrow(query))
    out$n_reference <- as.integer(nrow(reference))
    out
}

run_native_cuda_knn <- function(data, k, method, metric, output,
                                target_recall, keep_gpu, retain_data) {
    allowed <- c("auto", "exact", "flat", "bruteforce", "ivf")
    if (!method %in% allowed) {
        stop("CUDA native KNN supports `auto`, `exact`, and `ivf`.",
            call. = FALSE
        )
    }
    exact <- method %in% c("exact", "flat", "bruteforce") ||
        (method == "auto" && nrow(data) < 100000L)
    if (isTRUE(keep_gpu)) {
        if (exact) {
            return(run_faissr_cuda_resident_exact(
                data, k, metric, target_recall, retain_data))
        }
        if (identical(metric, "euclidean")) {
            return(run_faissr_cuda_resident_ivf(
                data, k, target_recall, retain_data))
        }
    }
    if (!isTRUE(native_cuda_knn_available_cpp())) {
        stop("Native CUDA KNN is unavailable; no fallback was used.",
            call. = FALSE
        )
    }
    out <- native_cuda_knn_cpp(
        data,
        k = k,
        method = method,
        metric = metric,
        target_recall = target_recall,
        keep_gpu = isTRUE(keep_gpu),
        retain_data = isTRUE(retain_data)
    )
    if (keep_gpu) out else fastembedr_convert_knn_distances(out, output)
}

run_faissr_cuda_resident_exact <- function(data, k, metric,
                                            target_recall,
                                            retain_data = FALSE) {
    if (!isTRUE(faissR::faiss_gpu_available())) stop(
        "faissR CUDA KNN is unavailable; no fallback was used.",
        call. = FALSE)
    out <- faissr_cuda_resident_knn_cpp(
        data, k, "exact", metric, target_recall
    )
    if (!inherits(out, "faissR_gpu_knn") ||
        !identical(out$result_residency, "cuda") ||
        !isTRUE(out$exclude_self) || !isTRUE(out$exact) ||
        !identical(as.integer(out$k), as.integer(k)) ||
        !identical(as.integer(out$n_query), as.integer(nrow(data))) ||
        isTRUE(out$cpu_fallback)) stop(
        "faissR did not return exact resident CUDA KNN.",
        call. = FALSE)
    if (isTRUE(retain_data)) {
        out <- faissr_gpu_knn_retain_data_cpp(out, data)
    }
    out
}

fastembedr_audit_resident_ivf <- function(data, knn, k, target_recall) {
    rows <- unique(as.integer(round(seq(
        1, nrow(data), length.out = min(256L, nrow(data))
    ))))
    sampled <- faissr_gpu_knn_sample_cpp(knn, rows)
    found <- sampled$indices
    if (!identical(dim(found), c(length(rows), as.integer(k))) ||
        anyNA(found) || any(found < 1L | found > nrow(data))) {
        stop("faissR resident IVF returned invalid pilot IDs.",
            call. = FALSE)
    }
    reference <- faissR::nn(
        data, points = data[rows, , drop = FALSE],
        k = k + 1L, exclude_self = FALSE, backend = "cpu",
        method = "flat", metric = "euclidean", output = "double"
    )
    if (!isTRUE(reference$exact) ||
        !startsWith(reference$backend_used %||% "", "faiss_flat_")) {
        stop("faissR did not return an exact CPU pilot reference.",
            call. = FALSE)
    }
    row_recall <- vapply(seq_along(rows), function(i) {
        ids <- found[i, ]
        if (anyDuplicated(ids) || rows[i] %in% ids) return(0)
        truth <- reference$indices[i, ]
        truth <- truth[truth != rows[i]][seq_len(k)]
        sum(ids %in% truth) / k
    }, numeric(1L))
    mean_recall <- mean(row_recall)
    if (mean_recall < target_recall) stop(
        "faissR resident IVF pilot recall ",
        format(mean_recall, digits = 5L), " is below target ",
        target_recall, ".", call. = FALSE)
    list(
        rows = rows, mean = mean_recall, minimum = min(row_recall),
        reference_backend = reference$backend_used
    )
}

run_faissr_cuda_resident_ivf <- function(data, k, target_recall,
                                            retain_data = FALSE) {
    if (!isTRUE(faissR::faiss_gpu_available())) stop(
        "faissR CUDA KNN is unavailable; no fallback was used.",
        call. = FALSE)
    params <- utils::getFromNamespace("cuda_ivf_params", "faissR")(
        nrow(data), ncol(data), k, "euclidean", target_recall
    )
    out <- faissr_gpu_multi_knn_cpp(
        data, as.integer(k), "ivf", 0L,
        as.integer(params$nlist), as.integer(params$nprobe),
        FALSE, "gpu"
    )
    if (!inherits(out, "faissR_gpu_knn") ||
        !identical(out$result_residency, "cuda") ||
        !isTRUE(out$exclude_self) || isTRUE(out$exact) ||
        !identical(as.integer(out$k), as.integer(k)) ||
        !identical(as.integer(out$n_query), as.integer(nrow(data))) ||
        isTRUE(out$cpu_fallback)) stop(
        "faissR did not return resident CUDA IVF KNN.",
        call. = FALSE)
    pilot <- fastembedr_audit_resident_ivf(data, out, k, target_recall)
    out$nlist <- as.integer(params$nlist)
    out$nprobe <- as.integer(params$nprobe)
    out$target_recall <- target_recall
    out$pilot_recall <- pilot$mean
    out$pilot_min_row_recall <- pilot$minimum
    out$pilot_rows <- pilot$rows
    out$pilot_reference_backend <- pilot$reference_backend
    out$recall_audited <- TRUE
    out$recall_status <- "fixed_rows_exact_reference"
    out$target_met <- TRUE
    if (isTRUE(retain_data)) {
        out <- faissr_gpu_knn_retain_data_cpp(out, data)
    }
    out
}

run_faissr_cuda_embedding_knn <- function(data, k, method, metric,
                                            output, target_recall,
                                            devices = NULL) {
    if (identical(metric, "euclidean")) {
        return(run_faissr_cuda_multi_knn(
            data, k, method, metric, output, target_recall,
            if (is.null(devices)) 0L else devices
        ))
    }
    if (!is.null(devices) && !identical(as.integer(devices), 0L)) {
        stop(
            "Non-Euclidean faissR CUDA KNN currently uses GPU 0.",
            call. = FALSE
        )
    }
    method <- if (method %in% c("exact", "flat")) "flat" else method
    if (!method %in% c("flat", "ivf")) {
        stop("Unsupported faissR CUDA embedding KNN method.",
            call. = FALSE)
    }
    if (!requireNamespace("faissR", quietly = TRUE) ||
        !isTRUE(faissR::faiss_gpu_available())) {
        stop("The faissR FAISS GPU backend is unavailable.", call. = FALSE)
    }
    out <- faissR::nn(
        data, k = k, exclude_self = TRUE, backend = "cuda",
        method = method, metric = metric, tuning = "auto",
        target_recall = target_recall, output = output
    )
    expected <- if (identical(method, "flat")) {
        paste0("faiss_gpu_flat_", if (identical(metric, "euclidean")) {
            "l2"
        } else metric)
    } else "faiss_gpu_ivf_flat"
    actual <- out$backend_used %||% attr(out, "backend_used")
    if (!identical(actual, expected) ||
        !identical(out$accelerator, "cuda") ||
        !identical(dim(out$indices), c(nrow(data), as.integer(k))) ||
        !identical(out$index_base, 1L) ||
        isTRUE(out$cpu_fallback)) {
        stop("faissR did not return the requested FAISS CUDA KNN.",
            call. = FALSE)
    }
    out$result_residency <- "host"
    out$exclude_self <- TRUE
    out
}

run_faissr_cuda_multi_knn <- function(data, k, method, metric,
                                        output, target_recall, devices) {
    if (!identical(metric, "euclidean")) {
        stop("faissR multi-GPU KNN currently requires Euclidean distance.",
            call. = FALSE)
    }
    method <- if (method %in% c("exact", "flat")) "flat" else method
    if (!method %in% c("flat", "ivf")) {
        stop("faissR multi-GPU KNN requires Flat or IVF.", call. = FALSE)
    }
    if (!requireNamespace("faissR", quietly = TRUE) ||
        !isTRUE(faissR::faiss_gpu_available())) {
        stop("The faissR FAISS GPU backend is unavailable.", call. = FALSE)
    }
    params <- if (identical(method, "ivf")) {
        utils::getFromNamespace("cuda_ivf_params", "faissR")(
            nrow(data), ncol(data), k, metric, target_recall
        )
    } else list(nlist = 1L, nprobe = 1L)
    out <- faissr_gpu_multi_knn_cpp(
        data, as.integer(k), method, as.integer(devices),
        as.integer(params$nlist), as.integer(params$nprobe),
        FALSE, output
    )
    expected_mode <- if (length(devices) == 1L) "single" else "replicate"
    expected_backend <- if (length(devices) == 1L) {
        "faiss_gpu_single_self"
    } else "faiss_gpu_multi_replicate"
    if (!identical(out$accelerator, "cuda") ||
        !identical(out$multi_gpu_mode, expected_mode) ||
        !identical(out$backend_used, expected_backend) ||
        !identical(out$devices, as.integer(devices)) ||
        !identical(dim(out$indices), c(nrow(data), as.integer(k))) ||
        !identical(out$index_base, 1L) || isTRUE(out$cpu_fallback)) {
        stop("faissR did not return the requested multi-GPU KNN.",
            call. = FALSE)
    }
    out$target_recall <- target_recall
    out$recall_audited <- FALSE
    out$recall_status <- if (identical(method, "flat")) {
        "exact_by_construction"
    } else "calibration_informed_not_runtime_audited"
    out
}

faissr_hnsw_input <- function(x) {
    if (is_float32_matrix(x)) return(float::dbl(x))
    if (is.integer(x)) return(matrix(as.double(x), nrow(x), ncol(x)))
    x
}

run_faissr_hnsw <- function(data, query, k, metric, output,
                            target_recall, n_threads) {
    loadNamespace("faissR")
    n <- nrow(data)
    self <- is.null(query)
    if (self) query <- data
    float_input <- is_float32_matrix(data) && is_float32_matrix(query)
    if (!float_input) {
        data <- faissr_hnsw_input(data)
        query <- if (self) data else faissr_hnsw_input(query)
    }
    out <- if (identical(metric, "euclidean")) {
        faissr_hnsw_search_cpp(
            data, if (self) NULL else query,
            as.integer(n), as.integer(ncol(data)), as.integer(k),
            target_recall, normalize_nn_threads(n_threads), output)
    } else {
        args <- list(data = data, k = k, exclude_self = self,
            backend = "cpu", method = "hnsw", metric = metric,
            output = output, target_recall = target_recall,
            n_threads = normalize_nn_threads(n_threads))
        if (!self) args$points <- query
        do.call(faissR::nn, args)
    }
    expected_rows <- nrow(query)
    if (!identical(out$index_type, "IndexHNSWFlat") ||
        !identical(out$exact, FALSE) ||
        !identical(dim(out$indices), c(expected_rows, as.integer(k))) ||
        !identical(dim(out$distances), c(expected_rows, as.integer(k)))) {
        stop("faissR did not return the requested CPU HNSW result.",
            call. = FALSE)
    }
    out$method <- if (self) "faiss_hnsw" else "faiss_hnsw_query"
    out$backend_used <- "faiss_hnsw"
    out$exclude_self <- self
    out$M <- out$m
    out$efConstruction <- out$ef_construction
    out$efSearch <- out$ef_search
    out$recall_audited <- FALSE
    out$target_met <- NA
    out$recall_status <- "calibration_informed_not_runtime_audited"
    attr(out, "backend") <- "cpu"
    attr(out, "method") <- out$method
    attr(out, "exclude_self") <- self
    fastembedr_convert_knn_distances(out, output)
}

run_faissr_cpu_exact <- function(data, query, k, metric, output,
                                    target_recall, n_threads) {
    if (!requireNamespace("faissR", quietly = TRUE) ||
        !isTRUE(faissR::faiss_available())) {
        stop("The faissR CPU Flat backend is unavailable.",
            call. = FALSE)
    }
    self <- is.null(query)
    args <- list(
        data = data, k = k, exclude_self = self,
        backend = "cpu", method = "flat", metric = metric,
        output = output, n_threads = normalize_nn_threads(n_threads),
        target_recall = target_recall
    )
    if (!self) args$points <- query
    out <- do.call(faissR::nn, args)
    if (!isTRUE(out$exact) ||
        !identical(dim(out$indices), c(if (self) nrow(data) else
            nrow(query), as.integer(k))) ||
        !identical(out$index_base, 1L) ||
        !startsWith(out$backend_used %||% "", "faiss_flat_")) {
        stop("faissR did not return CPU Flat KNN.", call. = FALSE)
    }
    out$backend <- "cpu"
    out$method <- if (self) "faissr_flat" else "faissr_flat_query"
    out$target_recall <- target_recall
    out$target_met <- TRUE
    out$exact_recall_by_construction <- TRUE
    out$expected_recall_at_k <- 1
    out$recall_audited <- TRUE
    out$recall_status <- "exact_by_construction"
    out$tuning_policy <- "exact_below_5000"
    out$cpu_fallback <- FALSE
    attr(out, "backend") <- "cpu"
    attr(out, "method") <- out$method
    attr(out, "exclude_self") <- self
    out
}

run_native_cpu_knn <- function(data, k, method, metric, output,
                                target_recall, n_threads) {
    if (!method %in% c("auto", "exact", "hnsw")) {
        stop("Native CPU KNN supports `exact` and `hnsw`.",
            call. = FALSE
        )
    }
    if (!metric %in% c("euclidean", "cosine", "correlation")) {
        stop("Native CPU KNN does not support this metric.",
            call. = FALSE
        )
    }
    if (identical(method, "auto")) {
        method <- fastembedr_cpu_knn_method(nrow(data))
    }
    if (identical(method, "hnsw")) {
        return(run_faissr_hnsw(
            data, NULL, k, metric, output, target_recall, n_threads
        ))
    }
    if (identical(method, "exact")) return(run_faissr_cpu_exact(
        data, NULL, k, metric, output, target_recall, n_threads))
    stop("Unsupported CPU KNN method.", call. = FALSE)
}

fastembedr_metal_knn_available <- function() {
    requireNamespace("faissR", quietly = TRUE) &&
        isTRUE(utils::getFromNamespace("nn_metal_available_cpp",
            "faissR")())
}

validate_faissr_metal_knn <- function(out) {
    failed <- identical(out$method, "native_metal_ivf") &&
        !isTRUE(out$target_met)
    if (failed) {
        stop(
            "Native Metal IVF did not meet the requested recall target; ",
            "no fallback was used.",
            call. = FALSE
        )
    }
    out
}

run_faissr_metal_knn <- function(data, k, method, metric, output,
                                    target_recall) {
    if (!method %in% c("auto", "exact", "ivf")) {
        stop("Native Metal KNN supports `auto`, `exact`, and `ivf`.",
            call. = FALSE
        )
    }
    if (!metric %in% c("euclidean", "cosine", "correlation")) {
        stop("Native Metal KNN does not support this metric.",
            call. = FALSE
        )
    }
    if (!fastembedr_metal_knn_available()) {
        stop("Native Metal KNN is unavailable in this build.",
            call. = FALSE
        )
    }
    out <- faissr_metal_knn_cpp(
        data,
        k = k,
        method = method,
        metric = metric,
        target_recall = target_recall
    )
    out <- validate_faissr_metal_knn(out)
    attr(out, "backend") <- "metal"
    attr(out, "exclude_self") <- TRUE
    fastembedr_convert_knn_distances(out, output)
}

fastembedr_nn_without_self <- function(data,
                                        k,
                                        backend,
                                        method = "auto",
                                        metric = "euclidean",
                                        output = "double",
                                        n_threads = NULL,
                                        tuning = "auto",
                                        target_recall = NULL,
                                        keep_gpu = FALSE,
                                        retain_data = FALSE) {
    k <- integer_scalar(k)
    if (is.na(k) || k < 1L) {
        stop("`k` must be a positive integer.", call. = FALSE)
    }
    target_recall <- target_recall %||% 0.99
    switch(backend,
        cuda = if (!isTRUE(keep_gpu) && !isTRUE(retain_data)) {
            resolved <- if (method == "auto") {
                fastembedr_embedding_nn_policy("cuda", nrow(data))$method
            } else if (method == "bruteforce") "exact" else method
            run_faissr_cuda_embedding_knn(
                data, k, resolved, metric, output, target_recall
            )
        } else run_native_cuda_knn(
            data, k, method, metric, output, target_recall, keep_gpu,
            retain_data
        ),
        cpu = run_native_cpu_knn(
            data, k, method, metric, output, target_recall, n_threads
        ),
        metal = run_faissr_metal_knn(
            data, k, method, metric, output, target_recall
        ),
        stop("Unknown native KNN backend: ", backend, call. = FALSE)
    )
}

run_native_cpu_query_knn <- function(data, query, k, method, metric,
                                        output, target_recall, n_threads) {
    if (!method %in% c("auto", "exact", "hnsw")) {
        stop("Native CPU query KNN supports `exact` and `hnsw`.",
            call. = FALSE
        )
    }
    if (!metric %in% c("euclidean", "cosine", "correlation")) {
        stop("Native CPU query KNN does not support this metric.",
            call. = FALSE
        )
    }
    if (identical(method, "auto")) {
        method <- fastembedr_cpu_knn_method(nrow(data))
    }
    if (identical(method, "hnsw")) {
        return(run_faissr_hnsw(
            data, query, k, metric, output, target_recall, n_threads
        ))
    }
    if (identical(method, "exact")) return(run_faissr_cpu_exact(
        data, query, k, metric, output, target_recall, n_threads))
    stop("Unsupported CPU KNN method.", call. = FALSE)
}

fastembedr_native_query_knn <- function(data, query, k,
                                        metric = "euclidean", n_threads = NULL,
                                        target_recall = 0.99,
                                        output = "double", backend = "cpu",
                                        method = "auto", keep_gpu = FALSE) {
    k <- integer_scalar(k)
    if (is.na(k) || k < 1L) {
        stop("`k` must be a positive integer.", call. = FALSE)
    }
    if (identical(backend, "cuda")) {
        return(run_faissr_cuda_query_knn(
            data, query, k, method, metric, output,
            target_recall, keep_gpu
        ))
    }
    if (identical(backend, "metal")) {
        out <- faissr_metal_query_knn_cpp(
            data, query,
            k = k, method = method, metric = metric,
            target_recall = target_recall
        )
        if (identical(out$method, "native_metal_ivf_query") &&
            !isTRUE(out$target_met)) {
            stop(
                "Native Metal query IVF missed the requested recall target; ",
                "no fallback was used.",
                call. = FALSE
            )
        }
        attr(out, "backend") <- "metal"
        attr(out, "exclude_self") <- FALSE
        return(fastembedr_convert_knn_distances(out, output))
    }
    run_native_cpu_query_knn(
        data, query, k, method, metric, output,
        target_recall, n_threads
    )
}

run_faissr_cuda_query_knn <- function(data, query, k, method, metric,
                                        output, target_recall, keep_gpu) {
    if (!requireNamespace("faissR", quietly = TRUE) ||
        !isTRUE(faissR::faiss_gpu_available())) {
        stop("The faissR FAISS GPU backend is unavailable.",
            call. = FALSE)
    }
    if (identical(method, "auto")) {
        method <- fastembedr_cuda_query_method(
            nrow(data), nrow(query), ncol(data)
        )
    }
    if (!method %in% c("exact", "flat", "bruteforce", "ivf")) {
        stop("Unsupported faissR CUDA query KNN method.", call. = FALSE)
    }
    if (isTRUE(keep_gpu) && identical(method, "ivf")) {
        stop("faissR has no GPU-resident IVF query result.",
            call. = FALSE)
    }
    args <- list(
        data = data, points = query, k = k, exclude_self = FALSE,
        method = method, metric = metric,
        target_recall = target_recall
    )
    if (isTRUE(keep_gpu)) {
        out <- do.call(faissR::nn_gpu, args)
        if (!inherits(out, "faissR_gpu_knn") ||
            !identical(out$result_residency, "cuda") ||
            !identical(as.integer(out$n_query), as.integer(nrow(query))) ||
            !identical(as.integer(out$k), as.integer(k)) ||
            isTRUE(out$cpu_fallback)) stop(
            "faissR did not return resident CUDA query KNN.",
            call. = FALSE)
        return(out)
    }
    out <- do.call(faissR::nn, c(args, list(
        backend = "cuda", tuning = "auto", output = output
    )))
    if (!identical(out$accelerator, "cuda") ||
        !startsWith(out$backend_used %||% "", "faiss_gpu") ||
        !identical(dim(out$indices), c(nrow(query), as.integer(k))) ||
        !identical(out$index_base, 1L) || isTRUE(out$cpu_fallback)) {
        stop("faissR did not return CUDA query KNN.", call. = FALSE)
    }
    attr(out, "backend") <- "cuda"
    attr(out, "exclude_self") <- FALSE
    out
}
