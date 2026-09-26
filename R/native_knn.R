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

fastembedr_embedding_nn_policy <- function(embedding_backend, n = NULL) {
    embedding_backend <- resolve_embedding_backend(embedding_backend)
    n <- integer_scalar(n %||% NA_integer_)
    small_enough_for_exact <- length(n) == 1L && !is.na(n) && n < 100000L
    method <- if (isTRUE(small_enough_for_exact)) "exact" else "ivf"
    if (identical(embedding_backend, "cuda")) {
        return(list(
            backend = "cuda", method = method, tuning = "auto",
            target_recall = 0.99
        ))
    }
    if (identical(embedding_backend, "metal")) {
        return(list(
            backend = "metal",
            method = if (length(n) == 1L && !is.na(n) && n < 4096L) {
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

fastembedr_nn_policy_engine <- function(policy, keep_gpu = FALSE) {
    if (is.list(policy) && identical(policy$backend, "cuda")) {
        prefix <- if (isTRUE(keep_gpu)) {
            "native_cuvs_gpu_"
        } else {
            "native_cuvs_cuda_host_"
        }
        return(paste0(prefix, policy$method %||% "auto"))
    }
    if (is.list(policy) && identical(policy$backend, "cpu")) {
        return(paste0("native_cpu_", policy$method %||% "hnsw"))
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
    out$engine <- fastembedr_nn_policy_engine(policy, keep_gpu)
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

#' Precompute native nearest neighbors
#'
#' `precompute_knn()` exposes the same package-native nearest-neighbor search
#' used internally by [umap()] and [tsne()]. The search algorithm is chosen
#' by fastEmbedR for the requested backend and is deliberately not a user
#' parameter.
#'
#' @param data Numeric matrix, numeric data frame, or a `float::float32` matrix
#'   with observations in rows.
#' @param k Number of non-self nearest neighbors to return.
#' @param metric Distance metric: `"euclidean"`, `"cosine"`, or
#'   `"correlation"`.
#' @param backend Search backend: `"cpu"`, `"cuda"`, or `"metal"`.
#' @param n.cores Number of CPU cores. Native GPU backends ignore
#'   this argument.
#'
#' @details
#' CPU uses native exhaustive float32 search below 5,000 observations and HNSW
#' otherwise. HNSW applies a metric-, shape-, and `k`-aware policy calibrated
#' for target recall 0.99; its recall is not audited during each call.
#' `tuning_reference_target_met` describes the faissR calibration cell and is
#' not measured recall for the current call. Metal uses native exact search for
#' small inputs and recall-tuned IVF-Flat for larger inputs. CUDA uses RAPIDS
#' cuVS brute-force exact search below 100,000 observations and cuVS IVF-Flat
#' above that threshold. Approximate Metal and CUDA routes use an internal
#' recall target of 0.99 and report whether their pilot audit met it.
#'
#' The CUDA result remains on the GPU and can be passed directly to
#' [umap_knn()] or [tsne_knn()] with `backend = "cuda"`. CPU and Metal
#' results contain one-based `indices` and `distances` matrices. Every result
#' excludes the observation itself. An unavailable requested backend raises an
#' error; no CPU fallback is reported as GPU work.
#'
#' @return A `fastEmbedR_knn` object. Host results contain `indices` and
#'   `distances`; CUDA results additionally inherit from `fastEmbedR_gpu_knn`
#'   and own device-resident index and distance buffers.
#' @examples
#' x <- scale(as.matrix(iris[, 1:4]))
#' knn <- precompute_knn(x, k = 15, backend = "cpu", n.cores = 2)
#' layout <- umap_knn(knn, backend = "cpu", seed = 1)
#' @export
precompute_knn <- function(data,
                            k = 30L,
                            metric = c("euclidean", "cosine", "correlation"),
                            backend = NULL,
                            n.cores = NULL) {
    backend <- resolve_embedding_backend(backend)
    metric <- resolve_embedding_metric(metric, data)
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
    policy <- fastembedr_embedding_nn_policy(backend, n = n)
    keep_gpu <- identical(backend, "cuda")
    result <- timed_do_call(run_precompute_knn, list(
        x, k, metric, policy, n.cores, keep_gpu
    ))
    metadata <- list(
        n = n, k = k, metric = metric, exclude_self = TRUE,
        backend = backend, keep_gpu = keep_gpu
    )
    finish_precomputed_knn(
        result$value,
        metadata,
        policy,
        result$time
    )
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
#' a query matrix. It uses the same package-native backend family and routing
#' policy as [precompute_knn()], but does not compute unnecessary
#' reference-to-reference or query-to-query neighbors.
#'
#' @param reference Reference observations in rows.
#' @param query Query observations in rows and the same feature space as
#'   `reference`.
#' @inheritParams precompute_knn
#'
#' @details
#' CPU uses exhaustive reference-query search when the reference has fewer than
#' 5,000 rows and HNSW otherwise. HNSW uses metric-, shape-, and `k`-aware
#' target-0.99 parameters and does not claim a per-call recall audit.
#' `tuning_reference_target_met` describes the faissR calibration cell and is
#' not measured recall for the current call. Metal routes between a native
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
    native <- if (identical(method, "exact")) {
        native_exact_knn_cpp
    } else {
        native_hnsw_knn_cpp
    }
    out <- native(
        data,
        k = k,
        n_threads = normalize_nn_threads(n_threads),
        metric = metric,
        target_recall = target_recall
    )
    attr(out, "backend") <- "cpu"
    attr(out, "method") <- out$method
    attr(out, "exclude_self") <- TRUE
    fastembedr_convert_knn_distances(out, output)
}

validate_native_metal_knn <- function(out) {
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

run_native_metal_knn <- function(data, k, method, metric, output,
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
    if (!isTRUE(native_metal_knn_available_cpp())) {
        stop("Native Metal KNN is unavailable in this build.",
            call. = FALSE
        )
    }
    out <- native_metal_knn_cpp(
        data,
        k = k,
        method = method,
        metric = metric,
        target_recall = target_recall
    )
    out <- validate_native_metal_knn(out)
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
        cuda = run_native_cuda_knn(
            data, k, method, metric, output, target_recall, keep_gpu,
            retain_data
        ),
        cpu = run_native_cpu_knn(
            data, k, method, metric, output, target_recall, n_threads
        ),
        metal = run_native_metal_knn(
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
    native <- if (identical(method, "exact")) {
        native_exact_query_cpp
    } else {
        native_hnsw_query_cpp
    }
    out <- native(
        data, query, k = k,
        n_threads = normalize_nn_threads(n_threads),
        metric = metric, target_recall = target_recall
    )
    attr(out, "backend") <- "cpu"
    attr(out, "method") <- out$method
    attr(out, "exclude_self") <- FALSE
    fastembedr_convert_knn_distances(out, output)
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
        out <- native_cuda_query_knn_cpp(
            data, query,
            k = k, method = method, metric = metric,
            target_recall = target_recall, keep_gpu = isTRUE(keep_gpu)
        )
        if (isTRUE(keep_gpu)) {
            return(out)
        }
        return(fastembedr_convert_knn_distances(out, output))
    }
    if (identical(backend, "metal")) {
        out <- native_metal_query_knn_cpp(
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
