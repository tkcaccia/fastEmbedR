normalize_nn_threads <- function(n_threads) {
    resolve_n_cores(n_threads)
}

fastembedr_optional_namespace_available <- function(package) {
    requireNamespace(package, quietly = TRUE)
}

fastembedr_optional_export <- function(package, name) {
    getExportedValue(package, name)
}

fastembedr_cuda_knn_available <- function() {
    requireNamespace("faissR", quietly = TRUE) &&
        isTRUE(faissR::cuda_available()) &&
        isTRUE(faissR::faiss_gpu_available())
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
    stop("External GPU KNN objects require explicit materialization ",
        "before passing them to fastEmbedR.", call. = FALSE)
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
    fastembedr_knn_selection("cpu", n, NA_integer_, 30L,
        "euclidean", 0.99)$method
}

fastembedr_target_recall <- function(value = NULL) {
    value <- value %||% getOption("fastEmbedR.target_recall", 0.99)
    allowed <- c(0.90, 0.95, 0.99)
    if (!is.numeric(value) || length(value) != 1L ||
        !is.finite(value) || !any(abs(value - allowed) < 1e-8)) {
        stop("`target_recall` must be 0.9, 0.95, or 0.99.",
            call. = FALSE)
    }
    allowed[[which.min(abs(value - allowed))]]
}

fastembedr_knn_provenance <- function(nn) {
    field <- function(name) {
        value <- if (is.list(nn)) nn[[name]] else NULL
        value %||% attr(nn, name)
    }
    list(
        provider = field("knn_provider") %||% field("provider"),
        backend_used = field("backend_used") %||%
            field("execution_backend") %||% field("backend"),
        selection = field("knn_selection"),
        target_recall = field("target_recall"),
        measured_recall = field("measured_recall"),
        recall_status = field("recall_status"),
        exact_recall_by_construction =
            isTRUE(field("exact_recall_by_construction"))
    )
}

fastembedr_one_call_target_recall <- function(value, massive,
        landmarks, supplied_knn) {
    if (supplied_knn && !is.null(value)) stop(
        "Precomputed KNN cannot be retuned with `target_recall`.",
        call. = FALSE)
    if (is.null(value) && (identical(massive,
            "out_of_core_graph") || supplied_knn)) return(0.99)
    value <- fastembedr_target_recall(value)
    if (value != 0.99 &&
        !massive %in% c("off", "out_of_core_data", "landmark")) stop(
        "Non-default `target_recall` requires a data-input mode.",
        call. = FALSE)
    value
}

fastembedr_knn_selection <- function(backend, n, p, k, metric,
        target_recall, n_query = n, self_query = TRUE,
        file_backed = FALSE, allow_ivf = TRUE,
        allow_cuda_profile = FALSE, memory_limit_bytes = NULL) {
    backend <- resolve_embedding_backend(backend)
    cuda_method <- if (backend == "cuda" && self_query) {
        getOption("fastEmbedR.cuda_knn_method", "auto")
    } else "auto"
    metal_method <- if (backend == "metal" && self_query) {
        getOption("fastEmbedR.metal_knn_method", "auto")
    } else "auto"
    if (!is.character(cuda_method) || length(cuda_method) != 1L ||
        !cuda_method %in% c("auto", "cagra", "cagra_width4", "cagra_sharded")) {
        stop("Invalid `fastEmbedR.cuda_knn_method` option.", call. = FALSE)
    }
    if (!is.character(metal_method) || length(metal_method) != 1L ||
        !metal_method %in% c("auto", "cagra")) {
        stop("Invalid `fastEmbedR.metal_knn_method` option.", call. = FALSE)
    }
    request <- list(
        backend = backend, metric = metric, cuda_method = cuda_method,
        metal_method = metal_method,
        n_reference = integer_scalar(n %||% NA_integer_),
        n_features = integer_scalar(p %||% NA_integer_),
        n_query = integer_scalar(n_query %||% NA_integer_),
        k = integer_scalar(k), self_query = self_query,
        file_backed = file_backed, allow_ivf = allow_ivf,
        allow_cuda_profile = allow_cuda_profile,
        target_recall = fastembedr_target_recall(target_recall))
    if (!is.null(memory_limit_bytes))
        request$memory_limit_bytes <- memory_limit_bytes
    if (backend == "cuda" && (allow_cuda_profile || file_backed))
        request$host_available_bytes <- massive_available_ram_bytes()
    selection <- faissr_knn_select_cpp(request)
    requested <- if (backend == "cuda") cuda_method else metal_method
    if (requested != "auto" && !identical(selection$method, requested)) {
        stop("The installed faissR does not support CAGRA selection.",
            call. = FALSE)
    }
    selection
}

fastembedr_resident_knn_policy <- function(selection) {
    if (identical(selection$reason, "cuda_cagra_resident_capacity_exceeded")) {
        stop("Resident CUDA CAGRA exceeds the available RAM or VRAM. ",
            "Precompute KNN with a file-backed massive_matrix().",
            call. = FALSE)
    }
    policy <- selection[c("backend", "method", "target_recall")]
    if (identical(policy$backend, "cpu")) {
        policy$recall_status <- if (identical(policy$method, "exact")) {
            "exact_by_construction"
        } else {
            "calibration_informed_not_runtime_audited"
        }
    } else {
        policy$tuning <- "auto"
        policy <- policy[c("backend", "method", "tuning", "target_recall")]
    }
    policy$selection <- selection
    policy
}

fastembedr_embedding_nn_policy <- function(embedding_backend, n = NULL,
        p = NULL, target_recall = NULL, k = 30L, metric = "euclidean",
        allow_cuda_profile = FALSE) {
    fastembedr_resident_knn_policy(fastembedr_knn_selection(
        embedding_backend, n, p, k, metric, target_recall,
        allow_cuda_profile = allow_cuda_profile))
}

fastembedr_query_nn_policy <- function(embedding_backend,
        n_reference = NULL, n_query = NULL, p = NULL,
        target_recall = NULL, k = 30L, metric = "euclidean",
        allow_ivf = TRUE) {
    fastembedr_resident_knn_policy(fastembedr_knn_selection(
        embedding_backend, n_reference, p, k, metric, target_recall, n_query,
        self_query = FALSE, allow_ivf = allow_ivf))
}

fastembedr_nn_policy_engine <- function(policy, keep_gpu = FALSE,
                                        metric = "euclidean") {
    if (is.list(policy) && identical(policy$backend, "cuda")) {
        if ((policy$method %||% "auto") %in% c("cagra", "cagra_width4")) {
            return("cuda_cuvs_cagra")
        }
        prefix <- if (isTRUE(keep_gpu)) {
            "faissr_cuda_gpu_"
        } else {
            "faissr_cuda_host_"
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
        if (identical(policy$method, "cagra")) {
            return("faissr_native_metal_cagra")
        }
        return(paste0("faissr_metal_", policy$method %||% "ivf"))
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
    out$knn_provider <- "faissR"
    out$knn_selection <- policy$selection
    out$engine <- fastembedr_nn_policy_engine(
        policy, keep_gpu, metadata$metric
    )
    if (identical(out$backend_used, "cuda_cuvs_multi_cagra")) {
        out$engine <- out$backend_used
    }
    out$elapsed_sec <- unname(elapsed[["elapsed"]])
    if (identical(policy$backend, "cpu") &&
        !identical(policy$method, "exact") &&
        !isTRUE(out$recall_audited)) {
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
                                keep_gpu, devices = NULL) {
    if (identical(policy$backend, "cuda") &&
        policy$method %in% c("cagra", "cagra_width4")) {
        return(run_faissr_cuda_embedding_knn(x, k, policy$method, metric,
            fastembedr_knn_output_type(x, policy$backend),
            policy$target_recall, devices,
            cagra_profile = policy$selection$cuda_cagra_profile))
    }
    if (!is.null(devices)) return(run_faissr_cuda_multi_knn(
        x, k, policy$method, metric,
        fastembedr_knn_output_type(x, policy$backend),
        policy$target_recall, devices))
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
#' used internally by [umap()] and [tsne()]. The native faissR selector
#' chooses the search method for the requested backend.
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
#' @param memory_limit Working-memory budget for file-backed KNN. CUDA also
#'   uses it to cap estimated VRAM. When omitted,
#'   `options(memory_limit = ...)` is used, or detected available RAM.
#' @param devices Optional distinct CUDA device indices. Device work
#'   division depends on the selected KNN method. See Details.
#' @param checkpoint Save completed file-backed KNN query blocks.
#' @param resume Continue a matching file-backed KNN checkpoint.
#' @param checkpoint_every Save progress after this many query blocks.
#' @param target_recall Requested approximate-search recall tier: `0.9`,
#'   `0.95`, or `0.99` (default). When omitted, uses
#'   `options(fastEmbedR.target_recall = ...)`. This is a tuning target,
#'   not a guarantee of measured recall. Exact search remains exact.
#' @param audit_rows Number of fixed file-backed query rows to compare with
#'   an exact full-reference search (default zero). At most 10,000 rows.
#'   The returned graph reports measured sampled recall separately from
#'   `target_recall`. This control is unavailable for resident input.
#'
#' @details
#' For resident inputs, CPU uses Flat below 5,000 observations and
#' HNSW for larger inputs. Euclidean HNSW checks a full-reference pilot.
#' Cosine and correlation use shape-based HNSW tuning and report recall as
#' unaudited. CUDA and Metal choose between Flat and IVF-Flat according to
#' the input shape, search effort, and backend limits. The policy uses `k`
#' and `target_recall` where they affect the available search. For GPU
#' inputs with at least 500,000 rows and 256 columns, a target of 0.99
#' selects Flat. Inputs with at least 500,000 rows and 1,024 columns select
#' Flat at all three tiers.
#'
#' Approximate search uses the requested tier to tune search effort. Pilot
#' and sampled recall refer to the rows checked. Exact routes report exact
#' search by construction.
#'
#' A file-backed descriptor uses bounded KNN graph construction and writes
#' indices and distances under `output`. CPU uses Flat up to 5,000 rows if
#' estimated pair-feature work is at most 125 billion, and sharded HNSW
#' otherwise. CUDA and Metal use the native selector. On one GPU, file-backed
#' Flat holds one reference shard at a time and reuses it across query
#' batches. Metal IVF requires a passing capacity check and checkpointing to
#' be off. A failed IVF pilot raises an error. File-backed KNN supports
#' Euclidean distance. The
#' result is a `fastEmbedR_massive_graph` descriptor; construct UMAP fuzzy
#' weights or t-SNE affinities from it before full-graph fitting.
#'
#' Resident CUDA Flat and IVF results stay on the device when `devices` is
#' omitted and can be passed directly to [umap_knn()] or [tsne_knn()] with
#' `backend = "cuda"`. An explicit device selection returns host matrices.
#' For large Euclidean CUDA self searches with cuVS, faissR can select CAGRA
#' with NN-descent and shard a file-backed index to fit the memory budget.
#' Set `options(fastEmbedR.cuda_knn_method = "cagra")` to use faissR's
#' cuVS CAGRA settings for the requested recall tier, or use
#' `"cagra_width4"` to request a width-four graph search. These options
#' apply to CUDA Euclidean self KNN with at least 500,000 rows and require
#' a cuVS-enabled faissR build with CAGRA support.
#' Resident CAGRA returns host neighbor matrices. File-backed CAGRA stages
#' the source in batches and writes neighbors to disk. With
#' `"cagra_width4"` and two or more explicit CUDA `devices`, each GPU
#' builds a disjoint reference shard and searches all query batches.
#' faissR merges candidate neighbors before writing each batch.
#' File-backed CAGRA checks capacity and requires checkpointing off.
#' Resident multi-GPU CAGRA uses cuVS tree merging.
#' The target tier describes the settings used. Supply audit rows to
#' measure recall for a file-backed fit.
#' Set `options(fastEmbedR.metal_knn_method = "cagra")` for native Metal
#' CAGRA on resident or file-backed Euclidean input. This route requires
#' at least 129 rows, up to 1,024 features, and up to 128 neighbors.
#' A file-backed Metal CAGRA index uses shared memory on one device.
#' CPU and Metal
#' results contain one-based `indices` and `distances` matrices. Every result
#' excludes the observation itself. An unavailable requested backend raises an
#' error; no CPU fallback is reported as GPU work.
#'
#' @return A resident `fastEmbedR_knn` object, or a file-backed
#'   `fastEmbedR_massive_graph` descriptor. Resident host results contain
#'   `indices` and `distances`. CUDA results with device-resident buffers
#'   also inherit from `fastEmbedR_gpu_knn`. Resident
#'   results record `knn_provider = "faissR"` and the selected engine.
#'   Both storage modes return `knn_selection` with the method, selection
#'   reason, shape group, and requested recall tier reported by faissR.
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
        checkpoint_every = 100L, target_recall = NULL,
        audit_rows = 0L) {
    massive_checkpoint_validate_controls(checkpoint, resume)
    target_recall <- fastembedr_target_recall(target_recall)
    if (inherits(data, "fastEmbedR_massive_graph")) stop(
        "precompute_knn() needs a data matrix, not a prepared graph.",
        call. = FALSE)
    fastembedr_storage_option(data, storage)
    backend <- resolve_embedding_backend(backend)
    metric <- resolve_embedding_metric(metric, data)
    if (fastembedr_file_backed_input(data)) return(
        precompute_knn_backed(data, k, metric, backend, n.cores,
            output, memory_limit, devices, checkpoint, resume,
            checkpoint_every, target_recall, audit_rows))
    if (inherits(data, "fastEmbedR_massive_matrix")) stop(
        "A memory-backed massive_matrix() is not a disk-backed source; ",
        "pass its resident matrix to precompute_knn().", call. = FALSE)
    if (!is.numeric(audit_rows) || length(audit_rows) != 1L ||
        !is.finite(audit_rows) || audit_rows != 0) stop(
        "`audit_rows` requires file-backed input.", call. = FALSE)
    if (!is.null(output) || !is.null(memory_limit) ||
        checkpoint || resume || !identical(checkpoint_every, 100L)) stop(
        "File-backed KNN controls require a disk-backed massive_matrix().",
        call. = FALSE)
    prepared <- prepare_embedding_data(data, standardize = FALSE,
        pca_dims = NULL, seed = 4L, backend = backend)
    x <- prepared$data
    devices <- massive_validate_devices(devices, backend, nrow(x))
    n <- nrow(x)
    k <- validate_precompute_k(k, n - 1L, "nrow(data) - 1")
    policy <- fastembedr_embedding_nn_policy(
        backend, n = n, p = ncol(x),
        target_recall = target_recall, k = k, metric = metric,
        allow_cuda_profile = is.null(devices))
    keep_gpu <- identical(backend, "cuda") && is.null(devices) &&
        !policy$method %in% c("cagra", "cagra_width4")
    result <- timed_do_call(run_precompute_knn, list(
        x, k, metric, policy, n.cores, keep_gpu, devices
    ))
    metadata <- list(n = n, k = k, metric = metric, exclude_self = TRUE,
        backend = backend, keep_gpu = keep_gpu)
    finish_precomputed_knn(result$value, metadata, policy,
        result$time)
}

precompute_knn_backed <- function(data, k, metric, backend, n.cores,
        output, memory_limit, devices, checkpoint, resume,
        checkpoint_every, target_recall, audit_rows) {
    if (!identical(metric, "euclidean")) stop(
        "File-backed KNN currently supports Euclidean distance only.",
        call. = FALSE)
    if (is.null(output)) stop(
        "File-backed KNN requires a new `output` file prefix.",
        call. = FALSE)
    limit <- if (is.null(memory_limit))
        fastembedr_memory_option() else memory_limit
    memory_bytes <- massive_memory_bytes(limit)
    cuda_auto <- backend == "cuda" && identical(limit, "available")
    workers <- resolve_n_cores(n.cores)
    allow_ivf <- backend != "metal" || (!checkpoint && !resume && workers == 1L)
    select <- function(allow) fastembedr_knn_selection(backend, data$nrow,
        data$ncol, k, metric, target_recall, file_backed = TRUE,
        allow_ivf = allow,
        memory_limit_bytes = if (!cuda_auto) memory_bytes,
        allow_cuda_profile = data$format %in% c("f32", "fbin") &&
            is.null(devices) && !checkpoint && !resume)
    selection <- select(allow_ivf)
    if (selection$method == "cagra_sharded") {
        return(precompute_knn_file_cagra_sharded(data, k, output, selection,
            audit_rows, devices, checkpoint || resume))
    }
    if (selection$method %in% c("cagra", "cagra_width4")) {
        budget <- if (cuda_auto && length(devices) < 2L) {
            massive_cuda_memory_cpp()$free_bytes
        } else memory_bytes
        devices <- massive_validate_devices(devices, backend, data$nrow)
        return(precompute_knn_file_cagra(data, k, output,
            budget, selection, audit_rows, devices,
            checkpoint || resume))
    }
    if (backend == "metal" && selection$method == "ivf" &&
        !isTRUE(faissr_metal_file_ivf_plan_cpp(data, as.integer(k),
            memory_bytes)$fits)) selection <- select(FALSE)
    method <- switch(selection$method, exact = "exact", hnsw = "hnsw_sharded",
        ivf = if (backend == "metal") "metal_ivf" else "ivf_sharded")
    graph <- massive_full_knn_graph(data, k = k, output = output,
        backend = backend, n.cores = workers,
        memory_limit = limit, method = method, devices = devices,
        checkpoint = checkpoint, resume = resume,
        checkpoint_every = checkpoint_every,
        target_recall = target_recall, audit_rows = audit_rows)
    selection$reason <- graph$resources$selection_reason %||% selection$reason
    graph$knn_selection <- selection
    graph
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
#' uses the requested recall tier but does not claim a per-call audit. Metal
#' routes between a native
#' query-only exact kernel and recall-tuned IVF-Flat from the estimated
#' reference-query distance workload. CUDA routes between faissR FAISS GPU
#' exact search and IVF-Flat using the reference size, query batch size,
#' and feature count. CUDA results remain device-resident for direct consumption
#' by landmark UMAP and t-SNE transformations.
#'
#' @return A `fastEmbedR_knn` object with one row per query observation and
#'   one-based indices into `reference`. Its `knn_selection` field records
#'   the method, selection reason, shape group, and requested recall tier.
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
                                    n.cores = NULL,
                                    target_recall = NULL) {
    target_recall <- fastembedr_target_recall(target_recall)
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
        p = ncol(reference),
        target_recall = target_recall, k = k, metric = metric,
        allow_ivf = backend != "cuda"
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

run_faissr_cuda_resident_knn <- function(data, k, method, metric,
                                            target_recall, retain_data) {
    allowed <- c("auto", "exact", "flat", "bruteforce", "ivf")
    if (!method %in% allowed) {
        stop("faissR CUDA KNN supports `auto`, `exact`, and `ivf`.",
            call. = FALSE
        )
    }
    if (method == "auto") {
        method <- fastembedr_embedding_nn_policy("cuda", nrow(data),
            ncol(data), target_recall, k, metric)$method
    }
    exact <- method %in% c("exact", "flat", "bruteforce")
    if (exact) {
        return(run_faissr_cuda_resident_exact(
            data, k, metric, target_recall, retain_data))
    }
    run_faissr_cuda_resident_ivf(
        data, k, metric, target_recall, retain_data)
}

run_faissr_cuda_resident_exact <- function(data, k, metric,
                                            target_recall,
                                            retain_data = FALSE) {
    if (!fastembedr_cuda_knn_available()) stop(
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

fastembedr_euclidean_tie_audit <- function(data, rows, found,
                                            found_distances, reference,
                                            k) {
    if (!identical(dim(found_distances), dim(found)) ||
        any(!is.finite(found_distances))) stop(
        "faissR resident IVF returned invalid pilot distances.",
        call. = FALSE)
    values <- vapply(seq_along(rows), function(i) {
        ids <- found[i, ]
        truth <- reference[i, ]
        truth <- truth[truth != rows[i]][seq_len(k)]
        query <- as.vector(as.numeric(data[rows[i], ]))
        distance <- function(selected) {
            selected <- matrix(as.vector(as.numeric(
                data[selected, , drop = FALSE])), nrow = length(selected))
            rowSums(sweep(selected, 2L, query, "-")^2)
        }
        actual <- distance(ids)
        kth <- max(distance(truth))
        tolerance <- max(1e-5, 1e-6 * kth)
        c(tie = sum(actual <= kth + tolerance) / k,
            error = max(abs(found_distances[i, ] - sqrt(actual)) /
                pmax(1e-12, sqrt(actual))))
    }, numeric(2L))
    if (any(values["error", ] > 1e-3)) stop(
        "faissR resident IVF pilot distances disagree with input.",
        call. = FALSE)
    list(mean = mean(values["tie", ]),
        minimum = min(values["tie", ]),
        max_relative_distance_error = max(values["error", ]))
}

fastembedr_audit_resident_ivf <- function(data, knn, k, metric,
                                            target_recall) {
    rows <- unique(as.integer(round(seq(
        1, nrow(data), length.out = min(256L, nrow(data))
    ))))
    sampled <- if (identical(knn$result_residency, "host")) {
        list(indices = knn$indices[rows, , drop = FALSE],
            distances = matrix(as.double(
                knn$distances[rows, , drop = FALSE]), length(rows), k))
    } else faissr_gpu_knn_sample_cpp(knn, rows)
    found <- sampled$indices
    if (!identical(dim(found), c(length(rows), as.integer(k))) ||
        anyNA(found) || any(found < 1L | found > nrow(data))) {
        stop("faissR resident IVF returned invalid pilot IDs.",
            call. = FALSE)
    }
    reference <- faissR::nn(
        data, points = data[rows, , drop = FALSE],
        k = k + 1L, exclude_self = FALSE, backend = "cpu",
        method = "flat", metric = metric, output = "double"
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
    if (mean_recall < target_recall && !keep_knn_recall_miss()) stop(
        "faissR resident IVF pilot recall ",
        format(mean_recall, digits = 5L), " is below target ",
        target_recall, ".", call. = FALSE)
    tie <- if (identical(metric, "euclidean")) {
        fastembedr_euclidean_tie_audit(data, rows, found,
            sampled$distances, reference$indices, k)
    } else NULL
    list(
        rows = rows, mean = mean_recall, minimum = min(row_recall),
        reference_backend = reference$backend_used, tie = tie
    )
}

keep_knn_recall_miss <- function() {
    isTRUE(getOption("faissR.knn.report_recall_miss", FALSE))
}

fastembedr_cuda_ivf_probe_reference <- function(data, rows, k) {
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
    reference
}

fastembedr_cuda_ivf_probe_score <- function(data, rows, k,
                                            index, reference) {
    found <- faissr_gpu_index_search_cpp(index,
        data[rows, , drop = FALSE], k + 1L)
    if (!identical(dim(found$indices), c(length(rows), k + 1L)) ||
        !identical(dim(found$distances), c(length(rows), k + 1L))) {
        stop("faissR returned an invalid IVF pilot shape.",
            call. = FALSE)
    }
    keep <- lapply(seq_along(rows), function(i) {
        selected <- which(found$indices[i, ] != rows[i])
        if (length(selected) < k) stop(
            "faissR IVF pilot lacks non-self neighbors.", call. = FALSE)
        selected[seq_len(k)]
    })
    ids <- t(vapply(seq_along(rows), function(i)
        found$indices[i, keep[[i]]], integer(k)))
    distances <- t(vapply(seq_along(rows), function(i)
        found$distances[i, keep[[i]]], numeric(k)))
    recall <- vapply(seq_along(rows), function(i) {
        if (anyDuplicated(ids[i, ])) return(0)
        truth <- reference$indices[i, ]
        truth <- truth[truth != rows[i]][seq_len(k)]
        sum(ids[i, ] %in% truth) / k
    }, numeric(1L))
    tie <- fastembedr_euclidean_tie_audit(data, rows, ids,
        distances, reference$indices, k)
    list(mean = mean(recall), minimum = min(recall), tie = tie,
        rows = rows, reference_backend = reference$backend_used)
}

fastembedr_cuda_ivf_tune_persistent <- function(data, k, params,
                                                target_recall) {
    pilot_started <- proc.time()[["elapsed"]]
    pilot_guard <- min(1, target_recall + 0.005)
    rows <- unique(as.integer(round(seq(
        1, nrow(data), length.out = min(256L, nrow(data))
    ))))
    reference <- fastembedr_cuda_ivf_probe_reference(data, rows, k)
    index <- faissr_gpu_index_build_cpp(data, k + 1L, "ivf", 0L,
        as.integer(params$nlist), as.integer(params$nprobe), FALSE)
    nprobe <- as.integer(params$nprobe)
    trace <- list()
    repeat {
        faissr_gpu_index_set_nprobe_cpp(index$pointer, nprobe)
        pilot <- fastembedr_cuda_ivf_probe_score(data, rows, k,
            index$pointer, reference)
        trace[[length(trace) + 1L]] <- c(nprobe = nprobe,
            measured_recall = pilot$mean)
        if (pilot$mean >= pilot_guard) break
        if (nprobe >= params$nlist && keep_knn_recall_miss()) break
        if (nprobe >= params$nlist) stop(
            "faissR IVF missed recall even at full probe width.",
            call. = FALSE)
        nprobe <- as.integer(min(params$nlist,
            max(nprobe + 1L, ceiling(nprobe * 1.5))))
    }
    pilot_seconds <- proc.time()[["elapsed"]] - pilot_started
    search_seconds <- system.time({
        out <- faissr_gpu_index_search_cpp(index$pointer, data, k,
            "gpu_self_exclude")
    })[["elapsed"]]
    out$gpu_index_persistent <- FALSE
    out$index_residency <- "transient_gpu_index_during_call"
    out$cpu_training_seconds <- index$cpu_training_seconds
    out$gpu_clone_seconds <- index$gpu_clone_seconds
    out$gpu_add_seconds <- index$gpu_add_seconds
    out$pilot_tune_call_seconds <- pilot_seconds
    out$full_search_call_seconds <- unname(search_seconds)
    list(out = out, pilot = pilot, nprobe = nprobe,
        pilot_guard_recall = pilot_guard,
        trace = do.call(rbind, trace))
}

run_faissr_cuda_resident_ivf <- function(data, k, metric, target_recall,
                                            retain_data = FALSE) {
    if (!fastembedr_cuda_knn_available()) stop(
        "faissR CUDA KNN is unavailable; no fallback was used.",
        call. = FALSE)
    params <- utils::getFromNamespace("cuda_ivf_params", "faissR")(
        nrow(data), ncol(data), k, metric, target_recall
    )
    tuned <- if (identical(metric, "euclidean")) {
        fastembedr_cuda_ivf_tune_persistent(
            data, k, params, target_recall)
    } else NULL
    out <- if (!is.null(tuned)) tuned$out else
        faissr_gpu_metric_ivf_knn_cpp(
        data, as.integer(k), metric, 0L,
        as.integer(params$nlist), as.integer(params$nprobe))
    if (!inherits(out, "faissR_gpu_knn") ||
        !identical(out$result_residency, "cuda") ||
        !isTRUE(out$exclude_self) || isTRUE(out$exact) ||
        !identical(as.integer(out$k), as.integer(k)) ||
        !identical(as.integer(out$n_query), as.integer(nrow(data))) ||
        isTRUE(out$cpu_fallback)) stop(
        "faissR did not return resident CUDA IVF KNN.",
        call. = FALSE)
    pilot <- if (!is.null(tuned)) tuned$pilot else
        fastembedr_audit_resident_ivf(data, out, k, metric, target_recall)
    out$nlist <- as.integer(params$nlist)
    out$nprobe <- if (!is.null(tuned)) tuned$nprobe else
        as.integer(params$nprobe)
    out$pilot_probe_trace <- if (!is.null(tuned)) tuned$trace else NULL
    out$pilot_guard_recall <- tuned$pilot_guard_recall %||% NA_real_
    out$target_recall <- target_recall
    out$pilot_recall <- pilot$mean
    out$pilot_min_row_recall <- pilot$minimum
    out$pilot_rows <- pilot$rows
    out$pilot_reference_backend <- pilot$reference_backend
    out$pilot_distance_tie_recall <- pilot$tie$mean %||% NA_real_
    out$pilot_min_distance_tie_recall <-
        pilot$tie$minimum %||% NA_real_
    out$pilot_max_relative_distance_error <-
        pilot$tie$max_relative_distance_error %||% NA_real_
    out$recall_audited <- TRUE
    out$recall_status <- "fixed_rows_exact_reference"
    out$target_met <- pilot$mean >= target_recall
    if (isTRUE(retain_data)) {
        out <- faissr_gpu_knn_retain_data_cpp(out, data)
    }
    out
}

run_faissr_cuda_cagra <- function(data, k, method, metric, output,
        target_recall, devices, cagra_profile = NULL) {
    if (!identical(metric, "euclidean")) {
        stop("Resident CAGRA currently requires Euclidean distance.",
            call. = FALSE)
    }
    if (length(devices) > 1L) {
        return(run_faissr_cuda_multi_cagra(data, k, method, output,
            target_recall, devices, cagra_profile))
    }
    if (!is.null(devices) && !identical(as.integer(devices), 0L)) {
        stop("A single-device CAGRA request requires CUDA device 0.",
            call. = FALSE)
    }
    runner <- get0("cuvs_cagra_wide_self", envir = asNamespace("faissR"),
        inherits = FALSE)
    if (!is.function(runner)) {
        stop("The installed faissR build lacks the CAGRA presets.",
            call. = FALSE)
    }
    args <- list(data = data, k = k, target_recall = target_recall,
        output = output, width4 = identical(method, "cagra_width4"))
    if (!is.null(cagra_profile)) args$profile <- cagra_profile
    do.call(runner, args)
}

run_faissr_cuda_multi_cagra <- function(data, k, method, output,
        target_recall, devices, profile = NULL) {
    faiss <- asNamespace("faissR")
    preset <- get0("cuvs_cagra_wide_params", envir = faiss, inherits = FALSE)
    if (!is.function(preset)) stop(
        "The installed faissR build lacks the CAGRA presets.", call. = FALSE)
    params <- preset(target_recall, identical(method, "cagra_width4"))
    search <- getOption("faissR.cuvs_cagra_resident_search")
    if (!is.null(search)) {
        resolve <- get("cuvs_cagra_file_search", envir = faiss)
        params <- resolve(params, k, search)
    }
    profile <- profile %||% getOption("faissR.cuvs_cagra_profile")
    if (!is.null(profile)) {
        validate <- get("cuvs_validate_cagra_profile", envir = faiss)
        profile <- validate(profile, nrow(data), k)
    }
    devices <- sort(as.integer(devices))
    out <- faissr_cuvs_multi_cagra_cpp(list(
        data = data, k = as.integer(k), devices = devices, output = output,
        width = params$width, queue = params$queue,
        nn_descent = params$builder == "nn_descent",
        graph_degree = profile$graph_degree %||% 64L,
        intermediate_degree = profile$intermediate_degree %||% 128L,
        batch_rows = min(params$batch %||% 1024L,
            profile$batch_rows %||% 1024L),
        memory_limit_bytes = massive_available_ram_bytes()
    ))
    expected <- c(nrow(data), as.integer(k))
    if (!identical(out$backend_used, "cuda_cuvs_multi_cagra") ||
        !identical(out$devices, devices) || !identical(out$exact, FALSE) ||
        !identical(dim(out$indices), expected) ||
        !identical(dim(out$distances), expected)) stop(
        "faissR returned incompatible multi-GPU CAGRA KNN.", call. = FALSE)
    out$backend <- "cuda"
    out$accelerator <- "cuda"
    out$index_base <- 1L
    out$exclude_self <- TRUE
    out$knn_provider <- "faissR"
    out$result_residency <- "host"
    out$target_recall <- target_recall
    out$target_met <- NA
    out$recall_audited <- FALSE
    out$recall_status <- "preset_not_runtime_audited"
    out$cagra_settings <- params
    out
}

run_faissr_cuda_embedding_knn <- function(data, k, method, metric, output,
        target_recall, devices = NULL, prefer_resident = FALSE,
        retain_data = FALSE, cagra_profile = NULL) {
    if (method %in% c("cagra", "cagra_width4")) {
        return(run_faissr_cuda_cagra(data, k, method, metric, output,
            target_recall, devices, cagra_profile))
    }
    if (isTRUE(prefer_resident) && is.null(devices) &&
        method %in% c("exact", "flat")) {
        return(run_faissr_cuda_resident_knn(
            data, k, method, metric, target_recall, retain_data))
    }
    if (identical(metric, "euclidean")) return(run_faissr_cuda_multi_knn(
        data, k, method, metric, output, target_recall, devices %||% 0L
    ))
    if (!is.null(devices) && !identical(as.integer(devices), 0L)) {
        stop("Non-Euclidean faissR CUDA KNN currently uses GPU 0.",
            call. = FALSE)
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
        paste0("faiss_gpu_flat_",
            if (identical(metric, "euclidean")) "l2" else metric)
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

fastembedr_cuda_multi_recall_result <- function(out, data, k, method,
                                                metric, target_recall) {
    pilot <- if (identical(method, "ivf")) {
        fastembedr_audit_resident_ivf(data, out, k, metric, target_recall)
    } else NULL
    if (!is.null(pilot) && (length(pilot$mean) != 1L ||
        !is.finite(pilot$mean) ||
        (pilot$mean < target_recall && !keep_knn_recall_miss()))) stop(
        "faissR CUDA IVF pilot recall is below target.", call. = FALSE)
    out$target_recall <- target_recall
    out$pilot_recall <- pilot$mean %||% NA_real_
    out$pilot_min_row_recall <- pilot$minimum %||% NA_real_
    out$pilot_rows <- pilot$rows %||% integer()
    out$pilot_reference_backend <- pilot$reference_backend %||%
        NA_character_
    out$pilot_distance_tie_recall <- pilot$tie$mean %||% NA_real_
    out$target_met <- if (identical(method, "flat")) TRUE else
        pilot$mean >= target_recall
    out$recall_audited <- !is.null(pilot)
    out$recall_status <- if (identical(method, "flat")) {
        "exact_by_construction"
    } else if (out$target_met) "runtime_audited_target_met"
    else "runtime_audited_target_missed"
    out
}

fastembedr_cuda_host_pilot <- function(index, data, rows, k,
        reference) {
    found <- faissr_gpu_index_search_cpp(index,
        data[rows, , drop = FALSE], k + 1L, "double")
    if (!identical(dim(found$indices), c(length(rows), k + 1L))) stop(
        "faissR CUDA IVF pilot returned an invalid shape.", call. = FALSE)
    recall <- vapply(seq_along(rows), function(i) {
        ids <- found$indices[i, ]
        ids <- ids[ids != rows[i]][seq_len(k)]
        truth <- reference$indices[i, ]
        truth <- truth[truth != rows[i]][seq_len(k)]
        if (anyNA(ids) || anyDuplicated(ids)) return(0)
        mean(ids %in% truth)
    }, numeric(1L))
    mean(recall)
}

fastembedr_cuda_tune_host_index <- function(data, k, params,
                                            target_recall, devices,
                                            output) {
    rows <- unique(as.integer(round(seq(1, nrow(data),
        length.out = min(256L, nrow(data))))))
    reference <- fastembedr_cuda_ivf_probe_reference(data, rows, k)
    index <- faissr_gpu_index_build_cpp(data, k + 1L, "ivf",
        as.integer(devices), as.integer(params$nlist),
        as.integer(params$nprobe), FALSE)
    nprobe <- as.integer(params$nprobe)
    trace <- list()
    repeat {
        faissr_gpu_index_set_nprobe_cpp(index$pointer, nprobe)
        measured <- fastembedr_cuda_host_pilot(index$pointer, data,
            rows, k, reference)
        trace[[length(trace) + 1L]] <- c(nprobe = nprobe,
            measured_recall = measured)
        if (measured >= target_recall) break
        if (nprobe >= params$nlist && keep_knn_recall_miss()) break
        if (nprobe >= params$nlist) stop(
            "faissR CUDA IVF missed target at full probe width.",
            call. = FALSE)
        nprobe <- as.integer(min(params$nlist,
            max(nprobe + 1L, ceiling(nprobe * 1.5))))
    }
    out <- faissr_gpu_index_search_cpp(index$pointer, data, k,
        paste0(output, "_self_exclude"))
    out$cpu_training_seconds <- index$cpu_training_seconds
    out$gpu_clone_seconds <- index$gpu_clone_seconds
    out$gpu_add_seconds <- index$gpu_add_seconds
    out$pilot_probe_trace <- do.call(rbind, trace)
    out$pilot_tuned_nprobe <- nprobe
    out$index_reused <- FALSE
    out$gpu_index_persistent <- FALSE
    out$index_residency <- "transient_gpu_index_during_call"
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
    tuned_host <- identical(method, "ivf") && output %in%
        c("double", "float")
    out <- if (tuned_host) fastembedr_cuda_tune_host_index(
        data, k, params, target_recall, devices, output) else
        faissr_gpu_multi_knn_cpp(
            data, as.integer(k), method, as.integer(devices),
            as.integer(params$nlist), as.integer(params$nprobe),
            FALSE, output)
    expected_mode <- if (length(devices) == 1L) "single" else "replicate"
    expected_backend <- if (length(devices) == 1L) {
        "faiss_gpu_single_self"
    } else "faiss_gpu_multi_replicate"
    if (tuned_host) expected_backend <- "faiss_gpu_persistent_query"
    if (!identical(out$accelerator, "cuda") ||
        !identical(out$multi_gpu_mode, expected_mode) ||
        !identical(out$backend_used, expected_backend) ||
        !identical(out$devices, as.integer(devices)) ||
        !identical(dim(out$indices), c(nrow(data), as.integer(k))) ||
        !identical(out$index_base, 1L) ||
        (tuned_host && !isTRUE(out$exclude_self)) ||
        isTRUE(out$cpu_fallback)) {
        stop("faissR did not return the requested multi-GPU KNN.",
            call. = FALSE)
    }
    fastembedr_cuda_multi_recall_result(out, data, k, method,
        metric, target_recall)
}

faissr_hnsw_input <- function(x) {
    if (is_float32_matrix(x)) return(float::dbl(x))
    if (is.integer(x)) return(matrix(as.double(x), nrow(x), ncol(x)))
    x
}

faissr_hnsw_recall_check <- function(out, metric) {
    if (!identical(metric, "euclidean")) {
        out$recall_audited <- FALSE
        out$target_met <- NA
        out$recall_status <-
            "calibration_informed_not_runtime_audited"
        return(out)
    }
    pilot_ok <- isTRUE(out$target_met) &&
        identical(out$recall_status, "pilot_exact_reference_passed")
    pilot_reported <- keep_knn_recall_miss() &&
        identical(out$recall_status, "pilot_exact_reference_missed_guard")
    if (!isTRUE(out$recall_audited) || !(pilot_ok || pilot_reported)) {
        stop("faissR CPU HNSW pilot did not meet the recall target.",
            call. = FALSE)
    }
    out
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
    out <- faissr_hnsw_recall_check(out, metric)
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

validate_faissr_metal_knn <- function(out, target_recall) {
    if (!out$method %in% c("native_metal_exact", "native_metal_ivf",
            "native_metal_cagra") ||
        !identical(out$backend_used, paste0("faissr_", out$method)) ||
        !identical(out$accelerator, "metal") ||
        !identical(out$target_recall, target_recall)) stop(
        "faissR Metal KNN changed the backend or recall target.",
        call. = FALSE)
    if (identical(out$method, "native_metal_ivf")) {
        measured <- out$pilot_recall
        if (!is.numeric(measured) || length(measured) != 1L ||
            !is.finite(measured) ||
            (!keep_knn_recall_miss() && (measured < target_recall ||
                !isTRUE(out$target_met)))) stop(
            "Native Metal IVF did not meet the requested recall target; ",
            "no fallback was used.", call. = FALSE)
        out$recall_audited <- TRUE
        out$recall_status <- if (isTRUE(out$target_met))
            "runtime_audited_target_met" else "runtime_audited_target_missed"
        out$exact_recall_by_construction <- FALSE
    } else if (identical(out$method, "native_metal_cagra")) {
        if (!identical(out$exact, FALSE) || !is.na(out$target_met)) stop(
            "faissR Metal CAGRA omitted its unaudited search status.",
            call. = FALSE)
        out$recall_audited <- FALSE
        out$recall_status <- "preset_not_runtime_audited"
    } else {
        if (!isTRUE(out$exact) || !isTRUE(out$target_met)) stop(
            "faissR Metal exact KNN changed method or result status.",
            call. = FALSE)
        out$pilot_recall <- NA_real_
        out$recall_audited <- FALSE
        out$recall_status <- "exact_by_construction"
        out$exact_recall_by_construction <- TRUE
    }
    out
}

run_faissr_metal_knn <- function(data, k, method, metric, output,
                                    target_recall) {
    if (!method %in% c("auto", "exact", "ivf", "cagra")) {
        stop("Native Metal KNN supports `auto`, `exact`, `ivf`, and `cagra`.",
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
    out <- validate_faissr_metal_knn(out, target_recall)
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
        cuda = if (!isTRUE(keep_gpu)) {
            if (isTRUE(retain_data)) stop(
                "`retain_data` requires `keep_gpu = TRUE`.",
                call. = FALSE)
            resolved <- if (method == "auto") {
                fastembedr_embedding_nn_policy("cuda", nrow(data),
                    ncol(data), target_recall, k, metric)$method
            } else if (method == "bruteforce") "exact" else method
            run_faissr_cuda_embedding_knn(
                data, k, resolved, metric, output, target_recall
            )
        } else run_faissr_cuda_resident_knn(
            data, k, method, metric, target_recall, retain_data
        ),
        cpu = run_native_cpu_knn(
            data, k, method, metric, output, target_recall, n_threads
        ),
        metal = run_faissr_metal_knn(data, k,
            if (method == "auto") fastembedr_embedding_nn_policy(
                "metal", nrow(data), ncol(data), target_recall, k,
                metric)$method else method, metric, output, target_recall),
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
    if (method == "auto") {
        method <- fastembedr_query_nn_policy(backend, nrow(data), nrow(query),
            ncol(data), target_recall, k, metric,
            allow_ivf = backend != "cuda" || !keep_gpu)$method
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
        fastembedr_check_metal_query_knn(
            out, nrow(data), nrow(query), k, method, metric,
            target_recall)
        attr(out, "backend") <- "metal"
        attr(out, "exclude_self") <- FALSE
        return(fastembedr_convert_knn_distances(out, output))
    }
    run_native_cpu_query_knn(
        data, query, k, method, metric, output,
        target_recall, n_threads
    )
}

fastembedr_check_metal_query_knn <- function(out, n_reference,
        n_query, k, requested_method, metric, target_recall) {
    expected <- paste0("native_metal_",
        c("exact", "ivf"), "_query")
    allowed <- if (identical(requested_method, "auto")) expected else
        paste0("native_metal_", requested_method, "_query")
    if (!out$method %in% allowed ||
        !identical(out$backend, "metal") ||
        !identical(out$accelerator, "metal") ||
        !identical(out$backend_used,
            paste0("faissr_", out$method)) ||
        !identical(out$index_base, 1L) ||
        !identical(out$metric, metric) ||
        !identical(out$target_recall, target_recall) ||
        !identical(out$n_reference, as.integer(n_reference)) ||
        !identical(out$n_query, as.integer(n_query)) ||
        !identical(dim(out$indices), c(n_query, as.integer(k))) ||
        !identical(dim(out$distances), c(n_query, as.integer(k))) ||
        isTRUE(out$exclude_self) ||
        !identical(out$exact,
            identical(out$method, "native_metal_exact_query"))) {
        stop("faissR did not return the requested Metal query KNN.",
            call. = FALSE)
    }
    if (!isTRUE(out$target_met) && !keep_knn_recall_miss()) stop(
        "Native Metal query IVF missed the requested recall target; ",
        "no fallback was used.", call. = FALSE)
    invisible(out)
}

fastembedr_check_cuda_query_knn <- function(out, n_query, k, method,
                                            target_recall) {
    if (!identical(out$accelerator, "cuda") ||
        !startsWith(out$backend_used %||% "", "faiss_gpu") ||
        !identical(dim(out$indices), c(n_query, as.integer(k))) ||
        !identical(out$index_base, 1L) || isTRUE(out$cpu_fallback)) {
        stop("faissR did not return CUDA query KNN.", call. = FALSE)
    }
    if (identical(method, "ivf") &&
        (!is.numeric(out$pilot_recall) ||
            length(out$pilot_recall) != 1L ||
            !is.finite(out$pilot_recall) ||
            (!keep_knn_recall_miss() && (!isTRUE(out$target_met) ||
                out$pilot_recall < target_recall)))) stop(
        "faissR CUDA query IVF missed the requested recall target.",
        call. = FALSE)
    attr(out, "backend") <- "cuda"
    attr(out, "exclude_self") <- FALSE
    out
}

run_faissr_cuda_query_knn <- function(data, query, k, method, metric,
                                        output, target_recall, keep_gpu) {
    if (!requireNamespace("faissR", quietly = TRUE) ||
        !isTRUE(faissR::faiss_gpu_available())) {
        stop("The faissR FAISS GPU backend is unavailable.",
            call. = FALSE)
    }
    if (identical(method, "auto")) {
        method <- fastembedr_query_nn_policy("cuda", nrow(data), nrow(query),
            ncol(data), target_recall, k, metric, allow_ivf = !keep_gpu)$method
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
        backend = "cuda",
        tuning = if (identical(method, "ivf")) "pilot" else "auto",
        output = output
    )))
    fastembedr_check_cuda_query_knn(
        out, nrow(query), k, method, target_recall)
}
