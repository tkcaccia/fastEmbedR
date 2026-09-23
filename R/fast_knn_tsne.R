# Public KNN-input and one-call t-SNE interfaces.

#' Run native interpolation-based t-SNE from precomputed KNN
#'
#' `tsne_knn()` is the direct KNN-input entry point for the native
#' interpolation-based optimizer. It accepts either a list containing KNN
#' `indices`
#' and `distances` or separate KNN index and distance matrices. The supplied
#' neighbors are used directly: this function performs no neighbor search or
#' input scaling. PCA is computed only when `init_data` is supplied and
#' `Y_init` is absent.
#' The optimizer follows the sparse-affinity and interpolation-based t-SNE
#' literature while using fastEmbedR's own R API and native kernels.
#'
#' @param indices A list containing KNN `indices` and `distances`, or an integer
#'   KNN index matrix.
#' @param distances Numeric KNN distance matrix matching `indices`. Leave
#'   `NULL` when `indices` is a KNN list.
#' @param n_neighbors Optional number of non-self neighbor columns to use from
#'   the supplied KNN graph. This lets you compute a wide KNN once and reuse
#'   its first columns. If omitted, the compact affinity policy uses
#'   `ceiling(perplexity)` columns. An explicit value must equal that width.
#' @param perplexity t-SNE perplexity. If `NULL`, the optimizer chooses a safe
#'   value from the supplied KNN width and sample size.
#' @param init_data Optional original high-dimensional data matrix used only to
#'   compute PCA initialization for KNN-input runs. It is not used for neighbor
#'   search or optimization.
#' @inheritParams tsne
#' @param backend Optimizer backend: `"cpu"`, `"cuda"`, or `"metal"`. This
#'   selects affinity construction and layout optimization only; it does not
#'   trigger a nearest-neighbor search. Host KNN matrices can be consumed by
#'   any compiled backend. A CUDA-resident KNN object can be consumed directly
#'   only by the CUDA backend. Unavailable GPU requests fail without CPU
#'   fallback.
#' @param n.cores Number of CPU cores used for CPU affinity construction and
#'   t-SNE optimization. `NULL` uses the package t-SNE thread option, which
#'   defaults to four. No KNN search is performed. Metal and CUDA optimizers
#'   ignore this argument.
#' @param ... Additional low-level optimizer controls, including `theta` and
#'   `min_gain`.
#' @return An embedding matrix with settings stored in
#'   `attr(layout, "fastEmbedR_config")`. Float32 KNN distances, including a
#'   CUDA-resident KNN object, produce a `float::float32` layout; host double
#'   distances produce a standard R double matrix. Native optimization uses
#'   float32 in both cases. The exact returned representation is recorded in
#'   `attr(layout, "precision")` and in `fastEmbedR_config$output_precision`.
#' @examples
#' x <- scale(as.matrix(iris[, 1:4]))
#' d <- as.matrix(stats::dist(x))
#' diag(d) <- Inf
#' k <- 15L
#' idx <- t(apply(d, 1L, order))[, seq_len(k), drop = FALSE]
#' dst <- matrix(d[cbind(rep(seq_len(nrow(d)), each = k), as.vector(t(idx)))],
#'     nrow = nrow(d), byrow = TRUE
#' )
#' layout <- tsne_knn(idx, dst,
#'     init_data = x, perplexity = 5,
#'     early_exaggeration_iter = 5, n_iter = 10
#' )
#' @name tsne_knn
NULL

validate_gpu_tsne_knn <- function(info, n_neighbors, perplexity) {
    if (isTRUE(info$has_self)) {
        stop(
            "CUDA GPU-resident t-SNE requires non-self KNN. Provide a ",
            "native fastEmbedR GPU KNN object with non-self neighbors.",
            call. = FALSE
        )
    }
    policy <- opentsne_neighbor_policy(info$n, perplexity, info$k)
    n_neighbors <- n_neighbors %||% policy$n_neighbors
    n_neighbors <- as.integer(n_neighbors)
    invalid <- length(n_neighbors) != 1L || is.na(n_neighbors) ||
        !is.finite(n_neighbors) || n_neighbors < 1L ||
        n_neighbors > info$k || n_neighbors >= info$n
    if (invalid) {
        stop(
            "`n_neighbors` must be a positive integer available in the ",
            "GPU KNN object.",
            call. = FALSE
        )
    }
    perplexity <- perplexity %||% policy$perplexity
    validate_tsne_support_width(n_neighbors, perplexity)
    list(n_neighbors = n_neighbors, perplexity = perplexity)
}

validate_tsne_support_width <- function(n_neighbors, perplexity) {
    required <- opentsne_support_width(perplexity)
    if (n_neighbors != required) {
        stop(
            "Compact t-SNE affinity support requires `n_neighbors = ",
            required, "` for this perplexity.",
            call. = FALSE
        )
    }
    invisible(n_neighbors)
}

run_gpu_tsne_knn <- function(indices, init_data, settings, extra) {
    info <- fastembedr_gpu_knn_info(indices)
    policy <- validate_gpu_tsne_knn(
        info, settings$n_neighbors, settings$perplexity
    )
    Y_init <- resolve_opentsne_y_init(
        settings$Y_init, info$n, settings$n_components
    )
    cuda_init_data <- NULL
    if (is.null(Y_init) && !is.null(init_data)) {
        cuda_init_data <- opentsne_pca_input_matrix(init_data)
        if (nrow(cuda_init_data) != info$n) {
            stop("`init_data` must have one row per GPU KNN row.",
                call. = FALSE
            )
        }
    }
    args <- c(list(
        indices = NULL, distances = NULL,
        n_components = settings$n_components,
        perplexity = policy$perplexity, Y_init = Y_init,
        seed = settings$seed, verbose = settings$verbose,
        backend = settings$backend, n_threads = settings$n_threads,
        gpu_knn = indices, gpu_n = info$n, gpu_k = policy$n_neighbors,
        cuda_init_data = cuda_init_data, input_backend = info$input_backend
    ), settings$optimizer, extra)
    layout <- do.call(fast_knn_opentsne_materialized, args)
    annotate_opentsne_affinity_support(
        layout, policy$n_neighbors, policy$perplexity
    )
}

resolve_host_tsne_knn <- function(
    indices, distances, n_neighbors, perplexity
) {
    if (inherits(indices, "fastEmbedR_tsne_prepared")) {
        if (!is.null(distances)) {
            stop(
                "Do not pass `distances` with a prepared t-SNE object.",
                call. = FALSE
            )
        }
        return(list(
            knn = indices$knn,
            perplexity = perplexity %||% indices$perplexity,
            n_neighbors = n_neighbors %||% indices$n_neighbors
        ))
    }
    raw <- coerce_knn_input(indices, distances)
    policy <- opentsne_neighbor_policy(
        nrow(raw$indices), perplexity, raw$n_neighbors
    )
    n_neighbors <- n_neighbors %||% policy$n_neighbors
    list(
        knn = normalize_opentsne_knn_input(
            indices, distances, n_neighbors
        ),
        perplexity = perplexity %||% policy$perplexity,
        n_neighbors = n_neighbors
    )
}

run_host_tsne_knn <- function(resolved, init_data, settings, extra) {
    knn <- resolved$knn
    validate_tsne_support_width(knn$n_neighbors, resolved$perplexity)
    Y_init <- resolve_opentsne_y_init(
        settings$Y_init, knn$n, settings$n_components
    )
    if (is.null(Y_init) && !is.null(init_data)) {
        init_backend <- if (settings$backend %in% c("metal", "cuda")) {
            settings$backend
        } else {
            "cpu"
        }
        Y_init <- make_opentsne_pca_init_from_data(
            init_data, knn$n, settings$n_components, settings$seed,
            init_backend, settings$n_threads
        )
    }
    args <- c(list(
        indices = knn$indices, distances = knn$distances,
        n_components = settings$n_components,
        perplexity = resolved$perplexity, Y_init = Y_init,
        seed = settings$seed, verbose = settings$verbose,
        backend = settings$backend, n_threads = settings$n_threads,
        input_had_self = knn$has_self, input_backend = knn$input_backend
    ), settings$optimizer, extra)
    layout <- do.call(fast_knn_opentsne_materialized, args)
    annotate_opentsne_affinity_support(
        layout, knn$n_neighbors, resolved$perplexity
    )
}

#' @rdname tsne_knn
#' @export
tsne_knn <- function(
    indices, distances = NULL, n_neighbors = NULL, perplexity = NULL,
    n_components = 2L,
    init_data = NULL, Y_init = NULL, seed = 4L, verbose = FALSE,
    backend = NULL, n.cores = NULL, learning_rate = "auto",
    early_exaggeration_iter = NULL, early_exaggeration = "auto",
    n_iter = NULL, exaggeration = NULL, initial_momentum = 0.8,
    final_momentum = 0.8, max_step_norm = "auto",
    negative_gradient_method = "auto", record_costs = FALSE,
    auto_config = TRUE, ...
) {
    extra <- list(...)
    if ("affinity_support" %in% names(extra)) {
        stop(
            "`affinity_support` is not configurable; t-SNE uses compact ",
            "support with `k = ceiling(perplexity)`.",
            call. = FALSE
        )
    }
    backend <- resolve_embedding_backend(backend)
    settings <- list(
        n_neighbors = n_neighbors, perplexity = perplexity,
        n_components = validate_opentsne_n_components(n_components, backend),
        Y_init = Y_init, seed = seed, verbose = verbose, backend = backend,
        n_threads = n.cores,
        optimizer = list(
            learning_rate = learning_rate,
            early_exaggeration_iter = early_exaggeration_iter,
            early_exaggeration = early_exaggeration, n_iter = n_iter,
            exaggeration = exaggeration, initial_momentum = initial_momentum,
            final_momentum = final_momentum, max_step_norm = max_step_norm,
            negative_gradient_method = negative_gradient_method,
            record_costs = record_costs, auto_config = auto_config
        )
    )
    if (fastembedr_is_gpu_knn(indices) && backend == "cuda") {
        if (!is.null(distances)) {
            stop("Do not pass `distances` with a GPU-resident KNN object.",
                call. = FALSE
            )
        }
        return(run_gpu_tsne_knn(
            fastembedr_as_gpu_knn(indices), init_data, settings, extra
        ))
    }
    resolved <- resolve_host_tsne_knn(
        indices, distances, n_neighbors, perplexity
    )
    run_host_tsne_knn(resolved, init_data, settings, extra)
}

tsne_knn_call_args <- function(data, settings, overrides, extra) {
    args <- c(list(
        indices = data,
        n_components = settings$n_components,
        perplexity = settings$perplexity,
        init_data = settings$init_data,
        Y_init = settings$Y_init,
        seed = settings$seed,
        verbose = settings$verbose,
        backend = settings$backend,
        n.cores = settings$n_threads
    ), settings$optimizer)
    args <- utils::modifyList(args, overrides)
    c(args, extra)
}

zero_tsne_timings <- function(embedding_time) {
    zero <- embedding_time
    zero[] <- 0
    rbind(preprocess = zero, knn = zero, embedding = embedding_time)
}

new_tsne_embedding <- function(
    layout, metrics, parameters, timings, knn = NULL,
    knn_with_self = NULL, preprocess = list(), diagnostics = list()
) {
    out <- list(
        layout = layout,
        labels = NULL,
        method = "tsne",
        metrics = metrics,
        parameters = parameters,
        timings = timings,
        knn = knn,
        knn_with_self = knn_with_self,
        preprocess = preprocess,
        diagnostics = diagnostics
    )
    class(out) <- "fastEmbedR_embedding"
    out
}

tsne_runtime_metrics <- function(layout, n, p, timings, n_neighbors) {
    cfg <- attr(layout, "fastEmbedR_config")
    data.frame(
        method = "tsne",
        n = n,
        p = p,
        n_neighbors = n_neighbors,
        perplexity = cfg$perplexity,
        elapsed = sum(timings[, "elapsed"]),
        preprocess_elapsed = timings["preprocess", "elapsed"],
        knn_elapsed = timings["knn", "elapsed"],
        embedding_elapsed = timings["embedding", "elapsed"],
        stringsAsFactors = FALSE
    )
}

run_gpu_input_tsne <- function(data, nn, settings, extra) {
    if (!is.null(nn)) {
        stop("When `data` is a GPU KNN object, do not also pass `nn`.",
            call. = FALSE
        )
    }
    data <- fastembedr_as_gpu_knn(data)
    result <- timed_do_call(
        tsne_knn,
        tsne_knn_call_args(data, settings, list(), extra)
    )
    layout <- result$value
    cfg <- attr(layout, "fastEmbedR_config")
    info <- fastembedr_gpu_knn_info(data)
    timings <- zero_tsne_timings(result$time)
    init_label <- if (is.null(settings$Y_init) &&
        is.null(settings$init_data)) {
        "none_precomputed_gpu_knn_cuda_random_init"
    } else {
        "none_precomputed_gpu_knn"
    }
    params <- c(list(
        method = "tsne", input = "gpu_resident_knn", n = nrow(layout),
        p = NA_integer_, n_neighbors = cfg$n_neighbors %||% info$k,
        k = cfg$n_neighbors %||% info$k,
        n_components = settings$n_components, seed = as.integer(settings$seed),
        nn_backend = info$input_backend, keep_knn = settings$keep_knn
    ), cfg, list(preprocess = init_label))
    metrics <- tsne_runtime_metrics(
        layout, nrow(layout), NA_integer_, timings,
        cfg$n_neighbors %||% info$k
    )
    new_tsne_embedding(
        layout, metrics, params, timings,
        knn = if (settings$keep_knn) data else NULL,
        preprocess = list(input = init_label)
    )
}

prepare_tsne_knn_input_run <- function(data, nn, settings) {
    if (!is.null(nn)) {
        stop("When `data` is a KNN object, do not also pass `nn`.",
            call. = FALSE
        )
    }
    full <- normalize_opentsne_knn_input(data, NULL, NULL)
    policy <- opentsne_neighbor_policy(
        full$n, settings$perplexity, full$n_neighbors
    )
    knn <- normalize_opentsne_knn_input(
        data, NULL, policy$n_neighbors
    )
    Y_init <- resolve_opentsne_y_init(
        settings$Y_init, knn$n, settings$n_components
    )
    if (is.null(Y_init) && !is.null(settings$init_data)) {
        init_backend <- if (settings$backend %in% c("metal", "cuda")) {
            settings$backend
        } else {
            "cpu"
        }
        Y_init <- make_opentsne_pca_init_from_data(
            settings$init_data, knn$n, settings$n_components,
            settings$seed, init_backend, settings$n_threads
        )
    }
    list(knn = knn, perplexity = policy$perplexity, Y_init = Y_init)
}

run_knn_input_tsne <- function(data, nn, settings, extra) {
    prepared <- prepare_tsne_knn_input_run(data, nn, settings)
    overrides <- list(
        n_neighbors = prepared$knn$n_neighbors,
        perplexity = prepared$perplexity,
        Y_init = prepared$Y_init
    )
    result <- timed_do_call(
        tsne_knn,
        tsne_knn_call_args(data, settings, overrides, extra)
    )
    layout <- result$value
    cfg <- attr(layout, "fastEmbedR_config")
    timings <- zero_tsne_timings(result$time)
    backend <- prepared$knn$input_backend %||% "supplied"
    if (is.na(backend)) backend <- "supplied"
    params <- c(list(
        method = "tsne", input = "knn", n = prepared$knn$n,
        p = NA_integer_, n_neighbors = prepared$knn$n_neighbors,
        k = prepared$knn$n_neighbors + 1L,
        n_components = settings$n_components, seed = as.integer(settings$seed),
        nn_backend = backend, keep_knn = settings$keep_knn
    ), cfg, list(preprocess = "none_precomputed_knn"))
    kept <- if (settings$keep_knn) {
        prepared$knn[c("indices", "distances")]
    } else {
        NULL
    }
    metrics <- tsne_runtime_metrics(
        layout, prepared$knn$n, NA_integer_, timings,
        prepared$knn$n_neighbors
    )
    diagnostics <- list(
        metal_trace = attr(layout, "metal_trace"),
        metal_stage_timing = attr(layout, "metal_stage_timing")
    )
    new_tsne_embedding(
        layout, metrics, params, timings, kept, NULL,
        list(input = "precomputed_knn"), diagnostics
    )
}

summarize_tsne_gpu_knn <- function(knn, n, n_neighbors) {
    info <- fastembedr_gpu_knn_info(knn)
    incompatible <- info$n != n || info$k < n_neighbors ||
        isTRUE(info$has_self)
    if (incompatible) {
        stop("GPU-resident KNN output is incompatible with t-SNE input.",
            call. = FALSE
        )
    }
    list(
        indices = NULL, distances = NULL,
        n_neighbors = as.integer(n_neighbors), has_self = FALSE,
        knn_with_self = NULL, nn_backend = info$input_backend
    )
}

compute_tsne_matrix_knn <- function(
    x, nn, n, n_neighbors, backend, metric, n_threads, keep_knn
) {
    engine <- "supplied"
    if (is.null(nn)) {
        policy <- fastembedr_embedding_nn_policy(backend, n = n)
        engine <- fastembedr_nn_policy_engine(
            policy,
            keep_gpu = policy$backend == "cuda"
        )
        nn <- fastembedr_nn_without_self(
            x,
            k = n_neighbors, backend = policy$backend,
            method = policy$method, metric = metric,
            output = fastembedr_knn_output_type(x, policy$backend),
            n_threads = n_threads, tuning = policy$tuning,
            target_recall = policy$target_recall,
            keep_gpu = policy$backend == "cuda"
        )
    }
    if (fastembedr_is_gpu_knn(nn)) {
        nn <- fastembedr_as_gpu_knn(nn)
        result <- summarize_tsne_gpu_knn(nn, n, n_neighbors)
    } else {
        result <- normalize_supplied_knn(
            nn, n, n_neighbors,
            keep_self = keep_knn
        )
        result$nn_backend <- attr(nn, "backend") %||% "supplied"
    }
    list(result = result, input = nn, engine = engine)
}

prepare_tsne_matrix_init <- function(x, knn_input, settings, n) {
    Y_init <- resolve_opentsne_y_init(
        settings$Y_init, n, settings$n_components
    )
    info <- list(method = "user", backend = NA_character_)
    cuda_data <- NULL
    if (!is.null(Y_init)) {
        return(list(Y_init = Y_init, info = info, cuda_data = cuda_data))
    }
    source <- settings$init_data %||% x
    if (settings$backend == "cuda" && fastembedr_is_gpu_knn(knn_input)) {
        cuda_data <- opentsne_pca_input_matrix(source)
        if (nrow(cuda_data) != n) {
            stop("`init_data` must have one row per input row.", call. = FALSE)
        }
        info$method <- "pca_cuda_auto_device"
        info$backend <- "cuda_native_pca_device"
        return(list(Y_init = NULL, info = info, cuda_data = cuda_data))
    }
    init_backend <- if (settings$backend %in% c("metal", "cuda")) {
        settings$backend
    } else {
        "cpu"
    }
    Y_init <- make_opentsne_pca_init_from_data(
        source, n, settings$n_components, settings$seed,
        init_backend, settings$n_threads
    )
    info$method <- attr(Y_init, "fastEmbedR_init_method") %||% "pca"
    info$backend <- attr(Y_init, "fastEmbedR_init_backend") %||% init_backend
    list(Y_init = Y_init, info = info, cuda_data = cuda_data)
}

validate_tsne_dots <- function(extra) {
    if ("init" %in% names(extra)) {
        stop(
            "`init` is not an argument of `tsne()`; use `Y_init` or ",
            "`init_data` for PCA initialization.",
            call. = FALSE
        )
    }
    if ("n_neighbors" %in% names(extra)) {
        stop(
            "`n_neighbors` is not an argument of `tsne()`; use ",
            "`perplexity` instead.",
            call. = FALSE
        )
    }
    if ("affinity_support" %in% names(extra)) {
        stop(
            "`affinity_support` is not configurable; t-SNE uses compact ",
            "support with `k = ceiling(perplexity)`.",
            call. = FALSE
        )
    }
    invisible(extra)
}

tsne_matrix_metrics <- function(layout, x, knn, timings) {
    cfg <- attr(layout, "fastEmbedR_config")
    data.frame(
        method = "tsne", n = nrow(x), p = ncol(x),
        n_neighbors = knn$n_neighbors, perplexity = cfg$perplexity,
        elapsed = sum(timings[, "elapsed"]),
        preprocess_elapsed = timings["preprocess", "elapsed"],
        knn_elapsed = timings["knn", "elapsed"],
        initialization_elapsed = timings["initialization", "elapsed"],
        embedding_elapsed = timings["embedding", "elapsed"],
        stringsAsFactors = FALSE
    )
}

assemble_matrix_tsne <- function(
    layout, prepared, knn_state, init_state, settings, timings, metric
) {
    x <- prepared$data
    knn <- knn_state$result
    cfg <- attr(layout, "fastEmbedR_config")
    params <- c(list(
        method = "tsne", n = nrow(x), p = ncol(x),
        n_neighbors = knn$n_neighbors, k = knn$n_neighbors + 1L,
        n_components = settings$n_components,
        seed = as.integer(settings$seed), nn_backend = knn$nn_backend,
        nn_engine = knn_state$engine, metric = metric,
        keep_knn = settings$keep_knn
    ), cfg, prepared$preprocess, list(
        init = init_state$info$method,
        init_backend = init_state$info$backend
    ))
    kept <- if (settings$keep_knn) {
        if (fastembedr_is_gpu_knn(knn_state$input)) {
            knn_state$input
        } else {
            knn[c("indices", "distances")]
        }
    } else {
        NULL
    }
    diagnostics <- list(
        metal_trace = attr(layout, "metal_trace"),
        metal_stage_timing = attr(layout, "metal_stage_timing")
    )
    new_tsne_embedding(
        layout, tsne_matrix_metrics(layout, x, knn, timings), params,
        timings, kept,
        if (settings$keep_knn) knn$knn_with_self else NULL,
        prepared$preprocess, diagnostics
    )
}

run_matrix_input_tsne <- function(data, nn, settings, extra, input_float) {
    preprocess <- timed_do_call(prepare_embedding_data, list(
        data, settings$standardize, settings$pca_dims, settings$seed,
        backend = resolve_preprocess_backend(settings$backend)
    ))
    x <- preprocess$value$data
    metric <- resolve_embedding_metric(settings$metric, x)
    policy <- opentsne_neighbor_policy(
        nrow(x), settings$perplexity
    )
    settings$perplexity <- policy$perplexity
    knn <- timed_do_call(compute_tsne_matrix_knn, list(
        x, nn, nrow(x), policy$n_neighbors, settings$backend, metric,
        settings$n_threads, settings$keep_knn
    ))
    init <- timed_do_call(prepare_tsne_matrix_init, list(
        x, knn$value$input, settings, nrow(x)
    ))
    overrides <- list(
        n_neighbors = knn$value$result$n_neighbors,
        perplexity = policy$perplexity,
        init_data = init$value$cuda_data %||% settings$init_data,
        Y_init = init$value$Y_init
    )
    embedding <- timed_do_call(
        tsne_knn,
        tsne_knn_call_args(knn$value$input, settings, overrides, extra)
    )
    layout <- finalize_embedding_layout(
        embedding$value, "TSNE",
        return_float32 = input_float && is_float32_matrix(x)
    )
    timings <- rbind(
        preprocess = preprocess$time, knn = knn$time,
        initialization = init$time, embedding = embedding$time
    )
    assemble_matrix_tsne(
        layout, preprocess$value, knn$value, init$value, settings,
        timings, metric
    )
}

#' Run native interpolation-based t-SNE from a data matrix
#'
#' `tsne()` computes or reuses a KNN graph, then runs the package-native
#' two-phase t-SNE optimizer. The default optimizer is CPU-native.
#' Explicit `backend = "metal"` and `backend = "cuda"` requests use the
#' matching package-native GPU optimizer when compiled and fail clearly if the
#' requested backend is unavailable.
#' The implementation follows published sparse-affinity and interpolation-based
#' t-SNE methods and provides independent CPU, Metal, and CUDA kernels.
#'
#' @param data Numeric matrix/data frame with observations in rows, or a list
#'   containing KNN `indices` and `distances`.
#' @param perplexity t-SNE perplexity. If `NULL`, uses the largest safe value
#'   up to 30 that is available for the input. Compact affinity support uses
#'   `ceiling(perplexity)` non-self neighbors.
#' @param n_components Output dimensionality, from 1 to 3. Dimensions other
#'   than two use CPU exact repulsion; the current Metal and CUDA
#'   interpolation/FFT optimizers support only `2L`.
#' @param init_data Optional original high-dimensional data matrix used only to
#'   compute PCA initialization with [tsne_pca_init()]. It is not used for
#'   neighbor search or optimization.
#' @param Y_init Optional explicit initial layout. Use [tsne_pca_init()] to
#'   precompute and reuse a PCA initialization.
#' @param standardize Center and scale columns before KNN. Defaults to `FALSE`
#'   so one-call results match a KNN object computed from the supplied matrix.
#' @param pca_dims Optional PCA dimension before KNN.
#' @param metric KNN distance metric for one-call matrix input: `"euclidean"`,
#'   `"cosine"`, `"correlation"`, or `"inner_product"`.
#' @param nn Optional precomputed KNN output when `data` is a data matrix.
#' @param seed Random seed.
#' @param backend Execution backend: `"cpu"`, `"cuda"`, or `"metal"`. CPU KNN
#'   uses package-native HNSW. Metal uses package-native exact or recall-tuned
#'   IVF-Flat search. CUDA uses cuVS brute-force exact search below 100,000
#'   rows and cuVS IVF-Flat above that threshold; an explicitly enabled FAISS
#'   GPU build may provide exact search. Device pointers pass directly to the
#'   native CUDA t-SNE optimizer.
#'   Unsupported GPU requests fail clearly and are not relabelled CPU runs.
#' @param keep_knn If `TRUE`, retain KNN matrices in the returned object.
#' @param verbose Print optimizer progress.
#' @param n.cores Number of CPU cores used by CPU KNN and CPU t-SNE
#'   optimization. `NULL` uses one CPU KNN worker and the package t-SNE thread
#'   option, which defaults to four. Native GPU optimizers ignore this
#'   argument.
#' @param learning_rate Positive number or `"auto"`. With `"auto"`, the native
#'   optimizer uses `n / exaggeration` separately for each phase.
#' @param early_exaggeration_iter Number of early-exaggeration iterations.
#' @param early_exaggeration Early-exaggeration multiplier, or `"auto"` for
#'   the default t-SNE rule.
#' @param n_iter Number of normal optimization iterations after early
#'   exaggeration.
#' @param exaggeration Normal-phase exaggeration. `NULL` means 1.
#' @param initial_momentum Momentum during early exaggeration.
#' @param final_momentum Momentum during normal optimization.
#' @param max_step_norm Maximum per-point update norm. `"auto"` uses the
#'   same validated limit on CPU, Metal, and CUDA. Use `NULL` or `NA` to
#'   disable clipping.
#' @param negative_gradient_method `"auto"`, `"exact"`, or
#'   `"fft"`. On CPU, `"auto"` resolves to the native grid-FFT
#'   FIt-SNE-style negative-gradient approximation. Native GPU FFT/exact paths
#'   are used only when the corresponding compiled symbols are available;
#'   otherwise GPU requests fail clearly rather than falling back to CPU.
#' @param record_costs If `TRUE`, compute diagnostic KL/cost traces.
#' @param auto_config If `TRUE`, choose missing t-SNE settings with a native
#'   C++ opt-SNE-inspired policy. The policy uses `n / early_exaggeration` for
#'   `"auto"` learning rate, chooses missing iteration limits, and enables
#'   KLD-based early stopping only on CPU/small exact runs where the monitor is
#'   not prohibitively expensive. Explicit user-supplied values are respected.
#' @param ... Additional low-level parameters passed to [tsne_knn()].
#' @details
#' The t-SNE API exposes the principal
#' scientifically consequential optimizer controls: perplexity,
#' initialization, iteration counts, early and normal exaggeration,
#' learning rate, momentum, clipping, and exact-versus-FFT repulsion. Setting
#' `auto_config = FALSE` disables automatic iteration and stopping choices;
#' explicit values always override automatic values. The matrix-input function
#' deliberately does not expose a nearest-neighbor index type or tuning
#' parameters: it uses the backend router with target recall 0.99. Supply an
#' externally generated KNN object to [tsne_knn()] when search policy must be
#' controlled independently. The resolved settings are returned in
#' `fit$parameters`. Affinity construction always uses compact support with
#' `k = ceiling(perplexity)` non-self neighbors.
#' @return A `fastEmbedR_embedding` object.
#' @examples
#' fit <- tsne(
#'     as.matrix(iris[, 1:4]),
#'     perplexity = 5,
#'     early_exaggeration_iter = 5, n_iter = 10, seed = 1
#' )
#' plot(fit, labels = iris$Species)
#' @export
tsne <- function(
    data, perplexity = NULL,
    n_components = 2L,
    init_data = NULL, Y_init = NULL, standardize = FALSE,
    pca_dims = NULL,
    metric = c("euclidean", "cosine", "correlation", "inner_product"),
    nn = NULL, seed = 4L, backend = NULL, keep_knn = FALSE,
    verbose = FALSE, n.cores = NULL, learning_rate = "auto",
    early_exaggeration_iter = NULL, early_exaggeration = "auto",
    n_iter = NULL, exaggeration = NULL, initial_momentum = 0.8,
    final_momentum = 0.8, max_step_norm = "auto",
    negative_gradient_method = "auto", record_costs = FALSE,
    auto_config = TRUE, ...
) {
    extra <- list(...)
    validate_tsne_dots(extra)
    backend <- resolve_embedding_backend(backend)
    settings <- list(
        perplexity = perplexity,
        n_components = validate_opentsne_n_components(
            n_components, backend
        ),
        init_data = init_data, Y_init = Y_init,
        standardize = standardize, pca_dims = pca_dims, metric = metric,
        seed = seed, backend = backend, keep_knn = isTRUE(keep_knn),
        verbose = verbose, n_threads = n.cores,
        optimizer = list(
            learning_rate = learning_rate,
            early_exaggeration_iter = early_exaggeration_iter,
            early_exaggeration = early_exaggeration, n_iter = n_iter,
            exaggeration = exaggeration,
            initial_momentum = initial_momentum,
            final_momentum = final_momentum,
            max_step_norm = max_step_norm,
            negative_gradient_method = negative_gradient_method,
            record_costs = record_costs, auto_config = auto_config
        )
    )
    if (fastembedr_is_gpu_knn(data)) {
        return(run_gpu_input_tsne(data, nn, settings, extra))
    }
    if (is_knn_input(data)) {
        return(run_knn_input_tsne(data, nn, settings, extra))
    }
    run_matrix_input_tsne(data, nn, settings, extra, is_float32_matrix(data))
}
