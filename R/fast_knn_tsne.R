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
#'   `ceiling(perplexity)` columns. A wider value such as
#'   `ceiling(3 * perplexity)` is an explicit affinity-model control.
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
#'   only by the CUDA backend. Coordinated CUDA fits materialize KNN in
#'   host RAM. Unavailable GPU requests fail without CPU fallback.
#' @param n.cores Number of CPU cores used for CPU affinity construction and
#'   t-SNE optimization. `NULL` uses the package t-SNE thread option, which
#'   defaults to four. No KNN search is performed. Coordinated CUDA fits
#'   use this count for host sparse-affinity construction.
#' @param devices Optional distinct CUDA device indices for one resident
#'   2D or 3D fit with NCCL. KNN affinities remain in host RAM.
#' @param ... Additional low-level optimizer controls, including `theta` and
#'   `min_gain`.
#' @return An embedding matrix with settings stored in
#'   `attr(layout, "fastEmbedR_config")`. Float32 KNN distances, including a
#'   CUDA-resident KNN object, produce a `float::float32` layout; host double
#'   distances produce a standard R double matrix. Native optimization uses
#'   float32 in both cases. The returned type is recorded in
#'   `attr(layout, "precision")` and the configuration's `output_precision`.
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
    n_neighbors <- integer_scalar(n_neighbors)
    invalid <- is.na(n_neighbors) || n_neighbors < 1L ||
        n_neighbors > info$k || n_neighbors >= info$n
    if (invalid) {
        stop(
            "`n_neighbors` must be a positive integer available in the ",
            "GPU KNN object.",
            call. = FALSE
        )
    }
    perplexity <- perplexity %||%
        min(policy$perplexity, n_neighbors)
    validate_tsne_support_width(n_neighbors, perplexity)
    list(n_neighbors = n_neighbors, perplexity = perplexity)
}

validate_tsne_support_width <- function(n_neighbors, perplexity) {
    required <- opentsne_support_width(perplexity)
    width <- integer_scalar(n_neighbors)
    if (is.na(width) || width < required) {
        stop(
            "`n_neighbors` must be a positive integer of at least ",
            required, " for this perplexity.",
            call. = FALSE
        )
    }
    invisible(width)
}

run_gpu_tsne_knn <- function(indices, init_data, settings, extra) {
    if (length(settings$devices) > 1L) {
        host <- fastembedr_gpu_knn_to_host(indices)
        resolved <- resolve_host_tsne_knn(host, NULL,
            settings$n_neighbors, settings$perplexity)
        return(run_host_tsne_knn(resolved, init_data, settings, extra))
    }
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
        if (!is.null(n_neighbors)) {
            width <- integer_scalar(n_neighbors)
            if (is.na(width) || width != indices$n_neighbors) stop(
                "A prepared t-SNE KNN cannot change its affinity width.",
                call. = FALSE)
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
        perplexity = perplexity %||%
            min(policy$perplexity, n_neighbors),
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
        devices = settings$devices,
        input_had_self = knn$has_self, input_backend = knn$input_backend
    ), settings$optimizer, extra)
    layout <- do.call(fast_knn_opentsne_materialized, args)
    annotate_opentsne_affinity_support(
        layout, knn$n_neighbors, resolved$perplexity
    )
}

#' @rdname tsne_knn
#' @export
#' @keywords internal
tsne_knn <- function(
    indices, distances = NULL, n_neighbors = NULL, perplexity = NULL,
    n_components = 2L,
    init_data = NULL, Y_init = NULL, seed = 4L, verbose = FALSE,
    backend = NULL, n.cores = NULL, learning_rate = "auto",
    early_exaggeration_iter = NULL, early_exaggeration = "auto",
    n_iter = NULL, exaggeration = NULL, initial_momentum = 0.8,
    final_momentum = 0.8, max_step_norm = "auto",
    negative_gradient_method = "auto", record_costs = FALSE,
    auto_config = TRUE, devices = NULL, ...
) {
    extra <- list(...)
    if ("affinity_support" %in% names(extra)) {
        stop(
            "`affinity_support` is not configurable; set ",
            "`n_neighbors` for an explicit support width.",
            call. = FALSE
        )
    }
    backend <- resolve_embedding_backend(backend)
    settings <- list(
        n_neighbors = n_neighbors, perplexity = perplexity,
        n_components = validate_opentsne_n_components(n_components),
        Y_init = Y_init, seed = seed, verbose = verbose, backend = backend,
        n_threads = n.cores, devices = devices,
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
        n_neighbors = settings$affinity_k,
        n_components = settings$n_components,
        perplexity = settings$perplexity,
        init_data = settings$init_data,
        Y_init = settings$Y_init,
        seed = settings$seed,
        verbose = settings$verbose,
        backend = settings$backend,
        n.cores = settings$n_threads,
        devices = settings$devices
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
    policy <- opentsne_neighbor_policy(full$n, settings$perplexity,
        full$n_neighbors, settings$affinity_k)
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
    x, nn, n, n_neighbors, backend, metric, n_threads, keep_knn,
    retain_cuda_data
) {
    engine <- "supplied"
    if (is.null(nn)) {
        policy <- fastembedr_embedding_nn_policy(
            backend, n = n, p = ncol(x))
        engine <- fastembedr_nn_policy_engine(
            policy, keep_gpu = policy$backend == "cuda",
            metric = metric
        )
        nn <- fastembedr_nn_without_self(
            x,
            k = n_neighbors, backend = policy$backend,
            method = policy$method, metric = metric,
            output = fastembedr_knn_output_type(x, policy$backend),
            n_threads = n_threads, tuning = policy$tuning,
            target_recall = policy$target_recall,
            keep_gpu = policy$backend == "cuda",
            retain_data = isTRUE(retain_cuda_data)
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
            "`affinity_k` for the affinity width.",
            call. = FALSE
        )
    }
    if ("affinity_support" %in% names(extra)) {
        stop(
            "`affinity_support` is not configurable; use `affinity_k`.",
            call. = FALSE
        )
    }
    invisible(extra)
}

validate_tsne_resident_devices <- function(devices, backend) {
    if (!is.null(devices) && backend != "cuda")
        stop("Resident `devices` requires CUDA t-SNE.",
            call. = FALSE)
}

tsne_optimizer_settings <- function(
    learning_rate, early_exaggeration_iter, early_exaggeration,
    n_iter, exaggeration, initial_momentum, final_momentum,
    max_step_norm, negative_gradient_method, record_costs, auto_config
) {
    list(
        learning_rate = learning_rate,
        early_exaggeration_iter = early_exaggeration_iter,
        early_exaggeration = early_exaggeration, n_iter = n_iter,
        exaggeration = exaggeration, initial_momentum = initial_momentum,
        final_momentum = final_momentum, max_step_norm = max_step_norm,
        negative_gradient_method = negative_gradient_method,
        record_costs = record_costs, auto_config = auto_config
    )
}

new_tsne_settings <- function(
    perplexity, n_components, init_data, Y_init, standardize, pca_dims,
    metric, seed, backend, keep_knn, verbose, n.cores, optimizer,
    devices = NULL, affinity_k = NULL
) {
    list(
        perplexity = perplexity, affinity_k = affinity_k,
        n_components = validate_opentsne_n_components(n_components),
        init_data = init_data, Y_init = Y_init,
        standardize = standardize, pca_dims = pca_dims, metric = metric,
        seed = seed, backend = backend, keep_knn = isTRUE(keep_knn),
        devices = devices,
        verbose = verbose, n_threads = n.cores, optimizer = optimizer
    )
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
        backend = resolve_preprocess_backend(settings$backend),
        n.cores = settings$n_threads
    ))
    x <- preprocess$value$data
    metric <- resolve_embedding_metric(settings$metric, x)
    policy <- opentsne_neighbor_policy(nrow(x), settings$perplexity,
        affinity_k = settings$affinity_k)
    settings$perplexity <- policy$perplexity
    retain_cuda_data <- is.null(nn) && settings$backend == "cuda" &&
        metric == "euclidean" && is.null(settings$Y_init) &&
        is.null(settings$init_data)
    knn <- timed_do_call(compute_tsne_matrix_knn, list(
        x, nn, nrow(x), policy$n_neighbors, settings$backend, metric,
        settings$n_threads, settings$keep_knn, retain_cuda_data
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

tsne_landmark_controls <- function(landmarks, transform_k,
                                    transform_perplexity, transform_iter,
                                    transform_early_exaggeration_iter,
                                    transform_n_negatives, initialization) {
    list(landmarks = landmarks, transform_k = transform_k,
        transform_perplexity = transform_perplexity,
        transform_iter = transform_iter,
        transform_early_exaggeration_iter =
            transform_early_exaggeration_iter,
        transform_n_negatives = transform_n_negatives,
        initialization = initialization)
}

dispatch_tsne_resident <- function(data, nn, settings, controls, extra,
        devices, landmarks) {
    validate_tsne_resident_devices(devices, settings$backend)
    if (landmark_embedding_requested(landmarks))
        return(run_landmark_tsne(data, nn, settings, controls, extra))
    if (fastembedr_is_gpu_knn(data))
        return(run_gpu_input_tsne(data, nn, settings, extra))
    if (is_knn_input(data))
        return(run_knn_input_tsne(data, nn, settings, extra))
    run_matrix_input_tsne(data, nn, settings, extra, is_float32_matrix(data))
}

validate_tsne_affinity_scope <- function(affinity_k, massive) {
    if (!is.null(affinity_k) &&
        !massive %in% c("off", "out_of_core_data")) stop(
        "`affinity_k` needs a data-matrix t-SNE fit.",
        call. = FALSE)
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
#' @param data Numeric matrix/data frame, a file-backed [massive_matrix()],
#'   a list containing KNN `indices` and `distances`, or a saved affinity
#'   graph. A file-backed matrix without landmarks triggers full-data graph
#'   preparation in the experimental one-call route.
#' @param perplexity t-SNE perplexity. If `NULL`, uses the largest safe value
#'   up to 30 that is available for the input. Compact affinity support uses
#'   `ceiling(perplexity)` non-self neighbors.
#' @param affinity_k Optional number of non-self neighbors for
#'   t-SNE affinities. Defaults to `ceiling(perplexity)`; use
#'   `ceiling(3 * perplexity)` for conventional-width support. This setting
#'   also controls the reference fit in resident landmark mode and KNN
#'   preparation for a file-backed one-call full fit. It is unavailable when
#'   a prepared affinity graph is supplied.
#' @param n_components Output dimensionality, 2 or 3. FFT repulsion is used
#'   on every backend.
#' @param init_data Optional original high-dimensional data matrix used only to
#'   compute PCA initialization. It is not used for
#'   neighbor search or optimization.
#' @param Y_init Optional explicit initial layout. In full-graph massive
#'   mode this must be a 2D or 3D file-backed `.f32` matrix. Use
#'   `pca(data, tsne_init = TRUE)$tsne_init` to precompute and reuse a PCA
#'   initialization.
#' @param standardize Center and scale columns before KNN. Defaults to `FALSE`
#'   so one-call results match a KNN object computed from the supplied matrix.
#' @param pca_dims Optional PCA dimension before KNN.
#' @param metric KNN distance metric for one-call matrix input: `"euclidean"`,
#'   `"cosine"`, or `"correlation"`.
#' @param nn Optional precomputed KNN output for resident matrix input. In
#'   EXPERIMENTAL `massive = "landmark"` mode, a previous massive landmark
#'   embedding reuses its saved selection and query-to-landmark KNN. Its
#'   source, reference, neighbor count, and backend must match.
#' @param seed Random seed.
#' @param backend Execution backend: `"cpu"`, `"cuda"`, or `"metal"`. CPU KNN
#'   uses native exact search below 5,000 rows, compiled FAISS HNSW for larger
#'   Euclidean searches, and native HNSW for other supported metrics.
#'   Metal uses package-native exact or recall-tuned IVF-Flat search. CUDA uses
#'   cuVS brute-force exact search below 100,000 rows and cuVS IVF-Flat above
#'   that threshold. Single-GPU device pointers pass directly to the native
#'   CUDA optimizer; coordinated fits materialize KNN in host RAM.
#'   Unsupported GPU requests fail clearly and are not relabelled CPU runs.
#'   Above 5,000 rows, one-call file-backed Metal t-SNE prepares neighbors
#'   with CPU HNSW; the returned pipeline records that stage separately.
#'   Metal bounds graph uploads, but retains coordinates, optimizer state,
#'   and FFT workspace in memory. It is not a fully streamed fit.
#' @param keep_knn If `TRUE`, retain KNN matrices or file-backed KNN files
#'   in the returned pipeline. One-call file-backed fits release KNN files
#'   after graph construction when this is `FALSE`.
#' @param verbose Print optimizer progress.
#' @param n.cores Number of CPU cores used by CPU PCA preprocessing, KNN, and
#'   t-SNE optimization. With `NULL`, PCA uses `options(n.cores)` (one by
#'   default), while KNN and t-SNE use their stage defaults. Native GPU stages
#'   ignore this argument.
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
#' @param negative_gradient_method `"auto"` or `"fft"`; both use FFT.
#'   Other methods fail explicitly. GPU FFT requires compiled native symbols.
#' @param record_costs If `TRUE`, compute diagnostic KL/cost traces.
#'   CUDA and three-dimensional Metal fits reject this option explicitly.
#' @param auto_config If `TRUE`, choose missing t-SNE settings with a native
#'   C++ opt-SNE-inspired policy. The policy uses `n / early_exaggeration` for
#'   `"auto"` learning rate, chooses missing iteration limits, and enables
#'   KLD-based early stopping on small CPU runs where the monitor is not
#'   prohibitively expensive. Explicit user-supplied values are respected.
#' @param landmarks `FALSE` (the default) embeds all observations. A fraction
#'   in `(0, 1)`, a positive landmark count, or explicit row indices enables
#'   landmark embedding followed by fixed-reference transformation of the
#'   remaining rows. File-backed landmark mode currently accepts a count
#'   only; fractions and explicit row indices require resident input.
#' @param transform_k Number of landmark neighbors used to place non-landmark
#'   observations. Used only when landmarking is enabled.
#' @param transform_perplexity Perplexity used to transform non-landmark rows.
#' @param transform_iter Number of normal transform iterations. Use zero for
#'   projection without transform refinement.
#' @param transform_early_exaggeration_iter Number of transform
#'   early-exaggeration iterations.
#' @param transform_n_negatives Number of sampled reference negatives used by
#'   large fixed-reference transforms.
#' @param initialization Initial placement for transformed rows: `"median"`,
#'   `"weighted"`, or `"random"`.
#' @param massive `"off"` (default), experimental `"landmark"`, or
#'   `"out_of_core_graph"` for a prepared t-SNE affinity graph. When
#'   omitted, a prepared file-backed graph selects full-graph mode; a
#'   file-backed matrix selects a full fit unless landmarks are supplied.
#' @param landmark_method Sampling method for experimental landmark mode:
#'   `"reservoir"` (default) or `"random"` for fast random-access storage.
#'   Auto mode uses ordinary t-SNE when its conservative memory estimate fits;
#'   otherwise it reports and runs approximate landmark t-SNE. The landmark
#'   route requires a file-backed [massive_matrix()] and a new `.f32` output.
#'   Full-graph mode streams affinities into the CPU, Metal, or CUDA FFT
#'   optimizer.
#' @param output New `.f32` path for experimental file-backed coordinates.
#'   A one-call full fit also writes KNN, affinity, and PCA initialization
#'   files beside this path and reports their stage times.
#' @param chunk_rows Maximum experimental query rows per batch.
#' @param memory_limit Conservative experimental RAM budget. When omitted for
#'   file-backed input, uses `options(memory_limit = ...)`, or
#'   `"8GB"`.
#' @param storage `"auto"`, `"resident"`, or `"file_backed"`. When omitted,
#'   uses `options(storage = ...)`. Explicit `"resident"` on a
#'   [massive_matrix()] file descriptor loads float32 input after a RAM and
#'   `memory_limit` check; the limit does not bound the subsequent fit.
#' @param checkpoint Save experimental workflow stages. Full-graph t-SNE
#'   also saves coordinates, momentum updates, and adaptive gains every
#'   100 iterations and at the final iteration. One-call full fits use a
#'   source-identified stage record and restartable global graph sorting,
#'   which needs more SSD space than the default bucketed sort.
#' @param resume Continue a matching interrupted experimental workflow.
#'   Changed input, backend, or fit controls fail without recomputation.
#'   One-call recovery requires a committed stage or native checkpoint;
#'   an incomplete stage with no valid checkpoint fails closed.
#' @param devices Optional CUDA device indices. In resident full-data mode,
#'   multiple devices coordinate one 2D or 3D fit with host KNN
#'   affinities and NCCL. The experimental file-backed full-graph route
#'   also accepts multiple CUDA devices.
#'   Resident landmark fits coordinate the reference optimizer across
#'   devices and shard the fixed-reference query transform across them.
#'   Query KNN is computed on one GPU and materialized in host RAM before
#'   its device shards; the scientific calculation uses no SSD files.
#' @param ... Additional low-level parameters passed to [tsne_knn()].
#' @details
#' The t-SNE API exposes the principal
#' scientifically consequential optimizer controls: perplexity,
#' initialization, iteration counts, early and normal exaggeration,
#' learning rate, momentum, and clipping. FFT repulsion is fixed. Setting
#' `auto_config = FALSE` disables automatic iteration and stopping choices;
#' explicit values always override automatic values. The matrix-input function
#' deliberately does not expose a nearest-neighbor index type or tuning
#' parameters: it uses the backend router with target recall 0.99. Supply an
#' externally generated KNN object to [tsne_knn()] when search policy must be
#' controlled independently. The resolved settings are returned in
#' `fit$parameters`. Affinities default to compact support with
#' `k = ceiling(perplexity)` non-self neighbors. Landmark mode embeds the
#' selected reference rows with the same t-SNE implementation and transforms
#' all remaining rows while keeping the reference coordinates fixed.
#' Resident precomputed KNN input and explicit `init_data` or `Y_init`
#' cannot be combined with landmarking.
#' @return A `fastEmbedR_embedding` object. Landmark fits also contain `model`,
#'   which can be passed to [project_landmark_model()] for new observations.
#' @examples
#' fit <- tsne(
#'     as.matrix(iris[, 1:4]),
#'     perplexity = 5,
#'     early_exaggeration_iter = 5, n_iter = 10, seed = 1
#' )
#' plot(fit, labels = iris$Species)
#' @export
tsne <- function(data, perplexity = NULL, n_components = 2L, init_data = NULL,
    Y_init = NULL, standardize = FALSE, pca_dims = NULL,
    metric = c("euclidean", "cosine", "correlation"), nn = NULL, seed = 4L,
    backend = NULL, keep_knn = FALSE, verbose = FALSE, n.cores = NULL,
    learning_rate = "auto", early_exaggeration_iter = NULL,
    early_exaggeration = "auto", n_iter = NULL, exaggeration = NULL,
    initial_momentum = 0.8, final_momentum = 0.8, max_step_norm = "auto",
    negative_gradient_method = "auto", record_costs = FALSE,
    auto_config = TRUE, landmarks = FALSE, transform_k = NULL,
    transform_perplexity = 5, transform_iter = 250L,
    transform_early_exaggeration_iter = 0L, transform_n_negatives = NULL,
    initialization = c("median", "weighted", "random"), massive = "off",
    output = NULL, chunk_rows = NULL, storage = NULL, memory_limit = "8GB",
    devices = NULL, landmark_method = "reservoir", checkpoint = FALSE,
    resume = FALSE, affinity_k = NULL, ...) {
    storage <- fastembedr_embedding_storage(data, landmarks, massive,
        missing(massive), memory_limit, missing(memory_limit), "t-SNE", storage,
        fastembedr_resident_controls_ok(output, chunk_rows,
            checkpoint, resume))
    data <- storage$data
    massive <- storage$massive
    memory_limit <- storage$memory_limit
    extra <- validate_tsne_dots(list(...))
    backend <- resolve_embedding_backend(backend)
    optimizer <- tsne_optimizer_settings(
        learning_rate, early_exaggeration_iter, early_exaggeration,
        n_iter, exaggeration, initial_momentum, final_momentum,
        max_step_norm, negative_gradient_method, record_costs, auto_config
    )
    settings <- new_tsne_settings(perplexity, n_components,
        init_data, Y_init, standardize, pca_dims, metric, seed,
        backend, keep_knn, verbose, n.cores, optimizer, devices,
        affinity_k)
    validate_tsne_affinity_scope(affinity_k, massive)
    controls <- if (!identical(massive, "off") ||
        landmark_embedding_requested(landmarks))
        tsne_landmark_controls(landmarks, transform_k,
            transform_perplexity, transform_iter,
            transform_early_exaggeration_iter,
            transform_n_negatives, initialization) else NULL
    if (!identical(massive, "off"))
        return(run_massive_tsne(data, nn, settings, extra, controls,
            massive, output, chunk_rows, memory_limit, devices,
            landmark_method, checkpoint, resume))
    massive_embedding_off(output, chunk_rows, storage$control_memory_limit,
        landmark_method = landmark_method,
        checkpoint = checkpoint, resume = resume)
    dispatch_tsne_resident(data, nn, settings, controls, extra,
        devices, landmarks)
}
