#' Fast UMAP from precomputed nearest neighbors
#'
#' @param indices Integer matrix of nearest-neighbor indices, one row per point,
#'   or a list containing `indices` and `distances`. Indices may be 1-based, as
#'   returned by R packages, or 0-based. If a self-neighbor first column is
#'   present it is removed automatically.
#' @param distances Numeric matrix matching `indices`. Leave as `NULL` when
#'   `indices` is a KNN list.
#' @param n_components Output dimensionality. The CPU backend supports positive
#'   dimensions, including three-dimensional embeddings. The current Metal and
#'   CUDA optimizers support only `2L`.
#' @param seed Integer random seed.
#' @param verbose Print progress from C++.
#' @param backend Execution backend: `"cpu"`, `"cuda"`, or `"metal"`.
#' @param graph_mode Graph weighting mode. `"fuzzy"` (the default) uses
#'   standard UMAP fuzzy graph weights. `"binary"` uses a symmetric
#'   unit-weight sensitivity graph.
#' @return A numeric matrix with `nrow(indices)` rows and `n_components`
#'   columns.
#' @details The public API intentionally keeps only the inputs that matter. The
#'   package chooses epochs, negative sampling, learning rate, spectral
#'   iterations, CPU thread count, and the UMAP repulsion weight internally
#'   using size-aware defaults.
#' @noRd
fast_knn_umap <- function(indices,
                            distances = NULL,
                            n_components = 2L,
                            seed = 42L,
                            verbose = FALSE,
                            backend = NULL,
                            n_threads = NULL,
                            graph_mode = c("fuzzy", "binary")) {
    fast_knn_umap_core(
        indices,
        distances,
        n_components = n_components,
        seed = seed,
        verbose = verbose,
        backend = backend,
        n_threads = n_threads,
        graph_mode = graph_mode
    )
}

dispatch_gpu_umap_core <- function(indices, distances, settings) {
    if (!is.null(distances)) {
        stop(
            "Do not pass `distances` with a GPU-resident KNN object.",
            call. = FALSE
        )
    }
    if (settings$backend != "cuda") {
        stop("GPU-resident KNN requires `backend = \"cuda\"`.",
            call. = FALSE
        )
    }
    fast_knn_umap_cuda_gpu_core(
        fastembedr_as_gpu_knn(indices),
        n_components = settings$n_components,
        seed = settings$seed,
        verbose = settings$verbose,
        n_threads = settings$n_threads,
        n_epochs = settings$n_epochs,
        config_override = settings$config_override,
        graph_mode = settings$graph_mode
    )
}

dispatch_host_umap_core <- function(indices, distances, settings) {
    knn <- coerce_knn_input(indices, distances)
    cfg <- configure_host_knn_umap(
        knn$indices, knn$distances, knn, settings$backend,
        settings$n_threads, settings$n_epochs,
        settings$config_override, settings$graph_mode,
        settings$seed
    )
    cfg$n_components <- as.integer(settings$n_components)
    if (cfg$backend %in% c("cuda", "metal")) {
        return(run_umap_gpu_host_knn(
            knn$indices, knn$distances, knn, cfg,
            settings$n_components, settings$graph_mode,
            settings$seed
        ))
    }
    run_umap_cpu_host_knn(
        knn$indices, knn$distances, knn, cfg,
        settings$n_components, settings$graph_mode,
        settings$seed, settings$verbose
    )
}

fast_knn_umap_core <- function(
    indices, distances = NULL, n_components = 2L,
    seed = 42L, verbose = FALSE, backend = NULL,
    n_threads = NULL, n_epochs = NULL, config_override = NULL,
    graph_mode = c("fuzzy", "binary")
) {
    prepared <- dispatch_prepared_umap_input(
        indices,
        distances,
        n_components,
        seed,
        verbose,
        backend,
        n_threads,
        n_epochs
    )
    if (isTRUE(prepared$handled)) {
        return(prepared$layout)
    }

    settings <- list(
        backend = resolve_embedding_backend(backend),
        graph_mode = match.arg(graph_mode),
        n_components = validate_n_components(n_components),
        seed = seed, verbose = verbose, n_threads = n_threads,
        n_epochs = n_epochs, config_override = config_override
    )
    if (fastembedr_is_gpu_knn(indices)) {
        return(dispatch_gpu_umap_core(indices, distances, settings))
    }
    dispatch_host_umap_core(indices, distances, settings)
}

validate_gpu_resident_umap <- function(gpu_knn, n_components) {
    if (n_components != 2L) {
        stop("Native CUDA UMAP supports only `n_components = 2`.",
            call. = FALSE
        )
    }
    info <- fastembedr_gpu_knn_info(gpu_knn)
    if (isTRUE(info$has_self)) {
        stop(
            "CUDA GPU-resident UMAP requires non-self KNN. ",
            "Provide a native non-self fastEmbedR GPU KNN object.",
            call. = FALSE
        )
    }
    info
}

gpu_resident_umap_metadata <- function(cfg, info, graph_mode, verbose) {
    cfg$input_had_self <- FALSE
    cfg$knn_col_start <- 0L
    cfg$knn_n_neighbors <- as.integer(info$k)
    cfg$knn_materialized <- FALSE
    cfg$knn_materialized_for_gpu <- FALSE
    cfg$knn_backend <- info$input_backend
    cfg$knn_distance_type <- info$distance_type
    cfg$knn_residency <- "cuda_device"
    cfg$graph_mode <- graph_mode
    cfg$graph_prep_backend <- paste0(
        "cuda_", graph_mode, "_union_device"
    )
    cfg$graph_storage <- "native_cuda_device_row_major"
    cfg$sgd_loop <- "cuda_graph_row_warp_fused"
    cfg$gpu_transfer_policy <-
        "gpu_knn_device_pointers_no_knn_host_copy"
    cfg$gpu_optimizer_mode <- "row_warp_atomic_tail"
    cfg$gpu_optimizer_update_rule <-
        "fused_attraction_repulsion_head_aggregated"
    cfg$gpu_optimizer_schedule <- "row_weight_epoch_schedule_device"
    cfg$gpu_edge_ordering <- "row_major_head_grouped"
    cfg$gpu_conflict_strategy <- "warp_aggregated_head_atomic_tail"
    cfg$gpu_negative_sampling <- "deterministic_on_device_per_edge_epoch"
    cfg$gpu_kernel_fusion <- "attraction_repulsion_coordinate_update"
    cfg$gpu_initial_backend <- "cuda"
    cfg$optimizer_backend <- "cuda"
    cfg$init_backend <- "cuda_fused_diffusion"
    cfg$init_backend_reason <- paste(
        "CUDA UMAP consumes native GPU KNN pointers and runs",
        "graph construction, initialization, and optimization on device."
    )
    cfg$gpu_umap_path <- paste0(
        "cuda_gpu_knn_", graph_mode, "_float32_row_warp"
    )
    cfg$gpu_initial_epochs <- as.integer(cfg$n_epochs)
    cfg$verbose <- isTRUE(verbose)
    cfg
}

configure_gpu_resident_umap <- function(info, n_components, n_threads,
                                        n_epochs, override, graph_mode,
                                        verbose) {
    cfg <- fast_knn_umap_config(
        n = info$n,
        k = info$k,
        backend = "cuda"
    )
    cfg$n_components <- as.integer(n_components)
    cfg$auto_parameter_backend <- "r_size_rule_gpu_resident_knn"
    cfg$auto_parameter_reason <- paste(
        "KNN distances stay on the CUDA device;",
        "host distance profiling is skipped."
    )
    cfg <- apply_umap_thread_override(cfg, n_threads)
    if (!is.null(n_epochs)) {
        cfg$n_epochs <- validate_epoch_count(n_epochs)
        cfg$preset <- "internal_epoch_override"
        cfg$epoch_source <- "internal_override"
    }
    cfg <- apply_fast_knn_umap_config_override(cfg, override)
    gpu_resident_umap_metadata(cfg, info, graph_mode, verbose)
}

run_gpu_resident_umap <- function(gpu_knn, info, cfg, seed,
                                    graph_mode) {
    if (!embedding_cuda_available_cpp()) {
        stop(
            "Native CUDA UMAP optimizer was requested, ",
            "but it is not available in this build.",
            call. = FALSE
        )
    }
    layout <- knn_umap_cuda_fused_gpu_cpp(
        gpu_knn,
        as.integer(info$k),
        as.integer(cfg$n_epochs),
        as.integer(cfg$negative_sample_rate),
        cfg$learning_rate,
        cfg$min_dist,
        cfg$repulsion_strength,
        as.integer(cfg$spectral_n_iter),
        as.integer(seed),
        0L,
        identical(graph_mode, "binary")
    )
    cfg$cuda_allocator <- attr(layout, "cuda_allocator")
    cfg$cuda_graph_capture <- attr(layout, "cuda_graph_capture")
    cfg$cuda_graph_scope <- attr(layout, "cuda_graph_scope")
    layout <- finalize_embedding_layout(layout, "UMAP", return_float32 = TRUE)
    attr(layout, "fastEmbedR_config") <- public_core_config(cfg)
    layout
}

fast_knn_umap_cuda_gpu_core <- function(
    gpu_knn, n_components = 2L, seed = 42L, verbose = FALSE,
    n_threads = NULL, n_epochs = NULL, config_override = NULL,
    graph_mode = c("fuzzy", "binary")
) {
    graph_mode <- match.arg(graph_mode)
    info <- validate_gpu_resident_umap(gpu_knn, n_components)
    cfg <- configure_gpu_resident_umap(
        info, n_components, n_threads, n_epochs,
        config_override, graph_mode, verbose
    )
    run_gpu_resident_umap(gpu_knn, info, cfg, seed, graph_mode)
}

#' Precompute reusable UMAP graph state from KNN
#'
#' `prepare_umap_knn()` builds the CSR graph once from a KNN object or index
#' and distance matrices. The result can be passed to [umap_knn()] repeatedly
#' with different seeds without rebuilding the graph or recomputing
#' `epochs_per_sample`.
#'
#' @inheritParams umap_knn
#' @return A prepared UMAP object containing the KNN, CSR graph, and resolved
#'   UMAP graph schedule.
#' @examples
#' x <- scale(as.matrix(iris[, 1:4]))
#' d <- as.matrix(stats::dist(x))
#' diag(d) <- Inf
#' k <- 15L
#' idx <- t(apply(d, 1L, order))[, seq_len(k), drop = FALSE]
#' dst <- matrix(d[cbind(rep(seq_len(nrow(d)), each = k), as.vector(t(idx)))],
#'     nrow = nrow(d), byrow = TRUE
#' )
#' prep <- prepare_umap_knn(idx, dst)
#' y1 <- umap_knn(prep, seed = 1)
#' y2 <- umap_knn(prep, seed = 2)
#' @name prepare_umap_knn
NULL

prepared_umap_auto_policy <- function(knn, cfg) {
    distances <- if (knn$col_start != 0L ||
        knn$n_neighbors != ncol(knn$distances)) {
        materialize_knn_range(
            knn$indices,
            knn$distances,
            knn$col_start,
            knn$n_neighbors
        )$distances
    } else {
        knn$distances
    }
    if (is_float32_matrix(distances)) {
        return(list(
            error = "float32 KNN auto-policy uses default parameters"
        ))
    }
    attempt <- capture_error(umap_auto_parameters_cpp(
        distances,
        as.integer(knn$n_neighbors),
        as.character(cfg$backend)
    ))
    if (is.null(attempt$value)) {
        list(error = attempt$error)
    } else {
        attempt$value
    }
}

apply_prepared_umap_policy <- function(cfg, policy) {
    if (!is.null(policy$error)) {
        cfg$auto_parameter_backend <- "r_size_rule_fallback"
        cfg$auto_parameter_error <- policy$error
        return(cfg)
    }
    fields <- c(
        "n_epochs", "min_dist", "negative_sample_rate",
        "learning_rate", "spectral_n_iter", "init_scale"
    )
    for (field in fields) {
        cfg[[field]] <- policy[[field]]
    }
    cfg$n_epochs <- as.integer(cfg$n_epochs)
    cfg$negative_sample_rate <- as.integer(cfg$negative_sample_rate)
    cfg$spectral_n_iter <- as.integer(cfg$spectral_n_iter)
    cfg$auto_parameter_backend <- "cpp_knn_distance_profile"
    cfg$auto_parameter_rule <- as.character(policy$rule)
    cfg$knn_distance_cv <- as.numeric(policy$knn_distance_cv)
    cfg$knn_distance_ratio_30_15 <- as.numeric(
        policy$knn_distance_ratio_30_15
    )
    cfg$knn_distance_ratio_50_15 <- as.numeric(
        policy$knn_distance_ratio_50_15
    )
    cfg
}

prepared_umap_metadata <- function(cfg, knn, graph_mode) {
    cfg$input_had_self <- isTRUE(knn$has_self)
    cfg$knn_col_start <- as.integer(knn$col_start)
    cfg$knn_n_neighbors <- as.integer(knn$n_neighbors)
    cfg$knn_materialized <- isTRUE(knn$materialized)
    cfg$knn_backend <- knn$input_backend
    cfg$graph_mode <- graph_mode
    cfg$sgd_loop <- "csr_float32_contiguous_inplace"
    apply_umap_connectivity_spectral_rule(
        cfg,
        knn$indices,
        col_start = knn$col_start,
        n_neighbors = knn$n_neighbors
    )
}

build_prepared_umap_graph <- function(knn, cfg, graph_mode) {
    graph <- umap_build_csr_graph(
        knn$indices,
        knn$distances,
        as.integer(knn$col_start),
        as.integer(knn$n_neighbors),
        as.integer(knn$n_neighbors),
        as.integer(cfg$n_threads),
        graph_mode = graph_mode
    )
    cfg$graph_prep_backend <- paste0("cpu_", graph_mode, "_csr")
    cfg$graph_storage <- cfg$graph_prep_backend
    cfg$graph_nnz <- as.integer(graph$nnz)
    cfg$graph_max_weight <- as.numeric(graph$max_weight)
    cfg$graph_cuda_like_width <- graph$cuda_like_width
    cfg$graph_builder <- graph$graph_builder
    cfg$prepared <- TRUE
    cfg$prepared_reuse <- "csr_graph_epochs_per_sample"
    list(graph = graph, config = cfg)
}

#' @rdname prepare_umap_knn
#' @export
prepare_umap_knn <- function(indices,
                                distances = NULL,
                                backend = NULL,
                                n.cores = NULL,
                                graph_mode = c("fuzzy", "binary")) {
    backend <- resolve_embedding_backend(backend)
    graph_mode <- match.arg(graph_mode)
    knn <- coerce_knn_input(indices, distances)
    cfg <- fast_knn_umap_config(
        n = nrow(knn$indices),
        k = knn$n_neighbors,
        backend = backend
    )
    policy <- tryCatch(
        prepared_umap_auto_policy(knn, cfg),
        error = function(e) list(error = conditionMessage(e))
    )
    cfg <- apply_prepared_umap_policy(cfg, policy)
    cfg <- apply_umap_thread_override(cfg, n.cores)
    cfg <- prepared_umap_metadata(cfg, knn, graph_mode)
    built <- build_prepared_umap_graph(knn, cfg, graph_mode)
    out <- list(
        knn = knn,
        graph = built$graph,
        config = built$config
    )
    class(out) <- c("fastEmbedR_umap_prepared", "list")
    out
}

configure_prepared_umap <- function(
    prepared,
    backend,
    n_components,
    n_threads,
    n_epochs
) {
    if (n_components != 2L) {
        stop(
            "Prepared UMAP reuse supports only `n_components = 2`.",
            call. = FALSE
        )
    }
    cfg <- prepared$config
    cfg$n_components <- as.integer(n_components)
    cfg$backend <- backend
    cfg$optimizer_backend <- backend
    cfg$prepared_reuse_hit <- TRUE
    cfg$sgd_loop <- "csr_float32_contiguous_inplace"
    cfg <- apply_umap_thread_override(cfg, n_threads)
    if (!is.null(n_epochs)) {
        cfg$n_epochs <- validate_epoch_count(n_epochs)
        cfg$preset <- "internal_epoch_override"
        cfg$epoch_source <- "internal_override"
    }
    cfg
}

validate_reusable_umap_init <- function(init, n, n_components) {
    if (!identical(dim(init), c(n, as.integer(n_components)))) {
        stop(
            "The reusable UMAP initialization has incompatible dimensions.",
            call. = FALSE
        )
    }
    finite <- if (is_float32_matrix(init)) {
        float32_all_finite_cpp(init)
    } else {
        all(is.finite(init))
    }
    if (!isTRUE(finite)) {
        stop(
            "The reusable UMAP initialization contains non-finite values.",
            call. = FALSE
        )
    }
    init
}

compute_prepared_umap_init <- function(prepared, cfg, seed) {
    knn <- prepared$knn
    spectral <- cfg$backend %in% c("cuda", "metal") &&
        cfg$graph_mode == "fuzzy"
    init <- if (spectral) {
        distances <- if (is_float32_matrix(knn$distances)) {
            matrix(
                as.numeric(knn$distances),
                nrow = nrow(knn$indices),
                ncol = ncol(knn$indices)
            )
        } else {
            knn$distances
        }
        spectral_knn_init(
            knn$indices, distances,
            n_components = 2L,
            min_dist = cfg$min_dist,
            spectral_n_iter = cfg$spectral_n_iter,
            seed = seed, backend = "cpu",
            n_threads = cfg$n_threads
        )
    } else {
        umap_init_from_csr_graph(
            prepared$graph,
            n_components = 2L,
            cfg = cfg,
            seed = seed,
            verbose = FALSE
        )
    }
    if (spectral) {
        init <- scale_embedding_sdev_r(init, cfg$init_scale)
    }
    list(value = init, backend = attr(init, "backend") %||% "cpu")
}

prepare_reusable_umap_init <- function(prepared, cfg, n_components, seed) {
    if (!is.null(prepared$initialization)) {
        init <- validate_reusable_umap_init(
            prepared$initialization,
            nrow(prepared$knn$indices),
            n_components
        )
        backend <- prepared$initialization_parameters$init_backend %||%
            attr(init, "backend") %||%
            "external"
        return(list(value = init, backend = backend, reused = TRUE))
    }
    result <- compute_prepared_umap_init(prepared, cfg, seed)
    result$reused <- FALSE
    result
}

optimize_prepared_umap_metal <- function(graph, init, cfg, seed) {
    if (!embedding_metal_available_cpp()) {
        stop("Native Metal UMAP optimizer is unavailable in this build.",
            call. = FALSE
        )
    }
    knn_embed_metal_csr_cpp(
        graph$offsets,
        graph$neighbors,
        graph$weights,
        init,
        as.integer(cfg$n_epochs),
        as.integer(cfg$negative_sample_rate),
        cfg$learning_rate,
        cfg$min_dist,
        as.numeric(graph$max_weight),
        cfg$repulsion_strength,
        as.integer(seed),
        1L
    )
}

optimize_prepared_umap_cuda <- function(graph, init, cfg, seed) {
    if (!embedding_cuda_available_cpp()) {
        stop("Native CUDA UMAP optimizer is unavailable in this build.",
            call. = FALSE
        )
    }
    umap_cuda_optimize_csr_cpp(
        graph$offsets,
        graph$neighbors,
        graph$weights,
        graph$epochs_per_sample,
        init,
        as.integer(cfg$n_epochs),
        as.integer(cfg$negative_sample_rate),
        cfg$learning_rate,
        cfg$min_dist,
        cfg$repulsion_strength,
        as.integer(seed),
        0L
    )
}

optimize_prepared_umap_cpu <- function(graph, init, cfg, seed, verbose) {
    fast_knn_umap_csr_init_cpp(
        graph$offsets,
        graph$neighbors,
        graph$weights,
        init,
        as.integer(cfg$n_epochs),
        cfg$min_dist,
        as.integer(cfg$negative_sample_rate),
        cfg$learning_rate,
        cfg$repulsion_strength,
        as.integer(cfg$n_threads),
        as.integer(seed),
        isTRUE(verbose)
    )
}

run_prepared_umap_optimizer <- function(graph, init, cfg, seed, verbose) {
    if (cfg$backend == "metal") {
        return(optimize_prepared_umap_metal(graph, init, cfg, seed))
    }
    if (cfg$backend == "cuda") {
        return(optimize_prepared_umap_cuda(graph, init, cfg, seed))
    }
    optimize_prepared_umap_cpu(graph, init, cfg, seed, verbose)
}

finalize_prepared_umap <- function(layout, graph, init, cfg) {
    cfg$initialization_reuse_hit <- init$reused
    cfg$init_backend <- init$backend
    cfg$optimizer_mode <- "csr_epoch_schedule"
    cfg$optimizer_schedule <- paste0(
        "epochs_per_sample_",
        cfg$graph_mode,
        "_csr"
    )
    if (cfg$backend == "metal") {
        cfg$metal_graph_input <- attr(layout, "metal_graph_input")
        cfg$metal_csr_width <- attr(layout, "metal_csr_width")
        cfg$metal_truncated_edges <- attr(layout, "metal_truncated_edges")
    }
    layout <- finalize_embedding_layout(
        layout,
        "UMAP",
        return_float32 = is_float32_matrix(graph$weights)
    )
    attr(layout, "fastEmbedR_config") <- public_core_config(cfg)
    layout
}

fast_knn_umap_prepared_core <- function(
    prepared, n_components = 2L, seed = 42L, verbose = FALSE,
    backend = NULL, n_threads = NULL, n_epochs = NULL
) {
    backend <- resolve_embedding_backend(backend)
    n_components <- validate_n_components(n_components)
    cfg <- configure_prepared_umap(
        prepared, backend, n_components, n_threads, n_epochs
    )
    init <- prepare_reusable_umap_init(
        prepared, cfg, n_components, seed
    )
    layout <- run_prepared_umap_optimizer(
        prepared$graph, init$value, cfg, seed, verbose
    )
    finalize_prepared_umap(
        layout, prepared$graph, init, cfg
    )
}

umap_build_csr_graph <- function(indices,
                                    distances,
                                    col_start,
                                    n_cols,
                                    edge_budget,
                                    n_threads,
                                    graph_mode = c("fuzzy", "binary")) {
    graph_mode <- match.arg(graph_mode)
    distance_is_float32 <- is_float32_matrix(distances)
    if (identical(graph_mode, "binary")) {
        graph <- umap_graph_csr_binary_union_cpp(
            indices,
            as.integer(col_start),
            as.integer(n_cols),
            as.integer(n_threads)
        )
        graph$graph_mode <- "binary"
        return(umap_graph_keep_float32(graph, TRUE))
    }
    graph <- umap_graph_csr_float_cpp(
        indices,
        distances,
        as.integer(col_start),
        as.integer(n_cols),
        as.integer(edge_budget),
        as.integer(n_threads)
    )
    graph$graph_mode <- "fuzzy"
    umap_graph_keep_float32(graph, TRUE)
}

umap_graph_keep_float32 <- function(graph, use_float32) {
    if (!isTRUE(use_float32) || !requireNamespace("float", quietly = TRUE)) {
        return(graph)
    }
    if (!is_float32_matrix(graph$weights)) {
        graph$weights <- float::fl(graph$weights)
    }
    if (!is.null(graph$epochs_per_sample) && !is_float32_matrix(
        graph$epochs_per_sample
    )) {
        graph$epochs_per_sample <- float::fl(graph$epochs_per_sample)
    }
    graph$weight_type <- "float32"
    graph$epoch_schedule_type <- if (!is.null(
        graph$epochs_per_sample
    )) {
        "float32"
    } else {
        NA_character_
    }
    graph
}

umap_init_from_csr_graph <- function(graph,
                                        n_components,
                                        cfg,
                                        seed,
                                        verbose = FALSE) {
    init <- fast_knn_umap_csr_cpp(
        graph$offsets,
        graph$neighbors,
        graph$weights,
        as.integer(n_components),
        0L,
        cfg$min_dist,
        0L,
        1,
        cfg$repulsion_strength,
        as.integer(cfg$spectral_n_iter),
        as.integer(cfg$n_threads),
        cfg$init_scale,
        as.integer(seed),
        isTRUE(verbose)
    )
    graph_mode <- graph$graph_mode %||% cfg$graph_mode %||% "fuzzy"
    attr(init, "backend") <- paste0("cpu_", graph_mode, "_csr")
    init
}

umap_override_source <- function(override) {
    if (is.null(override$tuning_source)) {
        "internal_override"
    } else {
        paste0(override$tuning_source, "_override")
    }
}

apply_umap_epoch_override <- function(cfg, override) {
    if (is.null(override$n_epochs)) {
        return(cfg)
    }
    cfg$n_epochs <- validate_epoch_count(override$n_epochs)
    cfg$epoch_source <- umap_override_source(override)
    cfg
}

apply_umap_spectral_override <- function(cfg, override) {
    if (is.null(override$spectral_n_iter)) {
        return(cfg)
    }
    value <- as.integer(override$spectral_n_iter)
    invalid <- length(value) != 1L ||
        is.na(value) ||
        !is.finite(value) ||
        value < 1L
    if (invalid) {
        stop(
            "`spectral_n_iter` override must be a positive integer.",
            call. = FALSE
        )
    }
    cfg$spectral_n_iter <- if (isTRUE(
        cfg$spectral_connectivity_checked
    )) {
        as.integer(max(cfg$spectral_n_iter, value))
    } else {
        value
    }
    cfg$spectral_rule <- umap_override_source(override)
    cfg
}

apply_umap_init_scale_override <- function(cfg, override) {
    if (is.null(override$init_scale)) {
        return(cfg)
    }
    value <- as.numeric(override$init_scale)
    invalid <- length(value) != 1L ||
        (!is.na(value) && (!is.finite(value) || value <= 0))
    if (invalid) {
        stop(
            "`init_scale` override must be NA or positive and finite.",
            call. = FALSE
        )
    }
    cfg$init_scale <- value
    cfg$init_scale_source <- umap_override_source(override)
    cfg
}

apply_umap_learning_rate_override <- function(cfg, override) {
    if (is.null(override$learning_rate)) {
        return(cfg)
    }
    value <- as.numeric(override$learning_rate)
    if (length(value) != 1L || !is.finite(value) || value <= 0) {
        stop(
            "`learning_rate` override must be positive and finite.",
            call. = FALSE
        )
    }
    cfg$learning_rate <- value
    cfg$learning_rate_source <- umap_override_source(override)
    cfg
}

apply_umap_negative_rate_override <- function(cfg, override) {
    if (is.null(override$negative_sample_rate)) {
        return(cfg)
    }
    value <- as.integer(override$negative_sample_rate)
    invalid <- length(value) != 1L ||
        is.na(value) ||
        !is.finite(value) ||
        value < 0L
    if (invalid) {
        stop(
            "`negative_sample_rate` override must be non-negative.",
            call. = FALSE
        )
    }
    cfg$negative_sample_rate <- value
    cfg$negative_sample_rate_source <- umap_override_source(override)
    cfg
}

apply_umap_repulsion_override <- function(cfg, override) {
    if (is.null(override$repulsion_strength)) {
        return(cfg)
    }
    value <- as.numeric(override$repulsion_strength)
    if (length(value) != 1L || !is.finite(value) || value <= 0) {
        stop(
            "`repulsion_strength` override must be positive and finite.",
            call. = FALSE
        )
    }
    cfg$repulsion_strength <- value
    cfg$repulsion_strength_source <- umap_override_source(override)
    cfg
}

apply_umap_pilot_metadata <- function(cfg, override) {
    fields <- list(
        pilot_sample_n = as.integer,
        pilot_score = as.numeric,
        pilot_cache_key = as.character,
        pilot_cache_hit = isTRUE
    )
    for (name in names(fields)) {
        if (!is.null(override[[name]])) {
            cfg[[name]] <- fields[[name]](override[[name]])
        }
    }
    cfg
}

apply_fast_knn_umap_config_override <- function(cfg, override) {
    if (is.null(override) || !length(override)) {
        cfg$config_override <- cfg$config_override %||% FALSE
        return(cfg)
    }
    cfg <- apply_umap_epoch_override(cfg, override)
    cfg <- apply_umap_spectral_override(cfg, override)
    cfg <- apply_umap_init_scale_override(cfg, override)
    cfg <- apply_umap_learning_rate_override(cfg, override)
    cfg <- apply_umap_negative_rate_override(cfg, override)
    cfg <- apply_umap_repulsion_override(cfg, override)
    cfg$config_override <- TRUE
    cfg$config_override_source <- override$tuning_source %||% "internal"
    apply_umap_pilot_metadata(cfg, override)
}

fast_knn_umap_should_auto_pilot <- function(cfg,
                                            indices,
                                            config_override = NULL,
                                            n_epochs = NULL) {
    if (!is.null(config_override) || !is.null(n_epochs)) {
        return(FALSE)
    }
    if (!identical(cfg$backend, "cpu")) {
        return(FALSE)
    }
    if (!cfg$epoch_source %in% c("clean_large_default")) {
        return(FALSE)
    }
    if (!isTRUE(getOption("fastEmbedR.knn_pilot", FALSE))) {
        return(FALSE)
    }
    n <- nrow(indices)
    k <- ncol(indices)
    n >= fast_knn_umap_auto_pilot_min_n() && k >= 10L
}

fast_knn_umap_auto_pilot_skip_reason <- function(cfg,
                                                    indices,
                                                    config_override = NULL,
                                                    n_epochs = NULL) {
    if (!is.null(config_override)) {
        return("explicit config override supplied")
    }
    if (!is.null(n_epochs)) {
        return("explicit epoch override supplied")
    }
    if (!identical(cfg$backend, "cpu")) {
        return("auto KNN pilot currently runs only on the CPU optimizer path")
    }
    if (!cfg$epoch_source %in% c("clean_large_default")) {
        return("default is not the clean atomic UMAP path")
    }
    if (!isTRUE(getOption("fastEmbedR.knn_pilot", FALSE))) {
        return(paste(
            "disabled by default; set option fastEmbedR.knn_pilot = TRUE",
            "for internal benchmarking"
        ))
    }
    if (nrow(indices) < fast_knn_umap_auto_pilot_min_n()) {
        return("below auto KNN pilot size threshold")
    }
    if (ncol(indices) < 10L) {
        return(
            "too few supplied neighbors for a stable pilot"
        )
    }
    "not selected"
}

fast_knn_umap_auto_pilot_min_n <- function() {
    value <- getOption("fastEmbedR.knn_pilot_min_n", 20000L)
    value <- integer_scalar(value)
    if (length(value) != 1L || is.na(value) || !is.finite(value) ||
        value < 50L) {
        return(20000L)
    }
    value
}

fast_knn_umap_auto_pilot_max_n <- function() {
    value <- getOption("fastEmbedR.knn_pilot_max_n", 2500L)
    value <- integer_scalar(value)
    if (length(value) != 1L || is.na(value) || !is.finite(value) ||
        value < 50L) {
        return(2500L)
    }
    value
}

fast_knn_umap_auto_pilot_max_configs <- function() {
    value <- getOption("fastEmbedR.knn_pilot_max_configs", 4L)
    value <- integer_scalar(value)
    if (length(value) != 1L || is.na(value) || !is.finite(value) ||
        value < 1L) {
        return(4L)
    }
    value
}

fast_knn_umap_auto_pilot_use_cache <- function() {
    !isFALSE(getOption("fastEmbedR.knn_pilot_use_cache", TRUE))
}

scale_embedding_sdev_r <- function(embedding, target_sdev) {
    target_sdev <- as.numeric(target_sdev)
    invalid_sdev <- length(target_sdev) != 1L ||
        is.na(target_sdev) ||
        !is.finite(target_sdev) ||
        target_sdev <= 0
    if (invalid_sdev) {
        return(embedding)
    }
    if (nrow(embedding) < 2L) {
        return(embedding)
    }
    attr_backend <- attr(embedding, "backend")
    center <- colMeans(embedding)
    embedding <- sweep(embedding, 2L, center, "-")
    scale <- sqrt(colSums(embedding * embedding) / max(1L, nrow(
        embedding
    ) - 1L))
    scale[!is.finite(scale) | scale == 0] <- 1
    embedding <- sweep(embedding, 2L, scale / target_sdev, "/")
    attr(embedding, "backend") <- attr_backend
    embedding
}

umap_size_defaults <- function(n, k) {
    if (n >= 10000L) {
        return(list(
            n_epochs = 200L,
            spectral_n_iter = if (k <= 15L) 30L else 20L,
            preset = "clean_atomic_large",
            epoch_source = "clean_large_default",
            spectral_rule = "adaptive_large_k"
        ))
    }
    list(
        n_epochs = 500L,
        spectral_n_iter = if (n >= 500L) 60L else 50L,
        preset = "clean_atomic_standard",
        epoch_source = "clean_size_rule",
        spectral_rule = "size_rule"
    )
}

umap_default_threads <- function(n, k) {
    cores <- parallel::detectCores(logical = FALSE)
    if (length(cores) != 1L ||
        is.na(cores) ||
        !is.finite(cores)) {
        cores <- 1L
    }
    cap <- if (n >= 500L && k >= 15L) {
        4L
    } else if (n >= 200L && k >= 15L) {
        3L
    } else {
        1L
    }
    max(1L, min(cap, as.integer(cores)))
}

umap_graph_config <- function(k) {
    scales <- fast_knn_umap_graph_scales(k)
    mid_near <- fast_knn_umap_mid_near_count(k)
    prune <- fast_knn_umap_prune_fraction(k)
    list(
        graph_storage = if (length(scales) > 1L || mid_near > 0L) {
            "native_csr_float_multiscale_midnear"
        } else {
            "native_csr_float_direct"
        },
        graph_scales = paste(scales, collapse = ","),
        graph_mid_near_edges_per_point = as.integer(mid_near),
        graph_mid_near_weight = fast_knn_umap_mid_near_weight(k),
        graph_pruning = if (prune > 0) {
            "adaptive_weight_with_connectivity_rescue"
        } else {
            "none"
        },
        graph_prune_fraction = as.numeric(prune),
        graph_prune_min_degree = as.integer(
            fast_knn_umap_prune_min_degree(k)
        )
    )
}

fast_knn_umap_config <- function(n, k, backend) {
    backend <- resolve_backend_request(backend, need_embedding = TRUE)
    defaults <- umap_size_defaults(n, k)
    base <- list(
        method = "umap",
        preset = defaults$preset,
        optimizer = defaults$preset,
        epoch_source = defaults$epoch_source,
        n_epochs = as.integer(defaults$n_epochs),
        min_dist = 0.01,
        negative_sample_rate = 5L,
        repulsion_strength = 1,
        learning_rate = 1,
        spectral_n_iter = as.integer(defaults$spectral_n_iter),
        spectral_rule = defaults$spectral_rule,
        init_scale = NA_real_,
        graph_mode = "native_csr_graph",
        optimizer_math = "clean_atomic_edge_sampler",
        n = as.integer(n),
        k = as.integer(k),
        n_threads = as.integer(umap_default_threads(n, k)),
        backend = if (backend == "auto") "cpu" else backend
    )
    c(base, umap_graph_config(k))
}

apply_fast_knn_umap_distance_profile_rule <- function(cfg, distances) {
    if (is.null(cfg$n) || cfg$n < 50000L || is.null(cfg$k) ||
        cfg$k < 30L) {
        return(cfg)
    }

    profile <- fast_knn_umap_distance_profile(distances)
    cfg$knn_distance_cv <- profile$cv
    cfg$knn_distance_ratio_50_15 <- profile$ratio_50_15
    cfg$knn_distance_ratio_30_15 <- profile$ratio_30_15
    cfg$knn_distance_profile_rule <- "large_default"

    if (is.finite(profile$ratio_50_15) && is.finite(profile$cv) &&
        profile$ratio_50_15 >= 1.25 && profile$cv >= 1.0) {
        cfg$n_epochs <- as.integer(max(cfg$n_epochs, 200L))
        cfg$min_dist <- 0.1
        cfg$init_scale <- 5
        cfg$learning_rate <- 1.25
        cfg$preset <- "large_wide_shell_balanced"
        cfg$epoch_source <- "distance_profile_wide_shell"
        cfg$init_scale_source <- "distance_profile_wide_shell"
        cfg$learning_rate_source <- "distance_profile_wide_shell"
        cfg$min_dist_source <- "distance_profile_wide_shell"
        cfg$knn_distance_profile_rule <- "wide_shell_balanced_quality_speed"
    } else if (is.finite(profile$cv) && profile$cv >= 0.60) {
        cfg$n_epochs <- as.integer(max(cfg$n_epochs, 300L))
        cfg$preset <- "large_high_variability_fidelity"
        cfg$epoch_source <- "distance_profile_high_variability"
        cfg$knn_distance_profile_rule <- "high_variability_more_epochs"
    }
    cfg
}

umap_distance_profile_matrix <- function(distances) {
    if (!is_float32_matrix(distances)) {
        out <- as.matrix(distances)
        if (!identical(typeof(out), "double")) {
            storage.mode(out) <- "double"
        }
        return(out)
    }
    cols <- unique(pmin(c(15L, 30L, 50L), ncol(distances)))
    matrix(
        as.numeric(distances[, cols, drop = FALSE]),
        nrow = nrow(distances),
        ncol = length(cols)
    )
}

umap_distance_rank <- function(distances, rank) {
    column <- min(as.integer(rank), ncol(distances))
    as.numeric(distances[, column, drop = TRUE])
}

umap_distance_ratio <- function(numerator, denominator) {
    denominator <- stats::median(
        denominator[is.finite(denominator)]
    )
    numerator <- stats::median(
        numerator[is.finite(numerator)]
    )
    if (!is.finite(denominator) || denominator <= 0) {
        return(NA_real_)
    }
    numerator / denominator
}

fast_knn_umap_distance_profile <- function(distances) {
    sampled <- umap_distance_profile_matrix(distances)
    finite <- is.finite(sampled)
    if (!any(finite)) {
        return(list(
            cv = NA_real_,
            ratio_50_15 = NA_real_,
            ratio_30_15 = NA_real_
        ))
    }
    mean_distance <- mean(sampled[finite])
    cv <- if (is.finite(mean_distance) && mean_distance > 0) {
        stats::sd(sampled[finite]) / mean_distance
    } else {
        NA_real_
    }
    d15 <- umap_distance_rank(distances, 15L)
    list(
        cv = cv,
        ratio_50_15 = umap_distance_ratio(
            umap_distance_rank(distances, 50L),
            d15
        ),
        ratio_30_15 = umap_distance_ratio(
            umap_distance_rank(distances, 30L),
            d15
        )
    )
}

fast_knn_umap_graph_scales <- function(k) {
    k <- as.integer(k)
    if (length(k) != 1L || is.na(k) || k < 1L) {
        return(1L)
    }
    k
}

fast_knn_umap_mid_near_count <- function(k) {
    0L
}

fast_knn_umap_mid_near_weight <- function(k) {
    0
}

fast_knn_umap_prune_fraction <- function(k) {
    0
}

fast_knn_umap_prune_min_degree <- function(k) {
    k <- as.integer(k)
    if (length(k) != 1L || is.na(k) || k < 1L) {
        return(1L)
    }
    if (k < 30L) {
        return(as.integer(max(2L, k)))
    }
    if (k < 50L) {
        return(15L)
    }
    if (k < 150L) {
        return(20L)
    }
    24L
}

compute_umap_connectivity <- function(indices, col_start, n_neighbors) {
    tryCatch(
        knn_connectivity_range_cpp(indices, as.integer(col_start), as.integer(
            n_neighbors
        )),
        error = function(e) {
            structure(
                list(),
                error = conditionMessage(e)
            )
        }
    )
}

record_umap_connectivity <- function(cfg, stats) {
    error <- attr(stats, "error")
    if (!is.null(error)) {
        cfg$spectral_connectivity_checked <- FALSE
        cfg$spectral_connectivity_error <- error
        return(cfg)
    }
    cfg$spectral_connectivity_checked <- TRUE
    cfg$graph_connected <- isTRUE(stats$connected)
    cfg$graph_component_count <- as.integer(stats$component_count)
    cfg$graph_largest_component_fraction <- as.numeric(
        stats$largest_component_fraction
    )
    cfg$graph_largest_component_size <- as.integer(stats$largest_component_size)
    cfg$graph_singleton_count <- as.integer(stats$singleton_count)
    cfg$graph_invalid_edge_count <- as.integer(stats$invalid_edge_count)
    cfg
}

select_umap_spectral_iterations <- function(cfg, n, k) {
    base_iter <- if (k <= 15L) 30L else 20L
    selected_iter <- base_iter
    reason <- "connected_graph"
    component_limit <- max(2L, as.integer(ceiling(n / 10000)))
    many_components <- cfg$graph_component_count > component_limit
    if (cfg$graph_invalid_edge_count > 0L) {
        selected_iter <- max(selected_iter, 25L)
        reason <- "invalid_knn_edges"
    }
    if (cfg$graph_largest_component_fraction < 0.98 || many_components) {
        selected_iter <- max(selected_iter, 30L)
        reason <- "fragmented_graph"
    } else if (!isTRUE(cfg$graph_connected) ||
        cfg$graph_largest_component_fraction < 0.995) {
        selected_iter <- max(selected_iter, 25L)
        reason <- "mildly_disconnected_graph"
    }

    cfg$spectral_base_n_iter <- as.integer(base_iter)
    cfg$spectral_n_iter <- as.integer(selected_iter)
    cfg$spectral_rule <- "connectivity_adaptive_large"
    cfg$spectral_connectivity_reason <- reason
    cfg
}

apply_umap_connectivity_spectral_rule <- function(
    cfg, indices, col_start = 0L,
    n_neighbors = ncol(indices) - col_start
) {
    n <- nrow(indices)
    if (n < 10000L) {
        cfg$spectral_connectivity_checked <- FALSE
        return(cfg)
    }
    stats <- compute_umap_connectivity(
        indices,
        col_start,
        n_neighbors
    )
    cfg <- record_umap_connectivity(cfg, stats)
    if (!isTRUE(cfg$spectral_connectivity_checked)) {
        return(cfg)
    }
    select_umap_spectral_iterations(
        cfg,
        n,
        as.integer(n_neighbors)
    )
}

prepare_spectral_knn_input <- function(
    indices,
    distances,
    col_start,
    n_neighbors
) {
    if (!is.matrix(indices)) indices <- as.matrix(indices)
    if (!is.matrix(distances)) distances <- as.matrix(distances)
    if (!is.integer(indices)) storage.mode(indices) <- "integer"
    if (!identical(typeof(distances), "double")) {
        storage.mode(distances) <- "double"
    }
    col_start <- as.integer(col_start)
    n_neighbors <- n_neighbors %||%
        (ncol(indices) - col_start)
    list(
        indices = indices,
        distances = distances,
        col_start = col_start,
        n_neighbors = as.integer(n_neighbors)
    )
}

materialize_spectral_knn <- function(input) {
    complete <- input$col_start == 0L &&
        input$n_neighbors == ncol(input$indices)
    if (complete) {
        return(input)
    }
    knn <- materialize_knn_range(
        input$indices,
        input$distances,
        input$col_start,
        input$n_neighbors
    )
    list(
        indices = knn$indices,
        distances = knn$distances,
        col_start = 0L,
        n_neighbors = ncol(knn$indices)
    )
}

spectral_umap_threads <- function(n_threads) {
    if (is.null(n_threads)) {
        n_threads <- parallel::detectCores(logical = FALSE)
    }
    n_threads <- as.integer(n_threads)
    if (length(n_threads) != 1L ||
        is.na(n_threads) ||
        !is.finite(n_threads) ||
        n_threads < 1L) {
        n_threads <- 1L
    }
    max(1L, min(4L, n_threads))
}

run_spectral_knn_gpu <- function(input, n_components, n_iter,
                                    seed, backend) {
    input <- materialize_spectral_knn(input)
    available <- if (backend == "cuda") {
        embedding_cuda_available_cpp()
    } else {
        embedding_metal_available_cpp()
    }
    if (!available) {
        stop(backend, " spectral initialization is unavailable.",
            call. = FALSE
        )
    }
    if (as.integer(n_components) != 2L) {
        stop(backend, " spectral initialization supports only 2D.",
            call. = FALSE
        )
    }
    fun <- if (backend == "cuda") {
        spectral_knn_init_cuda_cpp
    } else {
        spectral_knn_init_metal_cpp
    }
    out <- fun(
        input$indices,
        input$distances,
        as.integer(n_components),
        as.integer(n_iter),
        as.integer(seed)
    )
    attr(out, "backend") <- backend
    out
}

run_spectral_knn_cpu <- function(input, n_components, min_dist,
                                    n_iter, n_threads, seed) {
    out <- fast_knn_umap_range_cpp(
        input$indices,
        input$distances,
        input$col_start,
        input$n_neighbors,
        as.integer(n_components),
        0L,
        min_dist,
        0L,
        1,
        1.0,
        as.integer(n_iter),
        as.integer(n_threads),
        NA_real_,
        as.integer(seed),
        FALSE
    )
    attr(out, "backend") <- "cpu"
    out
}

spectral_knn_init <- function(
    indices, distances, n_components = 2L, min_dist = 0.1,
    spectral_n_iter = 50L, seed = 42L, backend = "cpu",
    n_threads = NULL, col_start = 0L, n_neighbors = NULL
) {
    input <- prepare_spectral_knn_input(
        indices,
        distances,
        col_start,
        n_neighbors
    )
    if (backend %in% c("cuda", "metal")) {
        return(run_spectral_knn_gpu(
            input,
            n_components,
            spectral_n_iter,
            seed,
            backend
        ))
    }
    run_spectral_knn_cpu(
        input,
        n_components,
        min_dist,
        spectral_n_iter,
        spectral_umap_threads(n_threads),
        seed
    )
}

#' Run UMAP from precomputed nearest neighbors
#'
#' @param indices Integer matrix of nearest-neighbor indices, one row per
#'   observation, or a KNN object containing `indices` and `distances`.
#'   One-based and zero-based indices are accepted. A self-neighbor first
#'   column is removed automatically.
#' @param distances Numeric matrix matching `indices`. Leave as `NULL` when
#'   `indices` is a KNN object.
#' @param n_components Output dimensionality. The CPU backend supports positive
#'   dimensions, including three-dimensional embeddings. The current Metal and
#'   CUDA optimizers support only `2L`.
#' @param seed Integer random seed.
#' @param verbose Print native optimizer progress.
#' @param backend Execution backend: `"cpu"`, `"cuda"`, or `"metal"`.
#' @param n.cores Requested CPU core count. CPU UMAP currently uses at most
#'   four workers and records both requested and effective values. `NULL`
#'   selects a size-aware default from one to four workers. Native GPU
#'   backends ignore this argument.
#' @param graph_mode Graph weighting mode. `"fuzzy"` (the default) uses
#'   standard UMAP fuzzy graph weights. `"binary"` uses a symmetric
#'   unit-weight sensitivity graph.
#' @details
#' This KNN-input function exposes the same deliberately constrained optimizer
#' policy as [umap()]. It reuses the supplied neighbors but still selects
#' epochs, minimum distance, spread, learning rate, repulsion strength,
#' negative-sample rate, spectral iterations, and backend update mode
#' internally. The resolved configuration is attached as
#' `attr(layout, "fastEmbedR_config")`. Pass an object from [umap_init()] to
#' reuse both the prepared graph and package-native initialization. This API is
#' intended for fixed-boundary comparisons and repeated seeds, not arbitrary
#' UMAP hyperparameter sweeps.
#' @return An embedding matrix with `nrow(indices)` rows and `n_components`
#'   columns. Float32 KNN input returns a `float::float32` layout; ordinary R
#'   input returns a numeric matrix. Resolved settings are stored in
#'   `attr(layout, "fastEmbedR_config")`.
#' @examples
#' x <- scale(as.matrix(iris[, 1:4]))
#' d <- as.matrix(stats::dist(x))
#' diag(d) <- Inf
#' k <- 10L
#' idx <- t(apply(d, 1L, order))[, seq_len(k), drop = FALSE]
#' dst <- matrix(
#'     d[cbind(rep(seq_len(nrow(d)), each = k), as.vector(t(idx)))],
#'     nrow = nrow(d), byrow = TRUE
#' )
#' layout <- umap_knn(idx, dst, seed = 1)
#' plot(layout, pch = 21, bg = iris$Species)
#' @export
umap_knn <- function(indices,
                        distances = NULL,
                        n_components = 2L,
                        seed = 42L,
                        verbose = FALSE,
                        backend = NULL,
                        n.cores = NULL,
                        graph_mode = c("fuzzy", "binary")) {
    fast_knn_umap(
        indices,
        distances,
        n_components = n_components,
        seed = seed,
        verbose = verbose,
        backend = backend,
        n_threads = n.cores,
        graph_mode = graph_mode
    )
}
