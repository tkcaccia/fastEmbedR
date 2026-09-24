# Internal optimizer control, initialization, and backend dispatch for t-SNE.

# Backend resolution fails explicitly when a requested native optimizer is
# absent.

resolve_opentsne_optimizer_backend <- function(backend) {
    if (identical(backend, "cpu")) {
        return("cpu")
    }
    if (identical(backend, "metal")) {
        if (!embedding_metal_available_cpp() ||
            !metal_opentsne_native_available()) {
            stop(
                sprintf(
                    "%s%s%s",
                    "Native Metal t-SNE optimizer was requested, but it is ",
                    "not available in this build. ",
                    "No CPU fallback is used for Metal-labelled runs."
                ),
                call. = FALSE
            )
        }
        return("metal")
    }
    if (identical(backend, "cuda")) {
        if (!embedding_cuda_available_cpp() ||
            !cuda_opentsne_native_available()) {
            stop(
                sprintf(
                    "%s%s%s",
                    "Native CUDA t-SNE optimizer was requested, but it is ",
                    "not available in this build. ",
                    "No CPU fallback is used for CUDA-labelled runs."
                ),
                call. = FALSE
            )
        }
        return("cuda")
    }
    "cpu"
}

validate_opentsne_iteration_counts <- function(auto_params) {
    early <- integer_scalar(auto_params$early_exaggeration_iter)
    normal <- integer_scalar(auto_params$n_iter)
    if (is.na(early) || early < 0L) {
        stop(
            "`early_exaggeration_iter` must be a non-negative integer.",
            call. = FALSE
        )
    }
    if (is.na(normal) || normal < 0L) {
        stop("`n_iter` must be a non-negative integer.", call. = FALSE)
    }
    if (early + normal < 1L) {
        stop("At least one optimization iteration is required.", call. = FALSE)
    }
    list(early = early, normal = normal)
}

resolve_opentsne_gradient_method <- function(method,
                                                optimizer_backend,
                                                n,
                                                n_components) {
    if (identical(method, "auto")) {
        method <- if (n_components != 2L) {
            "exact"
        } else if (identical(optimizer_backend, "cuda")) {
            "fft"
        } else if (n <= 3000L) {
            "exact"
        } else {
            "fft"
        }
    }
    if (identical(optimizer_backend, "cpu") &&
        identical(method, "fft") &&
        n_components != 2L) {
        stop(
            sprintf(
                "%s%s",
                "`negative_gradient_method = \"fft\"` currently supports two ",
                "output components."
            ),
            call. = FALSE
        )
    }
    method
}

# Initialization metadata is resolved before optimizer-specific dispatch.
prepare_opentsne_initialization <- function(
    Y_init, gpu_resident_knn, cuda_init_data, optimizer_backend,
    indices, distances, n_components, seed, negative_gradient_method
) {
    if (!is.null(Y_init)) {
        return(list(
            Y_init = Y_init,
            method = attr(Y_init, "fastEmbedR_init_method") %||% "user",
            backend = attr(
                Y_init,
                "fastEmbedR_init_backend"
            ) %||% NA_character_,
            spectral_n_iter = attr(
                Y_init,
                "fastEmbedR_init_spectral_n_iter"
            ) %||% NA_integer_
        ))
    }

    if (isTRUE(gpu_resident_knn) &&
        identical(optimizer_backend, "cuda") &&
        !is.null(cuda_init_data)) {
        return(list(
            Y_init = NULL,
            method = "pca_cuda_auto_device",
            backend = "cuda_native_pca_device",
            spectral_n_iter = NA_integer_
        ))
    }
    if (isTRUE(gpu_resident_knn)) {
        return(list(
            Y_init = NULL,
            method = "cuda_random",
            backend = "cuda",
            spectral_n_iter = NA_integer_
        ))
    }
    make_opentsne_default_init(
        indices,
        distances,
        n_components,
        seed,
        optimizer_backend,
        negative_gradient_method
    )
}

validate_opentsne_min_gain <- function(min_gain) {
    value <- as.numeric(min_gain)
    invalid <- length(value) != 1L ||
        is.na(value) ||
        !is.finite(value) ||
        value <= 0
    if (invalid) {
        stop("`min_gain` must be a positive number.", call. = FALSE)
    }
    value
}

resolve_opentsne_max_step_norm <- function(max_step_norm) {
    is_auto <- is.character(max_step_norm) &&
        length(max_step_norm) == 1L &&
        identical(tolower(max_step_norm), "auto")
    if (is_auto) {
        return(5)
    }
    if (is.null(max_step_norm) ||
        (length(max_step_norm) == 1L && is.na(max_step_norm))) {
        return(NA_real_)
    }
    value <- as.numeric(max_step_norm)
    invalid <- length(value) != 1L ||
        is.na(value) ||
        !is.finite(value) ||
        value <= 0
    if (invalid) {
        stop(
            "`max_step_norm` must be NULL/NA or a positive number.",
            call. = FALSE
        )
    }
    value
}

# Shared controls are normalized once for CPU, Metal, and CUDA.
resolve_opentsne_controls <- function(
    n, n_components, perplexity, theta, early_iter, normal_iter, verbose,
    Y_init, initial_momentum, final_momentum, learning_rate,
    early_exaggeration, exaggeration, auto_params, min_gain, max_step_norm,
    optimizer_backend, negative_gradient_method, record_costs
) {
    args <- check_tsne_neighbor_params(
        n = n,
        n_components = n_components,
        perplexity = perplexity,
        theta = theta,
        max_iter = early_iter + normal_iter,
        verbose = verbose,
        Y_init = Y_init,
        momentum = initial_momentum,
        final_momentum = final_momentum
    )
    lr <- normalize_opentsne_learning_rate(learning_rate)
    if (isTRUE(auto_params$opt_sne_learning_rate) &&
        is.finite(auto_params$learning_rate_value) &&
        auto_params$learning_rate_value > 0) {
        lr <- list(auto = FALSE, value = auto_params$learning_rate_value)
    }
    list(
        args = args,
        learning_rate = lr,
        exaggeration = normalize_opentsne_exaggeration(
            early_exaggeration,
            exaggeration
        ),
        min_gain = validate_opentsne_min_gain(min_gain),
        max_step_norm = resolve_opentsne_max_step_norm(max_step_norm),
        record_costs = isTRUE(record_costs) || isTRUE(verbose)
    )
}

# This is the only t-SNE helper that dispatches to native optimizers.
run_opentsne_metal_optimizer <- function(state, controls, auto_params) {
    args <- controls$args
    lr <- controls$learning_rate
    ex <- controls$exaggeration
    knn_tsne_opentsne_metal_cpp(
        state$indices, state$distances, args$Y_init, args$init,
        args$n_components, args$perplexity, state$early_iter,
        state$normal_iter, ex$early, ex$normal, lr$value, lr$auto,
        args$momentum, args$final_momentum, controls$min_gain,
        controls$max_step_norm, state$gradient_method,
        as.integer(state$seed), controls$record_costs,
        isTRUE(auto_params$auto_kld_stop), auto_params$auto_iter_end
    )
}

run_opentsne_cuda_resident_optimizer <- function(state, controls) {
    args <- controls$args
    lr <- controls$learning_rate
    ex <- controls$exaggeration
    knn_tsne_opentsne_cuda_gpu_cpp(
        state$gpu_knn, as.integer(state$k), args$Y_init, args$init,
        state$cuda_init_data, args$n_components, args$perplexity,
        state$early_iter, state$normal_iter, ex$early, ex$normal,
        lr$value, lr$auto, args$momentum, args$final_momentum,
        controls$min_gain, controls$max_step_norm, state$gradient_method,
        as.integer(state$seed), controls$record_costs
    )
}

run_opentsne_cuda_host_optimizer <- function(state, controls) {
    args <- controls$args
    lr <- controls$learning_rate
    ex <- controls$exaggeration
    optimizer <- if (is_float32_matrix(state$distances)) {
        knn_tsne_opentsne_cuda_float_cpp
    } else {
        knn_tsne_opentsne_cuda_cpp
    }
    optimizer(
        state$indices, state$distances, args$Y_init, args$init,
        args$n_components, args$perplexity, state$early_iter,
        state$normal_iter, ex$early, ex$normal, lr$value, lr$auto,
        args$momentum, args$final_momentum, controls$min_gain,
        controls$max_step_norm, state$gradient_method,
        as.integer(state$seed), controls$record_costs
    )
}

run_opentsne_cpu_optimizer <- function(state, controls, auto_params) {
    args <- controls$args
    lr <- controls$learning_rate
    ex <- controls$exaggeration
    knn_tsne_opentsne_float_cpp(
        state$indices, state$distances, args$Y_init, args$init,
        args$n_components, args$perplexity, args$theta, state$early_iter,
        state$normal_iter, ex$early, ex$normal, lr$value, lr$auto,
        args$momentum, args$final_momentum, controls$min_gain,
        controls$max_step_norm, state$gradient_method,
        as.integer(state$n_threads), as.integer(state$seed), args$verbose,
        controls$record_costs, isTRUE(auto_params$auto_kld_stop),
        auto_params$auto_iter_end
    )
}

run_opentsne_native_optimizer <- function(
    optimizer_backend, indices, distances, gpu_knn, gpu_resident_knn,
    cuda_init_data, k, controls, early_iter, normal_iter,
    negative_gradient_method, auto_params, n_threads, seed
) {
    state <- list(
        indices = indices, distances = distances, gpu_knn = gpu_knn,
        cuda_init_data = cuda_init_data, k = k, early_iter = early_iter,
        normal_iter = normal_iter, gradient_method = negative_gradient_method,
        n_threads = n_threads, seed = seed
    )
    if (optimizer_backend == "metal") {
        return(run_opentsne_metal_optimizer(state, controls, auto_params))
    }
    if (optimizer_backend == "cuda" && isTRUE(gpu_resident_knn)) {
        return(run_opentsne_cuda_resident_optimizer(state, controls))
    }
    if (optimizer_backend == "cuda") {
        return(run_opentsne_cuda_host_optimizer(state, controls))
    }
    run_opentsne_cpu_optimizer(state, controls, auto_params)
}

opentsne_provenance <- function(optimizer_backend) {
    if (identical(optimizer_backend, "metal")) {
        return("interpolation_tsne_native_metal_directed_knn_gpu_optimizer")
    }
    if (identical(optimizer_backend, "cuda")) {
        return("interpolation_tsne_native_cuda_directed_knn_gpu_optimizer")
    }
    "interpolation_tsne_native_cpp_two_phase_optimizer"
}

# Result assembly is kept independent of the optimizer implementation.
opentsne_result_identity <- function(
    out, optimizer_backend, n, k, controls, early_iter, normal_iter
) {
    args <- controls$args
    list(
        method = "tsne",
        backend = optimizer_backend,
        n = n,
        n_neighbors = as.integer(k),
        perplexity = args$perplexity,
        theta = args$theta,
        early_exaggeration_iter = early_iter,
        n_iter = normal_iter,
        early_exaggeration_iter_actual =
            out$early_exaggeration_iter_actual %||% early_iter,
        n_iter_actual = out$n_iter_actual %||% normal_iter,
        max_iter = early_iter + normal_iter,
        max_iter_actual = out$max_iter_actual %||%
            (early_iter + normal_iter)
    )
}

opentsne_result_controls <- function(
    out, controls, auto_params, init_info, negative_gradient_method
) {
    args <- controls$args
    lr <- controls$learning_rate
    ex <- controls$exaggeration
    list(
        learning_rate = if (isTRUE(auto_params$opt_sne_learning_rate)) {
            "auto_opt_sne_n_over_early_exaggeration"
        } else if (isTRUE(lr$auto)) {
            "auto"
        } else {
            lr$value
        },
        learning_rate_early = out$learning_rate_early,
        learning_rate_normal = out$learning_rate_normal,
        early_exaggeration = ex$early,
        exaggeration = ex$normal,
        initial_momentum = args$momentum,
        final_momentum = args$final_momentum,
        min_gain = controls$min_gain,
        max_step_norm = controls$max_step_norm,
        initialization = out$initialization %||% init_info$method,
        initialization_requested = init_info$method,
        initialization_spectral_n_iter = init_info$spectral_n_iter,
        negative_gradient_method = negative_gradient_method
    )
}

opentsne_result_runtime <- function(
    out, controls, auto_params, return_float32, gpu_resident_knn,
    input_had_self, input_backend, optimizer_backend
) {
    list(
        fft_grid_size = out$fft_grid_size %||% NA_integer_,
        auto_config = isTRUE(auto_params$auto_config),
        auto_config_rule = auto_params$auto_rule,
        auto_kld_stop = isTRUE(out$auto_kld_stop %||% FALSE),
        auto_stop_reason = out$auto_stop_reason %||% "not_reported",
        auto_iter_end = out$auto_iter_end %||% auto_params$auto_iter_end,
        auto_perplexity = auto_params$auto_perplexity,
        auto_n_neighbors = auto_params$auto_n_neighbors,
        record_costs = controls$record_costs,
        optimizer = out$optimizer,
        repulsion = out$repulsion,
        precision = out$precision %||% "float32",
        output_precision = if (return_float32) "float32" else "double",
        probabilities = out$probabilities %||% "symmetric_sparse_knn_cpu",
        n_negatives = out$n_negatives %||% NA_integer_,
        n.cores = out$n_threads,
        n.cores_requested = out$n_threads_requested %||% out$n_threads,
        affinity_elapsed_sec = out$affinity_elapsed_sec %||% NA_real_,
        optimization_elapsed_sec =
            out$optimization_elapsed_sec %||% NA_real_,
        native_total_elapsed_sec = out$native_total_elapsed_sec %||% NA_real_,
        input_had_self = isTRUE(input_had_self),
        knn_backend = input_backend,
        knn_residency = if (isTRUE(gpu_resident_knn)) {
            "cuda_device"
        } else {
            "host"
        },
        provenance = opentsne_provenance(optimizer_backend)
    )
}

finalize_opentsne_native_result <- function(
    out, optimizer_backend, gpu_resident_knn, distances, n, k, controls,
    auto_params, init_info, negative_gradient_method, early_iter,
    normal_iter, input_had_self, input_backend
) {
    return_float32 <- isTRUE(gpu_resident_knn) ||
        is_float32_matrix(distances)
    layout <- finalize_embedding_layout(
        out$Y,
        "TSNE",
        return_float32 = return_float32
    )
    cfg <- c(
        opentsne_result_identity(
            out, optimizer_backend, n, k, controls, early_iter, normal_iter
        ),
        opentsne_result_controls(
            out, controls, auto_params, init_info,
            negative_gradient_method
        ),
        opentsne_result_runtime(
            out, controls, auto_params, return_float32,
            gpu_resident_knn, input_had_self, input_backend,
            optimizer_backend
        )
    )
    metal_stage_timing <- out$metal_stage_timing
    if (!is.null(metal_stage_timing) && NROW(metal_stage_timing) > 0L) {
        cfg$metal_stage_timing <- metal_stage_timing
    }
    attr(layout, "fastEmbedR_config") <- cfg
    attr(layout, "costs") <- out$costs
    attr(layout, "itercosts") <- out$itercosts
    attr(layout, "itercost_iterations") <- out$itercost_iterations
    attr(layout, "metal_trace") <- out$metal_trace
    if (!is.null(metal_stage_timing) && NROW(metal_stage_timing) > 0L) {
        attr(layout, "metal_stage_timing") <- metal_stage_timing
    }
    layout
}
