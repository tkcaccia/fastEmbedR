# Parameter validation and optimizer-control helpers for t-SNE workflows.

is_whole_number <- function(x, tol = .Machine$double.eps^0.5) {
    abs(x - round(x)) < tol
}

`%||%` <- function(x, y) {
    if (is.null(x)) y else x
}

auto_tsne_perplexity <- function(n, k) {
    n <- as.integer(n)
    k <- as.integer(k)
    max_from_n <- floor((n - 1L) / 3L)
    max_from_k <- k
    as.numeric(max(1L, min(30L, max_from_n, max_from_k)))
}

auto_tsne_k <- function(n, perplexity = NULL) {
    n <- as.integer(n)
    if (is.null(perplexity)) {
        perplexity <- min(30, floor((n - 1L) / 3L))
    }
    k <- as.integer(ceiling(as.numeric(perplexity)))
    max(1L, min(n - 1L, k))
}

opentsne_support_width <- function(perplexity) {
    as.integer(ceiling(as.numeric(perplexity)))
}

classify_opentsne_affinity_support <- function(k, perplexity) {
    k <- as.integer(k)
    perplexity <- as.numeric(perplexity)
    compact_k <- as.integer(ceiling(perplexity))
    if (k == compact_k) {
        "compact"
    } else {
        "custom"
    }
}

annotate_opentsne_affinity_support <- function(layout,
                                                k,
                                                perplexity) {
    cfg <- attr(layout, "fastEmbedR_config")
    cfg$affinity_support <- classify_opentsne_affinity_support(k, perplexity)
    cfg$affinity_support_policy <- "compact"
    cfg$affinity_support_k <- as.integer(k)
    cfg$affinity_support_multiplier <- as.numeric(k) / as.numeric(perplexity)
    cfg$compact_affinity_support <- as.integer(k) ==
        as.integer(ceiling(as.numeric(perplexity)))
    attr(layout, "fastEmbedR_config") <- cfg
    layout
}

validate_opentsne_neighbor_counts <- function(n, available) {
    n <- as.integer(n)
    if (length(n) != 1L || is.na(n) || n < 2L) {
        stop("`data` must contain at least two rows.", call. = FALSE)
    }
    if (!is.null(available)) {
        available <- as.integer(available)
        if (length(available) != 1L || is.na(available) || available < 1L) {
            stop(
                "The supplied KNN object has no usable non-self ",
                "neighbor columns.",
                call. = FALSE
            )
        }
    }
    list(n = n, available = available)
}

default_opentsne_perplexity <- function(max_perplexity, available) {
    max_from_support <- if (is.null(available)) {
        max_perplexity
    } else {
        available
    }
    as.numeric(max(1L, min(30L, max_perplexity, max_from_support)))
}

validate_opentsne_perplexity <- function(perplexity, max_perplexity) {
    perplexity <- numeric_scalar(perplexity)
    if (length(perplexity) != 1L || is.na(perplexity) ||
        !is.finite(perplexity) || perplexity <= 0) {
        stop("`perplexity` must be a positive finite number.", call. = FALSE)
    }
    if (perplexity > max_perplexity) {
        stop(
            "`perplexity` must be no larger than ",
            "floor((nrow(data) - 1) / 3).",
            call. = FALSE
        )
    }
    perplexity
}

opentsne_neighbor_policy <- function(n, perplexity = NULL, available = NULL) {
    counts <- validate_opentsne_neighbor_counts(n, available)
    n <- counts$n
    available <- counts$available
    max_perplexity <- floor((n - 1L) / 3L)
    if (is.null(perplexity)) {
        perplexity <- default_opentsne_perplexity(max_perplexity, available)
    } else {
        perplexity <- validate_opentsne_perplexity(
            perplexity,
            max_perplexity
        )
    }
    n_neighbors <- opentsne_support_width(perplexity)
    n_neighbors <- max(1L, min(n - 1L, n_neighbors))
    if (!is.null(available) && n_neighbors > available) {
        stop(
            "The supplied KNN object has fewer non-self columns than ",
            "required by compact t-SNE affinity support (need ",
            n_neighbors,
            ", have ", available, ").",
            call. = FALSE
        )
    }
    list(
        perplexity = as.numeric(perplexity),
        n_neighbors = n_neighbors,
        affinity_support = "compact",
        affinity_support_multiplier = 1
    )
}

opentsne_auto_parameter_values <- function(
    n, k, perplexity, optimizer_backend, negative_gradient_method,
    auto_config
) {
    perplexity_missing <- is.null(perplexity)
    if (isTRUE(auto_config)) {
        return(tsne_auto_parameters_cpp(
            as.integer(n),
            as.integer(k),
            if (perplexity_missing) NA_real_ else as.numeric(perplexity),
            isTRUE(perplexity_missing),
            as.character(optimizer_backend),
            as.character(negative_gradient_method)
        ))
    }
    list(
        perplexity = if (perplexity_missing) {
            auto_tsne_perplexity(n, k)
        } else {
            as.numeric(perplexity)
        },
        early_exaggeration_iter = 250L,
        n_iter = 500L,
        learning_rate = NA_real_,
        auto_kld_stop = FALSE,
        auto_iter_end = 5000,
        rule = "manual"
    )
}

resolve_opentsne_auto_parameters <- function(
    n, k, perplexity, early_exaggeration_iter, n_iter, learning_rate,
    optimizer_backend, negative_gradient_method, auto_config
) {
    perplexity_missing <- is.null(perplexity)
    early_iter_missing <- is.null(early_exaggeration_iter) ||
        identical(as.logical(is.na(early_exaggeration_iter)), TRUE)
    n_iter_missing <- is.null(n_iter) ||
        identical(as.logical(is.na(n_iter)), TRUE)
    auto <- opentsne_auto_parameter_values(
        n, k, perplexity, optimizer_backend, negative_gradient_method,
        auto_config
    )
    if (perplexity_missing) {
        perplexity <- auto$perplexity
    }
    if (early_iter_missing) {
        early_exaggeration_iter <- auto$early_exaggeration_iter
    }
    if (n_iter_missing) {
        n_iter <- auto$n_iter
    }
    opt_sne_learning_rate <- isTRUE(auto_config) &&
        is.character(learning_rate) &&
        length(learning_rate) == 1L &&
        identical(tolower(learning_rate), "auto")
    list(
        perplexity = as.numeric(perplexity),
        early_exaggeration_iter = early_exaggeration_iter,
        n_iter = n_iter,
        opt_sne_learning_rate = opt_sne_learning_rate,
        learning_rate_value = as.numeric(auto$learning_rate %||% NA_real_),
        auto_config = isTRUE(auto_config),
        auto_kld_stop = isTRUE(auto$auto_kld_stop) && early_iter_missing &&
            n_iter_missing,
        auto_iter_end = as.numeric(auto$auto_iter_end %||% 5000),
        auto_rule = as.character(auto$rule %||% "manual"),
        auto_perplexity = as.numeric(auto$perplexity %||% perplexity),
        auto_n_neighbors = as.integer(auto$n_neighbors %||% k)
    )
}

default_tsne_threads <- function() {
    resolve_n_cores(default = 4L)
}

metal_opentsne_exact_dense_threshold <- function() {
    6000L
}

cuda_opentsne_exact_dense_threshold <- function() {
    6000L
}

metal_opentsne_native_available <- function() {
    exists("knn_tsne_opentsne_metal_cpp",
        envir = asNamespace("fastEmbedR"),
        inherits = FALSE
    )
}

cuda_opentsne_native_available <- function() {
    exists("knn_tsne_opentsne_cuda_cpp",
        envir = asNamespace("fastEmbedR"),
        inherits = FALSE
    )
}

normalize_tsne_negative_gradient_method <- function(method) {
    method <- tolower(gsub("-", "_", as.character(method)))
    if (length(method) != 1L || is.na(method)) {
        stop("`negative_gradient_method` must be a single string.",
            call. = FALSE
        )
    }
    aliases <- c(
        auto = "auto",
        exact = "exact",
        pair = "exact",
        pair_symmetric = "exact",
        fft = "fft",
        interpolation = "fft",
        fitsne = "fft",
        fit_sne = "fft"
    )
    if (method %in% c("bh", "barnes_hut", "barnes", "barnes-hut")) {
        stop(
            sprintf(
                "%s%s%s%s",
                "`negative_gradient_method = \"bh\"` has been removed ",
                "from fastEmbedR. Use `negative_gradient_method = \"fft\"` ",
                "for the standard CPU t-SNE path, ",
                "or `\"exact\"` for small reference runs."
            ),
            call. = FALSE
        )
    }
    if (method %in% c("sampled", "negative_sampling", "sample")) {
        stop(
            "`negative_gradient_method = \"sampled\"` is not part of the ",
            "standard GPU t-SNE path because it changes the optimization ",
            "mathematics. Use `\"exact\"` for small native GPU checks, or CPU ",
            "`\"fft\"`; native Metal/CUDA FFT paths are used when compiled.",
            call. = FALSE
        )
    }
    out <- unname(aliases[method])
    if (is.na(out)) {
        stop(
            "`negative_gradient_method` must be one of ",
            "`\"auto\"`, `\"exact\"`, or `\"fft\"`.",
            call. = FALSE
        )
    }
    out
}

check_tsne_neighbor_params <- function(
    n, n_components, perplexity, theta, max_iter, verbose, Y_init,
    momentum, final_momentum
) {
    n_components <- validate_opentsne_n_components(n_components)
    if (!is_whole_number(max_iter) || max_iter <= 0L) {
        stop("Total optimization iterations must be positive.", call. = FALSE)
    }
    if (!is.null(Y_init) && (n != nrow(Y_init) || ncol(Y_init) !=
        n_components)) {
        stop("incorrect format for `Y_init`.", call. = FALSE)
    }
    if (!is.numeric(perplexity) || perplexity <= 0) {
        stop("`perplexity` should be a positive number.", call. = FALSE)
    }
    if (!is.numeric(theta) || theta < 0 || theta > 1) {
        stop("`theta` should lie in [0, 1].", call. = FALSE)
    }
    if (!is.numeric(momentum) || momentum < 0) {
        stop("`initial_momentum` should be non-negative.", call. = FALSE)
    }
    if (!is.numeric(final_momentum) || final_momentum < 0) {
        stop("`final_momentum` should be non-negative.", call. = FALSE)
    }
    if (n - 1L < 3 * perplexity) {
        stop("perplexity is too large for the number of samples.",
            call. = FALSE
        )
    }

    list(
        n_components = as.integer(n_components),
        perplexity = as.numeric(perplexity),
        theta = as.numeric(theta),
        max_iter = as.integer(max_iter),
        verbose = isTRUE(verbose),
        init = !is.null(Y_init),
        Y_init = if (is.null(Y_init)) {
            matrix(0, 0L, 0L)
        } else {
            opentsne_dense_numeric_matrix(Y_init)
        },
        momentum = as.numeric(momentum),
        final_momentum = as.numeric(final_momentum)
    )
}

validate_opentsne_n_components <- function(n_components, backend = NULL) {
    if (!is_whole_number(n_components) || n_components < 1L ||
        n_components > 3L) {
        stop("`n_components` should be 1, 2, or 3.", call. = FALSE)
    }
    n_components <- as.integer(n_components)
    if (!is.null(backend) && n_components != 2L && backend %in% c(
        "metal",
        "cuda"
    )) {
        stop(
            "Native Metal and CUDA t-SNE optimizers currently support only ",
            "`n_components = 2`.",
            call. = FALSE
        )
    }
    n_components
}

normalize_opentsne_learning_rate <- function(learning_rate) {
    if (is.character(learning_rate) && length(learning_rate) == 1L) {
        value <- tolower(learning_rate)
        if (identical(value, "auto")) {
            return(list(auto = TRUE, value = 1))
        }
    }
    value <- numeric_scalar(learning_rate)
    if (length(value) != 1L || is.na(value) || !is.finite(value) || value <=
        0) {
        stop("`learning_rate` must be a positive number or \"auto\".",
            call. = FALSE
        )
    }
    list(auto = FALSE, value = value)
}

normalize_opentsne_exaggeration <- function(early_exaggeration, exaggeration) {
    normal <- if (is.null(exaggeration)) {
        1
    } else {
        value <- numeric_scalar(exaggeration)
        if (length(value) != 1L || is.na(value) || !is.finite(value) || value <=
            0) {
            stop("`exaggeration` must be NULL or a positive number.",
                call. = FALSE
            )
        }
        value
    }
    early <- if (is.character(early_exaggeration) &&
        length(early_exaggeration) == 1L &&
        identical(tolower(early_exaggeration), "auto")) {
        if (is.null(exaggeration)) 12 else max(12, normal)
    } else {
        value <- numeric_scalar(early_exaggeration)
        if (length(value) != 1L || is.na(value) || !is.finite(value) || value <=
            0) {
            stop("`early_exaggeration` must be a positive number or \"auto\".",
                call. = FALSE
            )
        }
        value
    }
    list(early = early, normal = normal)
}

make_opentsne_random_init <- function(n, n_components, seed) {
    seed <- integer_scalar(seed)
    if (length(seed) != 1L || is.na(seed)) seed <- 5489L
    restore_seed <- set_local_seed(seed)
    on.exit(restore_seed(), add = TRUE)
    init <- matrix(stats::rnorm(n * n_components, sd = 1e-4),
        nrow = n,
        ncol = n_components
    )
    sweep(init, 2L, colMeans(init), check.margin = FALSE)
}

make_opentsne_default_init <- function(indices,
                                        distances,
                                        n_components,
                                        seed,
                                        optimizer_backend,
                                        negative_gradient_method) {
    n <- nrow(indices)
    if (identical(optimizer_backend, "metal") &&
        identical(negative_gradient_method, "fft") &&
        n_components == 2L &&
        n >= 10000L &&
        isTRUE(embedding_metal_available_cpp())) {
        spectral <- tryCatch(
            spectral_knn_init(
                indices,
                distances,
                n_components = 2L,
                spectral_n_iter = 10L,
                backend = "metal",
                seed = seed
            ),
            error = function(e) NULL
        )
        if (!is.null(spectral)) {
            spectral <- as.matrix(spectral)
            spectral <- sweep(spectral, 2L, colMeans(spectral),
                check.margin = FALSE
            )
            scale <- max(stats::sd(spectral[, 1L]), stats::sd(spectral[, 2L]))
            if (is.finite(scale) && scale > 0) {
                spectral <- spectral * (1e-4 / scale)
            }
            return(list(
                Y_init = spectral,
                method = "metal_spectral_knn",
                spectral_n_iter = 10L
            ))
        }
    }
    list(
        Y_init = make_opentsne_random_init(n, n_components, seed),
        method = "random_normal",
        spectral_n_iter = NA_integer_
    )
}

opentsne_dense_numeric_matrix <- function(x) {
    embedding_dense_double_matrix(x)
}

opentsne_pca_input_matrix <- function(x) {
    if (is_float32_matrix(x)) {
        if (!requireNamespace("float", quietly = TRUE)) {
            stop("The float package is required to use float32 input.",
                call. = FALSE
            )
        }
        return(x)
    }
    x <- as.matrix(x)
    storage.mode(x) <- "double"
    x
}

opentsne_pca_has_nonfinite <- function(x) {
    if (is_float32_matrix(x)) {
        if (!requireNamespace("float", quietly = TRUE)) {
            stop("The float package is required to use float32 input.",
                call. = FALSE
            )
        }
        return(!isTRUE(float32_all_finite_cpp(x)))
    }
    any(!is.finite(x))
}

normalize_opentsne_pca_scores <- function(scores, n_components) {
    init <- scores[, seq_len(n_components), drop = FALSE]
    if (is_float32_matrix(init)) {
        if (!requireNamespace("float", quietly = TRUE)) {
            stop(
                "The float package is required to normalize float32 ",
                "PCA scores.",
                call. = FALSE
            )
        }
        init <- float::sweep(init, 2L, float::colMeans(init),
            check.margin = FALSE
        )
        mean_squares <- float::dbl(float::colMeans(init * init))
        variance <- mean_squares
        if (nrow(init) > 1L) {
            variance <- variance * nrow(init) / (nrow(init) - 1L)
        }
        init_scale <- sqrt(max(variance))
        if (is.finite(init_scale) && init_scale > 0) {
            init <- init * float::fl(1e-4 / init_scale)
        }
        return(init)
    }
    init <- as.matrix(init)
    init <- sweep(init, 2L, colMeans(init), check.margin = FALSE)
    init_scale <- max(apply(init, 2L, stats::sd))
    if (is.finite(init_scale) && init_scale > 0) {
        init <- init * (1e-4 / init_scale)
    }
    init
}
