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
  max_from_k <- floor(k / 3L)
  as.numeric(max(1L, min(30L, max_from_n, max_from_k)))
}

auto_tsne_k <- function(n, perplexity = NULL) {
  n <- as.integer(n)
  if (is.null(perplexity)) {
    perplexity <- min(30, floor((n - 1L) / 3L))
  }
  k <- as.integer(floor(3 * as.numeric(perplexity)))
  max(1L, min(n - 1L, k))
}

opentsne_neighbor_policy <- function(n, perplexity = NULL, available = NULL) {
  n <- as.integer(n)
  if (length(n) != 1L || is.na(n) || n < 2L) {
    stop("`data` must contain at least two rows.", call. = FALSE)
  }
  max_perplexity <- floor((n - 1L) / 3L)
  if (!is.null(available)) {
    available <- as.integer(available)
    if (length(available) != 1L || is.na(available) || available < 1L) {
      stop("The supplied KNN object has no usable non-self neighbour columns.", call. = FALSE)
    }
  }
  if (is.null(perplexity)) {
    n_neighbors <- max(1L, min(15L, n - 1L))
    if (!is.null(available)) {
      n_neighbors <- min(n_neighbors, available)
    }
    return(list(perplexity = NULL, n_neighbors = as.integer(n_neighbors)))
  } else {
    perplexity <- suppressWarnings(as.numeric(perplexity))
    if (length(perplexity) != 1L || is.na(perplexity) ||
        !is.finite(perplexity) || perplexity <= 0) {
      stop("`perplexity` must be a positive finite number.", call. = FALSE)
    }
  }
  if (perplexity > max_perplexity) {
    stop(
      "`perplexity` must be no larger than floor((nrow(data) - 1) / 3).",
      call. = FALSE
    )
  }
  n_neighbors <- as.integer(ceiling(perplexity))
  n_neighbors <- max(1L, min(n - 1L, n_neighbors))
  if (!is.null(available) && n_neighbors > available) {
    stop(
      "The supplied KNN object has fewer non-self columns than `ceiling(perplexity)`.",
      call. = FALSE
    )
  }
  list(perplexity = as.numeric(perplexity), n_neighbors = n_neighbors)
}

resolve_opentsne_auto_parameters <- function(n,
                                             k,
                                             perplexity,
                                             early_exaggeration_iter,
                                             n_iter,
                                             learning_rate,
                                             optimizer_backend,
                                             negative_gradient_method,
                                             auto_config) {
  perplexity_missing <- is.null(perplexity)
  early_iter_missing <- is.null(early_exaggeration_iter) ||
    (length(early_exaggeration_iter) == 1L && is.na(early_exaggeration_iter))
  n_iter_missing <- is.null(n_iter) ||
    (length(n_iter) == 1L && is.na(n_iter))
  auto <- if (isTRUE(auto_config)) {
    tsne_auto_parameters_cpp(
      as.integer(n),
      as.integer(k),
      if (perplexity_missing) NA_real_ else as.numeric(perplexity),
      isTRUE(perplexity_missing),
      as.character(optimizer_backend),
      as.character(negative_gradient_method)
    )
  } else {
    list(
      perplexity = if (perplexity_missing) auto_tsne_perplexity(n, k) else as.numeric(perplexity),
      early_exaggeration_iter = 250L,
      n_iter = 500L,
      learning_rate = NA_real_,
      auto_kld_stop = FALSE,
      auto_iter_end = 5000,
      rule = "manual"
    )
  }
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
    early_exaggeration_iter = as.integer(early_exaggeration_iter),
    n_iter = as.integer(n_iter),
    opt_sne_learning_rate = opt_sne_learning_rate,
    learning_rate_value = as.numeric(auto$learning_rate %||% NA_real_),
    auto_config = isTRUE(auto_config),
    auto_kld_stop = isTRUE(auto$auto_kld_stop) && early_iter_missing && n_iter_missing,
    auto_iter_end = as.numeric(auto$auto_iter_end %||% 5000),
    auto_rule = as.character(auto$rule %||% "manual"),
    auto_perplexity = as.numeric(auto$perplexity %||% perplexity),
    auto_n_neighbors = as.integer(auto$n_neighbors %||% k)
  )
}

default_tsne_threads <- function() {
  value <- getOption("fastEmbedR.tsne_threads", 4L)
  value <- suppressWarnings(as.integer(value))
  if (length(value) != 1L || is.na(value) || !is.finite(value) || value < 0L) {
    return(4L)
  }
  value
}

metal_opentsne_exact_dense_threshold <- function() {
  6000L
}

cuda_opentsne_exact_dense_threshold <- function() {
  6000L
}

metal_opentsne_native_available <- function() {
  exists("knn_tsne_opentsne_metal_cpp", envir = asNamespace("fastEmbedR"), inherits = FALSE)
}

cuda_opentsne_native_available <- function() {
  exists("knn_tsne_opentsne_cuda_cpp", envir = asNamespace("fastEmbedR"), inherits = FALSE)
}

normalize_tsne_negative_gradient_method <- function(method) {
  method <- tolower(gsub("-", "_", as.character(method)))
  if (length(method) != 1L || is.na(method)) {
    stop("`negative_gradient_method` must be a single string.", call. = FALSE)
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
      "`negative_gradient_method = \"bh\"` has been removed from fastEmbedR. ",
      "Use `negative_gradient_method = \"fft\"` for the standard CPU openTSNE path, ",
      "or `\"exact\"` for small reference runs.",
      call. = FALSE
    )
  }
  if (method %in% c("sampled", "negative_sampling", "sample")) {
    stop(
      "`negative_gradient_method = \"sampled\"` is not part of the ",
      "standard GPU openTSNE path because it changes the optimization ",
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

check_tsne_neighbor_params <- function(n,
                                       n_components,
                                       perplexity,
                                       theta,
                                       max_iter,
                                       verbose,
                                       Y_init,
                                       momentum,
                                       final_momentum) {
  if (!is_whole_number(n_components) || n_components < 1L || n_components > 3L) {
    stop("`n_components` should be 1, 2, or 3.", call. = FALSE)
  }
  if (!is_whole_number(max_iter) || max_iter <= 0L) {
    stop("Total optimization iterations must be positive.", call. = FALSE)
  }
  if (!is.null(Y_init) && (n != nrow(Y_init) || ncol(Y_init) != n_components)) {
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
    stop("perplexity is too large for the number of samples.", call. = FALSE)
  }

  list(
    n_components = as.integer(n_components),
    perplexity = as.numeric(perplexity),
    theta = as.numeric(theta),
    max_iter = as.integer(max_iter),
    verbose = isTRUE(verbose),
    init = !is.null(Y_init),
    Y_init = if (is.null(Y_init)) matrix(0, 0L, 0L) else {
      Y_init <- as.matrix(Y_init)
      storage.mode(Y_init) <- "double"
      Y_init
    },
    momentum = as.numeric(momentum),
    final_momentum = as.numeric(final_momentum)
  )
}

normalize_opentsne_learning_rate <- function(learning_rate) {
  if (is.character(learning_rate) && length(learning_rate) == 1L) {
    value <- tolower(learning_rate)
    if (identical(value, "auto")) {
      return(list(auto = TRUE, value = 1))
    }
  }
  value <- suppressWarnings(as.numeric(learning_rate))
  if (length(value) != 1L || is.na(value) || !is.finite(value) || value <= 0) {
    stop("`learning_rate` must be a positive number or \"auto\".", call. = FALSE)
  }
  list(auto = FALSE, value = value)
}

normalize_opentsne_exaggeration <- function(early_exaggeration, exaggeration) {
  normal <- if (is.null(exaggeration)) 1 else {
    value <- suppressWarnings(as.numeric(exaggeration))
    if (length(value) != 1L || is.na(value) || !is.finite(value) || value <= 0) {
      stop("`exaggeration` must be NULL or a positive number.", call. = FALSE)
    }
    value
  }
  early <- if (is.character(early_exaggeration) &&
               length(early_exaggeration) == 1L &&
               identical(tolower(early_exaggeration), "auto")) {
    if (is.null(exaggeration)) 12 else max(12, normal)
  } else {
    value <- suppressWarnings(as.numeric(early_exaggeration))
    if (length(value) != 1L || is.na(value) || !is.finite(value) || value <= 0) {
      stop("`early_exaggeration` must be a positive number or \"auto\".", call. = FALSE)
    }
    value
  }
  list(early = early, normal = normal)
}

make_opentsne_random_init <- function(n, n_components, seed) {
  old_seed <- if (exists(".Random.seed", envir = .GlobalEnv, inherits = FALSE)) {
    get(".Random.seed", envir = .GlobalEnv, inherits = FALSE)
  } else {
    NULL
  }
  on.exit({
    if (is.null(old_seed)) {
      if (exists(".Random.seed", envir = .GlobalEnv, inherits = FALSE)) {
        rm(".Random.seed", envir = .GlobalEnv)
      }
    } else {
      assign(".Random.seed", old_seed, envir = .GlobalEnv)
    }
  }, add = TRUE)

  seed <- suppressWarnings(as.integer(seed))
  if (length(seed) != 1L || is.na(seed)) seed <- 5489L
  set.seed(seed)
  init <- matrix(stats::rnorm(n * n_components, sd = 1e-4), nrow = n, ncol = n_components)
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
      spectral <- sweep(spectral, 2L, colMeans(spectral), check.margin = FALSE)
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

make_opentsne_pca_init <- function(x, n_components, seed, backend = "cpu") {
  x <- as.matrix(x)
  storage.mode(x) <- "double"
  n_components <- as.integer(n_components)
  if (length(n_components) != 1L || is.na(n_components) || n_components < 1L) {
    stop("`n_components` must be a positive integer.", call. = FALSE)
  }
  centered <- sweep(x, 2L, colMeans(x), check.margin = FALSE)
  pca <- fastpls_rsvd_pca_scores(
    centered,
    rank = n_components,
    seed = seed,
    backend = backend
  )
  init <- as.matrix(pca$scores[, seq_len(n_components), drop = FALSE])
  init <- sweep(init, 2L, colMeans(init), check.margin = FALSE)
  scale <- max(apply(init, 2L, stats::sd))
  if (is.finite(scale) && scale > 0) {
    init <- init * (1e-4 / scale)
  }
  attr(init, "fastEmbedR_init_method") <- paste0("pca_", pca$method)
  attr(init, "fastEmbedR_init_backend") <- pca$backend
  init
}

#' Compute or reuse a PCA initialization for openTSNE
#'
#' `opentsne_pca_init()` creates the small-scale PCA initialization used by
#' [opentsne()] and [opentsne_knn()]. Supplying `cache_file` stores the result
#' as an RDS file; later calls with the same path reuse the saved matrix instead
#' of recomputing PCA. This is useful when comparing several KNN backends with
#' exactly the same initialization.
#'
#' @param data Numeric matrix/data frame with observations in rows.
#' @param n_components Output dimensionality, usually `2`.
#' @param seed Random seed used by the randomized SVD PCA helper.
#' @param backend Backend used for PCA when available: `"cpu"`, `"metal"`,
#'   or `"cuda"`.
#' @param cache_file Optional `.rds` file path. If it exists and
#'   `force_recompute = FALSE`, the saved initialization is loaded and
#'   validated.
#' @param force_recompute If `TRUE`, ignore any existing cache and recompute.
#' @return A numeric initialization matrix suitable for `Y_init`.
#' @export
opentsne_pca_init <- function(data,
                              n_components = 2L,
                              seed = 4L,
                              backend = c("cpu", "metal", "cuda"),
                              cache_file = NULL,
                              force_recompute = FALSE) {
  backend <- match.arg(backend)
  n_components <- as.integer(n_components)
  if (length(n_components) != 1L || is.na(n_components) || n_components < 1L) {
    stop("`n_components` must be a positive integer.", call. = FALSE)
  }
  x <- as.matrix(data)
  storage.mode(x) <- "double"
  if (nrow(x) < 2L || ncol(x) < 1L) {
    stop("`data` must have at least two rows and one column.", call. = FALSE)
  }
  if (any(!is.finite(x))) {
    stop("`data` must contain only finite values.", call. = FALSE)
  }
  if (!is.null(cache_file)) {
    cache_file <- path.expand(as.character(cache_file)[1L])
    if (!isTRUE(force_recompute) && file.exists(cache_file)) {
      init <- readRDS(cache_file)
      init <- resolve_opentsne_y_init(
        init,
        n = nrow(x),
        n_components = n_components
      )
      attr(init, "fastEmbedR_init_cache_file") <- cache_file
      attr(init, "fastEmbedR_init_cache_hit") <- TRUE
      return(init)
    }
  }
  init <- make_opentsne_pca_init(
    x,
    n_components = n_components,
    seed = seed,
    backend = backend
  )
  if (!is.null(cache_file)) {
    dir.create(dirname(cache_file), recursive = TRUE, showWarnings = FALSE)
    saveRDS(init, cache_file, version = 2)
    attr(init, "fastEmbedR_init_cache_file") <- cache_file
    attr(init, "fastEmbedR_init_cache_hit") <- FALSE
  }
  init
}

resolve_opentsne_y_init <- function(Y_init, n, n_components) {
  if (is.null(Y_init)) return(NULL)
  if (is.character(Y_init) && length(Y_init) == 1L) {
    path <- path.expand(Y_init)
    if (!file.exists(path)) {
      stop("`Y_init` file does not exist: ", path, call. = FALSE)
    }
    Y_init <- readRDS(path)
    attr(Y_init, "fastEmbedR_init_cache_file") <- path
    attr(Y_init, "fastEmbedR_init_cache_hit") <- TRUE
  }
  Y_init <- as.matrix(Y_init)
  storage.mode(Y_init) <- "double"
  if (nrow(Y_init) != n || ncol(Y_init) != n_components) {
    stop(
      "`Y_init` must have ", n, " rows and ", n_components,
      " columns.",
      call. = FALSE
    )
  }
  if (any(!is.finite(Y_init))) {
    stop("`Y_init` must contain only finite values.", call. = FALSE)
  }
  Y_init
}

make_opentsne_pca_init_from_data <- function(init_data,
                                             n,
                                             n_components,
                                             seed,
                                             backend = "cpu") {
  x <- as.matrix(init_data)
  storage.mode(x) <- "double"
  if (nrow(x) != n) {
    stop("`init_data` must have one row per KNN row.", call. = FALSE)
  }
  make_opentsne_pca_init(
    x,
    n_components = n_components,
    seed = seed,
    backend = backend
  )
}

normalize_opentsne_knn_input <- function(indices, distances = NULL, n_neighbors = NULL) {
  knn <- coerce_knn_input(indices, distances)
  n <- nrow(knn$indices)
  available <- knn$n_neighbors
  if (is.null(n_neighbors)) {
    n_neighbors <- available
  } else {
    n_neighbors <- as.integer(n_neighbors)
    if (length(n_neighbors) != 1L || is.na(n_neighbors) ||
        !is.finite(n_neighbors) || n_neighbors < 1L || n_neighbors >= n) {
      stop("`n_neighbors` must be a positive integer smaller than the number of rows.", call. = FALSE)
    }
    if (n_neighbors > available) {
      stop("`n_neighbors` is larger than the supplied KNN width.", call. = FALSE)
    }
  }
  materialized <- materialize_knn_range(
    knn$indices,
    knn$distances,
    knn$col_start,
    n_neighbors
  )
  distance_type <- knn$distance_type
  list(
    indices = materialized$indices,
    distances = materialized$distances,
    n = n,
    n_neighbors = as.integer(n_neighbors),
    has_self = isTRUE(knn$has_self),
    input_backend = knn$input_backend,
    distance_type = distance_type
  )
}

fast_knn_opentsne_materialized <- function(indices,
                                           distances,
                                           n_components = 2L,
                                           perplexity = NULL,
                                           theta = 0.5,
                                           early_exaggeration_iter = NULL,
                                           n_iter = NULL,
                                           learning_rate = "auto",
                                           early_exaggeration = "auto",
                                           exaggeration = NULL,
                                           Y_init = NULL,
                                           initial_momentum = 0.8,
                                           final_momentum = 0.8,
                                           min_gain = 0.01,
                                           max_step_norm = "auto",
                                           negative_gradient_method = "auto",
                                           record_costs = FALSE,
                                           n_threads = NULL,
                                           seed = 42L,
                                           verbose = FALSE,
                                           backend = c("cpu", "cuda", "metal"),
                                           auto_config = TRUE,
                                           input_had_self = FALSE,
                                           input_backend = NA_character_) {
  backend <- resolve_embedding_backend(backend)
  optimizer_backend <- if (identical(backend, "cpu")) {
    "cpu"
  } else if (identical(backend, "metal")) {
    if (!embedding_metal_available_cpp() || !metal_opentsne_native_available()) {
      stop(
        "Native Metal openTSNE optimizer was requested, but it is not available in this build. ",
        "No CPU fallback is used for Metal-labelled runs.",
        call. = FALSE
      )
    }
    "metal"
  } else if (identical(backend, "cuda")) {
    if (!embedding_cuda_available_cpp() || !cuda_opentsne_native_available()) {
      stop(
        "Native CUDA openTSNE optimizer was requested, but it is not available in this build. ",
        "No CPU fallback is used for CUDA-labelled runs.",
        call. = FALSE
      )
    }
    "cuda"
  } else {
    "cpu"
  }
  n <- nrow(indices)
  k <- ncol(indices)
  Y_init <- resolve_opentsne_y_init(Y_init, n, n_components)
  if (is.null(n_threads)) {
    n_threads <- default_tsne_threads()
  }
  negative_gradient_method <- normalize_tsne_negative_gradient_method(negative_gradient_method)
  auto_params <- resolve_opentsne_auto_parameters(
    n = n,
    k = k,
    perplexity = perplexity,
    early_exaggeration_iter = early_exaggeration_iter,
    n_iter = n_iter,
    learning_rate = learning_rate,
    optimizer_backend = optimizer_backend,
    negative_gradient_method = negative_gradient_method,
    auto_config = auto_config
  )
  perplexity <- auto_params$perplexity
  early_exaggeration_iter <- auto_params$early_exaggeration_iter
  n_iter <- auto_params$n_iter
  early_exaggeration_iter <- as.integer(early_exaggeration_iter)
  n_iter <- as.integer(n_iter)
  if (length(early_exaggeration_iter) != 1L || is.na(early_exaggeration_iter) || early_exaggeration_iter < 0L) {
    stop("`early_exaggeration_iter` must be a non-negative integer.", call. = FALSE)
  }
  if (length(n_iter) != 1L || is.na(n_iter) || n_iter < 0L) {
    stop("`n_iter` must be a non-negative integer.", call. = FALSE)
  }
  if (early_exaggeration_iter + n_iter < 1L) {
    stop("At least one optimization iteration is required.", call. = FALSE)
  }
  if (identical(optimizer_backend, "metal")) {
    if (identical(negative_gradient_method, "auto")) {
      negative_gradient_method <- "fft"
    }
  } else if (identical(optimizer_backend, "cuda")) {
    if (identical(negative_gradient_method, "auto")) {
      negative_gradient_method <- "fft"
    }
  } else if (identical(negative_gradient_method, "auto")) {
    negative_gradient_method <- "fft"
  }
  if (identical(optimizer_backend, "cpu") && identical(negative_gradient_method, "fft")) {
    if (n_components != 2L) {
      stop(
        "`negative_gradient_method = \"fft\"` currently supports two output components.",
        call. = FALSE
      )
    }
  }
  init_info <- if (is.null(Y_init)) {
    make_opentsne_default_init(
      indices,
      distances,
      n_components,
      seed,
      optimizer_backend,
      negative_gradient_method
    )
  } else {
    list(
      Y_init = Y_init,
      method = attr(Y_init, "fastEmbedR_init_method") %||% "user",
      backend = attr(Y_init, "fastEmbedR_init_backend") %||% NA_character_,
      spectral_n_iter = attr(Y_init, "fastEmbedR_init_spectral_n_iter") %||% NA_integer_
    )
  }
  if (is.null(Y_init)) {
    Y_init <- init_info$Y_init
  }
  args <- check_tsne_neighbor_params(
    n = n,
    n_components = n_components,
    perplexity = perplexity,
    theta = theta,
    max_iter = early_exaggeration_iter + n_iter,
    verbose = verbose,
    Y_init = Y_init,
    momentum = initial_momentum,
    final_momentum = final_momentum
  )
  lr <- normalize_opentsne_learning_rate(learning_rate)
  ex <- normalize_opentsne_exaggeration(early_exaggeration, exaggeration)
  if (isTRUE(auto_params$opt_sne_learning_rate) &&
      is.finite(auto_params$learning_rate_value) &&
      auto_params$learning_rate_value > 0) {
    lr <- list(auto = FALSE, value = auto_params$learning_rate_value)
  }
  min_gain <- as.numeric(min_gain)
  if (length(min_gain) != 1L || is.na(min_gain) || !is.finite(min_gain) || min_gain <= 0) {
    stop("`min_gain` must be a positive number.", call. = FALSE)
  }
  record_costs <- isTRUE(record_costs) || isTRUE(verbose)
  max_step_norm <- if (is.character(max_step_norm) &&
                       length(max_step_norm) == 1L &&
                       identical(tolower(max_step_norm), "auto")) {
    if (identical(optimizer_backend, "metal") &&
        identical(negative_gradient_method, "fft")) {
      0.5
    } else {
      5
    }
  } else if (is.null(max_step_norm) || (length(max_step_norm) == 1L && is.na(max_step_norm))) {
    NA_real_
  } else {
    value <- suppressWarnings(as.numeric(max_step_norm))
    if (length(value) != 1L || is.na(value) || !is.finite(value) || value <= 0) {
      stop("`max_step_norm` must be NULL/NA or a positive number.", call. = FALSE)
    }
    value
  }

  out <- if (identical(optimizer_backend, "metal")) {
    gpu_distances <- if (is_float32_matrix(distances)) {
      matrix(
        as.numeric(distances),
        nrow = nrow(indices),
        ncol = ncol(indices)
      )
    } else {
      distances
    }
    knn_tsne_opentsne_metal_cpp(
      indices,
      gpu_distances,
      args$Y_init,
      args$init,
      args$n_components,
      args$perplexity,
      early_exaggeration_iter,
      n_iter,
      ex$early,
      ex$normal,
      lr$value,
      lr$auto,
      args$momentum,
      args$final_momentum,
      min_gain,
      max_step_norm,
      negative_gradient_method,
      as.integer(seed),
      record_costs
    )
  } else if (identical(optimizer_backend, "cuda")) {
    if (is_float32_matrix(distances)) {
      knn_tsne_opentsne_cuda_float_cpp(
        indices,
        distances,
        args$Y_init,
        args$init,
        args$n_components,
        args$perplexity,
        early_exaggeration_iter,
        n_iter,
        ex$early,
        ex$normal,
        lr$value,
        lr$auto,
        args$momentum,
        args$final_momentum,
        min_gain,
        max_step_norm,
        negative_gradient_method,
        as.integer(seed),
        record_costs
      )
    } else {
      knn_tsne_opentsne_cuda_cpp(
        indices,
        distances,
        args$Y_init,
        args$init,
        args$n_components,
        args$perplexity,
        early_exaggeration_iter,
        n_iter,
        ex$early,
        ex$normal,
        lr$value,
        lr$auto,
        args$momentum,
        args$final_momentum,
        min_gain,
        max_step_norm,
        negative_gradient_method,
        as.integer(seed),
        record_costs
      )
    }
  } else {
    if (is_float32_matrix(distances)) {
      knn_tsne_opentsne_float_cpp(
        indices,
        distances,
        args$Y_init,
        args$init,
        args$n_components,
        args$perplexity,
        args$theta,
        early_exaggeration_iter,
        n_iter,
        ex$early,
        ex$normal,
        lr$value,
        lr$auto,
        args$momentum,
        args$final_momentum,
        min_gain,
        max_step_norm,
        negative_gradient_method,
        as.integer(n_threads),
        as.integer(seed),
        args$verbose,
        record_costs,
        isTRUE(auto_params$auto_kld_stop),
        auto_params$auto_iter_end
      )
    } else {
      knn_tsne_opentsne_cpp(
        indices,
        distances,
        args$Y_init,
        args$init,
        args$n_components,
        args$perplexity,
        args$theta,
        early_exaggeration_iter,
        n_iter,
        ex$early,
        ex$normal,
        lr$value,
        lr$auto,
        args$momentum,
        args$final_momentum,
        min_gain,
        max_step_norm,
        negative_gradient_method,
        as.integer(n_threads),
        as.integer(seed),
        args$verbose,
        record_costs,
        isTRUE(auto_params$auto_kld_stop),
        auto_params$auto_iter_end
      )
    }
  }
  layout <- finalize_embedding_layout(
    out$Y,
    "openTSNE",
    return_float32 = is_float32_matrix(distances)
  )
  probabilities <- out$probabilities
  if (is.null(probabilities)) probabilities <- "symmetric_sparse_knn_cpu"
  n_negatives <- out$n_negatives
  if (is.null(n_negatives)) n_negatives <- NA_integer_
  cfg <- list(
    method = "opentsne",
    backend = optimizer_backend,
    n = n,
    n_neighbors = as.integer(k),
    perplexity = args$perplexity,
    theta = args$theta,
    early_exaggeration_iter = early_exaggeration_iter,
    n_iter = n_iter,
    early_exaggeration_iter_actual = out$early_exaggeration_iter_actual %||% early_exaggeration_iter,
    n_iter_actual = out$n_iter_actual %||% n_iter,
    max_iter = early_exaggeration_iter + n_iter,
    max_iter_actual = out$max_iter_actual %||% (early_exaggeration_iter + n_iter),
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
    min_gain = min_gain,
    max_step_norm = max_step_norm,
    initialization = init_info$method,
    initialization_spectral_n_iter = init_info$spectral_n_iter,
    negative_gradient_method = negative_gradient_method,
    auto_config = isTRUE(auto_params$auto_config),
    auto_config_rule = auto_params$auto_rule,
    auto_kld_stop = isTRUE(out$auto_kld_stop %||% FALSE),
    auto_stop_reason = out$auto_stop_reason %||% "not_reported",
    auto_iter_end = out$auto_iter_end %||% auto_params$auto_iter_end,
    auto_perplexity = auto_params$auto_perplexity,
    auto_n_neighbors = auto_params$auto_n_neighbors,
    record_costs = record_costs,
    optimizer = out$optimizer,
    repulsion = out$repulsion,
    precision = out$precision %||% if (is_float32_matrix(distances) && identical(optimizer_backend, "cpu")) "float32" else "double",
    probabilities = probabilities,
    n_negatives = n_negatives,
    n_threads = out$n_threads,
    input_had_self = isTRUE(input_had_self),
    knn_backend = input_backend,
    provenance = if (identical(optimizer_backend, "metal")) {
      "openTSNE_style_native_metal_directed_knn_gpu_probability_optimizer"
    } else if (identical(optimizer_backend, "cuda")) {
      "openTSNE_style_native_cuda_directed_knn_gpu_probability_optimizer"
    } else {
      "openTSNE_native_cpp_two_phase_optimizer_bsd3_informed"
    }
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

fast_knn_opentsne_core <- function(indices,
                                   distances = NULL,
                                   n_components = 2L,
                                   perplexity = NULL,
                                   theta = 0.5,
                                   early_exaggeration_iter = NULL,
                                   n_iter = NULL,
                                   learning_rate = "auto",
                                   early_exaggeration = "auto",
                                   exaggeration = NULL,
                                   Y_init = NULL,
                                   initial_momentum = 0.8,
                                   final_momentum = 0.8,
                                   min_gain = 0.01,
                                   max_step_norm = 5,
                                   negative_gradient_method = "auto",
                                   record_costs = FALSE,
                                   n_threads = NULL,
                                   seed = 42L,
                                   verbose = FALSE,
                                   backend = c("cpu", "cuda", "metal"),
                                   auto_config = TRUE) {
  backend <- resolve_embedding_backend(backend)
  if (inherits(indices, "fastEmbedR_opentsne_prepared")) {
    if (!is.null(distances)) {
      stop("Do not pass `distances` when `indices` is a prepared openTSNE object.", call. = FALSE)
    }
    knn <- indices$knn
  } else {
    knn <- normalize_opentsne_knn_input(indices, distances)
  }
  Y_init <- resolve_opentsne_y_init(Y_init, knn$n, n_components)
  fast_knn_opentsne_materialized(
    knn$indices,
    knn$distances,
    n_components = n_components,
    perplexity = perplexity,
    theta = theta,
    early_exaggeration_iter = early_exaggeration_iter,
    n_iter = n_iter,
    learning_rate = learning_rate,
    early_exaggeration = early_exaggeration,
    exaggeration = exaggeration,
    Y_init = Y_init,
    initial_momentum = initial_momentum,
    final_momentum = final_momentum,
    min_gain = min_gain,
    max_step_norm = max_step_norm,
    negative_gradient_method = negative_gradient_method,
    record_costs = record_costs,
    n_threads = n_threads,
    seed = seed,
    verbose = verbose,
    backend = backend,
    auto_config = auto_config,
    input_had_self = knn$has_self,
    input_backend = knn$input_backend
  )
}

#' Precompute reusable openTSNE KNN state
#'
#' `prepare_opentsne_knn()` strips self-neighbours, trims to the requested
#' non-self width, and stores compact KNN matrices once. Pass the returned
#' object to [opentsne_knn()] for repeated seeds or backend comparisons without
#' repeating KNN normalization/materialization.
#'
#' @inheritParams opentsne_knn
#' @return A prepared openTSNE KNN object.
#' @examples
#' x <- scale(as.matrix(iris[, 1:4]))
#' knn <- faissR::nn(x, k = 15, exclude_self = TRUE)
#' prep <- prepare_opentsne_knn(knn, perplexity = 10)
#' y1 <- opentsne_knn(prep, seed = 1, early_exaggeration_iter = 50, n_iter = 100)
#' y2 <- opentsne_knn(prep, seed = 2, early_exaggeration_iter = 50, n_iter = 100)
#' @export
prepare_opentsne_knn <- function(indices,
                                 distances = NULL,
                                 n_neighbors = NULL,
                                 perplexity = NULL) {
  knn0 <- coerce_knn_input(indices, distances)
  policy <- opentsne_neighbor_policy(
    nrow(knn0$indices),
    perplexity = perplexity,
    available = knn0$n_neighbors
  )
  if (is.null(n_neighbors)) {
    n_neighbors <- policy$n_neighbors
  }
  knn <- normalize_opentsne_knn_input(indices, distances, n_neighbors)
  out <- list(
    knn = knn,
    perplexity = policy$perplexity,
    n_neighbors = as.integer(n_neighbors),
    affinity_state = "knn_materialized_affinity_builder_internal"
  )
  class(out) <- c("fastEmbedR_opentsne_prepared", "list")
  out
}

#' Run native openTSNE-style t-SNE from precomputed KNN
#'
#' `opentsne_knn()` is the direct KNN-input entry point for the native
#' openTSNE-style optimizer. It accepts either an object returned by
#' [faissR::nn()] or
#' separate KNN index and distance matrices. No neighbour search, scaling, or
#' PCA is done inside this function.
#'
#' @param indices A KNN object returned by [faissR::nn()], or an integer KNN index
#'   matrix.
#' @param distances Numeric KNN distance matrix matching `indices`. Leave
#'   `NULL` when `indices` is a [faissR::nn()] result.
#' @param n_neighbors Optional number of non-self neighbor columns to use from
#'   the supplied KNN graph. This lets you compute a wide KNN once and reuse
#'   its first columns for comparable tests.
#' @param perplexity t-SNE perplexity. If `NULL`, the optimizer chooses a safe
#'   value from the supplied KNN width and sample size.
#' @param init_data Optional original high-dimensional data matrix used only to
#'   compute PCA initialization for KNN-input runs. It is not used for neighbor
#'   search or optimization.
#' @inheritParams opentsne
#' @return A numeric embedding matrix with settings stored in
#'   `attr(layout, "fastEmbedR_config")`.
#' @examples
#' x <- scale(as.matrix(iris[, 1:4]))
#' knn <- faissR::nn(x, k = 31, exclude_self = TRUE)
#' layout <- opentsne_knn(knn, init_data = x, perplexity = 10,
#'   early_exaggeration_iter = 100, n_iter = 250)
#' if (all(is.finite(layout))) {
#'   plot(layout, pch = 21, bg = iris$Species)
#' }
#' @export
opentsne_knn <- function(indices,
                         distances = NULL,
                         n_neighbors = NULL,
                         perplexity = NULL,
                         n_components = 2L,
                         init_data = NULL,
                         Y_init = NULL,
                         seed = 4L,
                         verbose = FALSE,
                         backend = c("cpu", "cuda", "metal"),
                          n_threads = NULL,
                          learning_rate = "auto",
                          early_exaggeration_iter = NULL,
                          early_exaggeration = "auto",
                          n_iter = NULL,
                          exaggeration = NULL,
                         initial_momentum = 0.8,
                         final_momentum = 0.8,
                          max_step_norm = "auto",
                          negative_gradient_method = "auto",
                          record_costs = FALSE,
                          auto_config = TRUE,
                          ...) {
  backend <- resolve_embedding_backend(backend)
  if (inherits(indices, "fastEmbedR_opentsne_prepared")) {
    if (!is.null(distances)) {
      stop("Do not pass `distances` when `indices` is a prepared openTSNE object.", call. = FALSE)
    }
    knn <- indices$knn
    if (is.null(perplexity)) perplexity <- indices$perplexity
  } else {
    knn <- normalize_opentsne_knn_input(indices, distances, n_neighbors)
  }
  Y_init <- resolve_opentsne_y_init(Y_init, knn$n, n_components)
  if (is.null(Y_init) && !is.null(init_data)) {
    init_backend <- if (backend %in% c("metal", "cuda")) backend else "cpu"
    Y_init <- make_opentsne_pca_init_from_data(
      init_data,
      n = knn$n,
      n_components = n_components,
      seed = seed,
      backend = init_backend
    )
  }
  fast_knn_opentsne_materialized(
    knn$indices,
    knn$distances,
    n_components = n_components,
    perplexity = perplexity,
    Y_init = Y_init,
    seed = seed,
    verbose = verbose,
    backend = backend,
    n_threads = n_threads,
    learning_rate = learning_rate,
    early_exaggeration_iter = early_exaggeration_iter,
    early_exaggeration = early_exaggeration,
    n_iter = n_iter,
    exaggeration = exaggeration,
    initial_momentum = initial_momentum,
    final_momentum = final_momentum,
    max_step_norm = max_step_norm,
     negative_gradient_method = negative_gradient_method,
     record_costs = record_costs,
     auto_config = auto_config,
     input_had_self = knn$has_self,
    input_backend = knn$input_backend,
    ...
  )
}

#' Run native openTSNE-style t-SNE from a data matrix
#'
#' `opentsne()` computes or reuses a KNN graph, then runs the package-native
#' openTSNE-style two-phase optimizer. The default optimizer is CPU-native.
#' Explicit `backend = "metal"` and `backend = "cuda"` requests use the
#' matching package-native GPU optimizer when compiled and fail clearly if the
#' requested backend is unavailable.
#'
#' @param data Numeric matrix/data frame with observations in rows, or a KNN
#'   object returned by [faissR::nn()].
#' @param perplexity t-SNE perplexity. The one-call API uses
#'   `ceiling(perplexity)` non-self neighbours internally. If `NULL`, uses the
#'   largest safe value up to 30 that is available for the input.
#' @param n_components Output dimensionality, from 1 to 3.
#' @param init_data Optional original high-dimensional data matrix used only to
#'   compute PCA initialization with [opentsne_pca_init()]. It is not used for
#'   neighbour search or optimization.
#' @param Y_init Optional explicit initial layout. Use [opentsne_pca_init()] to
#'   precompute and reuse a PCA initialization.
#' @param standardize Center and scale columns before KNN. Defaults to `FALSE`
#'   so one-call results match a KNN object computed from the supplied matrix.
#' @param pca_dims Optional PCA dimension before KNN.
#' @param metric KNN distance metric for one-call matrix input.
#' @param nn Optional precomputed KNN output when `data` is a data matrix.
#' @param seed Random seed.
#' @param backend Execution backend: `"cpu"`, `"cuda"`, or `"metal"`. For
#'   matrix input, KNN is delegated to faissR with automatic method/tuning
#'   selection through an internal bridge. CPU and Metal request faissR CPU
#'   HNSW with `target_recall = 0.99`; CUDA requests faissR CUDA
#'   `method = "auto"` with `target_recall = 0.99`.
#'   Unsupported GPU requests fail clearly and are not relabelled CPU runs.
#' @param keep_knn If `TRUE`, retain KNN matrices in the returned object.
#' @param verbose Print optimizer progress.
#' @param n_threads Number of CPU worker threads used by CPU KNN and CPU
#'   openTSNE optimization. Native GPU optimizers ignore this argument.
#' @param learning_rate Positive number or `"auto"`. With `"auto"`, the native
#'   optimizer uses `n / exaggeration` separately for each phase.
#' @param early_exaggeration_iter Number of early-exaggeration iterations.
#' @param early_exaggeration Early-exaggeration multiplier, or `"auto"` for
#'   openTSNE's default rule.
#' @param n_iter Number of normal optimization iterations after early
#'   exaggeration.
#' @param exaggeration Normal-phase exaggeration. `NULL` means 1.
#' @param initial_momentum Momentum during early exaggeration.
#' @param final_momentum Momentum during normal optimization.
#' @param max_step_norm Maximum per-point update norm. `"auto"` uses the
#'   standard CPU limit and a tighter native Metal FFT-grid limit to avoid
#'   float32 outlier steps. Use `NULL` or `NA` to disable clipping.
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
#' @param ... Additional low-level parameters passed to [opentsne_knn()].
#' @return A `fastEmbedR_embedding` object.
#' @export
opentsne <- function(data,
                     perplexity = NULL,
                     n_components = 2L,
                     init_data = NULL,
                     Y_init = NULL,
                     standardize = FALSE,
                     pca_dims = NULL,
                     metric = c("euclidean", "cosine"),
                     nn = NULL,
                     seed = 4L,
                     backend = c("cpu", "cuda", "metal"),
                     keep_knn = FALSE,
                     verbose = FALSE,
                      n_threads = NULL,
                      learning_rate = "auto",
                      early_exaggeration_iter = NULL,
                      early_exaggeration = "auto",
                      n_iter = NULL,
                     exaggeration = NULL,
                     initial_momentum = 0.8,
                     final_momentum = 0.8,
                     max_step_norm = "auto",
                      negative_gradient_method = "auto",
                      record_costs = FALSE,
                      auto_config = TRUE,
                      ...) {
  backend <- resolve_embedding_backend(backend)
  input_is_float32 <- is_float32_matrix(data)
  dots <- list(...)
  if ("init" %in% names(dots)) {
    stop(
      "`init` is not an argument of `opentsne()`; use `Y_init` or `init_data` ",
      "for PCA initialization.",
      call. = FALSE
    )
  }
  if ("n_neighbors" %in% names(dots)) {
    stop(
      "`n_neighbors` is not an argument of `opentsne()`; use `perplexity`, ",
      "which also determines the internal non-self KNN width.",
      call. = FALSE
    )
  }
  optimizer_backend <- backend
  if (is_knn_input(data)) {
    if (!is.null(nn)) {
      stop("When `data` is a KNN object, do not also pass `nn`.", call. = FALSE)
    }
    full_knn <- normalize_opentsne_knn_input(data, NULL, NULL)
    n <- full_knn$n
    neighbour_policy <- opentsne_neighbor_policy(
      n,
      perplexity = perplexity,
      available = full_knn$n_neighbors
    )
    perplexity <- neighbour_policy$perplexity
    knn_result <- normalize_opentsne_knn_input(
      data,
      NULL,
      neighbour_policy$n_neighbors
    )
    Y_init <- resolve_opentsne_y_init(Y_init, n, n_components)
    if (is.null(Y_init) && !is.null(init_data)) {
      init_backend <- if (optimizer_backend %in% c("metal", "cuda")) optimizer_backend else "cpu"
      Y_init <- make_opentsne_pca_init_from_data(
        init_data,
        n = n,
        n_components = n_components,
        seed = seed,
        backend = init_backend
      )
    }

    embedding_time <- system.time({
      layout <- opentsne_knn(
        data,
        n_neighbors = knn_result$n_neighbors,
        n_components = n_components,
        perplexity = perplexity,
        init_data = init_data,
        Y_init = Y_init,
        seed = seed,
        verbose = verbose,
        backend = optimizer_backend,
        n_threads = n_threads,
        learning_rate = learning_rate,
        early_exaggeration_iter = early_exaggeration_iter,
        early_exaggeration = early_exaggeration,
        n_iter = n_iter,
        exaggeration = exaggeration,
        initial_momentum = initial_momentum,
        final_momentum = final_momentum,
        max_step_norm = max_step_norm,
         negative_gradient_method = negative_gradient_method,
         record_costs = record_costs,
         auto_config = auto_config,
        ...
      )
    })
    cfg <- attr(layout, "fastEmbedR_config")
    zero_time <- embedding_time
    zero_time[] <- 0
    timings <- rbind(
      preprocess = zero_time,
      knn = zero_time,
      embedding = embedding_time
    )
    knn_backend <- knn_result$input_backend
    if (is.na(knn_backend) || is.null(knn_backend)) knn_backend <- "supplied"
    metrics <- data.frame(
      method = "opentsne",
      n = n,
      p = NA_integer_,
      n_neighbors = knn_result$n_neighbors,
      perplexity = cfg$perplexity,
      elapsed = sum(timings[, "elapsed"]),
      preprocess_elapsed = 0,
      knn_elapsed = 0,
      embedding_elapsed = embedding_time["elapsed"],
      stringsAsFactors = FALSE
    )
    parameters <- c(
      list(
        method = "opentsne",
        input = "knn",
        n = n,
        p = NA_integer_,
        n_neighbors = knn_result$n_neighbors,
        k = knn_result$n_neighbors + 1L,
        n_components = as.integer(n_components),
        seed = as.integer(seed),
        nn_backend = knn_backend,
        keep_knn = keep_knn
      ),
      cfg,
      list(preprocess = "none_precomputed_knn")
    )
    out <- list(
      layout = layout,
      labels = NULL,
      method = "opentsne",
      metrics = metrics,
      parameters = parameters,
      timings = timings,
      knn = if (isTRUE(keep_knn)) {
        list(indices = knn_result$indices, distances = knn_result$distances)
      } else {
        NULL
      },
      knn_with_self = NULL,
      preprocess = list(input = "precomputed_knn"),
      diagnostics = list(
        metal_trace = attr(layout, "metal_trace"),
        metal_stage_timing = attr(layout, "metal_stage_timing")
      )
    )
    class(out) <- "fastEmbedR_embedding"
    return(out)
  }

  preprocess_time <- system.time({
    prepared <- prepare_embedding_data(
      data,
      standardize,
      pca_dims,
      seed,
      backend = resolve_preprocess_backend(backend)
    )
  })
  x <- prepared$data
  metric <- resolve_embedding_metric(metric, x)
  n <- nrow(x)
  neighbour_policy <- opentsne_neighbor_policy(n, perplexity = perplexity)
  perplexity <- neighbour_policy$perplexity
  n_neighbors <- neighbour_policy$n_neighbors

  knn_time <- system.time({
    if (is.null(nn)) {
      knn_policy <- fastembedr_embedding_nn_policy(backend)
      raw_knn <- fastembedr_nn_without_self(
        x,
        k = n_neighbors,
        backend = knn_policy$backend,
        method = knn_policy$method,
        metric = metric,
        output = fastembedr_faiss_float_output(x, knn_policy$backend),
        n_threads = n_threads,
        tuning = knn_policy$tuning,
        target_recall = knn_policy$target_recall
      )
      knn_result <- normalize_supplied_knn(raw_knn, n, n_neighbors)
      knn_result$nn_backend <- attr(raw_knn, "backend")
      embedding_knn_input <- raw_knn
    } else {
      knn_result <- normalize_supplied_knn(nn, n, n_neighbors, keep_self = keep_knn)
      knn_result$nn_backend <- attr(nn, "backend")
      if (is.null(knn_result$nn_backend)) knn_result$nn_backend <- "supplied"
      embedding_knn_input <- nn
    }
  })

  Y_init <- resolve_opentsne_y_init(Y_init, n, n_components)
  init_info <- list(method = "user", backend = NA_character_)
  if (is.null(Y_init) && !is.null(init_data)) {
    init_backend <- if (optimizer_backend %in% c("metal", "cuda")) optimizer_backend else "cpu"
    Y_init <- tryCatch(
      make_opentsne_pca_init_from_data(
        init_data,
        n = n,
        n_components = n_components,
        seed = seed,
        backend = init_backend
      ),
      error = function(e) {
        init_info$error <<- conditionMessage(e)
        NULL
      }
    )
    if (!is.null(Y_init)) {
      init_info$method <- attr(Y_init, "fastEmbedR_init_method") %||% "pca"
      init_info$backend <- attr(Y_init, "fastEmbedR_init_backend") %||% init_backend
    } else {
      init_info$method <- "random_fallback_after_pca_error"
      init_info$backend <- "cpu"
    }
  } else if (is.null(Y_init)) {
    init_info$method <- "knn_native_default"
    init_info$backend <- optimizer_backend
  }

  embedding_time <- system.time({
    layout <- opentsne_knn(
      embedding_knn_input,
      n_neighbors = knn_result$n_neighbors,
      n_components = n_components,
      perplexity = perplexity,
      init_data = init_data,
      Y_init = Y_init,
      seed = seed,
      verbose = verbose,
      backend = optimizer_backend,
      n_threads = n_threads,
      learning_rate = learning_rate,
      early_exaggeration_iter = early_exaggeration_iter,
      early_exaggeration = early_exaggeration,
      n_iter = n_iter,
      exaggeration = exaggeration,
      initial_momentum = initial_momentum,
      final_momentum = final_momentum,
      max_step_norm = max_step_norm,
            negative_gradient_method = negative_gradient_method,
            record_costs = record_costs,
            auto_config = auto_config,
      ...
    )
  })
  layout <- finalize_embedding_layout(
    layout,
    "openTSNE",
    return_float32 = input_is_float32 && is_float32_matrix(x)
  )
  cfg <- attr(layout, "fastEmbedR_config")
  timings <- rbind(
    preprocess = preprocess_time,
    knn = knn_time,
    embedding = embedding_time
  )
  metrics <- data.frame(
    method = "opentsne",
    n = n,
    p = ncol(x),
    n_neighbors = knn_result$n_neighbors,
    perplexity = cfg$perplexity,
    elapsed = sum(timings[, "elapsed"]),
    preprocess_elapsed = preprocess_time["elapsed"],
    knn_elapsed = knn_time["elapsed"],
    embedding_elapsed = embedding_time["elapsed"],
    stringsAsFactors = FALSE
  )
  parameters <- c(
    list(
      method = "opentsne",
      n = n,
      p = ncol(x),
      n_neighbors = knn_result$n_neighbors,
      k = knn_result$n_neighbors + 1L,
      n_components = as.integer(n_components),
      seed = as.integer(seed),
      nn_backend = knn_result$nn_backend,
      metric = metric,
      keep_knn = keep_knn
    ),
    cfg,
    prepared$preprocess,
    list(init = init_info$method, init_backend = init_info$backend)
  )
  out <- list(
    layout = layout,
    labels = NULL,
    method = "opentsne",
    metrics = metrics,
    parameters = parameters,
    timings = timings,
    knn = if (isTRUE(keep_knn)) {
      list(indices = knn_result$indices, distances = knn_result$distances)
    } else {
      NULL
    },
    knn_with_self = if (isTRUE(keep_knn)) knn_result$knn_with_self else NULL,
    preprocess = prepared$preprocess,
    diagnostics = list(
      metal_trace = attr(layout, "metal_trace"),
      metal_stage_timing = attr(layout, "metal_stage_timing")
    )
  )
  class(out) <- "fastEmbedR_embedding"
  out
}
