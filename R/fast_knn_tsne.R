is_whole_number <- function(x, tol = .Machine$double.eps^0.5) {
  abs(x - round(x)) < tol
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
      "`\"fft\"` until native Metal/CUDA FFT is implemented.",
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
  list(
    indices = materialized$indices,
    distances = materialized$distances,
    n = n,
    n_neighbors = as.integer(n_neighbors),
    has_self = isTRUE(knn$has_self),
    input_backend = knn$input_backend
  )
}

fast_knn_opentsne_materialized <- function(indices,
                                           distances,
                                           n_components = 2L,
                                           perplexity = NULL,
                                           theta = 0.5,
                                           early_exaggeration_iter = 250L,
                                           n_iter = 500L,
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
                                           backend = c("auto", "cpu", "gpu", "metal", "cuda"),
                                           input_had_self = FALSE,
                                           input_backend = NA_character_) {
  backend <- match.arg(backend)
  optimizer_backend <- if (backend %in% c("auto", "cpu")) {
    "cpu"
  } else if (backend %in% c("gpu", "metal")) {
    resolved_gpu <- if (identical(backend, "gpu")) {
      resolve_backend_request("gpu", need_embedding = TRUE)
    } else {
      "metal"
    }
    if (identical(resolved_gpu, "metal") &&
        (!embedding_metal_available_cpp() || !metal_opentsne_native_available())) {
      stop(
        "Native Metal openTSNE optimizer was requested, but it is not available in this build. ",
        "No CPU fallback is used for GPU-labelled runs.",
        call. = FALSE
      )
    }
    if (identical(resolved_gpu, "cuda") &&
        (!embedding_cuda_available_cpp() || !cuda_opentsne_native_available())) {
      stop(
        "Native CUDA openTSNE optimizer was requested, but it is not available in this build. ",
        "No CPU fallback is used for GPU-labelled runs.",
        call. = FALSE
      )
    }
    resolved_gpu
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
  if (is.null(perplexity)) {
    perplexity <- auto_tsne_perplexity(n, k)
  }
  if (is.null(n_threads)) {
    n_threads <- default_tsne_threads()
  }
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
  negative_gradient_method <- normalize_tsne_negative_gradient_method(negative_gradient_method)
  if (identical(optimizer_backend, "metal")) {
    if (identical(negative_gradient_method, "auto")) {
      negative_gradient_method <- "fft"
    }
  } else if (identical(optimizer_backend, "cuda")) {
    if (identical(negative_gradient_method, "auto")) {
      negative_gradient_method <- "fft"
    }
    if (identical(negative_gradient_method, "fft")) {
      stop(
        "openTSNE FFT/FIt-SNE interpolation is not yet ported to native CUDA. ",
        "Use CPU `negative_gradient_method = \"fft\"` for large runs, or ",
        "`\"exact\"` for small `backend = \"cuda\"` reference checks.",
        call. = FALSE
      )
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
    list(Y_init = Y_init, method = "user", spectral_n_iter = NA_integer_)
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
    knn_tsne_opentsne_metal_cpp(
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
  } else if (identical(optimizer_backend, "cuda")) {
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
      record_costs
    )
  }
  layout <- set_embedding_colnames(out$Y, "openTSNE")
  probabilities <- out$probabilities
  if (is.null(probabilities)) probabilities <- "symmetric_sparse_knn_cpu"
  n_negatives <- out$n_negatives
  if (is.null(n_negatives)) n_negatives <- NA_integer_
  repulsion_block_size <- out$repulsion_block_size
  if (is.null(repulsion_block_size)) repulsion_block_size <- NA_integer_
  cfg <- list(
    method = "opentsne",
    backend = optimizer_backend,
    n = n,
    n_neighbors = as.integer(k),
    perplexity = args$perplexity,
    theta = args$theta,
    early_exaggeration_iter = early_exaggeration_iter,
    n_iter = n_iter,
    max_iter = early_exaggeration_iter + n_iter,
    learning_rate = if (isTRUE(lr$auto)) "auto" else lr$value,
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
    record_costs = record_costs,
    optimizer = out$optimizer,
    repulsion = out$repulsion,
    probabilities = probabilities,
    n_negatives = n_negatives,
    repulsion_block_size = repulsion_block_size,
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
  attr(layout, "fastEmbedR_config") <- cfg
  attr(layout, "costs") <- out$costs
  attr(layout, "itercosts") <- out$itercosts
  layout
}

fast_knn_opentsne_core <- function(indices,
                                   distances = NULL,
                                   n_components = 2L,
                                   perplexity = NULL,
                                   theta = 0.5,
                                   early_exaggeration_iter = 250L,
                                   n_iter = 500L,
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
                                   backend = c("auto", "cpu", "gpu", "metal", "cuda")) {
  backend <- match.arg(backend)
  knn <- normalize_opentsne_knn_input(indices, distances)
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
    input_had_self = knn$has_self,
    input_backend = knn$input_backend
  )
}

#' Run native openTSNE-style t-SNE from precomputed KNN
#'
#' `opentsne_knn()` is the direct KNN-input entry point for the native
#' openTSNE-style optimizer. It accepts either an object returned by [nn()] or
#' separate KNN index and distance matrices. No neighbour search, scaling, or
#' PCA is done inside this function.
#'
#' @param indices A KNN object returned by [nn()], or an integer KNN index
#'   matrix.
#' @param distances Numeric KNN distance matrix matching `indices`. Leave
#'   `NULL` when `indices` is an [nn()] result.
#' @param n_neighbors Optional number of non-self neighbor columns to use from
#'   the supplied KNN graph. This lets you compute a wide KNN once and reuse
#'   its first columns for comparable tests.
#' @inheritParams opentsne
#' @return A numeric embedding matrix with settings stored in
#'   `attr(layout, "fastEmbedR_config")`.
#' @examples
#' x <- scale(as.matrix(iris[, 1:4]))
#' knn <- nn(x, k = 31)
#' layout <- opentsne_knn(knn, perplexity = 10,
#'   early_exaggeration_iter = 100, n_iter = 250)
#' plot(layout, pch = 21, bg = iris$Species)
#' @export
opentsne_knn <- function(indices,
                         distances = NULL,
                         n_neighbors = NULL,
                         perplexity = NULL,
                         n_components = 2L,
                         seed = 4L,
                         verbose = FALSE,
                         backend = c("auto", "cpu", "gpu", "metal", "cuda"),
                         n_threads = NULL,
                         learning_rate = "auto",
                         early_exaggeration_iter = 250L,
                         early_exaggeration = "auto",
                         n_iter = 500L,
                         exaggeration = NULL,
                         initial_momentum = 0.8,
                         final_momentum = 0.8,
                         max_step_norm = "auto",
                         negative_gradient_method = "auto",
                         record_costs = FALSE,
                         ...) {
  backend <- match.arg(backend)
  knn <- normalize_opentsne_knn_input(indices, distances, n_neighbors)
  fast_knn_opentsne_materialized(
    knn$indices,
    knn$distances,
    n_components = n_components,
    perplexity = perplexity,
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
    input_had_self = knn$has_self,
    input_backend = knn$input_backend,
    ...
  )
}

#' Run native openTSNE-style t-SNE from a data matrix
#'
#' `opentsne()` computes or reuses a KNN graph, then runs the package-native
#' openTSNE-style two-phase optimizer. The default optimizer is CPU-native.
#' Explicit `backend = "metal"` requests use the package-native Metal optimizer
#' and fail clearly if Metal is unavailable; CUDA optimizer requests are never
#' silently relabelled as CPU.
#'
#' @param data Numeric matrix/data frame with observations in rows, or a KNN
#'   object returned by [nn()].
#' @param labels Optional labels used only for scoring and plotting metadata.
#' @param n_neighbors Number of non-self neighbors. If `NULL`, the package uses
#'   `3 * perplexity`, matching the usual t-SNE neighbour convention.
#' @param perplexity t-SNE perplexity. If `NULL`, uses
#'   `min(30, floor((n - 1) / 3), floor(k / 3))`.
#' @param n_components Output dimensionality, from 1 to 3.
#' @param standardize Center and scale columns before KNN.
#' @param pca_dims Optional PCA dimension before KNN.
#' @param nn Optional precomputed KNN output when `data` is a data matrix.
#' @param seed Random seed.
#' @param backend KNN/preprocessing backend and, when the matching native
#'   openTSNE symbols are compiled, a real GPU optimizer backend. Unsupported
#'   GPU requests fail clearly and are not relabelled CPU runs. `backend =
#'   "auto"` keeps the CPU optimizer.
#' @param silhouette_sample Optional sample size for silhouette scoring.
#' @param preserve_sample Optional sample size for neighborhood scoring.
#' @param preserve_k Number of neighbors used for neighborhood scoring.
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
#' @param ... Additional low-level parameters passed to [opentsne_knn()].
#' @return A `fastEmbedR_embedding` object.
#' @export
opentsne <- function(data,
                     labels = NULL,
                     n_neighbors = NULL,
                     perplexity = NULL,
                     n_components = 2L,
                     standardize = TRUE,
                     pca_dims = NULL,
                     nn = NULL,
                     seed = 4L,
                     backend = c("auto", "cpu", "gpu", "metal", "cuda",
                                 "cuvs", "gpu_cuvs", "cuda_cuvs",
                                 "cuda_cuvs_cagra", "cuda_cuvs_bruteforce",
                                 "cuda_cuvs_exact", "cuda_cuvs_nndescent",
                                 "cuvs_bruteforce", "cuvs_nndescent"),
                     silhouette_sample = NULL,
                     preserve_sample = NULL,
                     preserve_k = NULL,
                     keep_knn = FALSE,
                     verbose = FALSE,
                     n_threads = NULL,
                     learning_rate = "auto",
                     early_exaggeration_iter = 250L,
                     early_exaggeration = "auto",
                     n_iter = 500L,
                     exaggeration = NULL,
                     initial_momentum = 0.8,
                     final_momentum = 0.8,
                     max_step_norm = "auto",
                     negative_gradient_method = "auto",
                     record_costs = FALSE,
                     ...) {
  backend <- match.arg(backend)
  optimizer_backend <- if (identical(backend, "gpu")) {
    resolve_backend_request("gpu", need_embedding = TRUE)
  } else if (backend %in% c("metal", "cuda")) {
    backend
  } else {
    "cpu"
  }
  if (is_knn_input(data)) {
    if (!is.null(nn)) {
      stop("When `data` is a KNN object, do not also pass `nn`.", call. = FALSE)
    }
    if (backend %in% c("cuvs", "gpu_cuvs", "cuda_cuvs",
                       "cuda_cuvs_cagra", "cuda_cuvs_bruteforce",
                       "cuda_cuvs_exact", "cuda_cuvs_nndescent",
                       "cuvs_bruteforce", "cuvs_nndescent")) {
      stop(
        "cuVS backends are KNN backends, not OpenTSNE optimizer backends. ",
        "Pass a plain KNN object with `backend = \"cuda\"` for native CUDA optimization.",
        call. = FALSE
      )
    }
    knn_result <- normalize_opentsne_knn_input(data, NULL, n_neighbors)
    n <- knn_result$n
    if (!is.null(labels) && length(labels) != n) {
      stop("`labels` must have one entry per KNN row.", call. = FALSE)
    }

    embedding_time <- system.time({
      layout <- fast_knn_opentsne_materialized(
        knn_result$indices,
        knn_result$distances,
        n_components = n_components,
        perplexity = perplexity,
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
        input_had_self = knn_result$has_self,
        input_backend = knn_result$input_backend,
        ...
      )
    })
    cfg <- attr(layout, "fastEmbedR_config")
    score_backend <- if (cfg$backend %in% c("metal", "cuda")) cfg$backend else "cpu"
    score_preserve_k <- if (is.null(preserve_k)) ncol(knn_result$indices) else {
      min(as.integer(preserve_k), ncol(knn_result$indices))
    }
    scores <- embedding_scores(
      layout,
      labels,
      knn_result$indices,
      silhouette_sample,
      preserve_sample,
      score_preserve_k,
      seed,
      backend = score_backend
    )
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
      scores,
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
      labels = labels,
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
      preprocess = list(input = "precomputed_knn")
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
  n <- nrow(x)
  if (!is.null(labels) && length(labels) != n) {
    stop("`labels` must have one entry per row of `data`.", call. = FALSE)
  }
  if (is.null(n_neighbors)) {
    n_neighbors <- auto_tsne_k(n, perplexity)
  } else {
    n_neighbors <- as.integer(n_neighbors)
    if (length(n_neighbors) != 1L || is.na(n_neighbors) || n_neighbors < 1L || n_neighbors >= n) {
      stop("`n_neighbors` must be a positive integer smaller than `nrow(data)`.", call. = FALSE)
    }
  }

  knn_time <- system.time({
    if (is.null(nn)) {
      raw_knn <- nn_without_self(
        x,
        k = n_neighbors,
        backend = backend,
        n_threads = n_threads
      )
      knn_result <- normalize_supplied_knn(raw_knn, n, n_neighbors)
      knn_result$nn_backend <- attr(raw_knn, "backend")
    } else {
      knn_result <- normalize_supplied_knn(nn, n, n_neighbors, keep_self = keep_knn)
      knn_result$nn_backend <- attr(nn, "backend")
      if (is.null(knn_result$nn_backend)) knn_result$nn_backend <- "supplied"
    }
  })

  embedding_time <- system.time({
    layout <- fast_knn_opentsne_materialized(
      knn_result$indices,
      knn_result$distances,
      n_components = n_components,
      perplexity = perplexity,
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
      input_had_self = knn_result$has_self,
      input_backend = knn_result$nn_backend,
      ...
    )
  })
  cfg <- attr(layout, "fastEmbedR_config")
  score_backend <- if (cfg$backend %in% c("metal", "cuda")) cfg$backend else "cpu"
  score_preserve_k <- if (is.null(preserve_k)) ncol(knn_result$indices) else {
    min(as.integer(preserve_k), ncol(knn_result$indices))
  }
  scores <- embedding_scores(
    layout,
    labels,
    knn_result$indices,
    silhouette_sample,
    preserve_sample,
    score_preserve_k,
    seed,
    backend = score_backend
  )
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
    scores,
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
      keep_knn = keep_knn
    ),
    cfg,
    prepared$preprocess
  )
  out <- list(
    layout = layout,
    labels = labels,
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
    preprocess = prepared$preprocess
  )
  class(out) <- "fastEmbedR_embedding"
  out
}
