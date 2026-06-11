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

normalize_tsne_negative_gradient_method <- function(method) {
  method <- tolower(gsub("-", "_", as.character(method)))
  if (length(method) != 1L || is.na(method)) {
    stop("`negative_gradient_method` must be a single string.", call. = FALSE)
  }
  aliases <- c(
    auto = "auto",
    bh = "bh",
    barnes_hut = "bh",
    barnes = "bh",
    exact = "exact",
    pair = "exact",
    pair_symmetric = "exact",
    fft = "fft",
    interpolation = "fft",
    fitsne = "fft",
    fit_sne = "fft",
    keops_blocked = "keops_blocked",
    blocked = "keops_blocked"
  )
  out <- aliases[[method]]
  if (is.null(out)) {
    stop(
      "`negative_gradient_method` must be one of ",
      "`\"auto\"`, `\"bh\"`, `\"exact\"`, or `\"fft\"`.",
      call. = FALSE
    )
  }
  out
}

resolve_tsne_optimizer_backend <- function(backend) {
  backend <- match.arg(backend, c("auto", "cpu", "gpu", "metal", "cuda"))
  if (backend %in% c("auto", "cpu")) {
    return("cpu")
  }
  if (identical(backend, "metal")) {
    stop(
      "Native full t-SNE from KNN is not supported on Metal yet. ",
      "Use `transform_tsne(..., backend = \"metal\")` or ",
      "`landmark_tsne(..., backend = \"metal\")` for the native Metal ",
      "fixed-reference transform path.",
      call. = FALSE
    )
  }
  if (identical(backend, "gpu") && !isTRUE(embedding_cuda_available_cpp())) {
    stop(
      "Native full t-SNE from KNN currently requires CUDA. ",
      "No CUDA embedding backend is available, and fastEmbedR will not ",
      "run CPU code and report it as GPU.",
      call. = FALSE
    )
  }
  if (!isTRUE(embedding_cuda_available_cpp())) {
    stop(
      "CUDA exact t-SNE is available only when fastEmbedR is built with ",
      "CUDA support and a CUDA device is visible.",
      call. = FALSE
    )
  }
  "cuda"
}

random_tsne_init <- function(n, n_components, seed) {
  had_seed <- exists(".Random.seed", envir = .GlobalEnv, inherits = FALSE)
  old_seed <- if (had_seed) {
    get(".Random.seed", envir = .GlobalEnv, inherits = FALSE)
  } else {
    NULL
  }
  on.exit({
    if (had_seed) {
      assign(".Random.seed", old_seed, envir = .GlobalEnv)
    } else if (exists(".Random.seed", envir = .GlobalEnv, inherits = FALSE)) {
      rm(".Random.seed", envir = .GlobalEnv)
    }
  }, add = TRUE)
  resolved_seed <- if (length(seed) == 1L && !is.na(seed)) as.integer(seed) else 5489L
  set.seed(resolved_seed)
  matrix(stats::rnorm(n * n_components, sd = 1e-4), nrow = n, ncol = n_components)
}

# Parameter checks and defaults intentionally mirror Rtsne::Rtsne_neighbors()
# from Rtsne 0.17, which is BSD-style licensed. The native optimizer called
# below is fastEmbedR code and does not vendor Rtsne's Barnes-Hut C++ files.
check_tsne_neighbor_params <- function(n,
                                       n_components,
                                       perplexity,
                                       theta,
                                       max_iter,
                                       verbose,
                                       Y_init,
                                       stop_lying_iter,
                                       mom_switch_iter,
                                       momentum,
                                       final_momentum,
                                       eta,
                                       exaggeration_factor) {
  if (!is_whole_number(n_components) || n_components < 1L || n_components > 3L) {
    stop("`n_components` should be 1, 2, or 3.", call. = FALSE)
  }
  if (!is_whole_number(max_iter) || max_iter <= 0L) {
    stop("`max_iter` should be a positive integer.", call. = FALSE)
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
  if (!is_whole_number(stop_lying_iter) || stop_lying_iter < 0L) {
    stop("`stop_lying_iter` should be a non-negative integer.", call. = FALSE)
  }
  if (!is_whole_number(mom_switch_iter) || mom_switch_iter < 0L) {
    stop("`mom_switch_iter` should be a non-negative integer.", call. = FALSE)
  }
  if (!is.numeric(momentum) || momentum < 0) {
    stop("`momentum` should be non-negative.", call. = FALSE)
  }
  if (!is.numeric(final_momentum) || final_momentum < 0) {
    stop("`final_momentum` should be non-negative.", call. = FALSE)
  }
  if (!is.numeric(eta) || eta <= 0) {
    stop("`eta` should be positive.", call. = FALSE)
  }
  if (!is.numeric(exaggeration_factor) || exaggeration_factor <= 0) {
    stop("`exaggeration_factor` should be positive.", call. = FALSE)
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
    stop_lying_iter = as.integer(stop_lying_iter),
    mom_switch_iter = as.integer(mom_switch_iter),
    momentum = as.numeric(momentum),
    final_momentum = as.numeric(final_momentum),
    eta = as.numeric(eta),
    exaggeration_factor = as.numeric(exaggeration_factor)
  )
}

fast_knn_tsne_core <- function(indices,
                               distances = NULL,
                               n_components = 2L,
                               perplexity = NULL,
                               theta = 0.5,
                               max_iter = 1000L,
                               Y_init = NULL,
                               stop_lying_iter = if (is.null(Y_init)) 250L else 0L,
                               mom_switch_iter = if (is.null(Y_init)) 250L else 0L,
                               momentum = 0.5,
                               final_momentum = 0.8,
                               eta = 200,
                               exaggeration_factor = 12,
                               negative_gradient_method = "auto",
                               n_threads = NULL,
                               seed = 42L,
                               verbose = FALSE,
                               backend = c("auto", "cpu", "gpu", "metal", "cuda")) {
  backend <- match.arg(backend)
  optimizer_backend <- resolve_tsne_optimizer_backend(backend)
  negative_gradient_method <- normalize_tsne_negative_gradient_method(negative_gradient_method)
  if (identical(negative_gradient_method, "fft")) {
    stop(
      "openTSNE-style FFT/FIt-SNE interpolation is not yet ported to ",
      "native fastEmbedR C++/Metal/CUDA. It will not be run through Python; ",
      "use `negative_gradient_method = \"bh\"` or `\"exact\"` for now.",
      call. = FALSE
    )
  }
  knn <- coerce_knn_input(indices, distances)
  materialized <- materialize_knn_range(
    knn$indices,
    knn$distances,
    knn$col_start,
    knn$n_neighbors
  )
  indices <- materialized$indices
  distances <- materialized$distances
  n <- nrow(indices)
  k <- ncol(indices)
  if (is.null(perplexity)) {
    perplexity <- auto_tsne_perplexity(n, k)
  }
  if (is.null(n_threads)) {
    n_threads <- default_tsne_threads()
  }
  args <- check_tsne_neighbor_params(
    n = n,
    n_components = n_components,
    perplexity = perplexity,
    theta = theta,
    max_iter = max_iter,
    verbose = verbose,
    Y_init = Y_init,
    stop_lying_iter = stop_lying_iter,
    mom_switch_iter = mom_switch_iter,
    momentum = momentum,
    final_momentum = final_momentum,
    eta = eta,
    exaggeration_factor = exaggeration_factor
  )

  if (identical(optimizer_backend, "cuda")) {
    if (args$n_components != 2L) {
      stop("CUDA exact t-SNE currently supports only `n_components = 2`.", call. = FALSE)
    }
    if (k < ceiling(3 * args$perplexity)) {
      stop(
        "CUDA exact t-SNE requires at least `ceiling(3 * perplexity)` ",
        "neighbors in the supplied KNN graph.",
        call. = FALSE
      )
    }
    cuda_init <- if (args$init) {
      args$Y_init
    } else {
      random_tsne_init(n, args$n_components, seed)
    }
    layout <- knn_tsne_exact_cuda_cpp(
      indices,
      distances,
      cuda_init,
      args$max_iter,
      args$perplexity,
      args$eta,
      args$stop_lying_iter,
      args$mom_switch_iter,
      args$momentum,
      args$final_momentum,
      args$exaggeration_factor,
      as.integer(seed)
    )
    out <- list(
      Y = layout,
      costs = NULL,
      itercosts = numeric(),
      optimizer = "cuda_exact_dense_from_knn",
      repulsion = "cuda_exact_dense",
      repulsion_block_size = NA_integer_,
      n_threads = NA_integer_
    )
  } else {
    out <- knn_tsne_rtsne_cpp(
      indices,
      distances,
      args$Y_init,
      args$init,
      args$n_components,
      args$perplexity,
      args$theta,
      args$max_iter,
      args$stop_lying_iter,
      args$mom_switch_iter,
      args$momentum,
      args$final_momentum,
      args$eta,
      args$exaggeration_factor,
      negative_gradient_method,
      as.integer(n_threads),
      as.integer(seed),
      args$verbose
    )
  }
  layout <- set_embedding_colnames(out$Y, "TSNE")
  cfg <- list(
    method = "tsne",
    backend = optimizer_backend,
    n = n,
    n_neighbors = as.integer(k),
    perplexity = args$perplexity,
    theta = args$theta,
    max_iter = args$max_iter,
    stop_lying_iter = args$stop_lying_iter,
    mom_switch_iter = args$mom_switch_iter,
    momentum = args$momentum,
    final_momentum = args$final_momentum,
    eta = args$eta,
    exaggeration_factor = args$exaggeration_factor,
    negative_gradient_method = negative_gradient_method,
    optimizer = out$optimizer,
    repulsion = out$repulsion,
    repulsion_block_size = out$repulsion_block_size,
    n_threads = out$n_threads,
    input_had_self = isTRUE(knn$has_self),
    knn_backend = knn$input_backend,
    provenance = if (identical(optimizer_backend, "cuda")) {
      "native_cuda_exact_tsne_from_knn_cannylab_tsne_cuda_informed"
    } else {
      "openTSNE_Rtsne_neighbors_compatible_clean_cpp_optimizer"
    }
  )
  attr(layout, "fastEmbedR_config") <- cfg
  attr(layout, "costs") <- out$costs
  attr(layout, "itercosts") <- out$itercosts
  layout
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
                                           max_step_norm = 5,
                                           negative_gradient_method = "auto",
                                           record_costs = FALSE,
                                           n_threads = NULL,
                                           seed = 42L,
                                           verbose = FALSE,
                                           backend = c("auto", "cpu", "gpu", "metal", "cuda"),
                                           input_had_self = FALSE,
                                           input_backend = NA_character_) {
  backend <- match.arg(backend)
  if (!backend %in% c("auto", "cpu")) {
    stop(
      "Native openTSNE-style full t-SNE from KNN currently supports only ",
      "`backend = \"cpu\"`. GPU requests do not fall back silently.",
      call. = FALSE
    )
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
  if (identical(negative_gradient_method, "auto")) {
    negative_gradient_method <- "bh"
  }
  if (identical(negative_gradient_method, "fft")) {
    stop(
      "openTSNE FFT/FIt-SNE interpolation is not yet ported to native C++. ",
      "The native openTSNE-style path currently supports `\"bh\"` and `\"exact\"`.",
      call. = FALSE
    )
  }
  if (identical(negative_gradient_method, "keops_blocked")) {
    stop(
      "`negative_gradient_method = \"keops_blocked\"` belongs to the ",
      "experimental `method = \"tsne\"` path, not the openTSNE-style path.",
      call. = FALSE
    )
  }
  args <- check_tsne_neighbor_params(
    n = n,
    n_components = n_components,
    perplexity = perplexity,
    theta = theta,
    max_iter = early_exaggeration_iter + n_iter,
    verbose = verbose,
    Y_init = Y_init,
    stop_lying_iter = 0L,
    mom_switch_iter = 0L,
    momentum = initial_momentum,
    final_momentum = final_momentum,
    eta = 1,
    exaggeration_factor = 1
  )
  lr <- normalize_opentsne_learning_rate(learning_rate)
  ex <- normalize_opentsne_exaggeration(early_exaggeration, exaggeration)
  min_gain <- as.numeric(min_gain)
  if (length(min_gain) != 1L || is.na(min_gain) || !is.finite(min_gain) || min_gain <= 0) {
    stop("`min_gain` must be a positive number.", call. = FALSE)
  }
  record_costs <- isTRUE(record_costs) || isTRUE(verbose)
  max_step_norm <- if (is.null(max_step_norm) || (length(max_step_norm) == 1L && is.na(max_step_norm))) {
    NA_real_
  } else {
    value <- suppressWarnings(as.numeric(max_step_norm))
    if (length(value) != 1L || is.na(value) || !is.finite(value) || value <= 0) {
      stop("`max_step_norm` must be NULL/NA or a positive number.", call. = FALSE)
    }
    value
  }

  out <- knn_tsne_opentsne_cpp(
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
  layout <- set_embedding_colnames(out$Y, "openTSNE")
  cfg <- list(
    method = "opentsne",
    backend = "cpu",
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
    negative_gradient_method = negative_gradient_method,
    record_costs = record_costs,
    optimizer = out$optimizer,
    repulsion = out$repulsion,
    repulsion_block_size = out$repulsion_block_size,
    n_threads = out$n_threads,
    input_had_self = isTRUE(input_had_self),
    knn_backend = input_backend,
    provenance = "openTSNE_native_cpp_two_phase_optimizer_bsd3_informed"
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
                         learning_rate = "auto",
                         early_exaggeration_iter = 250L,
                         early_exaggeration = "auto",
                         n_iter = 500L,
                         exaggeration = NULL,
                         initial_momentum = 0.8,
                         final_momentum = 0.8,
                         max_step_norm = 5,
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

fast_knn_infotsne_core <- function(indices,
                                   distances = NULL,
                                   n_components = 2L,
                                   perplexity = NULL,
                                   max_iter = 1000L,
                                   Y_init = NULL,
                                   early_exaggeration_iter = if (is.null(Y_init)) 250L else 0L,
                                   momentum = 0.5,
                                   final_momentum = 0.8,
                                   learning_rate = NULL,
                                   eta = NULL,
                                   early_exaggeration_coeff = 12,
                                   repulsion_strength = 1,
                                   n_negatives = 300L,
                                   n_threads = NULL,
                                   seed = 42L,
                                   verbose = FALSE,
                                   backend = c("auto", "cpu", "gpu", "metal", "cuda")) {
  backend <- match.arg(backend)
  if (!backend %in% c("auto", "cpu")) {
    stop("Native InfoTSNE from KNN currently supports only `backend = \"cpu\"`; requested GPU backends do not fall back silently.", call. = FALSE)
  }
  knn <- coerce_knn_input(indices, distances)
  materialized <- materialize_knn_range(
    knn$indices,
    knn$distances,
    knn$col_start,
    knn$n_neighbors
  )
  indices <- materialized$indices
  distances <- materialized$distances
  n <- nrow(indices)
  k <- ncol(indices)
  if (is.null(perplexity)) {
    perplexity <- auto_tsne_perplexity(n, k)
  }
  if (is.null(n_threads)) {
    n_threads <- default_tsne_threads()
  }
  if (!is.null(eta) && is.null(learning_rate)) {
    learning_rate <- eta
  }
  if (is.null(learning_rate) || identical(learning_rate, "auto")) {
    learning_rate <- max(n / as.numeric(early_exaggeration_coeff) / 4, 50)
  }
  args <- check_tsne_neighbor_params(
    n = n,
    n_components = n_components,
    perplexity = perplexity,
    theta = 0.5,
    max_iter = max_iter,
    verbose = verbose,
    Y_init = Y_init,
    stop_lying_iter = early_exaggeration_iter,
    mom_switch_iter = early_exaggeration_iter,
    momentum = momentum,
    final_momentum = final_momentum,
    eta = learning_rate,
    exaggeration_factor = early_exaggeration_coeff
  )
  n_negatives <- as.integer(n_negatives)
  if (length(n_negatives) != 1L || is.na(n_negatives) || n_negatives < 1L) {
    stop("`n_negatives` must be a positive integer.", call. = FALSE)
  }
  if (!is.numeric(repulsion_strength) || repulsion_strength <= 0) {
    stop("`repulsion_strength` must be positive.", call. = FALSE)
  }

  out <- knn_infotsne_cpp(
    indices,
    distances,
    args$Y_init,
    args$init,
    args$n_components,
    args$perplexity,
    args$max_iter,
    args$stop_lying_iter,
    args$momentum,
    args$final_momentum,
    args$eta,
    args$exaggeration_factor,
    as.numeric(repulsion_strength),
    n_negatives,
    as.integer(n_threads),
    as.integer(seed),
    args$verbose
  )
  layout <- set_embedding_colnames(out$Y, "InfoTSNE")
  cfg <- list(
    method = "infotsne",
    backend = "cpu",
    n = n,
    n_neighbors = as.integer(k),
    perplexity = args$perplexity,
    max_iter = args$max_iter,
    early_exaggeration_iter = args$stop_lying_iter,
    momentum = args$momentum,
    final_momentum = args$final_momentum,
    learning_rate = args$eta,
    early_exaggeration_coeff = args$exaggeration_factor,
    repulsion_strength = as.numeric(repulsion_strength),
    n_negatives = out$n_negatives,
    optimizer = out$optimizer,
    objective = out$objective,
    n_threads = out$n_threads,
    input_had_self = isTRUE(knn$has_self),
    knn_backend = knn$input_backend,
    provenance = "TorchDR_InfoTSNE_inspired_native_negative_sampling_optimizer"
  )
  attr(layout, "fastEmbedR_config") <- cfg
  layout
}

#' Run t-SNE from a data matrix
#'
#' `tsne()` is a small convenience wrapper around `nn()` plus
#' `embed_knn(method = "tsne")`. The KNN input path follows the user-facing
#' behaviour of `Rtsne::Rtsne_neighbors()` while using package-native C++ code.
#' An internal KeOps-inspired exact blocked reduction is available for
#' development/GPU translation experiments, but the faster pair-symmetric CPU
#' loop remains the default.
#'
#' @param data Numeric matrix or data frame with observations in rows.
#' @param labels Optional labels used only for scoring and plotting metadata.
#' @param n_neighbors Number of non-self neighbors. If `NULL`, the package uses
#'   `3 * perplexity`, matching the usual `Rtsne_neighbors()` convention.
#' @param perplexity t-SNE perplexity. If `NULL`, uses
#'   `min(30, floor((n - 1) / 3), floor(k / 3))`.
#' @param n_components Output dimensionality, from 1 to 3.
#' @param standardize Center and scale columns before KNN.
#' @param pca_dims Optional PCA dimension before KNN.
#' @param nn Optional precomputed KNN output.
#' @param seed Random seed.
#' @param backend KNN backend. The t-SNE optimizer itself is currently CPU-only.
#'   GPU and RAPIDS cuVS values are used only for the neighbour search and are
#'   recorded separately as `nn_backend`; unavailable explicit GPU/cuVS requests
#'   fail rather than falling back silently.
#' @param silhouette_sample Optional sample size for silhouette scoring.
#' @param preserve_sample Optional sample size for neighborhood scoring.
#' @param preserve_k Number of neighbors used for neighborhood scoring.
#' @param keep_knn If `TRUE`, retain KNN matrices in the returned object.
#' @param verbose Print optimizer progress.
#' @param ... Additional t-SNE optimizer parameters: `theta`, `max_iter`,
#'   `Y_init`, `stop_lying_iter`, `mom_switch_iter`, `momentum`,
#'   `final_momentum`, `eta`, `exaggeration_factor`, and `n_threads`.
#' @return A `fastEmbedR_embedding` object.
#' @export
tsne <- function(data,
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
                 ...) {
  backend <- match.arg(backend)
  preprocess_time <- system.time({
    prepared <- prepare_embedding_data(
      data,
      standardize,
      pca_dims,
      seed,
      backend = "cpu"
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
      raw_knn <- nn_without_self(x, k = n_neighbors, backend = backend)
      knn_result <- normalize_supplied_knn(raw_knn, n, n_neighbors)
      knn_result$nn_backend <- attr(raw_knn, "backend")
    } else {
      knn_result <- normalize_supplied_knn(nn, n, n_neighbors, keep_self = keep_knn)
      knn_result$nn_backend <- attr(nn, "backend")
      if (is.null(knn_result$nn_backend)) knn_result$nn_backend <- "supplied"
    }
  })

  embedding_time <- system.time({
    layout <- fast_knn_tsne_core(
      knn_result$indices,
      knn_result$distances,
      n_components = n_components,
      perplexity = perplexity,
      seed = seed,
      verbose = verbose,
      backend = "cpu",
      ...
    )
  })
  cfg <- attr(layout, "fastEmbedR_config")
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
    backend = "cpu"
  )
  timings <- rbind(
    preprocess = preprocess_time,
    knn = knn_time,
    embedding = embedding_time
  )
  metrics <- data.frame(
    method = "tsne",
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
      method = "tsne",
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
    method = "tsne",
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

#' Run native openTSNE-style t-SNE from a data matrix
#'
#' `opentsne()` uses the same KNN-first workflow as `tsne()`, but its optimizer
#' follows openTSNE's two-phase contract: an early-exaggeration phase followed
#' by a normal optimization phase, `learning_rate = n / exaggeration` when
#' `learning_rate = "auto"`, openTSNE-style gains, and max-step clipping.
#' The implementation is native fastEmbedR C++ and does not call Python.
#'
#' @inheritParams tsne
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
#' @param max_step_norm Maximum per-point update norm. Use `NULL` or `NA` to
#'   disable clipping.
#' @param negative_gradient_method `"auto"`, `"bh"`, or `"exact"`. Native FFT
#'   interpolation is not ported yet and fails clearly.
#' @param record_costs If `TRUE`, compute diagnostic KL/cost traces. This does
#'   not affect the embedding, but costs extra time because it evaluates the
#'   objective for reporting.
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
                     learning_rate = "auto",
                     early_exaggeration_iter = 250L,
                     early_exaggeration = "auto",
                     n_iter = 500L,
                     exaggeration = NULL,
                     initial_momentum = 0.8,
                     final_momentum = 0.8,
                     max_step_norm = 5,
                     negative_gradient_method = "auto",
	                     record_costs = FALSE,
	                     ...) {
	  backend <- match.arg(backend)
  if (is_knn_input(data)) {
    if (!is.null(nn)) {
      stop("When `data` is a KNN object, do not also pass `nn`.", call. = FALSE)
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
        backend = backend,
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
      backend = "cpu"
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
      backend = "cpu"
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
      raw_knn <- nn_without_self(x, k = n_neighbors, backend = backend)
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
      backend = "cpu",
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
    backend = "cpu"
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

#' Run InfoTSNE from a data matrix
#'
#' `infotsne()` uses the same KNN-first input path as `tsne()`, but optimizes
#' the InfoTSNE negative-sampling objective inspired by TorchDR. Its per-epoch
#' complexity is roughly linear in `n * (k + n_negatives)` rather than all-pairs
#' exact t-SNE repulsion.
#'
#' @inheritParams tsne
#' @param n_negatives Number of sampled negatives per point and iteration.
#' @param learning_rate Learning rate. Use `NULL` or `"auto"` for the TorchDR
#'   style default `max(n / early_exaggeration_coeff / 4, 50)`.
#' @param early_exaggeration_iter Number of iterations using exaggerated
#'   attractive forces.
#' @param early_exaggeration_coeff Early exaggeration multiplier.
#' @param repulsion_strength Multiplier for the sampled repulsive term.
#' @return A `fastEmbedR_embedding` object.
#' @export
infotsne <- function(data,
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
                     n_negatives = 300L,
                     learning_rate = NULL,
                     early_exaggeration_iter = 250L,
                     early_exaggeration_coeff = 12,
                     repulsion_strength = 1,
                     ...) {
  backend <- match.arg(backend)
  preprocess_time <- system.time({
    prepared <- prepare_embedding_data(
      data,
      standardize,
      pca_dims,
      seed,
      backend = "cpu"
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
      raw_knn <- nn_without_self(x, k = n_neighbors, backend = backend)
      knn_result <- normalize_supplied_knn(raw_knn, n, n_neighbors)
      knn_result$nn_backend <- attr(raw_knn, "backend")
    } else {
      knn_result <- normalize_supplied_knn(nn, n, n_neighbors, keep_self = keep_knn)
      knn_result$nn_backend <- attr(nn, "backend")
      if (is.null(knn_result$nn_backend)) knn_result$nn_backend <- "supplied"
    }
  })

  embedding_time <- system.time({
    layout <- fast_knn_infotsne_core(
      knn_result$indices,
      knn_result$distances,
      n_components = n_components,
      perplexity = perplexity,
      seed = seed,
      verbose = verbose,
      backend = "cpu",
      n_negatives = n_negatives,
      learning_rate = learning_rate,
      early_exaggeration_iter = early_exaggeration_iter,
      early_exaggeration_coeff = early_exaggeration_coeff,
      repulsion_strength = repulsion_strength,
      ...
    )
  })
  cfg <- attr(layout, "fastEmbedR_config")
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
    backend = "cpu"
  )
  timings <- rbind(
    preprocess = preprocess_time,
    knn = knn_time,
    embedding = embedding_time
  )
  metrics <- data.frame(
    method = "infotsne",
    n = n,
    p = ncol(x),
    n_neighbors = knn_result$n_neighbors,
    perplexity = cfg$perplexity,
    n_negatives = cfg$n_negatives,
    elapsed = sum(timings[, "elapsed"]),
    preprocess_elapsed = preprocess_time["elapsed"],
    knn_elapsed = knn_time["elapsed"],
    embedding_elapsed = embedding_time["elapsed"],
    scores,
    stringsAsFactors = FALSE
  )
  parameters <- c(
    list(
      method = "infotsne",
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
    method = "infotsne",
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
