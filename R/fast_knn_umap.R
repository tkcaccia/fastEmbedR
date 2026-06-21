#' Fast UMAP from precomputed nearest neighbors
#'
#' @param indices Integer matrix of nearest-neighbor indices, one row per point,
#'   or a list returned by [faissR::nn()]. Indices may be 1-based, as returned by R
#'   packages, or 0-based. If a self-neighbor first column is present it is
#'   removed automatically.
#' @param distances Numeric matrix matching `indices`. Leave as `NULL` when
#'   `indices` is a [faissR::nn()] result.
#' @param n_components Output dimensionality.
#' @param seed Integer random seed.
#' @param verbose Print progress from C++.
#' @param backend Execution backend: `"cpu"`, `"cuda"`, or `"metal"`.
#' @param graph_mode Graph weighting mode. `"binary"` uses a symmetric
#'   unit-weight graph. `"fuzzy"` uses standard UMAP fuzzy graph weights.
#' @return A numeric matrix with `nrow(indices)` rows and `n_components` columns.
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
                          backend = c("cpu", "cuda", "metal"),
                          n_threads = NULL,
                          graph_mode = c("binary", "fuzzy")) {
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

fast_knn_umap_core <- function(indices,
                               distances = NULL,
                               n_components = 2L,
                               seed = 42L,
                               verbose = FALSE,
                               backend = c("cpu", "cuda", "metal"),
                               n_threads = NULL,
                               n_epochs = NULL,
                               config_override = NULL,
                               graph_mode = c("binary", "fuzzy")) {
  backend <- resolve_embedding_backend(backend)
  graph_mode <- match.arg(graph_mode)
  n_components <- validate_n_components(n_components)
  knn <- coerce_knn_input(indices, distances)
  indices <- knn$indices
  distances <- knn$distances

	  cfg <- fast_knn_umap_config(
	    n = nrow(indices),
	    k = knn$n_neighbors,
	    backend = backend
	  )
	  auto_policy <- tryCatch(
	    {
	      policy_distances <- if (knn$col_start != 0L || knn$n_neighbors != ncol(distances)) {
	        materialize_knn_range(indices, distances, knn$col_start, knn$n_neighbors)$distances
	      } else {
	        distances
	      }
	      umap_auto_parameters_cpp(
	        policy_distances,
	        as.integer(knn$n_neighbors),
	        as.character(cfg$backend)
	      )
	    },
	    error = function(e) {
	      list(error = conditionMessage(e))
	    }
	  )
	  if (is.null(auto_policy$error)) {
	    cfg$n_epochs <- as.integer(auto_policy$n_epochs)
	    cfg$min_dist <- as.numeric(auto_policy$min_dist)
	    cfg$negative_sample_rate <- as.integer(auto_policy$negative_sample_rate)
	    cfg$learning_rate <- as.numeric(auto_policy$learning_rate)
	    cfg$spectral_n_iter <- as.integer(auto_policy$spectral_n_iter)
	    cfg$init_scale <- as.numeric(auto_policy$init_scale)
	    cfg$auto_parameter_backend <- "cpp_knn_distance_profile"
	    cfg$auto_parameter_rule <- as.character(auto_policy$rule)
	    cfg$knn_distance_cv <- as.numeric(auto_policy$knn_distance_cv)
	    cfg$knn_distance_ratio_30_15 <- as.numeric(auto_policy$knn_distance_ratio_30_15)
	    cfg$knn_distance_ratio_50_15 <- as.numeric(auto_policy$knn_distance_ratio_50_15)
	  } else {
	    cfg$auto_parameter_backend <- "r_size_rule_fallback"
	    cfg$auto_parameter_error <- auto_policy$error
	  }
	  if (!is.null(n_threads)) {
    n_threads <- as.integer(n_threads)
    if (length(n_threads) != 1L || is.na(n_threads) || !is.finite(n_threads) || n_threads < 1L) {
      stop("`n_threads` must be NULL or a positive integer.", call. = FALSE)
    }
    cfg$n_threads <- as.integer(max(1L, min(4L, n_threads)))
  }
  cfg$input_had_self <- isTRUE(knn$has_self)
  cfg$knn_col_start <- as.integer(knn$col_start)
  cfg$knn_n_neighbors <- as.integer(knn$n_neighbors)
  cfg$knn_materialized <- isTRUE(knn$materialized)
  cfg$knn_backend <- knn$input_backend
  cfg$graph_mode <- graph_mode
  if (!is.null(n_epochs)) {
    cfg$n_epochs <- validate_epoch_count(n_epochs)
    cfg$preset <- "internal_epoch_override"
    cfg$epoch_source <- "internal_override"
  }
  cfg <- apply_umap_connectivity_spectral_rule(
    cfg,
    indices,
    col_start = knn$col_start,
    n_neighbors = knn$n_neighbors
  )
  if (fast_knn_umap_should_auto_pilot(
    cfg = cfg,
    indices = indices,
    config_override = config_override,
    n_epochs = n_epochs
  )) {
    pilot <- tryCatch(
      auto_umap_knn_pilot_tune(
        indices,
        distances,
        seed = seed,
        full_n = nrow(indices),
        pilot_min_n = fast_knn_umap_auto_pilot_min_n(),
        pilot_max_n = fast_knn_umap_auto_pilot_max_n(),
        pilot_max_configs = fast_knn_umap_auto_pilot_max_configs(),
        use_cache = fast_knn_umap_auto_pilot_use_cache(),
        cache_dir = getOption("fastEmbedR.knn_pilot_cache_dir", NULL),
        force_recompute = isTRUE(getOption("fastEmbedR.knn_pilot_force_recompute", FALSE))
      ),
      error = function(e) {
        list(status = "failed", reason = conditionMessage(e))
      }
    )
    cfg$auto_knn_pilot_status <- pilot$status
    cfg$auto_knn_pilot_reason <- pilot$reason
    if (identical(pilot$status, "success") && !is.null(pilot$config_override)) {
      cfg <- apply_fast_knn_umap_config_override(cfg, pilot$config_override)
    }
  } else {
    cfg$auto_knn_pilot_status <- "skipped"
    cfg$auto_knn_pilot_reason <- fast_knn_umap_auto_pilot_skip_reason(
      cfg = cfg,
      indices = indices,
      config_override = config_override,
      n_epochs = n_epochs
    )
  }
  cfg <- apply_fast_knn_umap_config_override(cfg, config_override)

  if (cfg$backend %in% c("cuda", "metal")) {
    if (knn$col_start != 0L || knn$n_neighbors != ncol(indices)) {
      gpu_knn <- materialize_knn_range(indices, distances, knn$col_start, knn$n_neighbors)
      indices <- gpu_knn$indices
      distances <- gpu_knn$distances
      cfg$knn_materialized_for_gpu <- TRUE
      cfg$knn_col_start <- 0L
      cfg$knn_n_neighbors <- ncol(indices)
    } else {
      cfg$knn_materialized_for_gpu <- FALSE
    }
    if (n_components != 2L) {
      stop("Native GPU embedding backends currently support only `n_components = 2`.", call. = FALSE)
    }
    gpu_backend <- cfg$backend
    cfg$gpu_transfer_policy <- "single_upload_optimizer"
    use_fused_cuda_umap <- FALSE
    cfg$gpu_optimizer_mode <- if (identical(gpu_backend, "metal")) {
      "clean_linear_atomic_inplace"
    } else {
      "atomic_coo"
    }
    cfg$gpu_optimizer_update_rule <- if (gpu_backend == "metal") {
      "native_metal_csr_clean_linear_atomic_inplace_edge_update"
    } else {
      "native_cuda_atomic_coo_umap_schedule"
    }
    cfg$gpu_optimizer_schedule <- if (gpu_backend == "metal") {
      "clean_edge_probability_linear_decay"
    } else {
      "coo_epochs_per_sample"
    }
    cfg$gpu_epochs_per_command <- if (gpu_backend == "metal") 64L else NA_integer_
    init <- NULL
    if (isTRUE(use_fused_cuda_umap)) {
      cfg$init_backend <- "cuda_fused_spectral"
      cfg$init_backend_reason <- "CUDA fused UMAP uploads KNN once, computes spectral initialization on device, keeps graph/layout on device, and returns only the final layout."
    } else if (!identical(graph_mode, "binary")) {
      init <- spectral_knn_init(
        indices,
        distances,
        n_components = 2L,
        min_dist = cfg$min_dist,
        spectral_n_iter = cfg$spectral_n_iter,
        seed = seed,
        backend = "cpu",
        n_threads = cfg$n_threads
      )
      init <- scale_embedding_sdev_r(init, cfg$init_scale)
      cfg$init_backend <- attr(init, "backend")
      cfg$init_backend_reason <- "CPU initialization avoids a separate GPU init round trip; the native optimizer uploads KNN and init once."
    } else {
      cfg$init_backend <- "pending_binary_csr"
      cfg$init_backend_reason <- "Binary UMAP initializes from the same unit-weight CSR graph used by the optimizer."
    }
    cfg$graph_prep_backend <- if (gpu_backend == "cuda" && isTRUE(use_fused_cuda_umap)) {
      "cuda_fused_binary_coo"
    } else if (identical(graph_mode, "binary")) {
      "cpu_binary_csr"
    } else {
      "cpu_fuzzy_csr"
    }
    cfg$graph_storage <- if (gpu_backend == "cuda" && isTRUE(use_fused_cuda_umap)) {
      "native_cuda_binary_coo_fused"
    } else if (gpu_backend == "cuda") {
      "cpu_csr_packed_to_cuda_coo"
    } else {
      "cpu_csr_packed_to_metal"
    }
    cfg$gpu_initial_backend <- gpu_backend
    cfg$optimizer_backend <- gpu_backend
    cfg <- add_gpu_transfer_metadata(
      cfg,
      indices,
      distances,
      init = init,
      n = nrow(indices),
      n_components = 2L,
      objective = "umap"
    )
    cfg$backend <- cfg$optimizer_backend
    cfg$gpu_umap_path <- if (identical(gpu_backend, "cuda") && isTRUE(use_fused_cuda_umap)) {
      "cuda_binary_atomic"
    } else if (identical(gpu_backend, "cuda")) {
      "cuda_fuzzy_graph_atomic"
    } else {
      "metal_atomic_inplace"
    }
    cfg$gpu_initial_epochs <- as.integer(cfg$n_epochs)
    layout <- if (isTRUE(use_fused_cuda_umap)) {
      knn_umap_cuda_fused_cpp(
        indices,
        distances,
        as.integer(cfg$n_epochs),
        as.integer(cfg$negative_sample_rate),
        cfg$learning_rate,
        cfg$min_dist,
        cfg$repulsion_strength,
        as.integer(cfg$spectral_n_iter),
        as.integer(seed),
        0L
      )
    } else if (identical(gpu_backend, "metal")) {
      graph <- umap_build_csr_graph(
        indices,
        distances,
        0L,
        as.integer(ncol(indices)),
        as.integer(ncol(indices)),
        as.integer(cfg$n_threads),
        graph_mode = graph_mode
      )
      if (is.null(init)) {
        init <- umap_init_from_csr_graph(
          graph,
          n_components = 2L,
          cfg = cfg,
          seed = seed,
          verbose = FALSE
        )
        cfg$init_backend <- attr(init, "backend")
      }
      cfg$graph_nnz <- as.integer(graph$nnz)
      cfg$graph_max_weight <- as.numeric(graph$max_weight)
      cfg$graph_cuda_like_width <- graph$cuda_like_width
      cfg$graph_builder <- graph$graph_builder
      out <- knn_embed_metal_csr_cpp(
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
      cfg$metal_graph_input <- attr(out, "metal_graph_input")
      cfg$metal_csr_width <- attr(out, "metal_csr_width")
      cfg$metal_truncated_edges <- attr(out, "metal_truncated_edges")
      out
    } else if (identical(gpu_backend, "cuda")) {
      graph <- umap_build_csr_graph(
        indices,
        distances,
        0L,
        as.integer(ncol(indices)),
        as.integer(ncol(indices)),
        as.integer(cfg$n_threads),
        graph_mode = graph_mode
      )
      if (is.null(init)) {
        init <- umap_init_from_csr_graph(
          graph,
          n_components = 2L,
          cfg = cfg,
          seed = seed,
          verbose = FALSE
        )
        cfg$init_backend <- attr(init, "backend")
      }
      coo <- umap_csr_to_coo(graph)
      cfg$graph_nnz <- as.integer(graph$nnz)
      cfg$graph_max_weight <- as.numeric(graph$max_weight)
      cfg$graph_cuda_like_width <- graph$cuda_like_width
      cfg$graph_builder <- graph$graph_builder
      umap_cuda_optimize_coo_cpp(
        coo$heads,
        coo$tails,
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
    } else {
      run_native_knn_optimizer(
        gpu_backend,
        indices,
        distances,
        init,
        "umap",
        cfg$n_epochs,
        cfg$negative_sample_rate,
        cfg$learning_rate,
        min_dist = cfg$min_dist,
        seed = seed
      )
    }
    layout <- set_embedding_colnames(layout, "UMAP")
    attr(layout, "fastEmbedR_config") <- cfg
    return(layout)
  }

  cfg$graph_prep_backend <- "cpu"
  if (n_components != 2L) {
    cfg <- add_gpu_transfer_metadata(
      cfg,
      indices,
      distances,
      n = nrow(indices),
      n_components = n_components,
      objective = "umap"
    )
    cfg$init_backend <- "cpu"
    cfg$optimizer_mode <- "legacy_range_optimizer_non_2d"
    layout <- fast_knn_umap_range_cpp(
      indices, distances, as.integer(knn$col_start), as.integer(knn$n_neighbors),
      n_components, as.integer(cfg$n_epochs),
      cfg$min_dist, as.integer(cfg$negative_sample_rate), cfg$learning_rate,
      cfg$repulsion_strength, as.integer(cfg$spectral_n_iter), as.integer(cfg$n_threads),
      cfg$init_scale, as.integer(seed), isTRUE(verbose)
    )
    layout <- set_embedding_colnames(layout, "UMAP")
    attr(layout, "fastEmbedR_config") <- cfg
    return(layout)
  }

  if (identical(graph_mode, "binary")) {
    graph <- umap_build_csr_graph(
      indices,
      distances,
      as.integer(knn$col_start),
      as.integer(knn$n_neighbors),
      as.integer(knn$n_neighbors),
      as.integer(cfg$n_threads),
      graph_mode = graph_mode
    )
    init <- umap_init_from_csr_graph(
      graph,
      n_components = n_components,
      cfg = cfg,
      seed = seed,
      verbose = FALSE
    )
    cfg$graph_prep_backend <- "cpu_binary_csr"
    cfg$graph_storage <- "cpu_binary_csr"
    cfg$graph_nnz <- as.integer(graph$nnz)
    cfg$graph_max_weight <- as.numeric(graph$max_weight)
    cfg$graph_cuda_like_width <- graph$cuda_like_width
    cfg$graph_builder <- graph$graph_builder
    cfg$init_backend <- attr(init, "backend")
    cfg$optimizer_mode <- "csr_epoch_schedule"
    cfg$optimizer_schedule <- "epochs_per_sample_binary_csr"
    layout <- fast_knn_umap_csr_init_cpp(
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
    layout <- set_embedding_colnames(layout, "UMAP")
    attr(layout, "fastEmbedR_config") <- cfg
    return(layout)
  }
  graph <- umap_build_csr_graph(
    indices,
    distances,
    as.integer(knn$col_start),
    as.integer(knn$n_neighbors),
    as.integer(knn$n_neighbors),
    as.integer(cfg$n_threads),
    graph_mode = graph_mode
  )
  init <- umap_init_from_csr_graph(
    graph,
    n_components = n_components,
    cfg = cfg,
    seed = seed,
    verbose = FALSE
  )
  cfg$graph_prep_backend <- "cpu_fuzzy_csr"
  cfg$graph_storage <- "cpu_fuzzy_csr"
  cfg$graph_nnz <- as.integer(graph$nnz)
  cfg$graph_max_weight <- as.numeric(graph$max_weight)
  cfg$graph_cuda_like_width <- graph$cuda_like_width
  cfg$graph_builder <- graph$graph_builder
  cfg$init_backend <- attr(init, "backend")
  cfg$optimizer_mode <- "cpu_cuda_style_atomic_coo"
  cfg$optimizer_schedule <- "coo_epochs_per_sample"
  coo <- umap_csr_to_coo(graph)
  layout <- fast_knn_umap_coo_replay_cpp(
    coo$heads,
    coo$tails,
    graph$weights,
    graph$epochs_per_sample,
    init,
    as.integer(cfg$n_epochs),
    cfg$min_dist,
    as.integer(cfg$negative_sample_rate),
    cfg$learning_rate,
    cfg$repulsion_strength,
    as.integer(cfg$n_threads),
    as.integer(seed),
    FALSE,
    isTRUE(verbose)
  )
  layout <- set_embedding_colnames(layout, "UMAP")
  attr(layout, "fastEmbedR_config") <- cfg
  layout
}

umap_build_csr_graph <- function(indices,
                                 distances,
                                 col_start,
                                 n_cols,
                                 edge_budget,
                                 n_threads,
                                 graph_mode = c("binary", "fuzzy")) {
  graph_mode <- match.arg(graph_mode)
  if (identical(graph_mode, "binary")) {
    graph <- umap_graph_csr_binary_union_cpp(
      indices,
      as.integer(col_start),
      as.integer(n_cols),
      as.integer(n_threads)
    )
    graph$graph_mode <- "binary"
    return(graph)
  }
  graph <- umap_graph_csr_cpp(
    indices,
    distances,
    as.integer(col_start),
    as.integer(n_cols),
    as.integer(edge_budget),
    as.integer(n_threads)
  )
  graph$graph_mode <- "fuzzy"
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
  attr(init, "backend") <- "cpu_binary_csr"
  init
}

umap_csr_to_coo <- function(graph) {
  offsets <- as.integer(graph$offsets)
  nnz_by_row <- diff(offsets)
  list(
    heads = as.integer(rep.int(seq_along(nnz_by_row) - 1L, nnz_by_row)),
    tails = as.integer(graph$neighbors)
  )
}


apply_fast_knn_umap_config_override <- function(cfg, override) {
  if (is.null(override) || length(override) == 0L) {
    if (is.null(cfg$config_override)) {
      cfg$config_override <- FALSE
    }
    return(cfg)
  }
  if (!is.null(override$n_epochs)) {
    cfg$n_epochs <- validate_epoch_count(override$n_epochs)
    cfg$epoch_source <- if (is.null(override$tuning_source)) {
      "internal_override"
    } else {
      paste0(override$tuning_source, "_override")
    }
  }
  if (!is.null(override$spectral_n_iter)) {
    spectral <- as.integer(override$spectral_n_iter)
    if (length(spectral) != 1L || is.na(spectral) || !is.finite(spectral) || spectral < 1L) {
      stop("`spectral_n_iter` override must be a positive integer.", call. = FALSE)
    }
    cfg$spectral_n_iter <- if (isTRUE(cfg$spectral_connectivity_checked)) {
      as.integer(max(cfg$spectral_n_iter, spectral))
    } else {
      as.integer(spectral)
    }
    cfg$spectral_rule <- if (is.null(override$tuning_source)) {
      "internal_override"
    } else {
      paste0(override$tuning_source, "_override")
    }
  }
  if (!is.null(override$init_scale)) {
    init_scale <- as.numeric(override$init_scale)
    if (length(init_scale) != 1L || (!is.na(init_scale) && (!is.finite(init_scale) || init_scale <= 0))) {
      stop("`init_scale` override must be NA or a positive finite number.", call. = FALSE)
    }
    cfg$init_scale <- init_scale
    cfg$init_scale_source <- if (is.null(override$tuning_source)) {
      "internal_override"
    } else {
      paste0(override$tuning_source, "_override")
    }
  }
  if (!is.null(override$learning_rate)) {
    learning_rate <- as.numeric(override$learning_rate)
    if (length(learning_rate) != 1L || !is.finite(learning_rate) || learning_rate <= 0) {
      stop("`learning_rate` override must be a positive finite number.", call. = FALSE)
    }
    cfg$learning_rate <- learning_rate
    cfg$learning_rate_source <- if (is.null(override$tuning_source)) {
      "internal_override"
    } else {
      paste0(override$tuning_source, "_override")
    }
  }
  if (!is.null(override$negative_sample_rate)) {
    negative_sample_rate <- as.integer(override$negative_sample_rate)
    if (length(negative_sample_rate) != 1L || is.na(negative_sample_rate) ||
        !is.finite(negative_sample_rate) || negative_sample_rate < 0L) {
      stop("`negative_sample_rate` override must be a non-negative integer.", call. = FALSE)
    }
    cfg$negative_sample_rate <- negative_sample_rate
    cfg$negative_sample_rate_source <- if (is.null(override$tuning_source)) {
      "internal_override"
    } else {
      paste0(override$tuning_source, "_override")
    }
  }
  if (!is.null(override$repulsion_strength)) {
    repulsion_strength <- as.numeric(override$repulsion_strength)
    if (length(repulsion_strength) != 1L || !is.finite(repulsion_strength) || repulsion_strength <= 0) {
      stop("`repulsion_strength` override must be a positive finite number.", call. = FALSE)
    }
    cfg$repulsion_strength <- repulsion_strength
    cfg$repulsion_strength_source <- if (is.null(override$tuning_source)) {
      "internal_override"
    } else {
      paste0(override$tuning_source, "_override")
    }
  }
  cfg$config_override <- TRUE
  cfg$config_override_source <- if (is.null(override$tuning_source)) {
    "internal"
  } else {
    override$tuning_source
  }
  if (!is.null(override$pilot_sample_n)) {
    cfg$pilot_sample_n <- as.integer(override$pilot_sample_n)
  }
  if (!is.null(override$pilot_score)) {
    cfg$pilot_score <- as.numeric(override$pilot_score)
  }
  if (!is.null(override$pilot_cache_key)) {
    cfg$pilot_cache_key <- as.character(override$pilot_cache_key)
  }
  if (!is.null(override$pilot_cache_hit)) {
    cfg$pilot_cache_hit <- isTRUE(override$pilot_cache_hit)
  }
  cfg
}

fast_knn_umap_should_auto_pilot <- function(cfg,
                                            indices,
                                            config_override = NULL,
                                            n_epochs = NULL) {
  if (!is.null(config_override) || !is.null(n_epochs)) return(FALSE)
  if (!identical(cfg$backend, "cpu")) return(FALSE)
  if (!cfg$epoch_source %in% c("clean_large_default")) return(FALSE)
  if (!isTRUE(getOption("fastEmbedR.knn_pilot", FALSE))) return(FALSE)
  n <- nrow(indices)
  k <- ncol(indices)
  n >= fast_knn_umap_auto_pilot_min_n() && k >= 10L
}

fast_knn_umap_auto_pilot_skip_reason <- function(cfg,
                                                 indices,
                                                 config_override = NULL,
                                                 n_epochs = NULL) {
  if (!is.null(config_override)) return("explicit config override supplied")
  if (!is.null(n_epochs)) return("explicit epoch override supplied")
  if (!identical(cfg$backend, "cpu")) return("auto KNN pilot currently runs only on the CPU optimizer path")
  if (!cfg$epoch_source %in% c("clean_large_default")) return("default is not the clean atomic UMAP path")
  if (!isTRUE(getOption("fastEmbedR.knn_pilot", FALSE))) return("disabled by default; set option fastEmbedR.knn_pilot = TRUE for internal benchmarking")
  if (nrow(indices) < fast_knn_umap_auto_pilot_min_n()) return("below auto KNN pilot size threshold")
  if (ncol(indices) < 10L) return("too few supplied neighbours for a stable pilot")
  "not selected"
}

fast_knn_umap_auto_pilot_min_n <- function() {
  value <- getOption("fastEmbedR.knn_pilot_min_n", 20000L)
  value <- suppressWarnings(as.integer(value))
  if (length(value) != 1L || is.na(value) || !is.finite(value) || value < 50L) {
    return(20000L)
  }
  value
}

fast_knn_umap_auto_pilot_max_n <- function() {
  value <- getOption("fastEmbedR.knn_pilot_max_n", 2500L)
  value <- suppressWarnings(as.integer(value))
  if (length(value) != 1L || is.na(value) || !is.finite(value) || value < 50L) {
    return(2500L)
  }
  value
}

fast_knn_umap_auto_pilot_max_configs <- function() {
  value <- getOption("fastEmbedR.knn_pilot_max_configs", 4L)
  value <- suppressWarnings(as.integer(value))
  if (length(value) != 1L || is.na(value) || !is.finite(value) || value < 1L) {
    return(4L)
  }
  value
}

fast_knn_umap_auto_pilot_use_cache <- function() {
  !isFALSE(getOption("fastEmbedR.knn_pilot_use_cache", TRUE))
}

scale_embedding_sdev_r <- function(embedding, target_sdev) {
  target_sdev <- as.numeric(target_sdev)
  if (length(target_sdev) != 1L || is.na(target_sdev) || !is.finite(target_sdev) || target_sdev <= 0) {
    return(embedding)
  }
  if (nrow(embedding) < 2L) {
    return(embedding)
  }
  attr_backend <- attr(embedding, "backend")
  center <- colMeans(embedding)
  embedding <- sweep(embedding, 2L, center, "-")
  scale <- sqrt(colSums(embedding * embedding) / max(1L, nrow(embedding) - 1L))
  scale[!is.finite(scale) | scale == 0] <- 1
  embedding <- sweep(embedding, 2L, scale / target_sdev, "/")
  attr(embedding, "backend") <- attr_backend
  embedding
}

fast_knn_umap_config <- function(n,
                                 k,
                                 backend) {
  backend <- resolve_backend_request(backend, need_embedding = TRUE)
  medium_or_large <- n >= 500L
  very_large <- n >= 10000L

  if (very_large) {
    n_epochs <- 200L
    min_dist <- 0.01
    negative_sample_rate <- 5L
    spectral_n_iter <- if (k <= 15L) 30L else 20L
    init_scale <- NA_real_
    preset <- "clean_atomic_large"
    epoch_source <- "clean_large_default"
  } else if (medium_or_large) {
    n_epochs <- 500L
    min_dist <- 0.01
    negative_sample_rate <- 5L
    spectral_n_iter <- 60L
    init_scale <- NA_real_
    preset <- "clean_atomic_standard"
    epoch_source <- "clean_size_rule"
  } else {
    n_epochs <- 500L
    min_dist <- 0.01
    negative_sample_rate <- 5L
    spectral_n_iter <- 50L
    init_scale <- NA_real_
    preset <- "clean_atomic_standard"
    epoch_source <- "clean_size_rule"
  }
  repulsion_strength <- 1
  learning_rate <- 1
  cores <- suppressWarnings(parallel::detectCores(logical = FALSE))
  if (length(cores) != 1L || is.na(cores) || !is.finite(cores)) cores <- 1L
  thread_cap <- if (very_large) {
    4L
  } else if (k >= 15L && n >= 500L) {
    4L
  } else if (k >= 15L && n >= 200L) {
    3L
  } else {
    1L
  }
  n_threads <- max(1L, min(thread_cap, as.integer(cores)))
  if (identical(backend, "auto")) backend <- "cpu"

  graph_scales <- fast_knn_umap_graph_scales(k)
  mid_near_count <- fast_knn_umap_mid_near_count(k)
  prune_fraction <- fast_knn_umap_prune_fraction(k)
  list(
    method = "umap",
    preset = preset,
    optimizer = preset,
    epoch_source = epoch_source,
    n_epochs = as.integer(n_epochs),
    min_dist = as.numeric(min_dist),
    negative_sample_rate = as.integer(negative_sample_rate),
    repulsion_strength = as.numeric(repulsion_strength),
    learning_rate = as.numeric(learning_rate),
    spectral_n_iter = as.integer(spectral_n_iter),
    spectral_rule = if (very_large) "adaptive_large_k" else "size_rule",
    init_scale = as.numeric(init_scale),
    graph_storage = if (length(graph_scales) > 1L || mid_near_count > 0L) {
      "native_csr_float_multiscale_midnear"
    } else {
      "native_csr_float_direct"
    },
    graph_scales = paste(graph_scales, collapse = ","),
    graph_mid_near_edges_per_point = as.integer(mid_near_count),
    graph_mid_near_weight = fast_knn_umap_mid_near_weight(k),
    graph_pruning = if (prune_fraction > 0) "adaptive_weight_with_connectivity_rescue" else "none",
    graph_prune_fraction = as.numeric(prune_fraction),
    graph_prune_min_degree = as.integer(fast_knn_umap_prune_min_degree(k)),
    graph_mode = "native_csr_graph",
    optimizer_math = "clean_atomic_edge_sampler",
    n = as.integer(n),
    k = as.integer(k),
    n_threads = as.integer(n_threads),
    backend = backend
  )
}

apply_fast_knn_umap_distance_profile_rule <- function(cfg, distances) {
  if (is.null(cfg$n) || cfg$n < 50000L || is.null(cfg$k) || cfg$k < 30L) return(cfg)

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

fast_knn_umap_distance_profile <- function(distances) {
  d <- as.matrix(distances)
  if (!identical(typeof(d), "double")) storage.mode(d) <- "double"
  finite <- is.finite(d)
  if (!any(finite)) {
    return(list(cv = NA_real_, ratio_50_15 = NA_real_, ratio_30_15 = NA_real_))
  }
  col_at <- function(rank) {
    d[, min(as.integer(rank), ncol(d)), drop = TRUE]
  }
  d15 <- col_at(15L)
  d30 <- col_at(30L)
  d50 <- col_at(50L)
  mean_d <- mean(d[finite])
  cv <- if (is.finite(mean_d) && mean_d > 0) stats::sd(d[finite]) / mean_d else NA_real_
  med15 <- stats::median(d15[is.finite(d15)])
  med30 <- stats::median(d30[is.finite(d30)])
  med50 <- stats::median(d50[is.finite(d50)])
  denom <- max(med15, .Machine$double.eps)
  list(
    cv = as.numeric(cv),
    ratio_50_15 = as.numeric(med50 / denom),
    ratio_30_15 = as.numeric(med30 / denom)
  )
}

fast_knn_umap_graph_scales <- function(k) {
  k <- as.integer(k)
  if (length(k) != 1L || is.na(k) || k < 1L) return(1L)
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
  if (length(k) != 1L || is.na(k) || k < 1L) return(1L)
  if (k < 30L) return(as.integer(max(2L, k)))
  if (k < 50L) return(15L)
  if (k < 150L) return(20L)
  24L
}

apply_umap_connectivity_spectral_rule <- function(cfg,
                                                  indices,
                                                  col_start = 0L,
                                                  n_neighbors = ncol(indices) - col_start) {
  n <- nrow(indices)
  k <- as.integer(n_neighbors)
  if (n < 10000L) {
    cfg$spectral_connectivity_checked <- FALSE
    return(cfg)
  }

  stats <- tryCatch(
    knn_connectivity_range_cpp(indices, as.integer(col_start), as.integer(n_neighbors)),
    error = function(e) {
      cfg$spectral_connectivity_checked <- FALSE
      cfg$spectral_connectivity_error <- conditionMessage(e)
      NULL
    }
  )
  if (is.null(stats)) {
    return(cfg)
  }

  cfg$spectral_connectivity_checked <- TRUE
  cfg$graph_connected <- isTRUE(stats$connected)
  cfg$graph_component_count <- as.integer(stats$component_count)
  cfg$graph_largest_component_fraction <- as.numeric(stats$largest_component_fraction)
  cfg$graph_largest_component_size <- as.integer(stats$largest_component_size)
  cfg$graph_singleton_count <- as.integer(stats$singleton_count)
  cfg$graph_invalid_edge_count <- as.integer(stats$invalid_edge_count)

  base_iter <- if (k <= 15L) 30L else 20L
  selected_iter <- base_iter
  reason <- "connected_graph"
  many_components <- cfg$graph_component_count > max(2L, as.integer(ceiling(n / 10000)))
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

spectral_knn_init <- function(indices,
                              distances,
                              n_components = 2L,
                              min_dist = 0.1,
                              spectral_n_iter = 50L,
                              seed = 42L,
                              backend = "cpu",
                              n_threads = NULL,
                              col_start = 0L,
                              n_neighbors = NULL) {
  if (!is.matrix(indices)) indices <- as.matrix(indices)
  if (!is.matrix(distances)) distances <- as.matrix(distances)
  if (!is.integer(indices)) storage.mode(indices) <- "integer"
  if (!identical(typeof(distances), "double")) storage.mode(distances) <- "double"
  col_start <- as.integer(col_start)
  if (is.null(n_neighbors)) n_neighbors <- ncol(indices) - col_start
  n_neighbors <- as.integer(n_neighbors)
  if (is.null(n_threads)) {
    cores <- suppressWarnings(parallel::detectCores(logical = FALSE))
    if (length(cores) != 1L || is.na(cores) || !is.finite(cores)) cores <- 1L
    n_threads <- max(1L, min(4L, as.integer(cores)))
  } else {
    n_threads <- as.integer(n_threads)
    if (length(n_threads) != 1L || is.na(n_threads) || !is.finite(n_threads) || n_threads < 1L) {
      n_threads <- 1L
    }
    n_threads <- max(1L, min(4L, n_threads))
  }
  if (identical(backend, "cuda")) {
    if (col_start != 0L || n_neighbors != ncol(indices)) {
      knn <- materialize_knn_range(indices, distances, col_start, n_neighbors)
      indices <- knn$indices
      distances <- knn$distances
      col_start <- 0L
      n_neighbors <- ncol(indices)
    }
    if (!embedding_cuda_available_cpp()) {
      stop("CUDA spectral initialization is not available on this system.", call. = FALSE)
    }
    if (as.integer(n_components) != 2L) {
      stop("CUDA spectral initialization currently supports only `n_components = 2`.", call. = FALSE)
    }
    out <- spectral_knn_init_cuda_cpp(
      indices,
      distances,
      as.integer(n_components),
      as.integer(spectral_n_iter),
      as.integer(seed)
    )
    attr(out, "backend") <- "cuda"
    return(out)
  }
  if (identical(backend, "metal")) {
    if (col_start != 0L || n_neighbors != ncol(indices)) {
      knn <- materialize_knn_range(indices, distances, col_start, n_neighbors)
      indices <- knn$indices
      distances <- knn$distances
      col_start <- 0L
      n_neighbors <- ncol(indices)
    }
    if (!embedding_metal_available_cpp()) {
      stop("Metal spectral initialization is not available on this system.", call. = FALSE)
    }
    if (as.integer(n_components) != 2L) {
      stop("Metal spectral initialization currently supports only `n_components = 2`.", call. = FALSE)
    }
    out <- spectral_knn_init_metal_cpp(
      indices,
      distances,
      as.integer(n_components),
      as.integer(spectral_n_iter),
      as.integer(seed)
    )
    attr(out, "backend") <- "metal"
    return(out)
  }
  out <- fast_knn_umap_range_cpp(
    indices, distances, as.integer(col_start), as.integer(n_neighbors),
    as.integer(n_components), 0L,
    min_dist, 0L, 1,
    1.0, as.integer(spectral_n_iter), as.integer(n_threads),
    NA_real_, as.integer(seed), FALSE
  )
  attr(out, "backend") <- "cpu"
  out
}

# Compatibility alias. The public KNN API is `embed_knn(method = "umap")`.
umap_knn <- fast_knn_umap
