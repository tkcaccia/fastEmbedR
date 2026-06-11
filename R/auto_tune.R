auto_tune_embedding_policy <- function(n,
                                       method,
                                       mode,
                                       n_neighbors,
                                       landmarks,
                                       quality,
                                       embedding_backend,
                                       nn_supplied = FALSE,
                                       x = NULL,
                                       labels = NULL,
                                       seed = 4L,
                                       knn_backend = "auto",
                                       pilot_min_n = 2000L,
                                       pilot_max_n = 5000L,
                                       pilot_max_configs = 7L) {
  n <- as.integer(n)
  if (length(n) != 1L || is.na(n) || n < 2L) {
    stop("`n` must be the number of rows and must be at least two.", call. = FALSE)
  }
  user_n_neighbors <- n_neighbors
  if (!identical(method, "umap")) {
    stop("Only UMAP is supported.", call. = FALSE)
  }
  policy <- list(
    auto_tuned = FALSE,
    n_neighbors = n_neighbors,
    landmarks = landmarks,
    quality = quality,
    k_selected_by = if (is.null(n_neighbors) && !isTRUE(nn_supplied)) "size_rule" else "user_or_supplied",
    quality_selected_by = "not_used",
    landmarks_selected_by = if (identical(mode, "auto") && is.null(landmarks) && !isTRUE(nn_supplied)) "size_rule" else "user_or_mode",
    refinement_selected_by = if (identical(mode, "auto")) "best_default" else "mode",
    epoch_selected_by = "size_rule",
    k_pilot = data.frame(),
    quality_pilot = data.frame(),
    landmark_pilot = data.frame(),
    selected_pilot_score = NA_real_,
    selected_refinement_pilot_score = NA_real_,
    pilot_sample_n = NA_integer_,
    pilot_backend = NA_character_,
    pilot_spectral_n_iter = NA_integer_,
    pilot_n_epochs = NA_integer_,
    pilot_init_scale = NA_real_,
    pilot_refinement_strength = NA_real_,
    embedding_config_override = NULL,
    landmark_refinement_strength = 1,
    reason = NA_character_
  )

  if (is.null(n_neighbors) && !isTRUE(nn_supplied)) {
    policy$n_neighbors <- auto_embedding_k(n, method, include_self = FALSE)
    policy$auto_tuned <- TRUE
    policy$reason <- append_auto_reason(
      policy$reason,
      paste0("selected k=", policy$n_neighbors, " by size-aware rule")
    )
  }

  l_choice <- auto_select_landmark_policy(
    method = method,
    mode = mode,
    landmarks = landmarks,
    full_n = n,
    nn_supplied = nn_supplied
  )
  policy["landmarks"] <- list(l_choice$landmarks)
  policy$landmark_refinement <- l_choice$refinement
  policy$landmark_pilot <- l_choice$scores
  policy$selected_refinement_pilot_score <- l_choice$score
  policy$reason <- append_auto_reason(policy$reason, l_choice$reason)
  if (isTRUE(l_choice$auto_tuned)) policy$auto_tuned <- TRUE

  if (auto_umap_should_run_pilot(
    x = x,
    n = n,
    method = method,
    n_neighbors = user_n_neighbors,
    nn_supplied = nn_supplied,
    pilot_min_n = pilot_min_n
  )) {
    pilot <- tryCatch(
      auto_umap_pilot_tune(
        x = x,
        labels = labels,
        seed = seed,
        knn_backend = knn_backend,
        full_n = n,
        base_k = policy$n_neighbors,
        pilot_min_n = pilot_min_n,
        pilot_max_n = pilot_max_n,
        pilot_max_configs = pilot_max_configs
      ),
      error = function(e) {
        list(
          status = "failed",
          reason = paste0("pilot auto-tune failed: ", conditionMessage(e))
        )
      }
    )
    if (identical(pilot$status, "success")) {
      policy$n_neighbors <- pilot$n_neighbors
      policy$k_selected_by <- "pilot"
      policy$epoch_selected_by <- "pilot"
      policy$refinement_selected_by <- if (identical(policy$landmark_refinement, "none")) {
        policy$refinement_selected_by
      } else {
        "pilot"
      }
      policy$k_pilot <- pilot$scores
      policy$selected_pilot_score <- pilot$selected_score
      policy$pilot_sample_n <- pilot$pilot_sample_n
      policy$pilot_backend <- pilot$pilot_backend
      policy$pilot_spectral_n_iter <- pilot$spectral_n_iter
      policy$pilot_n_epochs <- pilot$n_epochs
      policy$pilot_init_scale <- pilot$init_scale
      policy$pilot_refinement_strength <- pilot$refinement_strength
      policy$landmark_refinement_strength <- pilot$refinement_strength
      policy$embedding_config_override <- pilot$config_override
      policy$reason <- append_auto_reason(policy$reason, pilot$reason)
      policy$auto_tuned <- TRUE
    } else if (!is.null(pilot$reason)) {
      policy$reason <- append_auto_reason(policy$reason, pilot$reason)
    }
  }

  policy
}

auto_umap_should_run_pilot <- function(x,
                                       n,
                                       method,
                                       n_neighbors,
                                       nn_supplied,
                                       pilot_min_n) {
  if (!identical(method, "umap")) return(FALSE)
  if (isTRUE(nn_supplied)) return(FALSE)
  if (!is.null(n_neighbors)) return(FALSE)
  if (is.null(x) || is.null(dim(x)) || nrow(x) != n) return(FALSE)
  n >= as.integer(pilot_min_n)
}

auto_umap_pilot_tune <- function(x,
                                 labels = NULL,
                                 seed = 4L,
                                 knn_backend = "auto",
                                 full_n = nrow(x),
                                 base_k = NULL,
                                 pilot_min_n = 2000L,
                                 pilot_max_n = 5000L,
                                 pilot_max_configs = 7L,
                                 use_cache = TRUE,
                                 cache_dir = NULL,
                                 force_recompute = FALSE) {
  x <- as.matrix(x)
  storage.mode(x) <- "double"
  full_n <- as.integer(full_n)
  pilot_n <- auto_umap_pilot_size(
    full_n = full_n,
    available_n = nrow(x),
    pilot_min_n = pilot_min_n,
    pilot_max_n = pilot_max_n
  )
  if (pilot_n < 50L) {
    return(list(
      status = "skipped",
      reason = "pilot auto-tune skipped below minimum pilot size"
    ))
  }

  dataset_profile <- auto_umap_dataset_profile(x, labels, full_n)
  pilot_rows <- auto_pilot_sample_indices(
    n = nrow(x),
    labels = labels,
    sample_size = pilot_n,
    seed = seed,
    rare_protect = isTRUE(dataset_profile$highly_imbalanced)
  )
  x_pilot <- x[pilot_rows, , drop = FALSE]
  labels_pilot <- if (is.null(labels)) NULL else labels[pilot_rows]
  pilot_n <- nrow(x_pilot)
  if (is.null(base_k)) {
    base_k <- auto_embedding_k(full_n, "umap", include_self = FALSE)
  }
  base_k <- as.integer(min(max(2L, base_k), pilot_n - 1L))
  alternate_k_candidates <- auto_umap_pilot_k_candidates(
    full_n,
    pilot_n,
    base_k,
    dataset_profile = dataset_profile
  )
  alternate_k_candidates <- setdiff(alternate_k_candidates, base_k)
  if (length(base_k) != 1L || !is.finite(base_k) || base_k < 2L) {
    return(list(
      status = "skipped",
      reason = "pilot auto-tune skipped because no valid base k was available"
    ))
  }

  max_k <- max(c(base_k, alternate_k_candidates))
  pilot_backend <- auto_umap_pilot_knn_backend(knn_backend, pilot_n, ncol(x_pilot), max_k)
  raw_knn <- nn_without_self(
    x_pilot,
    k = max_k,
    backend = pilot_backend
  )
  pilot_backend_used <- attr(raw_knn, "backend")
  if (is.null(pilot_backend_used)) pilot_backend_used <- pilot_backend

  score_keep <- sample_indices(
    pilot_n,
    min(1000L, max(50L, pilot_n)),
    seed + 211L
  )
  labels_factor <- if (is.null(labels_pilot)) NULL else as.factor(labels_pilot)
  labels_int <- if (is.null(labels_factor)) integer(0L) else as.integer(labels_factor)
  n_label_levels <- if (is.null(labels_factor)) 0L else length(levels(labels_factor))
  proxy_k <- min(50L, max_k, pilot_n - 1L)
  dataset_hash <- auto_umap_dataset_hash(x_pilot, labels = labels_pilot)
  cache_dir <- auto_umap_pilot_cache_dir(cache_dir)

  run_config <- function(k, spectral_n_iter, n_epochs, init_scale, stage) {
    cached <- auto_umap_read_pilot_cache_row(
      cache_dir = cache_dir,
      dataset_hash = dataset_hash,
      k = k,
      seed = seed,
      spectral_n_iter = spectral_n_iter,
      n_epochs = n_epochs,
      init_scale = init_scale,
      kind = "data",
      use_cache = use_cache,
      force_recompute = force_recompute
    )
    if (!is.null(cached)) {
      cached$stage <- stage
      cached$cache_hit <- TRUE
      return(cached)
    }
    idx <- raw_knn$indices[, seq_len(k), drop = FALSE]
    dst <- raw_knn$distances[, seq_len(k), drop = FALSE]
    elapsed <- NA_real_
    layout <- NULL
    err <- NA_character_
    elapsed <- system.time({
      layout <- tryCatch(
        fast_knn_umap_core(
          idx,
          dst,
          n_components = 2L,
          seed = seed,
          verbose = FALSE,
          backend = "cpu",
          config_override = list(
            n_epochs = as.integer(n_epochs),
            spectral_n_iter = as.integer(spectral_n_iter),
            init_scale = init_scale,
            tuning_source = "pilot"
          )
        ),
        error = function(e) {
          err <<- conditionMessage(e)
          NULL
        }
      )
    })["elapsed"]
    if (is.null(layout)) {
      return(auto_umap_failed_pilot_row(
        stage, k, spectral_n_iter, n_epochs, init_scale, elapsed, err
      ))
    }
    primary_k <- as.integer(min(15L, k))
    structure <- tryCatch(
      knn_structure_score_cpp(
        layout,
        idx,
        score_keep,
        primary_k,
        labels_int,
        as.integer(n_label_levels)
      ),
      error = function(e) {
        err <<- conditionMessage(e)
        rep(NA_real_, 5L)
      }
    )
    proxy_structure <- if (proxy_k == primary_k && ncol(idx) >= proxy_k) {
      structure
    } else {
      tryCatch(
        knn_structure_score_cpp(
          layout,
          raw_knn$indices[, seq_len(proxy_k), drop = FALSE],
          score_keep,
          as.integer(proxy_k),
          labels_int,
          as.integer(n_label_levels)
        ),
        error = function(e) {
          err <<- conditionMessage(e)
          rep(NA_real_, 5L)
        }
      )
    }
    label_scores <- auto_umap_pilot_label_scores(
      layout = layout,
      labels = labels_pilot,
      keep = score_keep,
      k = primary_k,
      seed = seed
    )
    score <- auto_umap_pilot_score(
      structure,
      elapsed,
      label_scores = label_scores,
      proxy_structure = proxy_structure,
      dataset_profile = dataset_profile,
      k = k
    )
    row <- data.frame(
      stage = stage,
      k = as.integer(k),
      spectral_n_iter = as.integer(spectral_n_iter),
      n_epochs = as.integer(n_epochs),
      init_scale = as.numeric(init_scale),
      elapsed = as.numeric(elapsed),
      local_trustworthiness = unname(structure["local_trustworthiness"]),
      knn_preservation = unname(structure["knn_preservation"]),
      local_continuity = unname(structure["local_continuity"]),
      knn_preservation_50 = unname(proxy_structure["knn_preservation"]),
      local_continuity_50 = unname(proxy_structure["local_continuity"]),
      embedding_knn_accuracy = label_scores$label_knn_accuracy,
      rare_class_recall = label_scores$rare_class_recall,
      structure_score = unname(structure["structure_score"]),
      pilot_score = as.numeric(score),
      low_dimensional = isTRUE(dataset_profile$low_dimensional),
      tabular_like = isTRUE(dataset_profile$tabular_like),
      highly_imbalanced = isTRUE(dataset_profile$highly_imbalanced),
      avoid_aggressive_k_reduction = isTRUE(dataset_profile$avoid_aggressive_k_reduction),
      cache_hit = FALSE,
      status = "success",
      error = NA_character_,
      stringsAsFactors = FALSE
    )
    auto_umap_write_pilot_cache_row(
      row,
      cache_dir = cache_dir,
      dataset_hash = dataset_hash,
      k = k,
      seed = seed,
      kind = "data",
      use_cache = use_cache
    )
    row
  }

  optimizer_budget <- if (length(alternate_k_candidates) > 0L && pilot_max_configs >= 4L) {
    as.integer(pilot_max_configs) - 1L
  } else {
    as.integer(pilot_max_configs)
  }
  config_rows <- auto_umap_optimizer_pilot_configs(
    full_n = full_n,
    k = base_k,
    max_configs = optimizer_budget
  )
  quick_rows <- lapply(seq_len(nrow(config_rows)), function(i) {
    run_config(
      k = config_rows$k[i],
      spectral_n_iter = config_rows$spectral_n_iter[i],
      n_epochs = config_rows$n_epochs[i],
      init_scale = config_rows$init_scale[i],
      stage = "config"
    )
  })
  scores <- do.call(rbind, quick_rows)
  successful <- scores[is.finite(scores$pilot_score), , drop = FALSE]
  if (nrow(successful) == 0L) {
    return(list(
      status = "failed",
      reason = "pilot auto-tune failed for all fixed-k optimizer candidates",
      scores = scores
    ))
  }

  fixed_k_best <- successful[order(-successful$pilot_score, successful$elapsed), , drop = FALSE][1L, ]
  best_k <- as.integer(fixed_k_best$k)
  remaining <- max(0L, as.integer(pilot_max_configs) - nrow(scores))
  if (remaining > 0L && length(alternate_k_candidates) > 0L) {
    alternate_k_candidates <- auto_umap_rank_alternate_k_candidates(
      alternate_k_candidates,
      base_k = base_k,
      dataset_profile = dataset_profile,
      max_candidates = remaining
    )
    if (length(alternate_k_candidates) > 0L) {
      alt_rows <- lapply(alternate_k_candidates, function(candidate_k) {
        run_config(
          k = candidate_k,
          spectral_n_iter = fixed_k_best$spectral_n_iter,
          n_epochs = fixed_k_best$n_epochs,
          init_scale = fixed_k_best$init_scale,
          stage = "k_guard"
        )
      })
      scores <- rbind(scores, do.call(rbind, alt_rows))
    }
  }

  successful <- scores[is.finite(scores$pilot_score), , drop = FALSE]
  fixed_success <- successful[successful$k == base_k, , drop = FALSE]
  fixed_k_best <- fixed_success[order(-fixed_success$pilot_score, fixed_success$elapsed), , drop = FALSE][1L, ]
  best <- auto_umap_select_pilot_winner(
    successful = successful,
    fixed_k_best = fixed_k_best,
    labels_available = !is.null(labels_factor) && n_label_levels >= 2L
  )
  refinement_strength <- auto_umap_pilot_refinement_strength(best)
  config_override <- list(
    n_epochs = as.integer(best$n_epochs),
    spectral_n_iter = as.integer(best$spectral_n_iter),
    init_scale = as.numeric(best$init_scale),
    tuning_source = "pilot",
    pilot_sample_n = as.integer(pilot_n),
    pilot_score = as.numeric(best$pilot_score),
    pilot_cache_key = auto_umap_pilot_cache_key(dataset_hash, best$k, seed),
    pilot_cache_hit = any(scores$cache_hit %in% TRUE, na.rm = TRUE)
  )
  k_changed <- !identical(as.integer(best$k), as.integer(base_k))
  list(
    status = "success",
    n_neighbors = as.integer(best$k),
    spectral_n_iter = as.integer(best$spectral_n_iter),
    n_epochs = as.integer(best$n_epochs),
    init_scale = as.numeric(best$init_scale),
    refinement_strength = as.numeric(refinement_strength),
    selected_score = as.numeric(best$pilot_score),
    pilot_sample_n = as.integer(pilot_n),
    pilot_backend = pilot_backend_used,
    config_override = config_override,
    scores = scores,
    reason = paste0(
      "pilot auto-tune sampled ", pilot_n, " rows, tuned optimizer settings first at k=", base_k,
      ", selected k=", best$k,
      ", spectral_n_iter=", best$spectral_n_iter,
      ", epochs=", best$n_epochs,
      ", init_scale=", format(best$init_scale, trim = TRUE),
      ", refinement_strength=", format(refinement_strength, trim = TRUE),
      if (isTRUE(k_changed)) {
        "; changed k only after strong fixed-k pilot improvement check"
      } else {
        "; kept size-rule k"
      },
      if (any(scores$cache_hit %in% TRUE, na.rm = TRUE)) {
        "; reused cached pilot score(s)"
      } else {
        ""
      },
      if (isTRUE(dataset_profile$avoid_aggressive_k_reduction)) {
        paste0("; avoided aggressive k reduction for ", dataset_profile$reason)
      } else {
        ""
      }
    )
  )
}

auto_umap_optimizer_pilot_configs <- function(full_n,
                                              k,
                                              max_configs) {
  max_configs <- as.integer(max(1L, max_configs))
  base <- data.frame(
    k = as.integer(k),
    spectral_n_iter = auto_umap_default_pilot_spectral(k),
    n_epochs = auto_umap_pilot_quick_epochs(full_n),
    init_scale = auto_umap_default_pilot_init_scale(full_n),
    stringsAsFactors = FALSE
  )
  refined <- auto_umap_refined_pilot_configs(
    full_n = full_n,
    k = k,
    max_configs = max(0L, max_configs - 1L)
  )
  grid <- unique(rbind(base, refined))
  grid[seq_len(min(nrow(grid), max_configs)), , drop = FALSE]
}

auto_umap_rank_alternate_k_candidates <- function(candidates,
                                                  base_k,
                                                  dataset_profile = NULL,
                                                  max_candidates = 1L) {
  candidates <- unique(as.integer(candidates))
  candidates <- candidates[is.finite(candidates) & candidates >= 2L]
  candidates <- setdiff(candidates, as.integer(base_k))
  if (length(candidates) == 0L || max_candidates <= 0L) return(integer(0L))
  if (isTRUE(dataset_profile$avoid_aggressive_k_reduction)) {
    candidates <- candidates[candidates > base_k]
    ordered <- candidates[order(abs(candidates - base_k), candidates)]
  } else {
    ordered <- candidates[order(abs(candidates - base_k), candidates)]
  }
  as.integer(utils::head(ordered, max_candidates))
}

auto_umap_select_pilot_winner <- function(successful,
                                          fixed_k_best,
                                          labels_available = FALSE) {
  if (nrow(successful) == 0L) return(fixed_k_best)
  candidates <- successful[successful$k != fixed_k_best$k, , drop = FALSE]
  if (nrow(candidates) == 0L) return(fixed_k_best)
  candidates <- candidates[order(-candidates$pilot_score, candidates$elapsed), , drop = FALSE]
  best_alt <- candidates[1L, , drop = FALSE]
  if (!auto_umap_k_change_has_strong_evidence(fixed_k_best, best_alt, labels_available)) {
    return(fixed_k_best)
  }
  best_alt
}

auto_umap_k_change_has_strong_evidence <- function(base,
                                                   candidate,
                                                   labels_available = FALSE) {
  score_gain <- as.numeric(candidate$pilot_score) - as.numeric(base$pilot_score)
  if (!is.finite(score_gain) || score_gain < 0.02) return(FALSE)
  if (!auto_umap_metric_not_worse(candidate$knn_preservation_50, base$knn_preservation_50, tolerance = 0.005)) {
    return(FALSE)
  }
  if (!auto_umap_metric_not_worse(candidate$local_continuity_50, base$local_continuity_50, tolerance = 0.005)) {
    return(FALSE)
  }
  if (isTRUE(labels_available)) {
    if (!auto_umap_metric_not_worse(candidate$embedding_knn_accuracy, base$embedding_knn_accuracy, tolerance = 0.002)) {
      return(FALSE)
    }
    if (!auto_umap_metric_not_worse(candidate$rare_class_recall, base$rare_class_recall, tolerance = 0.01)) {
      return(FALSE)
    }
  }
  TRUE
}

auto_umap_metric_not_worse <- function(candidate, base, tolerance) {
  candidate <- as.numeric(candidate)
  base <- as.numeric(base)
  if (!is.finite(candidate) || !is.finite(base)) return(TRUE)
  candidate >= base - tolerance
}

auto_umap_pilot_cache_version <- function() {
  3L
}

auto_umap_pilot_cache_dir <- function(cache_dir = NULL) {
  if (!is.null(cache_dir) && nzchar(cache_dir)) {
    return(cache_dir)
  }
  opt <- getOption("fastEmbedR.pilot_cache_dir", NULL)
  if (!is.null(opt) && nzchar(opt)) {
    return(opt)
  }
  if ("R_user_dir" %in% getNamespaceExports("tools")) {
    return(file.path(tools::R_user_dir("fastEmbedR", "cache"), "pilot"))
  }
  file.path(tempdir(), "fastEmbedR_pilot_cache")
}

auto_umap_pilot_cache_key <- function(dataset_hash, k, seed) {
  paste0("hash", dataset_hash, "_k", as.integer(k), "_seed", as.integer(seed))
}

auto_umap_pilot_cache_path <- function(cache_dir,
                                       dataset_hash,
                                       k,
                                       seed,
                                       kind = "data") {
  key <- auto_umap_pilot_cache_key(dataset_hash, k, seed)
  file.path(cache_dir, paste0("umap_pilot_", kind, "_", key, ".rds"))
}

auto_umap_init_key <- function(init_scale) {
  init_scale <- as.numeric(init_scale)
  if (length(init_scale) != 1L || !is.finite(init_scale)) {
    return("NA")
  }
  format(signif(init_scale, 12L), scientific = FALSE, trim = TRUE)
}

auto_umap_read_pilot_cache_row <- function(cache_dir,
                                           dataset_hash,
                                           k,
                                           seed,
                                           spectral_n_iter,
                                           n_epochs,
                                           init_scale,
                                           kind = "data",
                                           use_cache = TRUE,
                                           force_recompute = FALSE) {
  if (!isTRUE(use_cache) || isTRUE(force_recompute)) return(NULL)
  path <- auto_umap_pilot_cache_path(cache_dir, dataset_hash, k, seed, kind = kind)
  if (!file.exists(path)) return(NULL)
  cached <- tryCatch(readRDS(path), error = function(e) NULL)
  if (!is.list(cached) || !identical(cached$version, auto_umap_pilot_cache_version())) {
    return(NULL)
  }
  rows <- cached$rows
  if (!is.data.frame(rows) || nrow(rows) == 0L) return(NULL)
  init_key <- auto_umap_init_key(init_scale)
  row_init <- vapply(rows$init_scale, auto_umap_init_key, character(1L))
  keep <- rows$spectral_n_iter == as.integer(spectral_n_iter) &
    rows$n_epochs == as.integer(n_epochs) &
    row_init == init_key &
    rows$status == "success"
  if (!any(keep, na.rm = TRUE)) return(NULL)
  out <- rows[which(keep)[1L], , drop = FALSE]
  out$cache_hit <- TRUE
  out
}

auto_umap_write_pilot_cache_row <- function(row,
                                            cache_dir,
                                            dataset_hash,
                                            k,
                                            seed,
                                            kind = "data",
                                            use_cache = TRUE) {
  if (!isTRUE(use_cache) || !is.data.frame(row) || nrow(row) != 1L) {
    return(invisible(FALSE))
  }
  dir.create(cache_dir, recursive = TRUE, showWarnings = FALSE)
  path <- auto_umap_pilot_cache_path(cache_dir, dataset_hash, k, seed, kind = kind)
  cached <- if (file.exists(path)) {
    tryCatch(readRDS(path), error = function(e) NULL)
  } else {
    NULL
  }
  rows <- if (is.list(cached) && identical(cached$version, auto_umap_pilot_cache_version()) &&
    is.data.frame(cached$rows)) {
    cached$rows
  } else {
    row[0L, , drop = FALSE]
  }
  init_key <- auto_umap_init_key(row$init_scale)
  if (nrow(rows) > 0L) {
    row_init <- vapply(rows$init_scale, auto_umap_init_key, character(1L))
    keep <- !(rows$spectral_n_iter == row$spectral_n_iter &
      rows$n_epochs == row$n_epochs &
      row_init == init_key)
    rows <- rows[keep, , drop = FALSE]
  }
  row$cache_hit <- FALSE
  rows <- rbind(rows, row)
  saveRDS(
    list(
      version = auto_umap_pilot_cache_version(),
      dataset_hash = dataset_hash,
      k = as.integer(k),
      seed = as.integer(seed),
      kind = kind,
      rows = rows
    ),
    path
  )
  invisible(TRUE)
}

auto_umap_dataset_hash <- function(x,
                                   labels = NULL,
                                   max_rows = 512L,
                                   max_cols = 64L) {
  x <- as.matrix(x)
  n <- nrow(x)
  p <- ncol(x)
  rows <- unique(as.integer(round(seq(1L, n, length.out = min(n, max_rows)))))
  cols <- unique(as.integer(round(seq(1L, p, length.out = min(p, max_cols)))))
  values <- if (length(rows) > 0L && length(cols) > 0L) {
    signif(as.numeric(x[rows, cols, drop = FALSE]), 12L)
  } else {
    numeric(0L)
  }
  label_part <- NULL
  if (!is.null(labels) && length(labels) == n) {
    label_part <- as.character(labels[rows])
  }
  auto_umap_raw_hash(serialize(
    list(
      dim = c(n, p),
      rows = rows,
      cols = cols,
      values = values,
      labels = label_part
    ),
    NULL,
    version = 2L
  ))
}

auto_umap_knn_hash <- function(indices,
                               distances,
                               max_rows = 512L,
                               max_cols = 64L) {
  indices <- as.matrix(indices)
  distances <- as.matrix(distances)
  n <- nrow(indices)
  p <- ncol(indices)
  rows <- unique(as.integer(round(seq(1L, n, length.out = min(n, max_rows)))))
  cols <- unique(as.integer(round(seq(1L, p, length.out = min(p, max_cols)))))
  auto_umap_raw_hash(serialize(
    list(
      dim = c(n, p),
      rows = rows,
      cols = cols,
      indices = as.integer(indices[rows, cols, drop = FALSE]),
      distances = signif(as.numeric(distances[rows, cols, drop = FALSE]), 12L)
    ),
    NULL,
    version = 2L
  ))
}

auto_umap_raw_hash <- function(raw) {
  bytes <- as.integer(raw)
  if (length(bytes) == 0L) return("0000000000000000")
  pos <- seq_along(bytes)
  h1 <- sum((bytes + 1) * ((pos %% 104729L) + 1L)) %% 2147483647
  h2 <- sum((bytes + 3) * (((pos * 13L) %% 130363L) + 1L)) %% 2147483647
  paste0(
    sprintf("%08x", as.integer(h1)),
    sprintf("%08x", as.integer(h2))
  )
}

auto_umap_dataset_profile <- function(x,
                                      labels = NULL,
                                      full_n = nrow(x)) {
  dims <- dim(x)
  n <- if (length(dims) >= 1L) as.integer(dims[1L]) else as.integer(full_n)
  p <- if (length(dims) >= 2L) as.integer(dims[2L]) else NA_integer_
  full_n <- as.integer(full_n)
  low_dimensional <- is.finite(p) && p <= 50L
  tabular_like <- is.finite(p) && p <= 128L

  n_labels <- 0L
  min_class_fraction <- NA_real_
  imbalance_ratio <- NA_real_
  highly_imbalanced <- FALSE
  if (!is.null(labels) && length(labels) == n) {
    labels <- as.factor(labels)
    tab <- table(labels)
    tab <- tab[tab > 0]
    n_labels <- length(tab)
    if (n_labels >= 2L) {
      counts <- as.numeric(tab)
      min_class_fraction <- min(counts) / sum(counts)
      imbalance_ratio <- max(counts) / max(1, min(counts))
      highly_imbalanced <- isTRUE(min_class_fraction < 0.05 || imbalance_ratio >= 20)
    }
  }

  avoid <- isTRUE(low_dimensional) || isTRUE(tabular_like) || isTRUE(highly_imbalanced)
  reasons <- c(
    if (isTRUE(low_dimensional)) paste0("low-dimensional p=", p),
    if (!isTRUE(low_dimensional) && isTRUE(tabular_like)) paste0("tabular-like p=", p),
    if (isTRUE(highly_imbalanced)) {
      paste0(
        "imbalanced labels min_fraction=",
        format(min_class_fraction, digits = 3),
        ", max_min_ratio=",
        format(imbalance_ratio, digits = 3)
      )
    }
  )
  list(
    n = n,
    p = p,
    full_n = full_n,
    low_dimensional = low_dimensional,
    tabular_like = tabular_like,
    n_label_levels = as.integer(n_labels),
    min_class_fraction = min_class_fraction,
    imbalance_ratio = imbalance_ratio,
    highly_imbalanced = highly_imbalanced,
    avoid_aggressive_k_reduction = avoid,
    reason = if (length(reasons) == 0L) "high-dimensional balanced data" else paste(reasons, collapse = ", ")
  )
}

auto_umap_pilot_size <- function(full_n,
                                 available_n,
                                 pilot_min_n,
                                 pilot_max_n) {
  full_n <- as.integer(full_n)
  available_n <- as.integer(available_n)
  pilot_min_n <- as.integer(pilot_min_n)
  pilot_max_n <- as.integer(pilot_max_n)
  if (full_n < pilot_min_n || available_n < 50L) {
    return(0L)
  }
  if (full_n <= pilot_max_n) {
    return(as.integer(min(available_n, full_n)))
  }
  as.integer(min(
    available_n,
    max(pilot_min_n, min(pilot_max_n, floor(full_n * 0.10)))
  ))
}

auto_pilot_sample_indices <- function(n,
                                      labels,
                                      sample_size,
                                      seed,
                                      rare_protect = FALSE) {
  sample_size <- as.integer(min(n, sample_size))
  if (sample_size >= n) return(seq_len(n))
  if (is.null(labels)) {
    return(sample_indices(n, sample_size, seed))
  }

  labels <- as.factor(labels)
  if (length(labels) != n || nlevels(labels) < 2L) {
    return(sample_indices(n, sample_size, seed))
  }
  groups <- split(seq_len(n), labels)
  set.seed(seed + 1409L)
  group_sizes <- lengths(groups)
  if (isTRUE(rare_protect)) {
    weights <- sqrt(group_sizes)
    quotas <- as.integer(pmax(1L, floor(weights / sum(weights) * sample_size)))
  } else {
    quotas <- vapply(groups, function(idx) {
      as.integer(max(1L, floor(length(idx) / n * sample_size)))
    }, integer(1))
  }
  quotas <- pmin(quotas, lengths(groups))
  while (sum(quotas) > sample_size) {
    can_drop <- which(quotas > 1L)
    if (length(can_drop) == 0L) break
    j <- can_drop[which.max(quotas[can_drop] / pmax(1L, group_sizes[can_drop]))]
    quotas[j] <- quotas[j] - 1L
  }
  while (sum(quotas) < sample_size) {
    can_add <- which(quotas < lengths(groups))
    if (length(can_add) == 0L) break
    j <- can_add[which.max(lengths(groups)[can_add] - quotas[can_add])]
    quotas[j] <- quotas[j] + 1L
  }
  selected <- unlist(
    Map(function(idx, q) {
      if (q >= length(idx)) idx else sample(idx, q)
    }, groups, quotas),
    use.names = FALSE
  )
  sort(as.integer(selected))
}

auto_umap_pilot_k_candidates <- function(full_n,
                                         pilot_n,
                                         base_k,
                                         dataset_profile = NULL) {
  base_k <- as.integer(base_k)
  avoid_reduction <- isTRUE(dataset_profile$avoid_aggressive_k_reduction)
  if (isTRUE(avoid_reduction)) {
    upper <- min(150L, pilot_n - 1L)
    candidates <- unique(as.integer(c(
      base_k,
      if (base_k < 50L) 50L else NA_integer_,
      ceiling(base_k * 1.5),
      min(upper, max(base_k, 2L * base_k))
    )))
  } else {
    candidates <- if (full_n >= 10000L) {
      c(base_k, 15L, 30L, 50L)
    } else {
      c(base_k, 15L, 30L)
    }
  }
  candidates <- unique(as.integer(candidates))
  candidates <- candidates[is.finite(candidates) & candidates >= 2L & candidates < pilot_n]
  if (isTRUE(avoid_reduction)) {
    candidates <- candidates[candidates >= min(base_k, pilot_n - 1L)]
  }
  candidates
}

auto_umap_pilot_knn_backend <- function(knn_backend,
                                        pilot_n,
                                        p,
                                        k) {
  if (identical(knn_backend, "cuda") || identical(knn_backend, "metal")) {
    return(knn_backend)
  }
  work_size <- as.double(pilot_n) * as.double(pilot_n) * as.double(p)
  if (pilot_n >= 5000L && k >= 10L && work_size >= 5e8) {
    return("auto")
  }
  "cpu"
}

auto_umap_default_pilot_spectral <- function(k) {
  as.integer(if (k <= 15L) 20L else 10L)
}

auto_umap_pilot_quick_epochs <- function(full_n) {
  if (full_n >= 10000L) 60L else 120L
}

auto_umap_default_pilot_init_scale <- function(full_n) {
  if (full_n >= 10000L) 10 else NA_real_
}

auto_umap_refined_pilot_configs <- function(full_n,
                                           k,
                                           max_configs) {
  if (max_configs <= 0L) {
    return(data.frame())
  }
  spectral <- unique(as.integer(c(
    auto_umap_default_pilot_spectral(k),
    if (k <= 15L) 30L else 20L
  )))
  epochs <- if (full_n >= 10000L) c(80L, 100L) else c(150L, 300L)
  init_scale <- if (full_n >= 10000L) c(10, 5) else c(NA_real_, 5)
  grid <- expand.grid(
    k = as.integer(k),
    spectral_n_iter = spectral,
    n_epochs = as.integer(epochs),
    init_scale = init_scale,
    KEEP.OUT.ATTRS = FALSE,
    stringsAsFactors = FALSE
  )
  grid <- unique(grid)
  grid[seq_len(min(nrow(grid), max_configs)), , drop = FALSE]
}

auto_umap_failed_pilot_row <- function(stage,
                                       k,
                                       spectral_n_iter,
                                       n_epochs,
                                       init_scale,
                                       elapsed,
                                       error) {
  data.frame(
    stage = stage,
    k = as.integer(k),
    spectral_n_iter = as.integer(spectral_n_iter),
    n_epochs = as.integer(n_epochs),
    init_scale = as.numeric(init_scale),
    elapsed = as.numeric(elapsed),
    local_trustworthiness = NA_real_,
    knn_preservation = NA_real_,
    local_continuity = NA_real_,
    knn_preservation_50 = NA_real_,
    local_continuity_50 = NA_real_,
    embedding_knn_accuracy = NA_real_,
    rare_class_recall = NA_real_,
    structure_score = NA_real_,
    pilot_score = NA_real_,
    low_dimensional = NA,
    tabular_like = NA,
    highly_imbalanced = NA,
    avoid_aggressive_k_reduction = NA,
    cache_hit = FALSE,
    status = "failed",
    error = error,
    stringsAsFactors = FALSE
  )
}

auto_umap_pilot_label_scores <- function(layout,
                                         labels,
                                         keep,
                                         k,
                                         seed = 4L) {
  if (is.null(labels)) {
    return(list(label_knn_accuracy = NA_real_, rare_class_recall = NA_real_))
  }
  labels <- as.factor(labels)
  if (nlevels(labels) < 2L || length(labels) != nrow(layout)) {
    return(list(label_knn_accuracy = NA_real_, rare_class_recall = NA_real_))
  }
  k <- as.integer(min(max(1L, k), nrow(layout) - 1L))
  if (k < 1L) {
    return(list(label_knn_accuracy = NA_real_, rare_class_recall = NA_real_))
  }
  keep <- as.integer(keep)
  keep <- keep[is.finite(keep) & keep >= 1L & keep <= nrow(layout)]
  if (length(keep) == 0L) {
    keep <- seq_len(nrow(layout))
  }
  embed_nn <- tryCatch(
    fastEmbedR::nn(layout, layout, k + 1L, backend = "cpu"),
    error = function(e) NULL
  )
  if (is.null(embed_nn)) {
    return(list(label_knn_accuracy = NA_real_, rare_class_recall = NA_real_))
  }
  embed_indices <- embed_nn$indices[, -1L, drop = FALSE]
  pred <- classification_from_embedding_nn(embed_indices, labels, k)
  acc <- mean(pred[keep] == labels[keep], na.rm = TRUE)
  recalls <- class_recall_metrics(labels[keep], pred[keep])
  list(
    label_knn_accuracy = acc,
    rare_class_recall = recalls$rare_class_recall
  )
}

auto_umap_pilot_score <- function(structure,
                                  elapsed,
                                  label_scores = NULL,
                                  proxy_structure = NULL,
                                  dataset_profile = NULL,
                                  k = NA_integer_) {
  trust <- unname(structure["local_trustworthiness"])
  preserve <- unname(structure["knn_preservation"])
  continuity <- unname(structure["local_continuity"])
  if (is.null(proxy_structure)) {
    proxy_structure <- structure
  }
  preserve50 <- unname(proxy_structure["knn_preservation"])
  continuity50 <- unname(proxy_structure["local_continuity"])
  label_acc <- if (is.null(label_scores)) NA_real_ else label_scores$label_knn_accuracy
  rare_recall <- if (is.null(label_scores)) NA_real_ else label_scores$rare_class_recall

  has_labels <- is.finite(label_acc) || is.finite(rare_recall)
  if (isTRUE(has_labels)) {
    values <- c(
      trust = trust,
      preserve = preserve,
      continuity = continuity,
      label = label_acc,
      rare = rare_recall
    )
    weights <- c(trust = 0.25, preserve = 0.20, continuity = 0.15, label = 0.25, rare = 0.15)
  } else {
    values <- c(
      trust = trust,
      preserve50 = preserve50,
      continuity50 = continuity50,
      preserve = preserve
    )
    weights <- c(trust = 0.25, preserve50 = 0.35, continuity50 = 0.25, preserve = 0.15)
  }
  finite <- is.finite(values)
  if (!any(finite)) return(NA_real_)
  quality <- sum(values[finite] * weights[finite]) / sum(weights[finite])
  if (isTRUE(dataset_profile$avoid_aggressive_k_reduction)) {
    base <- auto_embedding_k(dataset_profile$full_n, "umap", include_self = FALSE)
    if (is.finite(k) && k < base) {
      quality <- quality - 0.03 * (1 - as.numeric(k) / max(1, base))
    }
  }
  runtime_penalty <- if (is.finite(elapsed)) 0.01 * log1p(as.numeric(elapsed)) else 0
  quality - runtime_penalty
}

auto_umap_pilot_refinement_strength <- function(best) {
  trust <- as.numeric(best$local_trustworthiness)
  preserve <- as.numeric(best$knn_preservation)
  if (is.finite(trust) && is.finite(preserve) && trust >= 0.96 && preserve >= 0.80) {
    return(0.5)
  }
  if (is.finite(trust) && trust < 0.90) {
    return(1.5)
  }
  if (is.finite(preserve) && preserve < 0.55) {
    return(1.5)
  }
  1
}

auto_umap_knn_pilot_tune <- function(indices,
                                     distances,
                                     seed = 4L,
                                     full_n = nrow(indices),
                                     pilot_min_n = 2000L,
                                     pilot_max_n = 5000L,
                                     pilot_max_configs = 6L,
                                     use_cache = TRUE,
                                     cache_dir = NULL,
                                     force_recompute = FALSE) {
  indices <- as.matrix(indices)
  distances <- as.matrix(distances)
  if (!is.integer(indices)) storage.mode(indices) <- "integer"
  if (!identical(typeof(distances), "double")) storage.mode(distances) <- "double"
  full_n <- as.integer(full_n)
  if (full_n < pilot_min_n || ncol(indices) < 2L) {
    return(list(
      status = "skipped",
      reason = "KNN pilot auto-tune skipped below threshold or with too few neighbors"
    ))
  }

  pilot_n <- auto_umap_pilot_size(
    full_n = full_n,
    available_n = nrow(indices),
    pilot_min_n = pilot_min_n,
    pilot_max_n = pilot_max_n
  )
  if (pilot_n < 50L) {
    return(list(
      status = "skipped",
      reason = "KNN pilot auto-tune skipped below minimum pilot size"
    ))
  }

  supplied_k <- as.integer(ncol(indices))
  k_candidates <- auto_umap_knn_pilot_k_candidates(
    full_n = full_n,
    pilot_n = pilot_n,
    supplied_k = supplied_k
  )
  if (length(k_candidates) == 0L) {
    return(list(
      status = "skipped",
      reason = "KNN pilot auto-tune skipped because no valid k candidates were available"
    ))
  }

  rows <- auto_knn_pilot_rows(indices, pilot_n, seed)
  subgraph <- auto_knn_pilot_subgraph(
    indices = indices,
    distances = distances,
    rows = rows,
    max_k = max(k_candidates)
  )
  if (nrow(subgraph$indices) < 50L) {
    return(list(
      status = "skipped",
      reason = "KNN pilot auto-tune skipped because the sampled graph was too small"
    ))
  }

  score_keep <- sample_indices(
    nrow(subgraph$indices),
    min(1000L, max(50L, nrow(subgraph$indices))),
    seed + 307L
  )
  graph_hash <- auto_umap_knn_hash(subgraph$indices, subgraph$distances)
  cache_dir <- auto_umap_pilot_cache_dir(cache_dir)
  run_config <- function(k, spectral_n_iter, n_epochs, init_scale, stage) {
    cached <- auto_umap_read_pilot_cache_row(
      cache_dir = cache_dir,
      dataset_hash = graph_hash,
      k = k,
      seed = seed,
      spectral_n_iter = spectral_n_iter,
      n_epochs = n_epochs,
      init_scale = init_scale,
      kind = "knn",
      use_cache = use_cache,
      force_recompute = force_recompute
    )
    if (!is.null(cached)) {
      cached$stage <- stage
      cached$cache_hit <- TRUE
      return(cached)
    }
    idx <- subgraph$indices[, seq_len(k), drop = FALSE]
    dst <- subgraph$distances[, seq_len(k), drop = FALSE]
    elapsed <- NA_real_
    layout <- NULL
    err <- NA_character_
    elapsed <- system.time({
      layout <- tryCatch(
        fast_knn_umap_core(
          idx,
          dst,
          n_components = 2L,
          seed = seed,
          verbose = FALSE,
          backend = "cpu",
          config_override = list(
            n_epochs = as.integer(n_epochs),
            spectral_n_iter = as.integer(spectral_n_iter),
            init_scale = init_scale,
            tuning_source = "knn_pilot"
          )
        ),
        error = function(e) {
          err <<- conditionMessage(e)
          NULL
        }
      )
    })["elapsed"]
    if (is.null(layout)) {
      return(auto_umap_failed_pilot_row(
        stage, k, spectral_n_iter, n_epochs, init_scale, elapsed, err
      ))
    }
    structure <- tryCatch(
      knn_structure_score_cpp(
        layout,
        idx,
        score_keep,
        as.integer(min(50L, k)),
        integer(0L),
        0L
      ),
      error = function(e) {
        err <<- conditionMessage(e)
        rep(NA_real_, 5L)
      }
    )
    score <- auto_umap_pilot_score(structure, elapsed)
    row <- data.frame(
      stage = stage,
      k = as.integer(k),
      spectral_n_iter = as.integer(spectral_n_iter),
      n_epochs = as.integer(n_epochs),
      init_scale = as.numeric(init_scale),
      elapsed = as.numeric(elapsed),
      local_trustworthiness = unname(structure["local_trustworthiness"]),
      knn_preservation = unname(structure["knn_preservation"]),
      local_continuity = unname(structure["local_continuity"]),
      knn_preservation_50 = unname(structure["knn_preservation"]),
      local_continuity_50 = unname(structure["local_continuity"]),
      embedding_knn_accuracy = unname(structure["embedding_knn_accuracy"]),
      rare_class_recall = NA_real_,
      structure_score = unname(structure["structure_score"]),
      pilot_score = as.numeric(score),
      low_dimensional = NA,
      tabular_like = NA,
      highly_imbalanced = NA,
      avoid_aggressive_k_reduction = NA,
      cache_hit = FALSE,
      status = "success",
      error = NA_character_,
      stringsAsFactors = FALSE
    )
    auto_umap_write_pilot_cache_row(
      row,
      cache_dir = cache_dir,
      dataset_hash = graph_hash,
      k = k,
      seed = seed,
      kind = "knn",
      use_cache = use_cache
    )
    row
  }

  base_k <- as.integer(k_candidates[1L])
  config_rows <- auto_umap_optimizer_pilot_configs(
    full_n = full_n,
    k = base_k,
    max_configs = pilot_max_configs
  )
  quick_rows <- lapply(seq_len(nrow(config_rows)), function(i) {
    run_config(
      k = config_rows$k[i],
      spectral_n_iter = config_rows$spectral_n_iter[i],
      n_epochs = config_rows$n_epochs[i],
      init_scale = config_rows$init_scale[i],
      stage = "config"
    )
  })
  scores <- do.call(rbind, quick_rows)
  successful <- scores[is.finite(scores$pilot_score), , drop = FALSE]
  if (nrow(successful) == 0L) {
    return(list(
      status = "failed",
      reason = "KNN pilot auto-tune failed for all fixed-k optimizer candidates",
      scores = scores
    ))
  }

  best <- successful[order(-successful$pilot_score, successful$elapsed, successful$k), , drop = FALSE][1L, ]
  config_override <- list(
    n_neighbors = as.integer(best$k),
    n_epochs = as.integer(best$n_epochs),
    spectral_n_iter = as.integer(best$spectral_n_iter),
    init_scale = as.numeric(best$init_scale),
    tuning_source = "knn_pilot",
    pilot_sample_n = as.integer(nrow(subgraph$indices)),
    pilot_score = as.numeric(best$pilot_score),
    pilot_cache_key = auto_umap_pilot_cache_key(graph_hash, best$k, seed),
    pilot_cache_hit = any(scores$cache_hit %in% TRUE, na.rm = TRUE)
  )
  list(
    status = "success",
    n_neighbors = as.integer(best$k),
    spectral_n_iter = as.integer(best$spectral_n_iter),
    n_epochs = as.integer(best$n_epochs),
    init_scale = as.numeric(best$init_scale),
    selected_score = as.numeric(best$pilot_score),
    pilot_sample_n = as.integer(nrow(subgraph$indices)),
    config_override = config_override,
    scores = scores,
    reason = paste0(
      "KNN pilot auto-tune sampled ", nrow(subgraph$indices),
      " graph rows, kept supplied k=", supplied_k,
      ", spectral_n_iter=", best$spectral_n_iter,
      ", epochs=", best$n_epochs,
      ", init_scale=", format(best$init_scale, trim = TRUE),
      if (any(scores$cache_hit %in% TRUE, na.rm = TRUE)) {
        "; reused cached pilot score(s)"
      } else {
        ""
      }
    )
  )
}

auto_umap_knn_pilot_k_candidates <- function(full_n,
                                            pilot_n,
                                            supplied_k) {
  candidates <- supplied_k
  candidates <- unique(as.integer(candidates))
  candidates <- candidates[is.finite(candidates) & candidates >= 2L]
  candidates <- candidates[candidates <= supplied_k & candidates < pilot_n]
  candidates
}

auto_knn_indices_one_based <- function(indices) {
  if (identical(knn_index_base(indices, nrow(indices)), "zero")) {
    indices + 1L
  } else {
    indices
  }
}

auto_knn_pilot_rows <- function(indices,
                                sample_size,
                                seed) {
  n <- nrow(indices)
  sample_size <- as.integer(min(max(1L, sample_size), n))
  if (sample_size >= n) return(seq_len(n))
  idx <- auto_knn_indices_one_based(indices)
  selected <- rep(FALSE, n)
  rows <- integer(0L)
  set.seed(seed + 1709L)
  queue <- sample.int(n, min(32L, n))
  cursor <- 1L
  while (length(rows) < sample_size) {
    if (cursor > length(queue)) {
      remaining <- which(!selected)
      if (length(remaining) == 0L) break
      queue <- c(queue, sample(remaining, 1L))
    }
    row <- queue[cursor]
    cursor <- cursor + 1L
    if (selected[row]) next
    selected[row] <- TRUE
    rows <- c(rows, row)
    nb <- idx[row, ]
    nb <- nb[is.finite(nb) & nb >= 1L & nb <= n & !selected[nb]]
    if (length(nb) > 0L) {
      nb <- sample(unique(as.integer(nb)), min(length(unique(nb)), 12L))
      queue <- c(queue, nb)
    }
  }
  sort(as.integer(rows[seq_len(min(length(rows), sample_size))]))
}

auto_knn_pilot_subgraph <- function(indices,
                                    distances,
                                    rows,
                                    max_k) {
  rows <- sort(unique(as.integer(rows)))
  n_sub <- length(rows)
  max_k <- as.integer(min(max_k, n_sub - 1L, ncol(indices)))
  idx <- auto_knn_indices_one_based(indices)
  map <- integer(nrow(indices))
  map[rows] <- seq_along(rows)
  out_idx <- matrix(0L, n_sub, max_k)
  out_dst <- matrix(0, n_sub, max_k)
  storage.mode(out_idx) <- "integer"
  storage.mode(out_dst) <- "double"
  finite_dist <- distances[is.finite(distances) & distances >= 0]
  fallback_distance <- if (length(finite_dist) == 0L) 1 else stats::median(finite_dist)
  if (!is.finite(fallback_distance) || fallback_distance <= 0) fallback_distance <- 1

  for (local_i in seq_len(n_sub)) {
    global_i <- rows[local_i]
    row_idx <- idx[global_i, seq_len(min(ncol(idx), max_k)), drop = TRUE]
    row_dst <- distances[global_i, seq_len(min(ncol(distances), max_k)), drop = TRUE]
    valid_row_idx <- is.finite(row_idx) & row_idx >= 1L & row_idx <= nrow(indices)
    row_idx <- row_idx[valid_row_idx]
    row_dst <- row_dst[valid_row_idx]
    local_nb <- map[row_idx]
    keep <- local_nb > 0L & local_nb != local_i & is.finite(row_dst) & row_dst >= 0
    local_nb <- local_nb[keep]
    row_dst <- row_dst[keep]
    if (length(local_nb) > 0L) {
      dedup <- !duplicated(local_nb)
      local_nb <- local_nb[dedup]
      row_dst <- row_dst[dedup]
    }
    if (length(local_nb) < max_k) {
      fill <- setdiff(seq_len(n_sub), c(local_i, local_nb))
      need <- max_k - length(local_nb)
      fill <- fill[seq_len(min(need, length(fill)))]
      fill_distance <- if (length(row_dst) > 0L && is.finite(max(row_dst))) {
        max(row_dst) * 1.25
      } else {
        fallback_distance * 1.25
      }
      local_nb <- c(local_nb, fill)
      row_dst <- c(row_dst, rep(fill_distance, length(fill)))
    }
    out_idx[local_i, ] <- as.integer(local_nb[seq_len(max_k)])
    out_dst[local_i, ] <- as.numeric(row_dst[seq_len(max_k)])
  }
  list(indices = out_idx, distances = out_dst)
}

auto_select_landmark_policy <- function(method,
                                        mode,
                                        landmarks,
                                        full_n,
                                        nn_supplied) {
  if (isTRUE(nn_supplied)) {
    return(list(
      landmarks = landmarks,
      refinement = "none",
      scores = data.frame(),
      score = NA_real_,
      auto_tuned = FALSE,
      reason = "auto landmarking skipped for supplied KNN"
    ))
  }

  if (!is.null(landmarks) && identical(landmarks, FALSE)) {
    return(list(
      landmarks = landmarks,
      refinement = "none",
      scores = data.frame(),
      score = NA_real_,
      auto_tuned = FALSE,
      reason = "landmarks disabled by user"
    ))
  }

  if (!identical(mode, "auto")) {
    stop("Only `mode = \"auto\"` is supported. Use `landmarks` to request landmarking explicitly.", call. = FALSE)
  }

  if (is.null(landmarks) && full_n < auto_landmark_threshold(method)) {
    return(list(
      landmarks = NULL,
      refinement = "none",
      scores = data.frame(),
      score = NA_real_,
      auto_tuned = TRUE,
      reason = "auto selected full embedding below landmark threshold"
    ))
  }

  list(
    landmarks = if (is.null(landmarks)) TRUE else landmarks,
    refinement = "bucketed",
    scores = data.frame(),
    score = NA_real_,
    auto_tuned = TRUE,
    reason = "auto selected landmarking with bucketed refinement"
  )
}

auto_landmark_threshold <- function(method) {
  if (!identical(method, "umap")) {
    stop("Only UMAP is supported.", call. = FALSE)
  }
  2500L
}

auto_embedding_epoch_count <- function(method,
                                       n,
                                       k,
                                       backend,
                                       quality) {
  if (identical(method, "umap")) {
    return(as.integer(fast_knn_umap_config(n, k, backend)$n_epochs))
  }
  stop("Only UMAP is supported.", call. = FALSE)
}

auto_refinement_epoch_count <- function(n,
                                        refinement,
                                        method) {
  landmark_refinement_epoch_count(n, refinement, method)
}

append_auto_reason <- function(current, message) {
  values <- c(current, message)
  values <- values[!is.na(values) & nzchar(values)]
  if (length(values) == 0L) NA_character_ else paste(unique(values), collapse = "; ")
}

validate_epoch_count <- function(n_epochs) {
  n_epochs <- as.integer(n_epochs)
  if (length(n_epochs) != 1L || is.na(n_epochs) || !is.finite(n_epochs) || n_epochs < 1L) {
    stop("`n_epochs` must be a positive integer.", call. = FALSE)
  }
  n_epochs
}
