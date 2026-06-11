#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(fastEmbedR)
})

arg_value <- function(name, default) {
  args <- commandArgs(trailingOnly = TRUE)
  prefix <- paste0("--", name, "=")
  hit <- args[startsWith(args, prefix)]
  if (length(hit) == 0L) default else sub(prefix, "", hit[[1L]], fixed = TRUE)
}

as_int <- function(name, default) as.integer(as.numeric(arg_value(name, as.character(default))))

safe_chr <- function(x) {
  if (is.null(x) || length(x) == 0L) NA_character_ else as.character(x[[1L]])
}

load_cached_dataset <- function(path, n = NULL, p = NULL) {
  obj <- readRDS(path)
  x <- if (is.list(obj) && !is.null(obj$x)) obj$x else obj
  if (!is.matrix(x)) x <- as.matrix(x)
  storage.mode(x) <- "double"
  if (!is.null(n) && is.finite(n) && nrow(x) > n) {
    x <- x[seq_len(n), , drop = FALSE]
  }
  if (!is.null(p) && is.finite(p) && ncol(x) > p) {
    x <- x[, seq_len(p), drop = FALSE]
  }
  x
}

strip_self <- function(indices, query_rows, k) {
  out <- matrix(NA_integer_, nrow(indices), k)
  for (i in seq_len(nrow(indices))) {
    idx <- indices[i, indices[i, ] != query_rows[[i]]]
    out[i, ] <- idx[seq_len(k)]
  }
  out
}

recall_against_exact_subset <- function(x, knn, k, sample_size, seed) {
  set.seed(seed + 1009L)
  rows <- sort(sample.int(nrow(x), min(sample_size, nrow(x))))
  exact <- fastEmbedR::nn(
    x,
    x[rows, , drop = FALSE],
    k = k + 1L,
    backend = if (fastEmbedR::metal_available()) "metal" else "cpu",
    n_threads = 4L
  )
  exact_idx <- strip_self(exact$indices, rows, k)
  idx <- knn$indices[rows, , drop = FALSE]
  if (all(idx[, 1L] == rows)) idx <- idx[, -1L, drop = FALSE]
  idx <- idx[, seq_len(k), drop = FALSE]
  fastEmbedR:::knn_recall(list(indices = idx), list(indices = exact_idx), k = k)
}

time_expr <- function(expr) {
  gc(FALSE)
  value <- NULL
  elapsed <- system.time({
    value <- force(expr)
  })[["elapsed"]]
  list(value = value, elapsed = as.numeric(elapsed))
}

run_fastembedr_nn <- function(x, backend, k, n_threads, option_values = list()) {
  old_options <- if (length(option_values)) do.call(base::options, option_values) else NULL
  if (!is.null(old_options)) on.exit(options(old_options), add = TRUE)
  time_expr(fastEmbedR::nn(
    x,
    k = k + 1L,
    backend = backend,
    n_threads = n_threads
  ))
}

run_uwot_annoy_nn <- function(x, k, n_threads) {
  if (!requireNamespace("uwot", quietly = TRUE)) {
    stop("Package `uwot` is not installed.", call. = FALSE)
  }
  timed <- time_expr(uwot:::annoy_nn(
    x,
    k = k + 1L,
    metric = "euclidean",
    n_trees = 50L,
    search_k = 2L * (k + 1L) * 50L,
    n_threads = n_threads,
    verbose = FALSE
  ))
  timed$value <- list(indices = timed$value$idx, distances = timed$value$dist)
  timed
}

result_row <- function(dataset, variant, x, method, backend, n_threads,
                       elapsed, recall, status, error_message, knn = NULL,
                       paper_idea = NA_character_) {
  approximation <- if (!is.null(knn)) attr(knn, "approximation") else NULL
  data.frame(
    dataset = dataset,
    variant = variant,
    n = nrow(x),
    p = ncol(x),
    method = method,
    backend = backend,
    n_threads = n_threads,
    paper_idea = paper_idea,
    sec = elapsed,
    recall_at_k = if (is.data.frame(recall)) recall$recall_at_k[[1L]] else NA_real_,
    median_recall_at_k = if (is.data.frame(recall)) recall$median_recall_at_k[[1L]] else NA_real_,
    strategy = safe_chr(if (is.list(approximation)) approximation$strategy else NA_character_),
    grid_dims = suppressWarnings(as.integer(safe_chr(if (is.list(approximation)) approximation$grid_dims else NA))),
    grid_bins = suppressWarnings(as.integer(safe_chr(if (is.list(approximation)) approximation$bins_per_dim else NA))),
    grid_radius = suppressWarnings(as.integer(safe_chr(if (is.list(approximation)) approximation$radius else NA))),
    status = status,
    error_message = error_message,
    stringsAsFactors = FALSE
  )
}

main <- function() {
  n_threads <- as_int("n-threads", 4L)
  k <- as_int("k", 30L)
  sample_size <- as_int("recall-sample", 512L)
  n_limit <- as_int("n", NA_integer_)
  out_dir <- arg_value("out-dir", "/Users/stefano/Documents/fastEmbedR-results/gpu_nn_paper_ideas")
  dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

  default_paths <- c(
    fashion = "/Users/stefano/Documents/fastEmbedR-results/nn_umap_preprocessed_seed6/preprocessed/fashion_mnist_70000_pca50_seed6.rds",
    mnist = "/Users/stefano/Documents/fastEmbedR-results/nn_umap_preprocessed_seed6/preprocessed/mnist_idx_70000_pca50_seed6.rds"
  )
  paths <- default_paths[file.exists(default_paths)]
  if (length(paths) == 0L) {
    stop("No cached PCA benchmark datasets were found.", call. = FALSE)
  }

  configs <- list(
    list(
      method = "fastEmbedR_cpu_annoy",
      backend = "cpu_annoy",
      n_threads = n_threads,
      paper_idea = "annoy_cpu_reference",
      options = list()
    ),
    list(
      method = "fastEmbedR_cpu_ivf",
      backend = "cpu_ivf",
      n_threads = n_threads,
      paper_idea = "ivf_flat_cpu_reference",
      options = list()
    )
  )
  if (fastEmbedR::metal_available()) {
    configs <- c(configs, list(
      list(
        method = "fastEmbedR_metal_exact",
        backend = "metal",
        n_threads = NA_integer_,
        paper_idea = "exact_gpu_reference",
        options = list()
      ),
      list(
        method = "fastEmbedR_metal_ivf",
        backend = "metal_ivf",
        n_threads = NA_integer_,
        paper_idea = "ivf_flat_gpu_proxy_for_ivf_rabitq",
        options = list()
      ),
      list(
        method = "fastEmbedR_metal_nndescent",
        backend = "metal_nndescent",
        n_threads = NA_integer_,
        paper_idea = "mlx_vis_seeded_nndescent_refinement",
        options = list(
          fastEmbedR.metal_nndescent_iters = 1L,
          fastEmbedR.metal_nndescent_sources = 10L,
          fastEmbedR.metal_nndescent_neighbors = 15L
        )
      ),
      list(
        method = "fastEmbedR_metal_grid_r1",
        backend = "metal_grid",
        n_threads = NA_integer_,
        paper_idea = "fastgraph_grid_bins",
        options = list(fastEmbedR.grid_radius = 1L, fastEmbedR.grid_dims = 5L)
      ),
      list(
        method = "fastEmbedR_metal_grid_r2",
        backend = "metal_grid",
        n_threads = NA_integer_,
        paper_idea = "fastgraph_grid_bins",
        options = list(fastEmbedR.grid_radius = 2L, fastEmbedR.grid_dims = 5L)
      )
    ))
  }

  rows <- list()
  for (dataset in names(paths)) {
    for (variant in c("pca50", "pca5")) {
      x <- load_cached_dataset(
        paths[[dataset]],
        n = n_limit,
        p = if (identical(variant, "pca5")) 5L else NA_integer_
      )
      message("Dataset ", dataset, " ", variant, ": ", nrow(x), " x ", ncol(x))

      for (cfg in configs) {
        result <- tryCatch(
          run_fastembedr_nn(x, cfg$backend, k, cfg$n_threads, cfg$options),
          error = identity
        )
        if (inherits(result, "error")) {
          row <- result_row(
            dataset, variant, x, cfg$method, cfg$backend, cfg$n_threads,
            NA_real_, NULL, "failed", conditionMessage(result),
            paper_idea = cfg$paper_idea
          )
        } else {
          recall <- recall_against_exact_subset(x, result$value, k, sample_size, 7L)
          row <- result_row(
            dataset, variant, x, cfg$method,
            safe_chr(attr(result$value, "backend")),
            cfg$n_threads, result$elapsed, recall, "success", NA_character_,
            knn = result$value,
            paper_idea = cfg$paper_idea
          )
        }
        print(row)
        rows[[length(rows) + 1L]] <- row
      }

      uwot_result <- tryCatch(run_uwot_annoy_nn(x, k, n_threads), error = identity)
      if (inherits(uwot_result, "error")) {
        row <- result_row(
          dataset, variant, x, "uwot_internal_annoy_nn", "uwot_annoy",
          n_threads, NA_real_, NULL, "failed", conditionMessage(uwot_result),
          paper_idea = "uwot_annoy_reference"
        )
      } else {
        recall <- recall_against_exact_subset(x, uwot_result$value, k, sample_size, 7L)
        row <- result_row(
          dataset, variant, x, "uwot_internal_annoy_nn", "uwot_annoy",
          n_threads, uwot_result$elapsed, recall, "success", NA_character_,
          knn = uwot_result$value,
          paper_idea = "uwot_annoy_reference"
        )
      }
      print(row)
      rows[[length(rows) + 1L]] <- row
    }
  }

  out <- do.call(rbind, rows)
  stamp <- format(Sys.time(), "%Y%m%d_%H%M%S")
  write.csv(out, file.path(out_dir, paste0("gpu_nn_paper_ideas_", stamp, ".csv")), row.names = FALSE)
  write.csv(out, file.path(out_dir, "latest_gpu_nn_paper_ideas.csv"), row.names = FALSE)
  invisible(out)
}

main()
